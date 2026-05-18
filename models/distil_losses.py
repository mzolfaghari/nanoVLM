"""
Knowledge Distillation loss registry for nanoVLM.

All loss functions share the same signature:

    loss_fn(
        student_logits: Tensor[B, T, V],   # float32, base vocab only
        teacher_logits: Tensor[B, T, V],   # float32, base vocab only
        answer_mask:    Tensor[B, T],       # float32 or bool, 1=answer token
        **kwargs,                           # method-specific hyperparams
    ) -> scalar Tensor

Logits are ALREADY truncated to base_vocab_size (49 152) before being passed
here — callers in distill_train.py are responsible for that slice.

Loss registry:
    "fkl"  — Forward KL   KL(teacher || student)      standard KD
    "rkl"  — Reverse KL   KL(student || teacher)      mode-seeking
    "js"   — Jensen-Shannon divergence                 symmetric
    "tvd"  — Total Variation distance                  L1 in prob space
    "taid" — Temperature-Annealing Interpolation KD   requires full logit vec
    "dkd"  — Decoupled KD (TCKD + NCKD)              requires full logit vec + labels

Weighting registry:
    "equal"           — uniform average of all loss terms
    "heteroscedastic" — learnable log-variance per loss term (Kendall et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Shared utility
# ──────────────────────────────────────────────────────────────────────────────

def _masked_mean(loss_per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Average loss over positions where mask == 1.
    loss_per_token: [B, T]
    mask:           [B, T]  (float or bool)
    """
    mask = mask.float()
    denom = mask.sum() + 1e-8
    return (loss_per_token * mask).sum() / denom


# ──────────────────────────────────────────────────────────────────────────────
# Individual loss functions
# ──────────────────────────────────────────────────────────────────────────────

def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Forward KL: KL(teacher || student)  — standard knowledge distillation.
    Minimising this makes the student cover all modes the teacher assigns
    probability to (mean-seeking behaviour).
    """
    T = temperature
    log_p = F.log_softmax(student_logits / T, dim=-1)   # student log-probs
    q     = F.softmax(teacher_logits    / T, dim=-1)    # teacher probs
    # kl_div expects (log_input, target); reduction="none" -> [B, T, V]
    per_token = F.kl_div(log_p, q, reduction="none").sum(-1)  # [B, T]
    return _masked_mean(per_token, answer_mask) * (T ** 2)


def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Reverse KL: KL(student || teacher) — mode-seeking.
    Student concentrates on one mode of the teacher; penalises hallucination.
    """
    T = temperature
    log_p = F.log_softmax(student_logits / T, dim=-1)
    p     = F.softmax(student_logits    / T, dim=-1)
    log_q = F.log_softmax(teacher_logits / T, dim=-1)
    per_token = (p * (log_p - log_q)).sum(-1)           # [B, T]
    return _masked_mean(per_token, answer_mask) * (T ** 2)


def js_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Jensen-Shannon divergence — symmetric, bounded in [0, log 2].
    JS(p, q) = 0.5 * KL(p||m) + 0.5 * KL(q||m),  m = 0.5*(p+q)
    """
    T = temperature
    p = F.softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits / T, dim=-1)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-8))
    per_token = 0.5 * (
        F.kl_div(log_m, p, reduction="none").sum(-1) +
        F.kl_div(log_m, q, reduction="none").sum(-1)
    )   # [B, T]
    return _masked_mean(per_token, answer_mask) * (T ** 2)


def tv_distance(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Total Variation distance: 0.5 * Σ|p - q|
    Bounded in [0, 1]; does not scale with temperature.
    """
    T = temperature
    p = F.softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits / T, dim=-1)
    per_token = 0.5 * (p - q).abs().sum(-1)    # [B, T]
    return _masked_mean(per_token, answer_mask)


def taid_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    taid_alpha_start: float = 0.0,
    taid_alpha_end: float = 1.0,
    global_step: int = 0,
    total_steps: int = 40000,
    **kwargs,
) -> torch.Tensor:
    """
    Temperature-Annealing Interpolation Distillation (TAID).

    Interpolated target: p_t = (1-α)*student_logits + α*teacher_logits
    where α ramps linearly from taid_alpha_start to taid_alpha_end.

    Requires FULL logit vectors — top-K storage breaks this because the
    interpolation is done in raw logit space before softmax.

    Reference: arXiv:XXXX (Beta-KD paper)
    """
    alpha = taid_alpha_start + (taid_alpha_end - taid_alpha_start) * (
        global_step / max(total_steps, 1)
    )
    alpha = float(max(0.0, min(1.0, alpha)))

    T = temperature
    interpolated = (1.0 - alpha) * student_logits + alpha * teacher_logits
    log_p      = F.log_softmax(student_logits / T, dim=-1)
    q_interp   = F.softmax(interpolated       / T, dim=-1)
    per_token  = F.kl_div(log_p, q_interp, reduction="none").sum(-1)  # [B, T]
    return _masked_mean(per_token, answer_mask) * (T ** 2)


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    dkd_alpha: float = 1.0,
    dkd_beta: float = 5.0,
    labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Decoupled Knowledge Distillation (DKD).
    Splits the KD signal into:
      - TCKD: KL on the target-class vs rest (2-class collapsed) distribution
      - NCKD: KL on the non-target class distribution

    dkd_alpha weights TCKD, dkd_beta weights NCKD.

    Requires FULL logit vectors (GT token index must exist in the vocab) and
    the ground-truth token IDs (labels) for the answer positions.

    Reference: https://arxiv.org/abs/2203.08679
    """
    if labels is None:
        raise ValueError("dkd_loss requires `labels` (ground-truth token IDs).")

    T = temperature
    B, L, V = student_logits.shape
    mask_flat = answer_mask.float().reshape(-1)     # [B*L]

    s_flat = student_logits.reshape(-1, V)          # [B*L, V]
    t_flat = teacher_logits.reshape(-1, V)          # [B*L, V]

    # Ground-truth token IDs; labels has -100 for non-answer → clamp to 0
    targets = labels.reshape(-1).clone()
    targets[targets < 0] = 0
    targets = targets.clamp(0, V - 1)

    # Boolean masks: which vocab slot is the ground-truth token?
    gt_mask    = torch.zeros_like(s_flat, dtype=torch.bool).scatter_(
        1, targets.unsqueeze(1), True
    )
    other_mask = ~gt_mask

    p_s = F.softmax(s_flat / T, dim=-1)
    p_t = F.softmax(t_flat / T, dim=-1)

    def _collapse(dist: torch.Tensor) -> torch.Tensor:
        """Collapse to 2-class: [p_gt, p_rest]."""
        p_gt   = (dist * gt_mask).sum(1, keepdim=True)
        p_rest = (dist * other_mask).sum(1, keepdim=True)
        return torch.cat([p_gt, p_rest], dim=1).clamp(min=1e-8)  # [B*L, 2]

    # TCKD: KL on the collapsed 2-class distribution
    p_s2 = _collapse(p_s)
    p_t2 = _collapse(p_t)
    tckd_per = F.kl_div(p_s2.log(), p_t2, reduction="none").sum(-1)  # [B*L]
    tckd = (tckd_per * mask_flat).sum() / (mask_flat.sum() + 1e-8) * T ** 2

    # NCKD: KL on the non-target class distribution (mask out GT token)
    INF = 1e4
    log_p_s_ngt = F.log_softmax(s_flat / T - INF * gt_mask, dim=-1)
    p_t_ngt     = F.softmax(    t_flat / T - INF * gt_mask, dim=-1)
    nckd_per    = F.kl_div(log_p_s_ngt, p_t_ngt, reduction="none").sum(-1)  # [B*L]
    nckd = (nckd_per * mask_flat).sum() / (mask_flat.sum() + 1e-8) * T ** 2

    return dkd_alpha * tckd + dkd_beta * nckd


# ──────────────────────────────────────────────────────────────────────────────
# Loss registry
# ──────────────────────────────────────────────────────────────────────────────

LOSS_REGISTRY: dict = {
    "fkl":  forward_kl,
    "rkl":  reverse_kl,
    "js":   js_divergence,
    "tvd":  tv_distance,
    "taid": taid_loss,
    "dkd":  dkd_loss,
}


def get_distill_loss(name: str):
    """Return the loss function for the given name."""
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown distillation loss '{name}'. "
            f"Available: {sorted(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]


# ──────────────────────────────────────────────────────────────────────────────
# Loss weighting strategies
# ──────────────────────────────────────────────────────────────────────────────

class EqualWeighting(nn.Module):
    """Uniform average of all loss terms. No learnable parameters."""

    def __init__(self, num_losses: int):
        super().__init__()
        self.num_losses = num_losses

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        return sum(losses) / len(losses)


class HeteroscedasticWeighting(nn.Module):
    """
    Learnable per-task uncertainty weighting (Kendall et al., 2018).

    L_total = Σ_i  [ 0.5 * exp(-log_σ_i) * L_i  +  0.5 * log_σ_i ]

    log_σ_i is initialised to 0 (σ = 1, equal weights) and learned during
    training.  It is automatically added to the optimizer by distill_train.py.

    Reference: https://arxiv.org/abs/1705.07115
    """

    def __init__(self, num_losses: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = sum(
            0.5 * torch.exp(-self.log_vars[i]) * L + 0.5 * self.log_vars[i]
            for i, L in enumerate(losses)
        )
        return total


WEIGHTING_REGISTRY: dict = {
    "equal":           EqualWeighting,
    "heteroscedastic": HeteroscedasticWeighting,
}


def get_weighting_strategy(name: str, num_losses: int) -> nn.Module:
    """Instantiate and return a weighting strategy module."""
    name = name.lower()
    if name not in WEIGHTING_REGISTRY:
        raise ValueError(
            f"Unknown weighting strategy '{name}'. "
            f"Available: {sorted(WEIGHTING_REGISTRY.keys())}"
        )
    return WEIGHTING_REGISTRY[name](num_losses)
