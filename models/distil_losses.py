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
    "fkl"      — Forward KL   KL(teacher || student)      standard KD
    "rkl"      — Reverse KL   KL(student || teacher)      mode-seeking
    "js"       — Jensen-Shannon divergence                 asymmetric (teacher_weight param)
    "tvd"      — Total Variation distance                  L1 in prob space
    "taid"     — Temperature-Annealing Interpolation KD   requires full logit vec
    "dkd"      — Decoupled KD (TCKD + NCKD)              requires full logit vec + labels
    "skew_fkl" — Skewed Forward KL (DistiLLM)

Implementations cross-checked against:
    https://github.com/StevenLauHKHK/Beta-KD (fkl, rkl, tvd, js, taid, dkd)
    https://github.com/jongwooko/distillm (fkl, rkl, tvd, js, skew_fkl)

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
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _masked_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Weighted mean of per-token losses.
    per_token: [B, T]
    mask:      [B, T]  bool or float, 1 = answer token
    """
    mask = mask.float()
    denom = mask.sum() + 1e-8
    return (per_token * mask).sum() / denom


def _masked_sum_div(x_flat: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
    """
    Sum of (x * mask) / sum(mask).  Used by beta-kd style losses where the
    product is already summed over the vocab dimension.
    x_flat, mask_flat: [B*T]
    """
    denom = mask_flat.sum() + 1e-8
    return (x_flat * mask_flat).sum() / denom


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
    Forward KL: KL(teacher || student) — standard knowledge distillation.

    Implementation follows beta-kd / DistiLLM:
      loss = -sum(q * log_p) / n_tokens
    where q = softmax(teacher), p = softmax(student).
    Inf positions in student logits are masked to 0.

    Note: no T^2 rescaling here — temperature softens the distributions but
    does not rescale the loss magnitude (unlike Hinton 2015).
    """
    T = temperature
    # Cast to float32 for numerical stability (inputs may be bf16)
    s = student_logits.float() / T
    t = teacher_logits.float() / T

    q          = F.softmax(t, dim=-1)                   # teacher probs
    log_p      = F.log_softmax(s, dim=-1)               # student log-probs

    inf_mask   = torch.isinf(s)                         # guard against ±inf
    per_vocab  = torch.masked_fill(q * log_p, inf_mask, 0.0)
    per_token  = -per_vocab.sum(-1)                     # [B, T]

    return _masked_mean(per_token, answer_mask)


def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Reverse KL: KL(student || teacher) — mode-seeking.
    loss = sum(p * (log_p - log_q)) / n_tokens
    """
    T = temperature
    s = student_logits.float() / T
    t = teacher_logits.float() / T

    p         = F.softmax(s, dim=-1)
    log_p     = F.log_softmax(s, dim=-1)
    log_q     = F.log_softmax(t, dim=-1)

    inf_mask  = torch.isinf(s) | torch.isinf(t)
    per_vocab = torch.masked_fill(p * (log_p - log_q), inf_mask, 0.0)
    per_token = per_vocab.sum(-1)                       # [B, T]

    return _masked_mean(per_token, answer_mask)


def js_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    js_teacher_weight: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """
    (Asymmetric) Jensen-Shannon divergence, following beta-kd's JS class.

    mixed = w * teacher + (1-w) * student   (default w=0.1, teacher-skewed)
    JS = (1-w) * KL(student || mixed) + w * KL(teacher || mixed)

    The asymmetry means this is closer to reverse-KL when w is small, which
    is the beta-kd design choice (teacher_weight=0.1 by default).
    Set js_teacher_weight=0.5 for the standard symmetric JS divergence.
    """
    T  = temperature
    w  = js_teacher_weight
    s  = student_logits.float() / T
    t  = teacher_logits.float() / T

    p          = F.softmax(s, dim=-1)           # student probs
    q          = F.softmax(t, dim=-1)           # teacher probs
    mixed      = w * q + (1 - w) * p
    log_mixed  = torch.log(mixed.clamp(min=1e-8))
    log_p      = torch.log(p.clamp(min=1e-8))
    log_q      = torch.log(q.clamp(min=1e-8))

    inf_mask   = torch.isinf(s) | torch.isinf(t)

    # (1-w) * KL(student || mixed) — reverse KL component
    rkl_part = torch.masked_fill(p * (log_mixed - log_p), inf_mask, 0.0)
    x_rkl    = rkl_part.sum(-1).view(-1)

    # w * KL(teacher || mixed) — forward KL component
    fkl_part = torch.masked_fill(q * (log_mixed - log_q), inf_mask, 0.0)
    x_fkl    = fkl_part.sum(-1).view(-1)

    mask_flat = answer_mask.float().view(-1)
    loss = (1 - w) * _masked_sum_div(x_rkl, mask_flat) \
         +      w  * _masked_sum_div(x_fkl, mask_flat)
    return -loss   # negate because KL = -E[log(mixed/self)]


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
    s = student_logits.float() / T
    t = teacher_logits.float() / T

    p         = F.softmax(s, dim=-1)
    q         = F.softmax(t, dim=-1)
    inf_mask  = torch.isinf(s) | torch.isinf(t)

    per_vocab = torch.masked_fill(0.5 * (p - q).abs(), inf_mask, 0.0)
    per_token = per_vocab.sum(-1)                       # [B, T]

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
    Temperature-Annealing Interpolation Distillation (TAID), simplified
    (linear schedule).

    Interpolated target: p_t = softmax( (1-α)*student.detach() + α*teacher )
    KD loss:             FKL( student || p_t )

    The DETACH on student logits is critical — without it, gradients would flow
    through p_t back into the student a second time, corrupting the update.

    Full beta-kd TAID uses an adaptive momentum-based schedule for α; here we
    use a simple linear ramp from taid_alpha_start to taid_alpha_end.

    Requires FULL logit vectors — top-K storage breaks the interpolation.
    """
    alpha = taid_alpha_start + (taid_alpha_end - taid_alpha_start) * (
        global_step / max(total_steps, 1)
    )
    alpha = float(max(0.0, min(1.0, alpha)))

    T = temperature
    s = student_logits.float() / T
    t = teacher_logits.float() / T

    # Interpolate in logit space; detach student so only one gradient path exists
    interpolated = (1.0 - alpha) * s.detach() + alpha * t
    p_t    = F.softmax(interpolated, dim=-1)        # interpolated target probs

    log_p  = F.log_softmax(s, dim=-1)               # student log-probs (grad flows here)
    inf_mask = torch.isinf(s)
    per_vocab = torch.masked_fill(p_t * log_p, inf_mask, 0.0)
    per_token = -per_vocab.sum(-1)                  # [B, T]

    return _masked_mean(per_token, answer_mask)


def skew_fkl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.0,
    skew_target_weight: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """
    Skewed Forward KL (DistiLLM, https://arxiv.org/abs/2402.03898).

    mixed = w * teacher + (1-w) * student
    loss  = FKL(teacher || mixed) = -sum(q * log(mixed)) / n_tokens

    Useful when you want the teacher distribution to guide without fully
    overriding the student's learned modes.
    """
    T = temperature
    w = skew_target_weight
    s = student_logits.float() / T
    t = teacher_logits.float() / T

    p     = F.softmax(s, dim=-1)
    q     = F.softmax(t, dim=-1)
    mixed = w * q + (1 - w) * p
    log_m = torch.log(mixed.clamp(min=1e-8))

    inf_mask  = torch.isinf(s) | torch.isinf(t)
    per_vocab = torch.masked_fill(q * log_m, inf_mask, 0.0)
    per_token = -per_vocab.sum(-1)                  # [B, T]

    return _masked_mean(per_token, answer_mask)


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

    IMPORTANT: `labels` must be [B, T] matching the shape of student_logits.
    Callers must extract answer-position labels before calling this
    (distill_train.py does this via gather_answer_labels).

    Reference: https://arxiv.org/abs/2203.08679
    """
    if labels is None:
        raise ValueError("dkd_loss requires `labels` (ground-truth token IDs).")

    T = temperature
    B, L, V = student_logits.shape

    # Verify shape compatibility
    if labels.shape != (B, L):
        raise ValueError(
            f"dkd_loss: labels shape {labels.shape} must match "
            f"student_logits [B, T] = [{B}, {L}]. "
            "Pass answer-position labels via gather_answer_labels()."
        )

    mask_flat = answer_mask.float().reshape(-1)         # [B*L]
    s_flat    = student_logits.float().reshape(-1, V)   # [B*L, V]
    t_flat    = teacher_logits.float().reshape(-1, V)   # [B*L, V]

    # GT token IDs; -100 (ignore index) → clamp to 0 (masked out by mask_flat)
    targets = labels.reshape(-1).clone()
    targets[targets < 0] = 0
    targets = targets.clamp(0, V - 1)

    gt_mask    = torch.zeros_like(s_flat, dtype=torch.bool).scatter_(
        1, targets.unsqueeze(1), True
    )
    other_mask = ~gt_mask

    p_s = F.softmax(s_flat / T, dim=-1)
    p_t = F.softmax(t_flat / T, dim=-1)

    def _collapse(dist: torch.Tensor) -> torch.Tensor:
        """Collapse to 2-class: [p_gt, p_rest]."""
        p_gt   = (dist * gt_mask  ).sum(1, keepdim=True)
        p_rest = (dist * other_mask).sum(1, keepdim=True)
        return torch.cat([p_gt, p_rest], dim=1).clamp(min=1e-8)  # [B*L, 2]

    # TCKD
    p_s2     = _collapse(p_s)
    p_t2     = _collapse(p_t)
    tckd_per = F.kl_div(p_s2.log(), p_t2, reduction="none").sum(-1)  # [B*L]
    tckd     = (tckd_per * mask_flat).sum() / (mask_flat.sum() + 1e-8)

    # NCKD — mask out GT token with large negative value
    INF          = 1e4
    log_p_s_ngt  = F.log_softmax(s_flat / T - INF * gt_mask, dim=-1)
    p_t_ngt      = F.softmax(    t_flat / T - INF * gt_mask, dim=-1)
    nckd_per     = F.kl_div(log_p_s_ngt, p_t_ngt, reduction="none").sum(-1)
    nckd         = (nckd_per * mask_flat).sum() / (mask_flat.sum() + 1e-8)

    return dkd_alpha * tckd + dkd_beta * nckd


# ──────────────────────────────────────────────────────────────────────────────
# Loss registry
# ──────────────────────────────────────────────────────────────────────────────

LOSS_REGISTRY: dict = {
    "fkl":      forward_kl,
    "rkl":      reverse_kl,
    "js":       js_divergence,
    "tvd":      tv_distance,
    "taid":     taid_loss,
    "dkd":      dkd_loss,
    "skew_fkl": skew_fkl,
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

    log_σ_i is initialised to 0 (σ=1, equal weights) and learned during
    training. Added to the optimizer by distill_train.py.

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
