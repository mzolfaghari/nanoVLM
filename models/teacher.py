"""
Teacher wrapper abstraction for online knowledge distillation.

Hierarchy:
    BaseTeacher  (ABC)
      └─ SmolVLM2Teacher   — wraps HuggingFaceTB/SmolVLM2-*-Instruct
      └─ EnsembleTeacher   — weighted average of multiple BaseTeacher instances

Design invariants:
  - Teachers are ALWAYS frozen (no gradients, no optimizer group).
  - Teachers are loaded BEFORE DDP-wrapping the student to avoid accidental
    wrapping by DistributedDataParallel.
  - Every teacher exposes a single method:
        get_answer_logits(batch) -> Tensor[B, T_answer, base_vocab_size]  float32
    The batch is the raw collated dict produced by DistillCollator.
  - Vocab alignment: logits are truncated to base_vocab_size (49152) inside the
    teacher so callers never have to think about it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseTeacher(ABC):
    """
    Abstract interface all teacher wrappers must implement.

    Callers only ever call `get_answer_logits(batch)`.  All teacher-specific
    prompt formatting, tokenisation, and device management is encapsulated here.
    """

    @abstractmethod
    @torch.no_grad()
    def get_answer_logits(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: collated dict produced by DistillCollator, must contain:
                - "raw_images"        List[List[PIL.Image]]  per-sample image lists
                - "raw_answers"       List[str]              ground-truth answer text
                - "raw_conversations" List[List[dict]]       messages WITHOUT the answer

        Returns:
            Float32 tensor [B, T_answer, base_vocab_size] on the student's device.
            Positions that are shorter than T_answer (the batch maximum) are
            zero-padded — callers should use the answer_mask from the batch to
            ignore them in the loss.
        """
        ...

    # Teachers have no trainable parameters.
    def parameters(self):
        return iter([])


# ──────────────────────────────────────────────────────────────────────────────
# SmolVLM2 teacher
# ──────────────────────────────────────────────────────────────────────────────

class SmolVLM2Teacher(BaseTeacher):
    """
    Wraps a SmolVLM2-*-Instruct model as an online distillation teacher.

    Why SmolVLM2?
      - Its language backbone is SmolLM2-1.7B, which shares the exact same
        49 152-token base vocabulary as nanoVLM's SmolLM2-360M student.
      - Logit-level KD is therefore well-defined without any token remapping.

    VRAM budget (H100 80 GB):
      - SmolVLM2-1.7B in bf16 ≈ 3.4 GB.  Loaded once per node on GPU 0 via
        device_map="auto".  The student uses all 8 GPUs via DDP — no conflict.

    Usage:
        teacher = SmolVLM2Teacher(
            model_id="HuggingFaceTB/SmolVLM2-1.7B-Instruct",
            base_vocab_size=49152,
            student_device=torch.device("cuda:0"),
        )
        t_logits = teacher.get_answer_logits(batch)  # [B, T_ans, 49152]
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-1.7B-Instruct",
        base_vocab_size: int = 49152,
        dtype: torch.dtype = torch.bfloat16,
        student_device: Optional[torch.device] = None,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.base_vocab_size = base_vocab_size
        self.student_device = student_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading teacher: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # The processor's tokenizer ships with a small model_max_length that
        # causes it to silently truncate long sequences even when truncation=False
        # is passed to __call__ (truncation=False only prevents a ValueError;
        # model_max_length is checked separately in some HF processor versions).
        # SmolVLM2 image tokens are large (document pages → ~1400 tokens), so we
        # must raise the limit to the model's actual capacity.
        model_max_pos = getattr(self.model.config, "max_position_embeddings", 16384)
        self.processor.tokenizer.model_max_length = model_max_pos
        logger.info(f"Set processor tokenizer model_max_length={model_max_pos}")

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_answer_logits(self, batch: dict) -> torch.Tensor:
        """
        Run the teacher on each sample individually and return answer logits.

        Running sample-by-sample (rather than batch) is simpler and avoids
        left-padding gymnastics across samples with different image counts.
        The bottleneck is the student's forward + backward, not the teacher.
        """
        raw_images        = batch["raw_images"]        # List[List[PIL]]
        raw_answers       = batch["raw_answers"]       # List[str]
        raw_conversations = batch["raw_conversations"] # List[List[dict]]

        all_logits: List[torch.Tensor] = []

        teacher_device = next(self.model.parameters()).device

        for imgs, conv, answer_text in zip(raw_images, raw_conversations, raw_answers):
            # Build the teacher conversation in SmolVLM2's multimodal format.
            # SmolVLM2's processor requires {"type": "image"} blocks in the
            # content list of whichever message contains images — passing PIL
            # images separately without corresponding tokens raises a mismatch
            # error ("number of images in text [0] and images [N] differ").
            teacher_conv = []
            images_injected = False
            for msg in conv:
                if msg["role"] == "user" and not images_injected and imgs:
                    # First user turn: prepend one {"type":"image"} per PIL image
                    content: list = [{"type": "image"} for _ in imgs]
                    content.append({"type": "text", "text": msg["content"]})
                    teacher_conv.append({"role": "user", "content": content})
                    images_injected = True
                else:
                    teacher_conv.append(msg)

            # If there were no user turns (shouldn't happen), just add a bare user msg
            if not teacher_conv and imgs:
                content = [{"type": "image"} for _ in imgs]
                content.append({"type": "text", "text": ""})
                teacher_conv.append({"role": "user", "content": content})

            # ── Find answer start by tokenising the prompt-only prefix ────────
            # This is more robust than searching for a header token pattern,
            # because special tokens may be merged or encoded differently across
            # processor versions.  add_generation_prompt=True appends the
            # assistant header so T_prompt already includes it.
            prompt_text = self.processor.apply_chat_template(
                teacher_conv,                    # prompt only, no answer yet
                tokenize=False,
                add_generation_prompt=True,      # adds "<|im_start|>assistant\n"
            )
            prompt_inputs = self.processor(
                text=prompt_text,
                images=imgs if imgs else None,
                return_tensors="pt",
                truncation=False,   # SmolVLM2 image tokens are long; never truncate
            )
            answer_start = prompt_inputs["input_ids"].shape[1]  # exact boundary

            # ── Full conversation (prompt + answer) for the teacher forward ───
            teacher_conv_full = teacher_conv + [{"role": "assistant", "content": answer_text}]
            full_text = self.processor.apply_chat_template(
                teacher_conv_full,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.processor(
                text=full_text,
                images=imgs if imgs else None,
                return_tensors="pt",
                truncation=False,   # must match prompt tokenisation to get correct answer_start
            )
            inputs = {k: v.to(teacher_device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, T_full, V_teacher]

            # Causal LM convention: logits[t] predicts token[t+1].
            # Answer tokens are at positions [answer_start : T_full].
            # The logits that predict them are at [answer_start-1 : T_full-1].
            answer_logits = logits[0, answer_start - 1 : -1, :self.base_vocab_size]
            # Shape: [T_answer_i, base_vocab_size]

            # Sanity: warn if the answer slice is much shorter than expected.
            # Typical cause: tokenizer truncated the full sequence despite
            # truncation=False.  This means image tokens filled most of the
            # context window and the answer was cut off.
            T_ans_i = answer_logits.size(0)
            T_full_i = logits.size(1)
            if T_ans_i < 4 and (T_full_i - answer_start) < 4:
                logger.warning(
                    "Very short answer logits (T_ans=%d, T_full=%d, answer_start=%d). "
                    "Possible truncation — check model_max_length vs image token count.",
                    T_ans_i, T_full_i, answer_start,
                )

            all_logits.append(answer_logits.float().cpu())

        # Pad all samples to the longest answer in the batch
        max_len = max(t.size(0) for t in all_logits) if all_logits else 1
        B = len(all_logits)
        out = torch.zeros(B, max_len, self.base_vocab_size, dtype=torch.float32)
        for i, t in enumerate(all_logits):
            n = t.size(0)
            if n > 0:
                out[i, :n] = t

        return out.to(self.student_device)



# ──────────────────────────────────────────────────────────────────────────────
# Ensemble teacher (future multi-teacher support)
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleTeacher(BaseTeacher):
    """
    Weighted ensemble of multiple BaseTeacher instances.

    Combines teacher distributions in probability space:
        p_ensemble = Σ w_i * softmax(logits_i)

    then returns log(p_ensemble) so that KL losses receive numerically
    stable log-probabilities from a single "virtual" teacher.

    Example:
        teacher = EnsembleTeacher(
            teachers=[
                SmolVLM2Teacher("HuggingFaceTB/SmolVLM2-1.7B-Instruct", 49152),
                SmolVLM2Teacher("HuggingFaceTB/SmolVLM2-2.2B-Instruct",  49152),
            ],
            weights=[0.6, 0.4],
        )
    """

    def __init__(self, teachers: List[BaseTeacher], weights: List[float]):
        assert len(teachers) == len(weights), "Must provide one weight per teacher"
        assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"
        self.teachers = teachers
        w = torch.tensor(weights, dtype=torch.float32)
        self.weights = w / w.sum()  # re-normalise for safety

    @torch.no_grad()
    def get_answer_logits(self, batch: dict) -> torch.Tensor:
        """
        Returns log-probabilities of the weighted ensemble.
        Shape: [B, T_answer, base_vocab_size], dtype=float32.
        """
        logit_list = [t.get_answer_logits(batch) for t in self.teachers]

        # Ensemble over probability space, not logit space
        ensemble_prob = sum(
            w.item() * F.softmax(logits, dim=-1)
            for w, logits in zip(self.weights, logit_list)
        )

        # Return log-probs; KD losses that expect raw logits should use these
        # directly (they're equivalent for softmax-based losses).
        return torch.log(ensemble_prob.clamp(min=1e-8))


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_teacher(
    teacher_model_id: str,
    base_vocab_size: int = 49152,
    student_device: Optional[torch.device] = None,
    ensemble_teacher_ids: Optional[str] = None,
    ensemble_teacher_weights: Optional[str] = None,
) -> BaseTeacher:
    """
    Factory that builds the right teacher from DistillConfig fields.

    If ensemble_teacher_ids is set (comma-separated), returns an EnsembleTeacher.
    Otherwise returns a single SmolVLM2Teacher for teacher_model_id.
    """
    if ensemble_teacher_ids:
        ids = [s.strip() for s in ensemble_teacher_ids.split(",")]
        if ensemble_teacher_weights:
            weights = [float(w) for w in ensemble_teacher_weights.split(",")]
        else:
            weights = [1.0 / len(ids)] * len(ids)
        teachers = [
            SmolVLM2Teacher(mid, base_vocab_size, student_device=student_device)
            for mid in ids
        ]
        return EnsembleTeacher(teachers, weights)

    return SmolVLM2Teacher(
        model_id=teacher_model_id,
        base_vocab_size=base_vocab_size,
        student_device=student_device,
    )
