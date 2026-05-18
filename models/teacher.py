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
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Cache the token IDs for the assistant turn boundary so we don't
        # re-encode them on every forward pass.
        self._assistant_header_ids: Optional[List[int]] = None

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

            # Append ground-truth answer as the assistant turn
            teacher_conv.append({"role": "assistant", "content": answer_text})

            text = self.processor.apply_chat_template(
                teacher_conv,
                tokenize=False,
                add_generation_prompt=False,
            )

            inputs = self.processor(
                text=text,
                images=imgs if imgs else None,
                return_tensors="pt",
            )
            # Move inputs to whichever device the teacher's first parameter lives on
            teacher_device = next(self.model.parameters()).device
            inputs = {k: v.to(teacher_device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, T_teacher, V_teacher]

            # Find where the assistant answer starts in the teacher's sequence
            answer_start = self._find_answer_start(inputs["input_ids"][0])

            # Slice to answer positions; shift by -1 for causal LM convention
            # (position i predicts token i+1, so logits[answer_start-1:-1] predicts
            #  the answer tokens).
            answer_logits = logits[0, answer_start - 1 : -1, :self.base_vocab_size]
            # Shape: [T_answer_i, base_vocab_size]

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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_assistant_header_ids(self) -> List[int]:
        """Cache-and-return the token IDs for '<|im_start|>assistant'."""
        if self._assistant_header_ids is None:
            self._assistant_header_ids = self.processor.tokenizer.encode(
                "<|im_start|>assistant",
                add_special_tokens=False,
            )
        return self._assistant_header_ids

    def _find_answer_start(self, input_ids: torch.Tensor) -> int:
        """
        Return the index (inclusive) of the first answer token in the teacher
        sequence.  We scan backwards for the last occurrence of the assistant
        header so multi-turn conversations are handled correctly.

        The SmolVLM2 chat template uses:
            <|im_start|>assistant\n<answer tokens><|im_end|>
        so the answer starts at  header_end + 1  (the '\n' token).
        """
        header = self._get_assistant_header_ids()
        ids = input_ids.tolist()
        h_len = len(header)

        for pos in range(len(ids) - h_len, -1, -1):
            if ids[pos : pos + h_len] == header:
                # +h_len to skip the header itself, +1 for the '\n' newline token
                return pos + h_len + 1

        # Fallback: treat the very last token as the answer
        logger.warning(
            "Could not find assistant header in teacher input_ids. "
            "Falling back to last token position."
        )
        return len(ids) - 1


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
