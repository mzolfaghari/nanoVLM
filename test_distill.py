"""
Distillation sanity-check script.

Run before a full training run to verify:
  1. Teacher logits are non-degenerate (not random, has signal)
  2. Answer boundary extraction is correct (top-1 teacher prediction matches answer)
  3. Vocab alignment is correct (49152 base tokens, no OOB)
  4. All KD losses are numerically stable (no NaN/Inf)
  5. Gradients flow correctly through the student (not zero, not exploding)
  6. CE + KD loss combination is correct

Usage (single GPU, small dataset):
    CUDA_VISIBLE_DEVICES=0 python test_distill.py \
        --train_dataset_path /path/to/dataset \
        --train_dataset_name default \
        --teacher_model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
        --n_batches 4
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import models.config as config
from models.vision_language_model import VisionLanguageModel
from models.teacher import build_teacher
from models.distil_losses import LOSS_REGISTRY, get_distill_loss, get_weighting_strategy
from data.datasets import VQADataset
from data.collators import DistillCollator
from data.processors import get_image_processor, get_tokenizer
from datasets import load_from_disk, DatasetDict


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


def check(name, ok, detail=""):
    icon = PASS if ok else FAIL
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_dataset_name", default="default")
    parser.add_argument("--teacher_model_id",   default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max_sample_length",  type=int, default=2048)
    parser.add_argument("--n_batches",          type=int, default=4,
                        help="Number of batches to run each check on")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Configs ───────────────────────────────────────────────────────────────
    vlm_cfg     = config.VLMConfig()
    distill_cfg = config.DistillConfig()
    distill_cfg.teacher_model_id  = args.teacher_model_id
    distill_cfg.max_sample_length = args.max_sample_length

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    raw = load_from_disk(args.train_dataset_path)
    if isinstance(raw, DatasetDict):
        ds = raw["train"] if "train" in raw else raw[next(iter(raw))]
    else:
        ds = raw
    ds = ds.select(range(min(200, len(ds))))   # tiny slice for speed

    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len
    )
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )
    dataset = VQADataset(ds, tokenizer, image_processor,
                         vlm_cfg.mp_image_token_length,
                         max_chat_tokens=distill_cfg.max_sample_length)
    collator = DistillCollator(tokenizer, distill_cfg.max_sample_length)
    loader   = DataLoader(dataset, batch_size=1, collate_fn=collator,
                          num_workers=0, drop_last=False)

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading teacher...")
    teacher = build_teacher(
        teacher_model_id=distill_cfg.teacher_model_id,
        base_vocab_size=distill_cfg.base_vocab_size,
        student_device=device,
    )

    print("Loading student...")
    student = VisionLanguageModel(vlm_cfg, load_backbone=True).to(device)
    student.train()

    all_passed = True

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK 1 — Teacher logit quality")
    print("="*60)
    # A well-calibrated teacher should:
    #   - Have top-1 prediction matching the ground-truth answer token most of
    #     the time (or at least more than random)
    #   - Have lower entropy than random (log(49152) ≈ 10.78 nats)
    # ══════════════════════════════════════════════════════════════════════════
    entropies   = []
    top1_match  = []
    ans_lengths = []

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        if not isinstance(batch.get("input_ids"), torch.Tensor):
            continue

        with torch.no_grad():
            t_logits = teacher.get_answer_logits(batch)   # [1, T_ans, V]

        T_ans = t_logits.size(1)
        ans_lengths.append(T_ans)
        if T_ans == 0:
            continue

        probs   = F.softmax(t_logits[0], dim=-1)          # [T_ans, V]
        entropy = -(probs * torch.log(probs.clamp(1e-10))).sum(-1).mean().item()
        entropies.append(entropy)

        # Check top-1 teacher prediction vs raw answer text tokens
        raw_answer = batch["raw_answers"][0]
        ans_ids    = tokenizer.encode(raw_answer, add_special_tokens=False)
        t_top1     = t_logits[0].argmax(-1).tolist()        # [T_ans]
        n_match    = sum(
            1 for pos in range(min(len(ans_ids), T_ans))
            if ans_ids[pos] == t_top1[pos]
        )
        match_rate = n_match / max(len(ans_ids), 1)
        top1_match.append(match_rate)

        print(f"  batch {i}: T_ans={T_ans}  entropy={entropy:.2f}  "
              f"top1_match={match_rate:.0%}  "
              f"answer='{raw_answer[:40]}'")

    random_entropy = torch.tensor(distill_cfg.base_vocab_size).float().log().item()
    avg_entropy    = sum(entropies) / max(len(entropies), 1)
    avg_match      = sum(top1_match) / max(len(top1_match), 1)
    avg_len        = sum(ans_lengths) / max(len(ans_lengths), 1)

    ok1a = check("Teacher entropy < random (log V = {:.2f})".format(random_entropy),
                 avg_entropy < random_entropy,
                 f"avg={avg_entropy:.2f}")
    ok1b = check("Teacher top-1 match rate > 0% (not degenerate)",
                 avg_match > 0.0,
                 f"avg={avg_match:.1%}")
    ok1c = check("Average answer length > 1 token",
                 avg_len > 1.0,
                 f"avg={avg_len:.1f} tokens")
    all_passed &= ok1a and ok1b and ok1c

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK 2 — Vocab alignment (no out-of-bound indices)")
    print("="*60)
    # ══════════════════════════════════════════════════════════════════════════
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        if not isinstance(batch.get("input_ids"), torch.Tensor):
            continue
        input_ids = batch["input_ids"]
        # Student extra tokens are 49152–49217; teacher vocab is exactly 49152
        extra_mask = (input_ids >= distill_cfg.base_vocab_size)
        n_extra    = extra_mask.sum().item()
        labels     = batch["labels"]
        bad_labels = ((labels >= distill_cfg.base_vocab_size) & (labels != -100)).sum().item()
        break

    ok2a = check("Labels contain no extra-token IDs (>= base_vocab_size)",
                 bad_labels == 0,
                 f"bad_label_count={bad_labels}")
    ok2b = check("Extra image tokens present in input_ids (expected)",
                 n_extra > 0,
                 f"count={n_extra} (these are replaced with pad before teacher)")
    all_passed &= ok2a

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK 3 — All KD losses: no NaN / Inf, reasonable magnitude")
    print("="*60)
    # Expected FKL magnitude: between 0 and log(V)=10.78.
    # Very close to 10.78 ⟹ teacher is near-uniform (bad signal).
    # Very close to 0    ⟹ student already matches teacher (unlikely at init).
    # ══════════════════════════════════════════════════════════════════════════
    # Get one batch to use for all loss checks
    for batch in loader:
        if isinstance(batch.get("input_ids"), torch.Tensor):
            break

    with torch.no_grad():
        t_logits = teacher.get_answer_logits(batch)
    T_ans      = t_logits.size(1)
    answer_mask = batch["answer_mask"].to(device)

    with torch.no_grad():
        s_logits, _ = student(
            batch["input_ids"].to(device),
            batch["images"],
            attention_mask=batch["attention_mask"].to(device),
        )

    s_base = s_logits[:, :, :distill_cfg.base_vocab_size]

    # gather answer logits
    B = s_base.size(0)
    s_ans = torch.zeros(B, T_ans, distill_cfg.base_vocab_size,
                        dtype=s_base.dtype, device=device)
    for b in range(B):
        pos = answer_mask[b].nonzero(as_tuple=False).squeeze(-1)
        n   = min(pos.size(0), T_ans)
        if n > 0:
            s_ans[b, :n] = s_base[b, pos[:n]]

    labels = batch["labels"].to(device)
    ans_labels = torch.full((B, T_ans), -100, dtype=labels.dtype, device=device)
    for b in range(B):
        pos = answer_mask[b].nonzero(as_tuple=False).squeeze(-1)
        n   = min(pos.size(0), T_ans)
        if n > 0:
            ans_labels[b, :n] = labels[b, pos[:n]]

    t_ans_mask = (t_logits.abs().sum(-1) > 0).float()

    for loss_name, loss_fn in LOSS_REGISTRY.items():
        # DKD can exceed 15 because beta*NCKD is unbounded — only check finite + positive
        max_val = 200.0 if loss_name == "dkd" else 15.0
        try:
            val = loss_fn(
                student_logits=s_ans.float(),
                teacher_logits=t_logits.float().to(device),
                answer_mask=t_ans_mask.to(device),
                temperature=2.0,
                labels=ans_labels,
                dkd_alpha=1.0, dkd_beta=5.0,
                taid_alpha_start=0.0, taid_alpha_end=1.0,
                global_step=0, total_steps=40000,
                js_teacher_weight=0.1,
                skew_target_weight=0.1,
            )
            is_finite = torch.isfinite(val).item()
            in_range  = 0.0 < val.item() < max_val
            ok = check(f"{loss_name:10s}: {val.item():.4f}",
                       is_finite and in_range,
                       f"out of expected range [0, {max_val}]" if not in_range else "")
            all_passed &= ok
        except Exception as e:
            check(f"{loss_name:10s}", False, str(e))
            all_passed = False

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK 4 — Gradient flow through student")
    print("="*60)
    # ══════════════════════════════════════════════════════════════════════════
    student.zero_grad()
    s_logits2, ce_loss = student(
        batch["input_ids"].to(device),
        batch["images"],
        attention_mask=batch["attention_mask"].to(device),
        targets=batch["labels"].to(device),
    )
    s_base2 = s_logits2[:, :, :distill_cfg.base_vocab_size]
    s_ans2  = torch.zeros_like(s_ans)
    for b in range(B):
        pos = answer_mask[b].nonzero(as_tuple=False).squeeze(-1)
        n   = min(pos.size(0), T_ans)
        if n > 0:
            s_ans2[b, :n] = s_base2[b, pos[:n]]

    kd_fn   = get_distill_loss("fkl")
    kd_loss = kd_fn(s_ans2.float(), t_logits.float().to(device),
                    t_ans_mask.to(device), temperature=2.0)
    weighting = get_weighting_strategy("equal", 2)
    total     = weighting([ce_loss, kd_loss])
    total.backward()

    # Check that MP (modality projector) received gradient
    mp_grad = next(student.MP.parameters()).grad
    lm_grad = next(student.decoder.parameters()).grad

    ok4a = check("Gradient flows to modality projector",
                 mp_grad is not None and mp_grad.abs().max().item() > 0,
                 f"max_grad={mp_grad.abs().max().item():.2e}" if mp_grad is not None else "None")
    ok4b = check("Gradient flows to language model",
                 lm_grad is not None and lm_grad.abs().max().item() > 0,
                 f"max_grad={lm_grad.abs().max().item():.2e}" if lm_grad is not None else "None")
    ok4c = check("Gradient is not exploding (max < 100)",
                 mp_grad is not None and mp_grad.abs().max().item() < 100,
                 "possible exploding gradient" if (mp_grad is not None and mp_grad.abs().max().item() >= 100) else "")
    all_passed &= ok4a and ok4b and ok4c

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK 5 — answer_mask alignment")
    print("="*60)
    # answer_mask should be True exactly where labels != -100
    # ══════════════════════════════════════════════════════════════════════════
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        if not isinstance(batch.get("input_ids"), torch.Tensor):
            continue
        lbl  = batch["labels"]
        amsk = batch["answer_mask"]
        expected = (lbl != -100)
        match = (amsk == expected).all().item()
        n_ans = amsk.sum().item()
        ok5 = check(f"batch {i}: answer_mask matches labels!=-100  "
                    f"(n_answer_tokens={n_ans})", match)
        all_passed &= ok5

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    if all_passed:
        print(f"  {PASS} All checks passed — distillation pipeline is correct.")
    else:
        print(f"  {FAIL} Some checks failed — review output above.")
    print()


if __name__ == "__main__":
    main()
