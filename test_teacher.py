"""
Deep teacher sanity-check script.

Answers the question: is the teacher actually producing meaningful logits
for the correct answer, or is something misaligned?

Three independent tests:

  A. Token-level diagnostics
     - Rank of the ground-truth token in the teacher distribution
       (rank 1 = teacher's top pick IS the correct token)
     - Log-probability the teacher assigns to the correct token
     - Text-level match: decode teacher top-1 tokens and compare with answer
       (handles the space-prefix BPE mismatch that breaks exact token ID match)

  B. Teacher generation test
     - Actually call model.generate() and inspect the decoded output
     - If the teacher can generate the right answer it definitely has signal

  C. Answer boundary sanity
     - Print lengths of prompt-only vs full-sequence tokenizations
     - Verify answer_start is plausible (not 0, not == T_full)
     - Verify T_ans matches teacher tokenizer encoding of the answer

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_teacher.py \\
        --train_dataset_path /path/to/dataset \\
        --n_samples 8 \\
        --teacher_model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct
"""

import argparse
import torch
import torch.nn.functional as F

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import models.config as config
from models.teacher import SmolVLM2Teacher
from models.vision_language_model import VisionLanguageModel
from data.datasets import VQADataset
from data.collators import DistillCollator
from data.processors import get_image_processor, get_tokenizer
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader


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
    parser.add_argument("--n_samples",          type=int, default=8,
                        help="Number of samples to diagnose")
    parser.add_argument("--skip_generation",    action="store_true",
                        help="Skip the generate() test (saves time if you just want logit checks)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    vlm_cfg     = config.VLMConfig()
    distill_cfg = config.DistillConfig()
    distill_cfg.teacher_model_id  = args.teacher_model_id
    distill_cfg.max_sample_length = args.max_sample_length

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    raw = load_from_disk(args.train_dataset_path)
    if isinstance(raw, DatasetDict):
        ds = raw["train"] if "train" in raw else raw[next(iter(raw))]
    else:
        ds = raw
    ds = ds.select(range(min(200, len(ds))))

    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len
    )
    student_tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )
    dataset  = VQADataset(ds, student_tokenizer, image_processor,
                          vlm_cfg.mp_image_token_length,
                          max_chat_tokens=distill_cfg.max_sample_length)
    collator = DistillCollator(student_tokenizer, distill_cfg.max_sample_length)
    loader   = DataLoader(dataset, batch_size=1, collate_fn=collator,
                          num_workers=0, drop_last=False)

    # ── Teacher ───────────────────────────────────────────────────────────────
    print("Loading teacher...")
    teacher = SmolVLM2Teacher(
        model_id=distill_cfg.teacher_model_id,
        base_vocab_size=distill_cfg.base_vocab_size,
        student_device=device,
    )
    teacher_tok = teacher.processor.tokenizer   # teacher's own tokenizer

    all_passed = True

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TEST A — Token-rank & text-level answer match")
    print("="*60)
    print("  (Rank 1 = teacher top-1 is the correct token. Rank <10 = strong signal.)")
    print("  (Text match handles space-prefix BPE differences.)\n")
    # ══════════════════════════════════════════════════════════════════════════

    all_ranks:      list[int]   = []
    all_logprobs:   list[float] = []
    all_text_match: list[bool]  = []

    for i, batch in enumerate(loader):
        if i >= args.n_samples:
            break
        if not isinstance(batch.get("input_ids"), torch.Tensor):
            continue

        raw_answer = batch["raw_answers"][0]
        if not raw_answer.strip():
            print(f"  [{WARN}] sample {i}: empty answer, skipping")
            continue

        with torch.no_grad():
            t_logits = teacher.get_answer_logits(batch)  # [1, T_ans, V]

        T_ans = t_logits.size(1)
        if T_ans == 0:
            print(f"  [{WARN}] sample {i}: T_ans=0, skipping")
            continue

        # ── Ground-truth tokens from the teacher's own tokenizer ─────────────
        gt_ids_teacher = teacher_tok.encode(raw_answer, add_special_tokens=False)

        # ── Ground-truth tokens from the student's tokenizer ─────────────────
        gt_ids_student = student_tokenizer.encode(raw_answer, add_special_tokens=False)

        # ── Rank of each correct token in the teacher distribution ──────────
        sample_ranks   = []
        sample_logprobs = []
        for pos in range(min(len(gt_ids_teacher), T_ans)):
            gt_tok = gt_ids_teacher[pos]
            logit_row = t_logits[0, pos]                  # [V]
            sorted_ids = logit_row.argsort(descending=True)
            rank = (sorted_ids == gt_tok).nonzero(as_tuple=False)
            rank_val = rank[0].item() + 1 if len(rank) > 0 else -1
            lp = F.log_softmax(logit_row, dim=-1)[gt_tok].item()
            sample_ranks.append(rank_val)
            sample_logprobs.append(lp)

        avg_rank   = sum(sample_ranks) / len(sample_ranks) if sample_ranks else -1
        avg_lp     = sum(sample_logprobs) / len(sample_logprobs) if sample_logprobs else float("nan")

        # ── Text-level: decode teacher top-1 and compare ─────────────────────
        t_top1_ids   = t_logits[0].argmax(-1).tolist()[:len(gt_ids_teacher)]
        teacher_text = teacher_tok.decode(t_top1_ids, skip_special_tokens=True).strip()
        expected_text = raw_answer.strip()
        text_match   = expected_text.lower() in teacher_text.lower() or \
                       teacher_text.lower() in expected_text.lower()

        # ── Student vs teacher tokenizer vocab agreement for this answer ──────
        student_text = student_tokenizer.decode(gt_ids_student, skip_special_tokens=True).strip()
        teacher_text_gt = teacher_tok.decode(gt_ids_teacher, skip_special_tokens=True).strip()

        all_ranks.append(avg_rank)
        all_logprobs.append(avg_lp)
        all_text_match.append(text_match)

        rank_ok = avg_rank <= 50    # top-50 is already very informative
        print(f"  sample {i:2d}: T_ans={T_ans:3d}  "
              f"avg_rank={avg_rank:6.1f}  avg_logprob={avg_lp:6.3f}  "
              f"text_match={'YES' if text_match else 'no '}")
        print(f"             expected='{expected_text[:50]}'")
        print(f"             teacher_top1='{teacher_text[:50]}'")
        if gt_ids_student[:3] != gt_ids_teacher[:3]:
            print(f"             {WARN}  TOKENIZER MISMATCH: "
                  f"student_ids={gt_ids_student[:5]}  "
                  f"teacher_ids={gt_ids_teacher[:5]}")
        print()

    if all_ranks:
        median_rank = sorted(all_ranks)[len(all_ranks) // 2]
        pct_top1  = sum(1 for r in all_ranks if r == 1) / len(all_ranks)
        pct_top10 = sum(1 for r in all_ranks if r <= 10) / len(all_ranks)
        pct_text  = sum(all_text_match) / len(all_text_match)

        print()
        ok_a1 = check("Median rank of correct token ≤ 50 (teacher has answer signal)",
                       median_rank <= 50,
                       f"median_rank={median_rank:.0f}")
        ok_a2 = check("≥ 10% of positions: teacher top-1 == correct token",
                       pct_top1 >= 0.10,
                       f"top1_rate={pct_top1:.1%}")
        ok_a3 = check("≥ 50% of positions: correct token in teacher top-10",
                       pct_top10 >= 0.50,
                       f"top10_rate={pct_top10:.1%}")
        ok_a4 = check("Text-level match rate ≥ 50% (covers space-prefix BPE artifact)",
                       pct_text >= 0.50,
                       f"text_match={pct_text:.1%}")
        all_passed &= ok_a1 and ok_a3
    else:
        print(f"  {FAIL} No valid samples collected")
        all_passed = False

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TEST B — Answer boundary (answer_start) sanity")
    print("="*60)
    # ══════════════════════════════════════════════════════════════════════════

    for i, batch in enumerate(loader):
        if i >= min(args.n_samples, 4):
            break
        if not isinstance(batch.get("input_ids"), torch.Tensor):
            continue

        imgs         = batch["raw_images"][0]
        conv         = batch["raw_conversations"][0]
        answer_text  = batch["raw_answers"][0]

        teacher_conv = []
        images_injected = False
        for msg in conv:
            if msg["role"] == "user" and not images_injected and imgs:
                content = [{"type": "image"} for _ in imgs]
                content.append({"type": "text", "text": msg["content"]})
                teacher_conv.append({"role": "user", "content": content})
                images_injected = True
            else:
                teacher_conv.append(msg)

        prompt_text = teacher.processor.apply_chat_template(
            teacher_conv, tokenize=False, add_generation_prompt=True)
        prompt_inputs = teacher.processor(
            text=prompt_text, images=imgs if imgs else None,
            return_tensors="pt", truncation=False)
        answer_start = prompt_inputs["input_ids"].shape[1]

        full_conv_text = teacher.processor.apply_chat_template(
            teacher_conv + [{"role": "assistant",
                             "content": [{"type": "text", "text": answer_text}]}],
            tokenize=False, add_generation_prompt=False)

        # ── Template diff diagnostic ──────────────────────────────────────────
        # If the template is correctly including the answer, full_conv_text should
        # be longer than prompt_text by the length of the answer.
        text_len_diff = len(full_conv_text) - len(prompt_text)
        answer_in_full = answer_text[:20] in full_conv_text
        answer_portion = full_conv_text[len(prompt_text):]
        answer_portion_ids = teacher_tok(
            answer_portion, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        T_ans_concat = answer_portion_ids.size(1)

        # How many tokens does the teacher tokenizer assign to the answer text alone?
        ans_tok_ids = teacher_tok.encode(answer_text, add_special_tokens=False)

        print(f"  sample {i}: T_prompt={answer_start}")
        print(f"             template text len diff = {text_len_diff} chars "
              f"(expected ~{len(answer_text)} chars for the answer)")
        print(f"             answer text in full_conv_text: {answer_in_full}")
        print(f"             answer_portion: {repr(answer_portion[:60])}")
        print(f"             T_ans via concat tokenize = {T_ans_concat}  "
              f"(answer-only tokens = {len(ans_tok_ids)})")
        print(f"             expected_answer='{answer_text[:50]}'")

        # Verify the answer portion decodes back to (approximately) the answer
        decoded_portion = teacher_tok.decode(
            answer_portion_ids[0].tolist(), skip_special_tokens=True).strip()
        boundary_ok = (
            answer_text.strip()[:10] in decoded_portion or
            decoded_portion[:10] in answer_text.strip()
        ) and T_ans_concat > 3

        icon = PASS if boundary_ok else FAIL
        print(f"             {icon} boundary {'OK' if boundary_ok else 'BAD'}  "
              f"decoded_portion='{decoded_portion[:50]}'\n")
        all_passed &= boundary_ok

    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_generation:
        print("\n" + "="*60)
        print("TEST C — Teacher generation (model.generate)")
        print("="*60)
        print("  (If the teacher can generate the correct answer, logits are definitely good.)\n")
        # ══════════════════════════════════════════════════════════════════════════

        teacher_device = next(teacher.model.parameters()).device

        for i, batch in enumerate(loader):
            if i >= min(args.n_samples, 4):
                break
            if not isinstance(batch.get("input_ids"), torch.Tensor):
                continue

            imgs        = batch["raw_images"][0]
            conv        = batch["raw_conversations"][0]
            answer_text = batch["raw_answers"][0]

            teacher_conv = []
            images_injected = False
            for msg in conv:
                if msg["role"] == "user" and not images_injected and imgs:
                    content = [{"type": "image"} for _ in imgs]
                    content.append({"type": "text", "text": msg["content"]})
                    teacher_conv.append({"role": "user", "content": content})
                    images_injected = True
                else:
                    teacher_conv.append(msg)

            prompt_text = teacher.processor.apply_chat_template(
                teacher_conv, tokenize=False, add_generation_prompt=True)
            inputs = teacher.processor(
                text=prompt_text, images=imgs if imgs else None,
                return_tensors="pt", truncation=False)
            inputs = {k: v.to(teacher_device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = teacher.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,      # greedy
                )

            # Strip prompt tokens from the generation
            gen_text = teacher_tok.decode(
                gen_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            match = (answer_text.strip().lower() in gen_text.lower() or
                     gen_text.lower() in answer_text.strip().lower())
            icon = PASS if match else WARN
            print(f"  {icon} sample {i}:")
            print(f"     expected : '{answer_text[:80]}'")
            print(f"     generated: '{gen_text[:80]}'")
            print()

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    if all_passed:
        print(f"  {PASS} Teacher is producing valid, aligned logits.")
    else:
        print(f"  {FAIL} Problems detected — see details above.")
    print()


if __name__ == "__main__":
    main()
