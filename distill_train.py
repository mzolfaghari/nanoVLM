"""
distill_train.py — Online knowledge distillation training for nanoVLM.

Teacher : SmolVLM2-*-Instruct  (frozen, runs every micro-batch)
Student : nanoVLM (SigLIP2-base + SmolLM2-360M)

Key differences from train.py:
  - ConstantLengthDataset packing is DISABLED (teacher needs per-sample alignment).
  - DistillCollator used instead of VQACollator (adds raw_images, raw_answers, etc.).
  - Student forward always returns full vocab logits (VisionLanguageModel fix).
  - KD loss computed over answer positions only, after truncating both logit
    tensors to base_vocab_size (49 152) to align student and teacher vocabs.
  - Loss registry: swap methods with --distill_loss fkl|rkl|taid|dkd|js|tvd.
  - Optimizer uses named param groups so weighting module params can be added.
  - train.py is NOT modified — baseline stays reproducible.
"""

import os
import re
import json
import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import subprocess
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk, DatasetDict

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

PG_CPU = None

from data.datasets import VQADataset
from data.collators import DistillCollator
from data.data_utils import synchronized_dataloader_step
from data.processors import get_image_processor, get_tokenizer

import models.config as config
from models.vision_language_model import VisionLanguageModel
from models.teacher import build_teacher
from models.distil_losses import get_distill_loss, get_weighting_strategy

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers (identical to train.py)
# ──────────────────────────────────────────────────────────────────────────────

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def init_dist():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)
    return result

def dist_mean_scalar(x: float | int) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)
    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

def wrap_model(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

def get_lr(it, max_lr, max_steps):
    """Cosine LR schedule with linear warmup (from Karpathy)."""
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ──────────────────────────────────────────────────────────────────────────────
# Run name
# ──────────────────────────────────────────────────────────────────────────────

def get_run_name(train_cfg, vlm_cfg, distill_cfg):
    bs    = f"bs{int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}"
    steps = f"{train_cfg.max_training_steps}"
    lrs   = f"lr_v{train_cfg.lr_vision_backbone}-l{train_cfg.lr_language_backbone}-mp{train_cfg.lr_mp}"
    ngpu  = f"{get_world_size()}xGPU"
    date  = time.strftime("%m%d-%H%M%S")
    vit   = f"{vlm_cfg.vit_model_type.split('/')[-1]}_{vlm_cfg.max_img_size}"
    llm   = f"{vlm_cfg.lm_model_type.split('/')[-1]}"
    loss  = f"kd_{distill_cfg.distill_loss}_T{distill_cfg.temperature}"
    return f"distill_nanoVLM_{vit}_{llm}_{ngpu}_{bs}_{steps}_{lrs}_{loss}_{date}"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader  (no packing — each sample is independent)
# ──────────────────────────────────────────────────────────────────────────────

def _load_hf_dataset_saved_to_disk(path: str):
    raw = load_from_disk(path)
    if isinstance(raw, DatasetDict):
        return raw["train"] if "train" in raw else raw[next(iter(raw))]
    return raw


def get_dataloaders(train_cfg, vlm_cfg, distill_cfg):
    print(f"Getting dataloaders from {train_cfg.train_dataset_path}")
    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len
    )
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    combined_train_data = []
    dataset_path = train_cfg.train_dataset_path
    dataset_info = os.path.join(dataset_path, "dataset_info.json")

    if os.path.isdir(dataset_path) and os.path.isfile(dataset_info):
        if train_cfg.stream_dataset and is_master():
            print("Warning: stream_dataset ignored for on-disk datasets.")
        print(f"Loading dataset from disk: {dataset_path}")
        try:
            train_ds = _load_hf_dataset_saved_to_disk(dataset_path)
            train_ds[0]
            combined_train_data.append(train_ds)
        except Exception as e:
            if is_master():
                print(f"Warning: failed to load from disk: {e}")

    if not combined_train_data:
        dataset_names = train_cfg.train_dataset_name
        if "all" in dataset_names:
            dataset_names = get_dataset_config_names(train_cfg.train_dataset_path)
        for name in dataset_names:
            try:
                ds = load_dataset(
                    train_cfg.train_dataset_path, name,
                    streaming=train_cfg.stream_dataset, on_bad_files='warn'
                )['train']
                if train_cfg.stream_dataset:
                    next(iter(ds))
                else:
                    ds[0]
                combined_train_data.append(ds)
            except Exception as e:
                if is_master():
                    print(f"Warning: failed to load config '{name}': {e}")

    if not combined_train_data:
        raise ValueError("No valid datasets loaded.")

    train_ds = concatenate_datasets(combined_train_data)
    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=0)

    if is_dist():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    val_size = int(train_cfg.val_size / get_world_size())
    print(f"Val size per GPU: {val_size}")

    if train_cfg.stream_dataset:
        val_ds   = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)
    else:
        val_ds   = train_ds.select(range(val_size))
        train_ds = train_ds.select(range(val_size, len(train_ds)))

    common_kwargs = dict(
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=vlm_cfg.mp_image_token_length,
        relevance_min_rating=train_cfg.relevance_min_rating,
        image_correspondence_min_rating=train_cfg.image_correspondence_min_rating,
        visual_dependency_min_rating=train_cfg.visual_dependency_min_rating,
        formatting_min_rating=train_cfg.formatting_min_rating,
        max_chat_tokens=distill_cfg.max_sample_length,
    )
    train_dataset = VQADataset(train_ds, **common_kwargs)
    val_dataset   = VQADataset(val_ds,   **common_kwargs)

    # NOTE: ConstantLengthDataset is intentionally NOT used here.
    # Packing merges multiple samples into one sequence, which breaks the
    # teacher–student answer-position alignment.

    distill_collator = DistillCollator(tokenizer, distill_cfg.max_sample_length)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=distill_collator,
        num_workers=3,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=distill_collator,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("Warming up dataloaders...")
    iter_train_loader = iter(train_loader)
    iter_val_loader   = iter(val_loader)
    next(iter_train_loader)
    next(iter_val_loader)
    print("Warmup complete.")

    return train_loader, val_loader, iter_train_loader, iter_val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Logit alignment helper
# ──────────────────────────────────────────────────────────────────────────────

def gather_answer_logits(
    logits: torch.Tensor,      # [B, T_seq, V]
    answer_mask: torch.Tensor, # [B, T_seq]  bool
    max_ans_len: int,
) -> torch.Tensor:
    """
    Select logits at the positions where answer_mask == True, and return them
    as a dense [B, max_ans_len, V] tensor (zero-padded for shorter answers).

    This converts the student's full-sequence logits into the same shape as the
    teacher's answer-only logit tensor so that the KD loss can be applied
    element-wise without index gymnastics.
    """
    B, T, V = logits.shape
    out = torch.zeros(B, max_ans_len, V, dtype=logits.dtype, device=logits.device)
    for b in range(B):
        positions = answer_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [n_ans]
        n = min(positions.size(0), max_ans_len)
        if n > 0:
            out[b, :n] = logits[b, positions[:n]]
    return out  # [B, max_ans_len, V]


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def distill_train(train_cfg, vlm_cfg, distill_cfg):

    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(
        train_cfg, vlm_cfg, distill_cfg
    )

    if is_dist():
        if is_master():
            print("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            print("All workers ready.")

    run_name = get_run_name(train_cfg, vlm_cfg, distill_cfg)

    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM-distill",
            config={
                "VLMConfig":     asdict(vlm_cfg),
                "TrainConfig":   asdict(train_cfg),
                "DistillConfig": asdict(distill_cfg),
            },
            name=run_name,
        )

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True

    # ── Teacher (load BEFORE DDP-wrapping the student) ────────────────────────
    if is_master():
        print(f"Loading teacher: {distill_cfg.teacher_model_id}")
    teacher = build_teacher(
        teacher_model_id=distill_cfg.teacher_model_id,
        base_vocab_size=distill_cfg.base_vocab_size,
        student_device=device,
        ensemble_teacher_ids=distill_cfg.ensemble_teacher_ids,
        ensemble_teacher_weights=distill_cfg.ensemble_teacher_weights,
    )
    if is_master():
        print("Teacher loaded and frozen.")

    # ── Student ───────────────────────────────────────────────────────────────
    if train_cfg.resume_from_vlm_checkpoint:
        print(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    if is_master():
        print(f"Student: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Loss weighting module ─────────────────────────────────────────────────
    # Count how many loss terms we will combine
    num_losses = (1 if distill_cfg.ce_weight > 0 else 0) + 1  # CE + KD
    weighting = get_weighting_strategy(distill_cfg.weighting_strategy, num_losses)
    weighting = weighting.to(device)

    # ── Optimizer (named param groups) ────────────────────────────────────────
    param_groups = []
    if train_cfg.lr_mp > 0:
        param_groups.append({"name": "mp",  "params": list(model.MP.parameters()),
                             "lr": train_cfg.lr_mp})
    else:
        for p in model.MP.parameters():
            p.requires_grad_(False)

    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({"name": "vit", "params": list(model.vision_encoder.parameters()),
                             "lr": train_cfg.lr_vision_backbone})
    else:
        for p in model.vision_encoder.parameters():
            p.requires_grad_(False)

    if train_cfg.lr_language_backbone > 0:
        param_groups.append({"name": "lm",  "params": list(model.decoder.parameters()),
                             "lr": train_cfg.lr_language_backbone})
    else:
        for p in model.decoder.parameters():
            p.requires_grad_(False)

    # Add weighting module params (only HeteroscedasticWeighting has any)
    weighting_params = list(weighting.parameters())
    if weighting_params:
        param_groups.append({"name": "weighting", "params": weighting_params, "lr": 1e-3})

    optimizer = optim.AdamW(param_groups)
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    # ── Move student to device, compile, wrap ─────────────────────────────────
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        print("Wrapping student for DDP")
        model = wrap_model(model)

    # ── Distillation loss function ─────────────────────────────────────────────
    kd_loss_fn = get_distill_loss(distill_cfg.distill_loss)
    if is_master():
        print(f"KD loss: {distill_cfg.distill_loss}  "
              f"T={distill_cfg.temperature}  "
              f"λ_kd={distill_cfg.distill_weight}  λ_ce={distill_cfg.ce_weight}  "
              f"weighting={distill_cfg.weighting_strategy}")

    # ── Training state ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_model_path = None
    logged_eval_steps = set()
    global_step = 0
    epoch = 0

    accumulated_stats = {
        "tokens_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
        "kd_loss": [],
        "ce_loss": [],
    }

    autocast_context = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float16,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start = time.time()
        model.train()
        weighting.train()
        total_train_loss  = 0.0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        train_heartbeat_microbatches = 25
        print("Starting training loop", flush=True)

        for i, batch in enumerate(synchronized_dataloader_step(iter_train_loader, is_dist())):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            batch_start = time.time()

            # Move tensor batch fields to device
            input_ids      = batch["input_ids"].to(device)
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answer_mask    = batch["answer_mask"].to(device)   # [B, T_seq] bool
            images         = batch["images"]
            data_load_time = time.time() - data_load_start

            # DDP gradient sync control
            if is_dist() and train_cfg.gradient_accumulation_steps > 1 and not is_update_step:
                no_sync_ctx = model.no_sync()
            else:
                no_sync_ctx = contextlib.nullcontext()

            fw_bw_start = time.time()

            # ── Forward pass ──────────────────────────────────────────────────
            with autocast_context:
                with no_sync_ctx:
                    # Student: always returns full vocab logits now
                    student_logits, ce_loss = model(
                        input_ids, images,
                        attention_mask=attention_mask,
                        targets=labels,
                    )
                    # student_logits: [B, T_seq, V_student=49218]

                    # ── Teacher forward (outside autocast, always fp32) ────────
                    # We exit autocast for the teacher so it stays in its native
                    # bf16 precision (set at load time) and logits are cast to
                    # fp32 inside get_answer_logits().
            with torch.no_grad():
                # teacher returns [B, T_answer, base_vocab_size] fp32 on `device`
                teacher_logits = teacher.get_answer_logits(batch)

            # ── Back to autocast for the KD loss computation ───────────────────
            with autocast_context:
                with no_sync_ctx:
                    T_answer = teacher_logits.size(1)

                    # Truncate student logits to base vocab (removes image tokens)
                    # and gather only the answer positions
                    s_base = student_logits[:, :, :distill_cfg.base_vocab_size]  # [B, T_seq, 49152]
                    s_ans  = gather_answer_logits(s_base, answer_mask, T_answer)  # [B, T_ans, 49152]

                    # Build a mask over the teacher's answer dimension
                    # (zero-padded positions have all-zero logit rows)
                    t_ans_mask = (teacher_logits.abs().sum(-1) > 0).float()  # [B, T_ans]

                    # KD loss
                    kd_loss = kd_loss_fn(
                        student_logits=s_ans.float(),
                        teacher_logits=teacher_logits.float(),
                        answer_mask=t_ans_mask,
                        temperature=distill_cfg.temperature,
                        # TAID params
                        taid_alpha_start=distill_cfg.taid_alpha_start,
                        taid_alpha_end=distill_cfg.taid_alpha_end,
                        global_step=global_step,
                        total_steps=train_cfg.max_training_steps,
                        # DKD params
                        dkd_alpha=distill_cfg.dkd_alpha,
                        dkd_beta=distill_cfg.dkd_beta,
                        labels=labels,
                    )

                    # Combine losses
                    losses = []
                    if distill_cfg.ce_weight > 0 and ce_loss is not None:
                        losses.append(distill_cfg.ce_weight * ce_loss)
                    losses.append(distill_cfg.distill_weight * kd_loss)
                    total_loss = weighting(losses)

            if train_cfg.gradient_accumulation_steps > 1:
                total_loss = total_loss / train_cfg.gradient_accumulation_steps

            total_loss.backward()
            fw_bw_time = time.time() - fw_bw_start

            post_start = time.time()
            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, max_norm=train_cfg.max_grad_norm
                    )

                # Update LR for named param groups
                for pg in optimizer.param_groups:
                    name = pg.get("name", "")
                    if name == "mp":
                        pg["lr"] = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                    elif name == "vit":
                        pg["lr"] = get_lr(global_step, train_cfg.lr_vision_backbone, train_cfg.max_training_steps)
                    elif name == "lm":
                        pg["lr"] = get_lr(global_step, train_cfg.lr_language_backbone, train_cfg.max_training_steps)
                    # weighting group keeps constant lr

                optimizer.step()
                optimizer.zero_grad()

            # Unscale losses for logging
            batch_total_loss = total_loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_total_loss *= train_cfg.gradient_accumulation_steps
            total_train_loss += batch_total_loss

            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_start

            batch_duration = time.time() - batch_start
            tokens_per_second = get_world_size() * num_tokens / batch_duration
            images_per_sample = [len(img_pack) for img_pack in images]

            # Heartbeat logging
            if is_master():
                if i == 0:
                    n_img = sum(len(s) for s in images)
                    print(
                        f"[distill] First micro-batch: input_ids={tuple(input_ids.shape)} "
                        f"image_lists={len(images)} total_imgs={n_img} "
                        f"t_ans_len={T_answer} epoch={epoch}",
                        flush=True,
                    )
                elif (i + 1) % train_heartbeat_microbatches == 0:
                    print(
                        f"[distill] step={global_step} micro={i+1} "
                        f"loss={batch_total_loss:.4f} "
                        f"kd={kd_loss.item():.4f} "
                        f"ce={ce_loss.item() if ce_loss is not None else 0:.4f} "
                        f"tok/s≈{tokens_per_second:.0f}",
                        flush=True,
                    )

            # Accumulate stats
            accumulated_stats["tokens_per_second"].append(tokens_per_second)
            accumulated_stats["data_load_time"].append(data_load_time)
            accumulated_stats["fw_bw_time"].append(fw_bw_time)
            accumulated_stats["post_process_time"].append(post_process_time)
            accumulated_stats["images_per_sample"].extend(images_per_sample)
            accumulated_stats["kd_loss"].append(kd_loss.item())
            if ce_loss is not None:
                accumulated_stats["ce_loss"].append(ce_loss.item())

            # ── Evaluation ────────────────────────────────────────────────────
            if train_cfg.eval_in_epochs and global_step % train_cfg.eval_interval == 0 and is_update_step:
                print("Starting evaluation")
                model.eval()
                weighting.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    total_val_loss = 0.0
                    val_batches = 0
                    for val_batch in synchronized_dataloader_step(iter_val_loader, is_dist()):
                        if val_batches > 64:
                            break
                        v_input_ids      = val_batch["input_ids"].to(device)
                        v_labels         = val_batch["labels"].to(device)
                        v_attention_mask = val_batch["attention_mask"].to(device)
                        v_answer_mask    = val_batch["answer_mask"].to(device)
                        v_images         = val_batch["images"]

                        with autocast_context:
                            v_student_logits, v_ce_loss = model(
                                v_input_ids, v_images,
                                attention_mask=v_attention_mask,
                                targets=v_labels,
                            )

                        v_teacher_logits = teacher.get_answer_logits(val_batch)
                        v_T_answer = v_teacher_logits.size(1)

                        with autocast_context:
                            v_s_base = v_student_logits[:, :, :distill_cfg.base_vocab_size]
                            v_s_ans  = gather_answer_logits(v_s_base, v_answer_mask, v_T_answer)
                            v_t_mask = (v_teacher_logits.abs().sum(-1) > 0).float()
                            v_kd_loss = kd_loss_fn(
                                student_logits=v_s_ans.float(),
                                teacher_logits=v_teacher_logits.float(),
                                answer_mask=v_t_mask,
                                temperature=distill_cfg.temperature,
                                taid_alpha_start=distill_cfg.taid_alpha_start,
                                taid_alpha_end=distill_cfg.taid_alpha_end,
                                global_step=global_step,
                                total_steps=train_cfg.max_training_steps,
                                dkd_alpha=distill_cfg.dkd_alpha,
                                dkd_beta=distill_cfg.dkd_beta,
                                labels=v_labels,
                            )
                            v_losses = []
                            if distill_cfg.ce_weight > 0 and v_ce_loss is not None:
                                v_losses.append(distill_cfg.ce_weight * v_ce_loss)
                            v_losses.append(distill_cfg.distill_weight * v_kd_loss)
                            v_total_loss = weighting(v_losses)

                        total_val_loss += v_total_loss.item()
                        val_batches += 1

                    iter_val_loader = iter(val_loader)
                    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss

                    checkpoint_path_step = ""
                    if is_master():
                        checkpoint_path_step = os.path.join(
                            distill_cfg.distill_checkpoint_path, run_name, f"step_{global_step}"
                        )
                        os.makedirs(checkpoint_path_step, exist_ok=True)
                        save_m = model.module if is_dist() else model
                        save_m.save_pretrained(checkpoint_path_step)
                        print(f"Step {global_step}: val_loss={avg_val_loss:.4f}  "
                              f"checkpoint saved to {checkpoint_path_step}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            best_model_path = checkpoint_path_step

                    if is_master() and train_cfg.log_wandb:
                        run.log({"val/total_loss": avg_val_loss}, step=global_step)

                model.train()
                weighting.train()

            # ── Stats logging ─────────────────────────────────────────────────
            if global_step % train_cfg.stats_log_interval == 0 and is_update_step and accumulated_stats["tokens_per_second"]:
                stats = {}
                for key in ["tokens_per_second", "data_load_time", "fw_bw_time",
                            "post_process_time", "images_per_sample"]:
                    vals = accumulated_stats[key]
                    if is_dist():
                        all_v = [v for sublist in dist_gather(vals) for v in sublist]
                    else:
                        all_v = vals
                    stats[f"avg_{key}"] = mean(all_v) if all_v else 0

                avg_kd = mean(accumulated_stats["kd_loss"]) if accumulated_stats["kd_loss"] else 0
                avg_ce = mean(accumulated_stats["ce_loss"]) if accumulated_stats["ce_loss"] else 0

                if is_master():
                    print(f"[stats] step={global_step} avg_kd={avg_kd:.4f} avg_ce={avg_ce:.4f} "
                          f"tok/s={stats['avg_tokens_per_second']:.0f}")
                    if train_cfg.log_wandb:
                        run.log({
                            "train/kd_loss":  avg_kd,
                            "train/ce_loss":  avg_ce,
                            **{f"training_stats/{k}": v for k, v in stats.items()},
                        }, step=global_step)

                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # ── Batch loss logging ─────────────────────────────────────────────
            if is_update_step:
                gathered_loss = dist_mean_scalar(batch_total_loss) if is_dist() else batch_total_loss
                if is_master() and train_cfg.log_wandb:
                    run.log({
                        "train/total_loss": gathered_loss,
                        **({"grad_norm": grad_norm} if train_cfg.max_grad_norm else {}),
                    }, step=global_step)

                global_step += 1
                if global_step >= train_cfg.max_training_steps:
                    break

            data_load_start = time.time()

        iter_train_loader = iter(train_loader)
        avg_train_loss = total_train_loss / max(i, 1)
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss

        epoch_dur  = time.time() - epoch_start
        total_toks = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed

        if is_master():
            print(f"Epoch {epoch} | step {global_step}/{train_cfg.max_training_steps} | "
                  f"train_loss={avg_train_loss:.4f} | {total_toks/epoch_dur:.0f} tok/s")
            if train_cfg.log_wandb:
                run.log({"epoch/train_loss": avg_train_loss,
                         "epoch/duration":   epoch_dur,
                         "epoch/tokens_per_second": total_toks / epoch_dur})

    # ── End of training ────────────────────────────────────────────────────────
    if is_master():
        print("Training complete.")
        if best_model_path and vlm_cfg.hf_repo_name:
            print(f"Pushing best model from {best_model_path} to Hub...")
            hf_model = VisionLanguageModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)
        if train_cfg.log_wandb:
            run.finish()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global PG_CPU
    parser = argparse.ArgumentParser(description="nanoVLM distillation training")

    # ── Standard train.py args ────────────────────────────────────────────────
    parser.add_argument("--lr_mp",                  type=float)
    parser.add_argument("--lr_vision_backbone",     type=float)
    parser.add_argument("--lr_language_backbone",   type=float)
    parser.add_argument("--vlm_checkpoint_path",    type=str)
    parser.add_argument("--compile",                type=bool)
    parser.add_argument("--no_log_wandb",           action="store_true")
    parser.add_argument("--resume_from_vlm_checkpoint", action="store_true")
    parser.add_argument("--train_dataset_path",     type=str)
    parser.add_argument("--train_dataset_name",     type=str)
    parser.add_argument("--val_size",               type=int)
    parser.add_argument("--batch_size",             type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--max_training_steps",     type=int)
    parser.add_argument("--eval_interval",          type=int)
    parser.add_argument("--relevance_min_rating",   type=int)
    parser.add_argument("--image_correspondence_min_rating", type=int)
    parser.add_argument("--visual_dependency_min_rating",    type=int)
    parser.add_argument("--formatting_min_rating",  type=int)

    # ── Distillation args ─────────────────────────────────────────────────────
    parser.add_argument("--teacher_model_id",       type=str,
                        default="HuggingFaceTB/SmolVLM2-1.7B-Instruct",
                        help="HF model ID for the teacher (must share base vocab with student)")
    parser.add_argument("--distill_loss",           type=str, default="fkl",
                        help="KD loss: fkl | rkl | js | tvd | taid | dkd")
    parser.add_argument("--distill_weight",         type=float, default=1.0,
                        help="Weight for the KD loss term")
    parser.add_argument("--ce_weight",              type=float, default=1.0,
                        help="Weight for the CE loss term (0 = pure KD)")
    parser.add_argument("--temperature",            type=float, default=2.0,
                        help="Softmax temperature for KD (>1 softens distributions)")
    parser.add_argument("--weighting_strategy",     type=str, default="equal",
                        help="Loss weighting: equal | heteroscedastic")
    parser.add_argument("--taid_alpha_start",       type=float, default=0.0)
    parser.add_argument("--taid_alpha_end",         type=float, default=1.0)
    parser.add_argument("--dkd_alpha",              type=float, default=1.0)
    parser.add_argument("--dkd_beta",               type=float, default=5.0)
    parser.add_argument("--max_sample_length",      type=int, default=2048)
    parser.add_argument("--distill_checkpoint_path", type=str, default="checkpoints_distill")
    parser.add_argument("--ensemble_teacher_ids",   type=str, default=None,
                        help="Comma-separated teacher model IDs for EnsembleTeacher")
    parser.add_argument("--ensemble_teacher_weights", type=str, default=None,
                        help="Comma-separated weights for each ensemble teacher")

    args = parser.parse_args()

    vlm_cfg     = config.VLMConfig()
    train_cfg   = config.TrainConfig()
    distill_cfg = config.DistillConfig()

    # Apply standard train args
    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_vision_backbone is not None:
        train_cfg.lr_vision_backbone = args.lr_vision_backbone
    if args.lr_language_backbone is not None:
        train_cfg.lr_language_backbone = args.lr_language_backbone
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb:
        train_cfg.log_wandb = False
    if args.resume_from_vlm_checkpoint:
        train_cfg.resume_from_vlm_checkpoint = True
        vlm_cfg.vlm_load_backbone_weights = False
    if args.train_dataset_path is not None:
        train_cfg.train_dataset_path = args.train_dataset_path
    if args.train_dataset_name is not None:
        train_cfg.train_dataset_name = (args.train_dataset_name,)
    if args.val_size is not None:
        train_cfg.val_size = args.val_size
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_training_steps is not None:
        train_cfg.max_training_steps = args.max_training_steps
    if args.eval_interval is not None:
        train_cfg.eval_interval = args.eval_interval
    if args.relevance_min_rating is not None:
        train_cfg.relevance_min_rating = args.relevance_min_rating
    if args.image_correspondence_min_rating is not None:
        train_cfg.image_correspondence_min_rating = args.image_correspondence_min_rating
    if args.visual_dependency_min_rating is not None:
        train_cfg.visual_dependency_min_rating = args.visual_dependency_min_rating
    if args.formatting_min_rating is not None:
        train_cfg.formatting_min_rating = args.formatting_min_rating

    # Apply distillation args
    if args.teacher_model_id is not None:
        distill_cfg.teacher_model_id = args.teacher_model_id
    if args.distill_loss is not None:
        distill_cfg.distill_loss = args.distill_loss
    if args.distill_weight is not None:
        distill_cfg.distill_weight = args.distill_weight
    if args.ce_weight is not None:
        distill_cfg.ce_weight = args.ce_weight
    if args.temperature is not None:
        distill_cfg.temperature = args.temperature
    if args.weighting_strategy is not None:
        distill_cfg.weighting_strategy = args.weighting_strategy
    if args.taid_alpha_start is not None:
        distill_cfg.taid_alpha_start = args.taid_alpha_start
    if args.taid_alpha_end is not None:
        distill_cfg.taid_alpha_end = args.taid_alpha_end
    if args.dkd_alpha is not None:
        distill_cfg.dkd_alpha = args.dkd_alpha
    if args.dkd_beta is not None:
        distill_cfg.dkd_beta = args.dkd_beta
    if args.max_sample_length is not None:
        distill_cfg.max_sample_length = args.max_sample_length
    if args.distill_checkpoint_path is not None:
        distill_cfg.distill_checkpoint_path = args.distill_checkpoint_path
    if args.ensemble_teacher_ids is not None:
        distill_cfg.ensemble_teacher_ids = args.ensemble_teacher_ids
    if args.ensemble_teacher_weights is not None:
        distill_cfg.ensemble_teacher_weights = args.ensemble_teacher_weights

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        PG_CPU = dist.new_group(backend="gloo")

    if is_master():
        print("--- VLM Config ---");    print(vlm_cfg)
        print("--- Train Config ---");  print(train_cfg)
        print("--- Distill Config ---"); print(distill_cfg)

    distill_train(train_cfg, vlm_cfg, distill_cfg)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()
