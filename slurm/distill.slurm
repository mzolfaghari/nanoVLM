#!/bin/bash
#SBATCH --job-name=distill_nanoVLM
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/distill_nanoVLM/%A_%a.out
#SBATCH --error=logs/distill_nanoVLM/%A_%a.err
#SBATCH --time=95:59:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --partition=main
#SBATCH --qos=normal

# ── EFA / NCCL settings (same as nanoVLM.slurm) ──────────────────────────────
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_HUGE_PAGE=0
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_SOCKET_IFNAME=enp

export FI_EFA_ENABLE_SHM_TRANSFER=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# ── Environment ───────────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate vlm-distill

export TOKENIZERS_PARALLELISM=false

# HF token for gated datasets (FineVision etc.).
# Set this on the remote with:  echo "hf_..." > ~/.hf_token
# NEVER commit the actual token to git.
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null)

mkdir -p logs/distill_nanoVLM

cd /home/reza/projects/nanoVLM

# ── Port selection ─────────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) |
        shuf | head -n "$N"
}
export MASTER_PORT=$(unused_port)

ulimit -n 99999

# ─────────────────────────────────────────────────────────────────────────────
# Launch distillation
#
# Switch --distill_loss to compare methods:
#   fkl   — Forward KL (standard KD, default)
#   rkl   — Reverse KL (mode-seeking)
#   js    — Jensen-Shannon divergence
#   tvd   — Total Variation distance
#   taid  — Temperature-Annealing Interpolation (alpha ramps 0→1)
#   dkd   — Decoupled KD (TCKD + NCKD, best on classification-style tasks)
#
# Switch --weighting_strategy to compare loss combination methods:
#   equal           — uniform average (default)
#   heteroscedastic — learnable log-variance per loss term
# ─────────────────────────────────────────────────────────────────────────────
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    distill_train.py \
    --no_log_wandb \
    --train_dataset_path /home/khashmi/data/qvac-vlm/finevision_doc_filter/presampled_dataset/fvdoc_v1_45-30-20-5_2026-05-12_imgcast \
    --train_dataset_name default \
    --teacher_model_id HuggingFaceTB/SmolVLM2-1.7B-Instruct \
    --distill_loss fkl \
    --distill_weight 1.0 \
    --ce_weight 1.0 \
    --temperature 2.0 \
    --weighting_strategy equal \
    --max_sample_length 2048 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_training_steps 40000 \
    --eval_interval 500
