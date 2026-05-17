#!/bin/bash
#SBATCH --job-name=train_nanoVLM_torchrun
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/train_nanoVLM/%A_%a.out
#SBATCH --error=logs/train_nanoVLM/%A_%a.err
#SBATCH --time=95:59:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
# Partition: `main` — override if your site uses a different default pool.
#SBATCH --partition=main
# GPU GRES on your nodes: gpu:nvidia_h100_80gb_hbm3:8 — if jobs fail to allocate, try
# replacing `--gpus-per-node=8` with:  #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --qos=normal

# echo "--- Starting parallel data copy on all nodes... ---"
# # This srun command launches the copy script on all 4 nodes simultaneously.
# # The shell will not proceed to the next line until ALL nodes have finished.
# srun --ntasks-per-node=1 bash -c '
#   mkdir -p /scratch/cache/FineVision && \
#   cd /fsx/.cache/HuggingFaceM4___fine_vision && \
#   find . -type f | parallel -j 16 rsync -R {} /scratch/cache/FineVision/
# '
# echo "--- All nodes have finished copying data. ---"

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

# Optional: uncomment and adjust for your cluster's module system.
# module load cuda/12.9

mkdir -p logs/train_nanoVLM

cd /home/reza/projects/nanoVLM

# Conda (non-interactive batch): hook then activate
eval "$(conda shell.bash hook)"
conda activate vlm-distill

export TOKENIZERS_PARALLELISM=false
# Token weights still load from the Hub unless cached — set before sbatch if needed:
# export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

# -------------------------------------------------------------------------------

# Get the master node's address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# From https://i.hsfzxjy.site/2021-03-10-obtain-a-random-unused-tcp-port-with-bash/
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export MASTER_PORT=$(unused_port)

# Run using torchrun on all allocated nodes
ulimit -n 99999
srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --no_log_wandb \
    --train_dataset_path /home/khashmi/data/qvac-vlm/finevision_doc_filter/presampled_dataset/fvdoc_v1_45-30-20-5_2026-05-12_imgcast \
    --train_dataset_name default

