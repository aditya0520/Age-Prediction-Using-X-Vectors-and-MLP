#!/bin/bash
#SBATCH -p GPU-shared               # Partition (queue) name
#SBATCH --gres=gpu:v100:2           # Request 2 GPUs
#SBATCH --cpus-per-task=10          # Request 10 CPU cores per task
#SBATCH --mem=40G                   # Memory allocation
#SBATCH -t 4:00:00                  # Maximum runtime (8 hours)
#SBATCH -o torchrun_output.log      # File for standard output
#SBATCH -e torchrun_error.log       # File for error messages

# Load Python module
module load gcc
module load AI/pytorch_23.02-1.13.1-py3

# Install Python packages (use --user to avoid permissions issues)
pip install --user pandas numpy tqdm wandb
pip install --user torch torchvision torchaudio

# Set NCCL environment variables for better performance and debugging
export NCCL_DEBUG=INFO                     # Enable detailed NCCL logs
export NCCL_SOCKET_IFNAME=ens10f0             # Specify network interface (adjust if needed)
export NCCL_IB_DISABLE=1                   # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1                  # Disable peer-to-peer communication (debugging)

export MASTER_PORT=29550                   # Set a master port
export CUDA_VISIBLE_DEVICES=0,1

# Run the distributed training job with torchrun
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT MLP_train.py