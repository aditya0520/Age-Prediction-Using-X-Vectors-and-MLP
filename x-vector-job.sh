#!/bin/bash
#SBATCH -p GPU-shared               # Partition (queue) name
#SBATCH --gres=gpu:v100-32:2        # Request 4 GPUs
#SBATCH --cpus-per-task=10           # Request 8 CPU cores
#SBATCH --mem=32G                   # Memory allocation
#SBATCH -t 8:00:00                  # Maximum runtime (8 hours)
#SBATCH -o torchrun_output.log      # File for standard output
#SBATCH -e torchrun_error.log       # File for error messages

# Load Python module
module load gcc
module load AI/pytorch_23.02-1.13.1-py3

# Install dependencies
pip install pandas
pip install torchaudio
pip install numpy
pip install tqdm
pip install wandb
pip install speechbrain

# Run the `torchrun` command
python x-vector.py