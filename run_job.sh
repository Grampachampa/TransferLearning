#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:01:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C Titan
#SBATCH --gres=gpu:1

module load cuda12.1/toolkit
module load cuDNN/cuda12.1


source $HOME/.bashrc
conda activate ML

cd /var/scratch/tbt204/TransferLearning


## python selfplay_rework.py --parameter

python <<EOF
import torch
print(torch.cuda.is_available())
EOF