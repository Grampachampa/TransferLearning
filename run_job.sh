#!/bin/bash
#SBATCH --job-name=40lostDeath
#SBATCH --time=14:45:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C Titan
#SBATCH --gres=gpu:1

module load cuda12.1/toolkit

source $HOME/.bashrc
conda activate ML

cd /var/scratch/tbt204/TransferLearning


python main.py --parameter

