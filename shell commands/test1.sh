#!/bin/bash
#SBATCH --job-name=data1
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C Titan
#SBATCH --gres=gpu:1

module load cuda12.1/toolkit

source $HOME/.bashrc
conda activate ML

cd /var/scratch/tbt204/TransferLearning


<<<<<<< HEAD
python compile_data.py 1
=======
python compile_data.py 1
>>>>>>> 5a187448875ae8fa6283d862c0cbc25a68899d39
