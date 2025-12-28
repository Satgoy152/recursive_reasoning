#!/bin/bash
#SBATCH --job-name=train_pretrain
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=180GB

# Activate your venv
source /home/sagoyal/research/recursive_reasoning/.venv/bin/activate

# Already in the right directory, just run the command
accelerate launch --num_processes=4 --mixed_precision=bf16 train_pretrain.py