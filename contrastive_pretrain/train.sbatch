#!/bin/bash
#SBATCH --job-name=contrastive_pretrain
#SBATCH --output=logs/contrastive_pretrain.out
#SBATCH --error=logs/contrastive_pretrain.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate proto2seg

# Run training
python train.py --config config/config_bcss.yaml