#!/bin/bash
#SBATCH --job-name=proto2seg_refine
#SBATCH --output=logs/refine_%j.out
#SBATCH --error=logs/refine_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gputype=a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.11

# Activate virtual environment (if using)
# source /path/to/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run refinement training
python train_seg.py --config configs/bcss.yaml

echo "End time: $(date)"
echo "Job completed successfully"


