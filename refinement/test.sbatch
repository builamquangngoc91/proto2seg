#!/bin/bash
#SBATCH --job-name=proto2seg_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gputype=a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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

# Set the experiment directory (change this to your trained model directory)
EXPERIMENT_DIR="logs/bcss/004_bcss_128_refinement"

# Run evaluation
python test.py \
    --dir ${EXPERIMENT_DIR} \
    --dataset-name bcss

echo "End time: $(date)"
echo "Job completed successfully"


