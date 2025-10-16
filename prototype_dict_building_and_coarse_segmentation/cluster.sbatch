#!/bin/bash
#SBATCH --job-name=proto2seg_cluster
#SBATCH --output=logs/cluster_%j.out
#SBATCH --error=logs/cluster_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gputype=a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

# Step 1: Prototype identification (clustering)
echo "=========================================="
echo "Step 1: Prototype Identification"
echo "=========================================="
python cluster.py --config config/bcss.yaml

# Step 2: Coarse segmentation with cluster query
echo "=========================================="
echo "Step 2: Coarse Segmentation (Cluster Query)"
echo "=========================================="
python coarse_seg_cluster_query.py --config config/bcss.yaml --n 5

echo "End time: $(date)"
echo "Job completed successfully"


