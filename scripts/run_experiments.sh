#!/bin/bash
# run_experiments.sh

# SLURM parameters
#SBATCH --job-name=ts_forecast
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-15  # 16 total jobs (4 model groups × 4 datasets)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1  # Request 1 GPU for deep learning models

# Create arrays of parameters
datasets=("BPI2019_1" "BPI2019_2" "BPI2019_3" "BPI2019_4")
model_groups=("baseline" "statistical" "regression" "deep_learning")
time_interval="1-day"

# Calculate indices for dataset and model_group
dataset_idx=$((SLURM_ARRAY_TASK_ID / 4))
model_group_idx=$((SLURM_ARRAY_TASK_ID % 4))

# Get current dataset and model_group
dataset=${datasets[$dataset_idx]}
model_group=${model_groups[$model_group_idx]}

# Load required modules
module load python/3.8
module load cuda/11.3  # Adjust version as needed

# Activate virtual environment
source /path/to/your/venv/bin/activate

# Set different resource configurations based on model group
if [ "$model_group" = "deep_learning" ]; then
    # Deep learning models need GPU
    export CUDA_VISIBLE_DEVICES=0
elif [ "$model_group" = "statistical" ]; then
    # Statistical models need more CPU
    export MKL_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export OMP_NUM_THREADS=4
else
    # Baseline and regression models need less resources
    export MKL_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    export OMP_NUM_THREADS=2
fi

# Run the training script
python train.py \
    --dataset "$dataset" \
    --time_interval "$time_interval" \
    --model_group "$model_group" \
    --config "config/base_config.yaml"