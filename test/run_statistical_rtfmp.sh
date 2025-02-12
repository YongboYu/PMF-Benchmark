#!/bin/bash

# SLURM parameters
#SBATCH --job-name=rtfmp_stat
#SBATCH --output=logs/slurm/%x_%A_%a.o
#SBATCH --error=logs/slurm/%x_%A_%a.e
#SBATCH --cluster=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=lp_lirisnlp
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yongbo.yu@student.kuleuven.be
#SBATCH --array=0-5  # For 6 statistical models

# Load required modules
module purge
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
conda activate pmf-benchmark
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Define models array
MODELS=(
    "prophet"
    "exp_smoothing"
    "auto_arima"
    "theta"
    "tbats"
    "four_theta"
)

# Get current model from array
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Create scratch directory with unique name per array task
JOB_SCRATCH_DIR="$VSC_SCRATCH/pmf_benchmark_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $JOB_SCRATCH_DIR

# Copy project files
rsync -av --exclude='.git' --exclude='data/raw' --exclude='data/interim' \
    $VSC_DATA/PMF_Benchmark_2025/ $JOB_SCRATCH_DIR/

cd $JOB_SCRATCH_DIR

# Set CPU configurations
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run single model
python train.py \
    --dataset "RTFMP" \
    --model_group "statistical" \
    --model "$MODEL" \
    --horizon 1 \
    --config "config/base_config.yaml"

# Copy results back
rsync -av $JOB_SCRATCH_DIR/results/ $VSC_DATA/PMF_Benchmark_2025/results/
rsync -av $JOB_SCRATCH_DIR/logs/ $VSC_DATA/PMF_Benchmark_2025/logs/

# Cleanup
rm -rf $JOB_SCRATCH_DIR

echo "All jobs completed successfully!" 