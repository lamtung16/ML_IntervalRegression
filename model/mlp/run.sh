#!/bin/bash
#SBATCH --array=0-2729
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=slurm-out/slurm-%A_%a.out
#SBATCH --job-name=mlp

python run_one.py $SLURM_ARRAY_TASK_ID
