import pandas as pd
import os
from itertools import product

# CREATE PARAMETERS CSV
# Define hyperparameters
folder_path = '../../data'
datasets = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
param_combinations = []

# Iterate over each dataset and collect parameter combinations
for dataset in datasets:
    fold_df = pd.read_csv(os.path.join(folder_path, dataset, 'folds.csv'))
    
    test_fold = sorted(fold_df['fold'].unique())
    num_layers = [1, 2]
    layer_size = [4, 8]

    # Create parameter grid
    combinations = product([dataset], num_layers, layer_size, test_fold)
    param_combinations.extend(combinations)

# Create DataFrame
params_df = pd.DataFrame(param_combinations, columns=['dataset', 'num_layer', 'layer_size', 'test_fold'])

# Check for completed rows
predictions_dir = 'predictions_cv'  # Set the base directory for predictions
completed_indices = []

for index, row in params_df.iterrows():
    dataset = row['dataset']
    num_layer = row['num_layer']
    layer_size = row['layer_size']
    test_fold = row['test_fold']
    
    # Build the path to the prediction file
    prediction_file = os.path.join(predictions_dir, f"{dataset}/{num_layer}layers_{layer_size}neurons_fold{test_fold}.csv")
    
    # Check if the file exists
    if os.path.exists(prediction_file):
        completed_indices.append(index)

# Remove completed rows
params_df = params_df.drop(completed_indices).reset_index(drop=True)

# Save the updated DataFrame to CSV
params_df.to_csv("params.csv", index=False)

# CREATE RUN_ONE.SH
# Define job parameters
n_tasks, ncol = params_df.shape

# Create output directory for SLURM logs if it doesn't exist
output_dir = 'slurm-out'
os.makedirs(output_dir, exist_ok=True)

# Create SLURM script
run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output={output_dir}/slurm-%A_%a.out
#SBATCH --error={output_dir}/slurm-%A_%a.out
#SBATCH --job-name=mlp

python run_one.py $SLURM_ARRAY_TASK_ID
"""

# Write the SLURM script to a file
run_one_sh = os.path.join("run.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)