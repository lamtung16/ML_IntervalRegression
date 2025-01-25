import os
import shutil
import pandas as pd

# Define the stats folder and output file paths
stats_folder = "stats"
output_csv = "stats.csv"

# List to hold data
all_data = []

# Header for the combined CSV
header = ["dataset", "n_layer", "n_neuron", "fold", "val_loss", "time"]

# Traverse through each subfolder in the stats directory
if os.path.exists(stats_folder):
    for subfolder in os.listdir(stats_folder):
        subfolder_path = os.path.join(stats_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Locate the CSV file in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subfolder_path, file)
                    # Read the single line of the CSV
                    with open(file_path, "r") as f:
                        line = f.readline().strip().split(",")
                        all_data.append(line)

# Create a DataFrame and save it to a new CSV file
if all_data:
    df = pd.DataFrame(all_data, columns=header)
    df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved as {output_csv}")
else:
    print("No data found to combine.")

# Delete the stats folder
try:
    shutil.rmtree(stats_folder)
    print(f"Deleted folder: {stats_folder}")
except Exception as e:
    print(f"Error deleting folder {stats_folder}: {e}")