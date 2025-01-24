import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.model_selection import KFold


# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
params = params_df.iloc[param_row]

dataset    = params['dataset']
num_layer  = params['num_layer']
layer_size = params['layer_size']
test_fold  = params['test_fold']


# create folder for predictions
os.makedirs(f'predictions_cv/{dataset}', exist_ok=True)
os.makedirs(f'stats/{dataset}', exist_ok=True)


# Early stopping parameters
patience = 10
max_epochs = 200


# try to use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hinged Square Loss
class SquaredHingeLoss(nn.Module):
    def __init__(self, margin=0):
        super(SquaredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, y):
        low, high = y[:, 0:1], y[:, 1:2]
        loss_low = torch.relu(low - predicted + self.margin)
        loss_high = torch.relu(predicted - high + self.margin)
        loss = loss_low + loss_high
        return torch.mean(torch.square(loss))


# MLP models
class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))  # Output layer

        self.model = nn.Sequential(*layers)  # Combine layers into a sequential model

    def forward(self, x):
        return self.model(x)


# Load features, target and fold
folds_df = pd.read_csv(f'../../data/{dataset}/folds.csv')
target_df = pd.read_csv(f'../../data/{dataset}/targets.csv')
features_df = pd.read_csv(f'../../data/{dataset}/features.csv')


# Prepare CSV file for logging
report_path = f'stats/{dataset}/{num_layer}layers_{layer_size}neurons_fold{test_fold}.csv'
report_header = ['dataset', 'num_layer', 'layer_size', 'test_fold', 'val_loss', 'time']
if not os.path.exists(report_path):
    pd.DataFrame(columns=report_header).to_csv(report_path, index=False)


# main
# Record start time
fold_start_time = time.time()

# Split data into training and test sets based on the fold
train_ids = folds_df[folds_df['fold'] != test_fold].index
test_ids = folds_df[folds_df['fold'] == test_fold].index

# Prepare train and test sequences as arrays
features_train = features_df.loc[train_ids].values
target_train   = target_df.loc[train_ids].values
features_test  = features_df.loc[test_ids].values

# Normalize training features
scaler = MinMaxScaler()  # Create scaler instance
features_train = scaler.fit_transform(features_train)  # Fit on training data
features_test = scaler.transform(features_test)  # Transform test data using the same parameters

kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_models = []
total_best_val_loss = 0
for train_idx, val_idx in kf.split(features_train):
    best_val_loss = float('inf')
    model = MLP(input_size=features_train.shape[1], layer_sizes=[layer_size] * num_layer)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = SquaredHingeLoss()
    patience_counter = 0
    best_model_state = None
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(features_train[train_idx], dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(target_train[train_idx], dtype=torch.float32))
        loss.backward()
        optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(features_train[val_idx], dtype=torch.float32))
            val_loss = criterion(val_outputs, torch.tensor(target_train[val_idx], dtype=torch.float32))
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save best model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Stop training if validation loss does not improve
        if patience_counter >= patience:
            break
    total_best_val_loss += best_val_loss
    model.load_state_dict(best_model_state)
    best_models.append(model)

model_outputs = []
for model in best_models:
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(features_test, dtype=torch.float32))
        model_outputs.append(output.numpy())

# Compute the mean output of the best models
target_mat_pred = np.mean(np.array(model_outputs), axis=0).flatten()
prediction = pd.DataFrame({'pred': target_mat_pred})
prediction.to_csv(f"predictions_cv/{dataset}/{num_layer}layers_{layer_size}neurons_fold{test_fold}.csv", index=False)

# Record end time and calculate elapsed time
elapsed_time = time.time() - fold_start_time

# Log the results
report_entry = {
    'dataset': dataset,
    'num_layers': num_layer,
    'layer_size': layer_size,
    'test_fold': test_fold,
    'val_loss': total_best_val_loss.item(),
    'time': elapsed_time
}
pd.DataFrame([report_entry]).to_csv(report_path, mode='w', header=False, index=False)