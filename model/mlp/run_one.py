import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
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

dataset = params['dataset']
test_fold = params['test_fold']

# Create folder for predictions
os.makedirs('predictions', exist_ok=True)

# Early stopping parameters
patience = 100
max_epochs = 10000

# Try to use GPU if available
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

# Load features, target, and fold
folds_df = pd.read_csv(f'../../data/{dataset}/folds.csv')
target_df = pd.read_csv(f'../../data/{dataset}/targets.csv')
features_df = pd.read_csv(f'../../data/{dataset}/features.csv')

# Split data into training and test sets based on the fold
train_ids = folds_df[folds_df['fold'] != test_fold].index
test_ids = folds_df[folds_df['fold'] == test_fold].index

# Prepare train and test sequences as arrays
features_train = features_df.loc[train_ids].values
target_train = target_df.loc[train_ids].values
features_test = features_df.loc[test_ids].values

# Normalize training features
scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Convert data to PyTorch tensors
features_train = torch.tensor(features_train, dtype=torch.float32).to(device)
target_train = torch.tensor(target_train, dtype=torch.float32).to(device)
features_test = torch.tensor(features_test, dtype=torch.float32).to(device)

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
best_models = []

for train_idx, val_idx in kf.split(features_train):
    best_model = None
    best_val_loss = float('inf')

    for num_layer in [1, 2]:
        for layer_size in [10, 20]:
            model = MLP(input_size=features_train.shape[1], layer_sizes=[layer_size] * num_layer).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = SquaredHingeLoss()
            patience_counter = 0
            best_model_state = None
            best_val_loss_model = float('inf')

            for epoch in range(max_epochs):
                # Training step
                model.train()
                optimizer.zero_grad()
                outputs = model(features_train[train_idx])
                loss = criterion(outputs, target_train[train_idx])
                loss.backward()
                optimizer.step()

                # Validation step
                model.eval()
                with torch.no_grad():
                    val_outputs = model(features_train[val_idx])
                    val_loss = criterion(val_outputs, target_train[val_idx])

                # Early stopping logic
                if val_loss < best_val_loss_model:
                    best_val_loss_model = val_loss.item()
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            if best_model_state:
                model.load_state_dict(best_model_state)

            if best_val_loss_model < best_val_loss:
                best_val_loss = best_val_loss_model
                best_model = model

    best_models.append(best_model)

# Predict on test data
model_outputs = []
for model in best_models:
    model.eval()
    with torch.no_grad():
        output = model(features_test)
        model_outputs.append(output.cpu().numpy())

# Compute the mean output of the best models
target_mat_pred = np.mean(np.array(model_outputs), axis=0).flatten()
prediction = pd.DataFrame({'pred': target_mat_pred})
prediction.to_csv(f"predictions/{dataset}.{test_fold}.csv", index=False)