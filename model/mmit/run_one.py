import pandas as pd
import numpy as np
import random
import os
import sys
from sklearn.model_selection import KFold
# from mmit_functions import mmit
from mmit import MaxMarginIntervalTree

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
params = params_df.iloc[param_row]

dataset    = params['dataset']
test_fold  = params['test_fold']

# create folder for predictions
os.makedirs(f'predictions', exist_ok=True)

# load data
folds_df = pd.read_csv(f'../../data/{dataset}/folds.csv')
features_df = pd.read_csv(f'../../data/{dataset}/features.csv').astype(np.float32)
target_df = pd.read_csv(f'../../data/{dataset}/targets.csv').astype(np.float32)
train_indices = folds_df[folds_df['fold'] != test_fold].index
test_indices = folds_df[folds_df['fold'] == test_fold].index

X_train = features_df.loc[train_indices].values
X_test = features_df.loc[test_indices].values
y_train = target_df.loc[train_indices].values

max_depths = [2, 5, 10, 15, 20, 25]
min_split_samples = [2, 5, 10, 20, 50]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = []
for train_idx, val_idx in kf.split(X_train):
    X_subtrain, X_val = X_train[train_idx], X_train[val_idx]
    y_subtrain, y_val = y_train[train_idx], y_train[val_idx]
    best_model = None
    best_hinge_error = float('inf')
    for max_depth in max_depths:
        for min_split_sample in min_split_samples:
            tree = MaxMarginIntervalTree(loss='squared_hinge', max_depth=max_depth, min_samples_split=min_split_sample)
            tree.fit(X_subtrain, y_subtrain)

            # Compute hinge error
            hinge_error = - tree.score(X_val, y_val)

            if hinge_error < best_hinge_error:
                best_hinge_error = hinge_error
                best_model = tree
    best_models.append(best_model)
target_mat_pred = np.mean([model.predict(X_test) for model in best_models], axis=0)
prediction = pd.DataFrame({'pred': target_mat_pred})
prediction.to_csv(f"predictions/{dataset}.{test_fold}.csv", index=False)