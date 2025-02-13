import pandas as pd
import numpy as np
import random
import sys
import xgboost as xgb
import os
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import copy

# Create folder for predictions
os.makedirs('predictions', exist_ok=True)

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
params = params_df.iloc[param_row]

dataset = params['dataset']
test_fold = params['test_fold']

# Load features, target, and fold
folds_df = pd.read_csv(f'../../data/{dataset}/folds.csv')
target_df = pd.read_csv(f'../../data/{dataset}/targets.csv')
features_df = pd.read_csv(f'../../data/{dataset}/features.csv')

# Split data into training and test sets based on the fold
train_ids = folds_df[folds_df['fold'] != test_fold].index
test_ids = folds_df[folds_df['fold'] == test_fold].index

# Prepare train and test sequences as arrays
X_train = features_df.loc[train_ids].values
y_train = target_df.loc[train_ids].values
X_test = features_df.loc[test_ids].values

# Define parameter grid
param_grid = {
    'objective': ['survival:aft'],
    'eval_metric': ['aft-nloglik'],
    'tree_method': ['hist'],
    'aft_loss_distribution': ['normal'],
    'aft_loss_distribution_scale': [0.5, 1.0, 1.5],
    'learning_rate': [0.001, 0.1, 1.0],
    'max_depth': [2, 4, 6, 8, 10, 20],
    'min_child_weight': [0.001, 1.0, 10.0],
    'reg_alpha': [0.001, 1.0, 10.0],
    'reg_lambda': [0.001, 1.0, 10.0]
}

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

best_models = []
for train_idx, val_idx in kf.split(X_train):

    dtrain = xgb.DMatrix(X_train[train_idx])
    dtrain.set_float_info('label_lower_bound', y_train[train_idx][:, 0])
    dtrain.set_float_info('label_upper_bound', y_train[train_idx][:, 1])

    dval = xgb.DMatrix(X_train[val_idx])
    dval.set_float_info('label_lower_bound', y_train[val_idx][:, 0])
    dval.set_float_info('label_upper_bound', y_train[val_idx][:, 1])

    best_model = None
    best_val_loss = float('inf')
    for params in ParameterGrid(param_grid):
        model = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        val_loss = model.best_score
        if val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss
    best_models.append(best_model)

# Make predictions on the test set
dtest = xgb.DMatrix(X_test)
y_pred = np.zeros(X_test.shape[0])
for model in best_models:
    y_pred += model.predict(dtest)
y_pred /= len(best_models)
prediction_df = pd.DataFrame({'pred': y_pred})
prediction_df.to_csv(f"predictions/{dataset}.{test_fold}.csv", index=False)