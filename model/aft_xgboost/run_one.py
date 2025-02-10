import pandas as pd
import numpy as np
import random
import sys
import xgboost as xgb
import os
from sklearn.model_selection import ParameterGrid

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
    'aft_loss_distribution': ['normal'],
    'aft_loss_distribution_scale': [0.5, 1.0, 1.5],
    'tree_method': ['hist'],
    'learning_rate': [0.001, 0.1, 1.0],
    'max_depth': [2, 4, 6, 8, 10],
    'min_child_weight': [0.001, 1.0, 10.0],
    'reg_alpha': [0.001, 1.0, 10.0],
    'reg_lambda': [0.001, 1.0, 10.0]
}

def perform_cross_validation(params, dtrain):
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=2000,
        nfold=5,
        early_stopping_rounds=50,
        as_pandas=True,
        seed=42  # Set random seed for reproducibility
    )
    return cv_results

y_lower_bound, y_upper_bound = y_train[:, 0], y_train[:, 1]

# Create DMatrix with bounds
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label_lower_bound', y_lower_bound)
dtrain.set_float_info('label_upper_bound', y_upper_bound)

# Perform cross-validation to find the best parameters
best_params = None
best_cv_score = float('inf')
for params in ParameterGrid(param_grid):
    cv_results = perform_cross_validation(params, dtrain)
    mean_score = cv_results['test-aft-nloglik-mean'].mean()  # Minimize the negative log-likelihood

    if mean_score < best_cv_score:
        best_cv_score = mean_score
        best_params = params

# Use the best parameters from cross-validation
params = best_params

bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=2000, 
    evals=[(dtrain, 'train')],
    early_stopping_rounds=50
)

# Create DMatrix for prediction
dtest = xgb.DMatrix(X_test)

# Predict
y_pred = bst.predict(dtest)
prediction_df = pd.DataFrame({'pred': y_pred})
prediction_df.to_csv(f"predictions/{dataset}.{test_fold}.csv", index=False)