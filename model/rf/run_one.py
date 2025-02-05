import numpy as np
from sklearn.utils import resample
import pandas as pd
import random
import os
import sys
from mmit import MaxMarginIntervalTree

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

class MMIF:
    def __init__(self, n_trees=100, max_depth_values=[2, 5, 10, 15, 20, 25], min_split_samples=[2, 5, 10, 20, 50], feature_fraction=1/3, sample_ratio=0.66):
        self.n_trees = n_trees
        self.max_depth_values = max_depth_values
        self.min_split_samples = min_split_samples
        self.feature_fraction = feature_fraction
        self.sample_ratio = sample_ratio
        self.trees = []

    def fit(self, X, y):
        N, D = X.shape
        num_features = max(1, int(D * self.feature_fraction))
        
        for _ in range(self.n_trees):
            X_bootstrap, y_bootstrap = resample(X, y, n_samples=int(self.sample_ratio * N), replace=True)
            oob_indices = np.setdiff1d(np.arange(X.shape[0]), np.unique(np.where(X == X_bootstrap)[0]))
            X_oob = X[oob_indices]
            y_oob = y[oob_indices]
            
            best_tree, best_score = None, float('-inf')
            feature_indices = np.random.choice(D, num_features, replace=False)
            for depth in self.max_depth_values:
                for min_split in self.min_split_samples:
                    tree = MaxMarginIntervalTree(loss='squared_hinge', max_depth=depth, min_samples_split=min_split)
                    tree.fit(X_bootstrap[:, feature_indices], y_bootstrap)
                    score = tree.score(X_oob[:, feature_indices], y_oob)
                    if score > best_score:  # Higher score is better
                        best_tree, best_score = tree, score
            
            self.trees.append((best_tree, feature_indices, -best_score))

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        total_weight = 0
        
        for tree, feature_indices, error in self.trees:
            weight = 1 / (10*error + 1e-4)  # Use inverse error as weight
            predictions += weight * tree.predict(X[:, feature_indices])
            total_weight += weight
        
        return predictions / total_weight  # Weighted aggregation
    





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

rf = MMIF()
rf.fit(X_train, y_train)
target_mat_pred = rf.predict(X_test)

prediction = pd.DataFrame({'pred': target_mat_pred})
prediction.to_csv(f"predictions/{dataset}.{test_fold}.csv", index=False)