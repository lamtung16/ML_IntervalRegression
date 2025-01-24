import numpy as np

class mmit:
    def __init__(self, max_depth=20, margin_length=0, p=2, min_split_sample=1):
        # Initialize hyperparameters
        self.max_depth = max_depth  # Maximum depth of the tree
        self.margin_length = margin_length  # Margin length for hinge error
        self.p = p  # Power for hinge error calculation
        self.min_split_sample = min_split_sample  # Minimum samples required to split a node
        self.tree = None  # Placeholder for the tree structure

    def relu(self, x):
        # ReLU function for hinge error calculation
        return np.maximum(0, x)

    def hinge_error(self, y_pred, y_low, y_up):
        # Calculate hinge error based on prediction and interval targets
        return self.relu(y_low - y_pred + self.margin_length) ** self.p + self.relu(y_pred - y_up + self.margin_length) ** self.p

    def best_split(self, X, y):
        # Determine the best feature and threshold for splitting the data
        _, n_features = X.shape
        best_feature, best_threshold = None, None
        min_error = float('inf')

        for feature in range(n_features):
            # Iterate over all unique thresholds for the current feature
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                # Ensure both splits are valid
                if np.sum(left_mask) >= self.min_split_sample and np.sum(right_mask) >= self.min_split_sample:
                    left_y = y[left_mask]
                    right_y = y[right_mask]

                    # Evaluate the split using hinge error
                    error = self.evaluate_split(left_y) + self.evaluate_split(right_y)

                    # Update the best split if current split has lower error
                    if error < min_error:
                        min_error = error
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def evaluate_split(self, y):
        # Evaluate the hinge error for a split
        y_low = y[:, 0] + self.margin_length
        y_up = y[:, 1] - self.margin_length
        valid_values = np.concatenate([y_low[y_low != -np.inf], y_up[y_up != np.inf]])

        if len(valid_values) == 0:
            # No valid values, return zero error
            return 0

        # Compute hinge error for each potential prediction value
        errors = [np.sum(self.hinge_error(val, y_low, y_up)) for val in valid_values]
        return np.min(errors)

    def predict_leaf_value(self, y):
        # Predict the value at a leaf node
        y_low = y[:, 0] + self.margin_length
        y_up = y[:, 1] - self.margin_length
        valid_values = np.concatenate([y_low[y_low != -np.inf], y_up[y_up != np.inf]])

        if len(valid_values) == 0:
            # Default prediction if no valid values
            return 0

        # Find the prediction value that minimizes hinge error
        errors = [np.sum(self.hinge_error(val, y_low, y_up)) for val in valid_values]
        return valid_values[np.argmin(errors)]

    def build_tree(self, X, y, depth):
        # Recursively build the decision tree
        if depth == self.max_depth or len(X) < self.min_split_sample:
            # Stop splitting if max depth is reached or not enough samples
            return {
                'type': 'leaf',
                'value': self.predict_leaf_value(y)
            }

        # Find the best split for the current node
        feature, threshold = self.best_split(X, y)
        if feature is None:
            # If no valid split, create a leaf node
            return {
                'type': 'leaf',
                'value': self.predict_leaf_value(y)
            }

        # Split the data into left and right subsets
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        # Recursively build left and right branches
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        # Fit the decision tree to the data
        self.tree = self.build_tree(X, y, 0)

    def predict_one(self, x, node):
        # Predict the target for a single instance
        if node['type'] == 'leaf':
            return node['value']

        # Traverse the tree based on feature thresholds
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        # Predict the target for all instances in the dataset
        return np.array([self.predict_one(x, self.tree) for x in X])