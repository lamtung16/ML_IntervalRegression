# Interval Regression

## Overview

Interval regression is a type of regression analysis where the target variable is an interval rather than a single point. Unlike point regression, which aims to model the data points as closely as possible, interval regression focuses on intersecting with all the intervals while maximizing the distance between the model and the interval endpoints (lower and upper limits).

### Key Differences between Point Regression and Interval Regression

- **Point Regression:** The goal is to fit a model that closely follows the data points, minimizing the error between the predicted values and the actual data points. The most common loss function used here is the Mean Squared Error (MSE).
  
- **Interval Regression:** The goal is to create a model that intersects with all given intervals, aiming to maximize the distance between the model and the interval endpoints. This approach uses a different loss function, specifically the squared hinge error.

### Visual Comparison

Point regression and interval regression can be visually distinguished by how the model aligns with data points versus intervals.

![Regression Comparison](https://arxiv.org/html/2408.00856v2/x2.png)
![Loss Function Comparison](https://arxiv.org/html/2408.00856v2/x3.png)

## Previous Methods

- **Linear Approaches:**
  - **Max Margin Interval Regression:** A linear approach that focuses on maximizing the margin between the predicted line and the interval boundaries.
  - **Max Margin Interval Regression L1 Regularization:** A linear approach that focuses on maximizing the margin between the predicted line and the interval boundaries using L1 regularization to choose the most impactful features.
  
- **Tree-Based Approaches:**
  - **Maximal Margin Interval Tree (MMIT):** A decision tree-based method that builds trees by maximizing the margin at each split to best fit the intervals.

## Proposed Method

The proposed method enhances interval regression using a multi-layer perceptron (MLP) model with feature selection. The process involves:

1. **Feature Selection:**
   - Employing `n` (I used `n = 10`) random forest models to determine feature importance.
   - Selecting features based on their importance scores, ensuring that the most impactful features are included in the model.

2. **Model Architecture:**
   - Implementing a multi-layer perceptron (MLP) to capture complex relationships between features and the target intervals.
   - The architecture allows for greater flexibility and capacity compared to linear or tree-based methods.

3. **Training Strategy:**
   - The MLP is trained by minimizing the squared hinge loss, which is specifically tailored for interval regression tasks.
   - The loss function encourages the model to not only intersect with the intervals but also to maintain a maximum margin from the interval boundaries.

4. **Performance:** 
   - Preliminary results indicate that this approach outperforms traditional methods, especially in cases with complex data patterns where linear or tree-based methods may fall short.

## Future Work

- **Hyperparameter Tuning:** Further exploration into optimal configurations for the MLP, including the number of layers, neurons per layer, and activation functions.
- **Generalization:** Evaluating the proposed method across different datasets to confirm its robustness and generalizability.

## References
- Information from [mmit paper](https://arxiv.org/abs/1710.04234).