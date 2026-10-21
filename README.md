# Weekly-Challenge-39-GBM

## Description
For Week 39, I implemented the mathematical engine that powers state-of-the-art predictive models (like XGBoost and LightGBM): the **Gradient Boosting Machine (GBM)**. 

Unlike Random Forests that average independent trees, or AdaBoost which changes the statistical weights of misclassified students, GBM takes a direct mathematical approach to error correction. It trains a sequence of trees where each new tree is built exclusively to predict the *residual errors* (the exact difference between the actual value and the previous prediction) of the ensemble before it. 

## How it works (The Math)
GBM is fundamentally Gradient Descent applied in a function space.
1. **Initialization:** The model starts with a naive base prediction (usually the mean of the target variable $y$).
   $$F_0(x) = \bar{y}$$
2. **Pseudo-Residuals:** For each iteration $m$, the algorithm calculates the negative gradient of the loss function (Mean Squared Error). In MSE, the negative gradient is simply the residual:
   $$r_{im} = y_i - F_{m-1}(x_i)$$
3. **Sequential Learning:** A new Regression Stump (a weak decision tree) is trained to fit these residuals, not the original target $y$.
4. **Ensemble Update:** The new tree's predictions are scaled by a Learning Rate ($\eta$) and added to the cumulative model to push the overall error down:
   $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

## Technical Highlights
* **Pure NumPy Calculation:** The entire sequence of residual calculation, split evaluation, and gradient updates is handled mathematically without any ML libraries.
* **Continuous Risk Assessment:** Instead of simple binary classification (Dropout vs. Safe), this implementation functions as a Regressor, providing an exact percentage of dropout risk that gradually converges to the true target values over 15 estimators.

## Dependencies
* Python 3.14.3
* NumPy
