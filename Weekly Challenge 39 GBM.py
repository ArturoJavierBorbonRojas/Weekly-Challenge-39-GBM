import numpy as np

# Weekly Challenge 39: Gradient Boosting Machine (GBM) from Scratch
# Author: Ing. Arturo Javier Borbon Rojas

print("Initializing Gradient Boosting Machine (GBM) Engine...")
print("-" * 65)

# 1. Weak Learner: Regression Stump (Tree with a single split)
class RegressionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        m, n = X.shape
        min_mse = float('inf')
        
        # Iterate over all features and possible thresholds to find the best split
        for feature_i in range(n):
            thresholds = np.unique(X[:, feature_i])
            for thresh in thresholds:
                left_mask = X[:, feature_i] <= thresh
                right_mask = X[:, feature_i] > thresh
                
                # Skip if a split leaves a side empty
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                    
                # Calculate the mean of the residuals for both branches
                left_val = np.mean(y[left_mask])
                right_val = np.mean(y[right_mask])
                
                # Calculate the Mean Squared Error of this split
                pred = np.zeros(m)
                pred[left_mask] = left_val
                pred[right_mask] = right_val
                mse = np.mean((y - pred)**2)
                
                # Save the split if it minimizes the error
                if mse < min_mse:
                    min_mse = mse
                    self.feature_idx = feature_i
                    self.threshold = thresh
                    self.left_value = left_val
                    self.right_value = right_val
                    
    def predict(self, X):
        preds = np.zeros(X.shape[0])
        left_mask = X[:, self.feature_idx] <= self.threshold
        preds[left_mask] = self.left_value
        preds[~left_mask] = self.right_value
        return preds

# 2. Gradient Boosting Engine
class GradientBoostingRegressor:
    def __init__(self, n_estimators=10, learning_rate=0.2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        print(f"Training {self.n_estimators} sequential trees on pseudo-residuals:")
        
        # 1. Initial base prediction: The mean of the target variable
        self.base_pred = np.mean(y)
        current_preds = np.full(y.shape, self.base_pred)
        
        for i in range(self.n_estimators):
            # 2. Calculate Pseudo-Residuals (The negative gradient of the MSE loss function)
            residuals = y - current_preds
            
            # 3. Train a weak learner specifically to predict these residuals
            tree = RegressionStump()
            tree.fit(X, residuals)
            
            # 4. Update the ensemble's predictions (Base + Learning Rate * Tree Prediction)
            update = tree.predict(X)
            current_preds += self.learning_rate * update
            
            self.trees.append(tree)
            
            # Tracking convergence
            current_mse = np.mean((y - current_preds)**2)
            print(f"   ➔ Tree {i+1:02d} added | Residual Error (MSE) dropped to: {current_mse:.4f}")

    def predict(self, X):
        # Start with the base prediction and sequentially add the scaled tree corrections
        preds = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds

# --- Educational Dataset (Predicting Dropout Risk %) ---
# Features: [Absences (0-30), Stress Level (1-10), Grades (0-10)]
X_train = np.array([
    [2, 3, 9.5], [5, 4, 8.0], [1, 2, 9.0], # Safe: Low risk
    [15, 8, 6.0], [18, 9, 5.0], [20, 10, 4.0], # Critical: High risk
    [10, 7, 7.5], [12, 6, 6.5] # Borderline cases
])

# Actual dropout probability (0.0 to 1.0)
y_train = np.array([0.05, 0.10, 0.02, 0.85, 0.90, 0.95, 0.50, 0.60])

# Train the GBM
gbm = GradientBoostingRegressor(n_estimators=15, learning_rate=0.25)
gbm.fit(X_train, y_train)

print("-" * 65)
print("EVALUATING GBM ENSEMBLE:")

X_test = np.array([
    [4, 3, 8.8],   # Expected: Low Risk
    [16, 9, 5.5],  # Expected: High Risk
    [11, 8, 7.0]   # Borderline
])

predictions = gbm.predict(X_test)

for i, (student, pred) in enumerate(zip(X_test, predictions)):
    risk_percentage = min(max(pred * 100, 0), 100) # Clip between 0 and 100
    status = "High Dropout Risk" if risk_percentage > 70 else ("Borderline" if risk_percentage > 40 else "Safe (Retained)")
    print(f"Profile {i+1} [Absences: {student[0]:02.0f}, Stress: {student[1]}, Grades: {student[2]}] ➔ Predicted Risk: {risk_percentage:05.2f}% | {status}")