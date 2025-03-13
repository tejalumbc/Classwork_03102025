import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate a synthetic dataset for demonstration
# Replace this with your actual dataset if needed
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# (1) Fit an SVM model with a linear kernel
linear_svm = SVC(kernel="linear", random_state=42)
linear_svm.fit(X_train, y_train)

# Predictions and evaluation for linear kernel
y_pred_linear = linear_svm.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Linear Kernel SVM Results:")
print(f"Accuracy: {accuracy_linear:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_linear))
print("Classification Report:")
print(classification_report(y_test, y_pred_linear))

# (2) Fit an SVM model with an RBF kernel and use grid search to find the best gamma
# Define the parameter grid for gamma
param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize the SVM model with RBF kernel
rbf_svm = SVC(kernel="rbf", random_state=42)

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(rbf_svm, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best gamma value from grid search
best_gamma = grid_search.best_params_["gamma"]
print(f"\nBest Gamma for RBF Kernel: {best_gamma}")

# Train the RBF SVM model with the best gamma
rbf_svm_best = SVC(kernel="rbf", gamma=best_gamma, random_state=42)
rbf_svm_best.fit(X_train, y_train)

# Predictions and evaluation for RBF kernel
y_pred_rbf = rbf_svm_best.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("\nRBF Kernel SVM Results:")
print(f"Accuracy: {accuracy_rbf:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rbf))

# Visualization of decision boundaries for both kernels
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

# Plot decision boundaries for linear kernel
plot_decision_boundary(linear_svm, X_train, y_train, "Linear Kernel SVM Decision Boundary")

# Plot decision boundaries for RBF kernel
plot_decision_boundary(rbf_svm_best, X_train, y_train, f"RBF Kernel SVM Decision Boundary (Gamma={best_gamma})")