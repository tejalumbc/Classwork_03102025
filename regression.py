import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("student_data.csv")

# (1) Create a visualization of the data
plt.figure(figsize=(10, 6))

# Scatter plot for Hours_Studied vs Results, colored by Review_Session
plt.scatter(data[data["Review_Session"] == 0]["Hours_Studied"], 
            data[data["Review_Session"] == 0]["Results"], 
            color="blue", label="No Review Session", alpha=0.5)
plt.scatter(data[data["Review_Session"] == 1]["Hours_Studied"], 
            data[data["Review_Session"] == 1]["Results"], 
            color="red", label="Review Session", alpha=0.5)

plt.xlabel("Hours Studied")
plt.ylabel("Results (Pass=1, Fail=0)")
plt.title("Scatter Plot of Hours Studied vs Results (Colored by Review Session)")
plt.legend()
plt.grid(True)
plt.show()

# (2) Fit a logistic regression model
# Prepare features (X) and target (y)
X = data[["Hours_Studied", "Review_Session"]]
y = data["Results"]

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities and classes
y_pred_prob = model.predict_proba(X)[:, 1]  # Probabilities for the positive class
y_pred = model.predict(X)  # Predicted classes

# (3) Output model coefficients and performance metrics
# Model coefficients
coefficients = model.coef_
intercept = model.intercept_
print("Model Coefficients:")
print(f"Coefficient for Hours_Studied: {coefficients[0][0]}")
print(f"Coefficient for Review_Session: {coefficients[0][1]}")
print(f"Intercept: {intercept[0]}")

# Performance metrics
accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_pred_prob)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y, y_pred)
print("\nClassification Report:")
print(class_report)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()