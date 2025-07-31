import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("student_scores.csv")  # Ensure this file is in the same folder

# -------------------------------
# Define features and target
# -------------------------------
X = df[['StudyHours', 'SleepHours']].values
y = df['Marks'].values

# -------------------------------
# Train the model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Prediction for new input (change here)
# -------------------------------
X_test = np.array([
    [5, 6],
    [8, 5],
    [10, 7]
])
y_pred = model.predict(X_test)

# -------------------------------
# Display results
# -------------------------------
print("\n========== Multiple Linear Regression Results ==========\n")
print(f"➡️  StudyHours Coefficient: {model.coef_[0]:.4f}")
print(f"➡️  SleepHours Coefficient: {model.coef_[1]:.4f}")
print(f"➡️  Intercept: {model.intercept_:.4f}\n")

for i, (study, sleep) in enumerate(X_test):
    print(f"🧑‍🎓 Student {i+1}: Study = {study}h, Sleep = {sleep}h → Predicted Marks = {y_pred[i]:.2f}")

mse = mean_squared_error(y, model.predict(X))
r2 = r2_score(y, model.predict(X))
print(f"\n📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")
print("\n========================================================\n")

# -------------------------------
# Plot: Actual vs Predicted
# -------------------------------
y_train_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(y, y_train_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction Line')
plt.title("📊 Actual vs Predicted Marks")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
