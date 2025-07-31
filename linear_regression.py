import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)  # House size in square feet
y = np.array([100000, 150000, 200000, 250000, 300000])     # House price in dollars

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Predict prices for new sizes
x_test = np.array([1000, 1800]).reshape(-1, 1)
y_pred = model.predict(x_test)

print("Predicted prices:", y_pred)
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Visualize
plt.scatter(x, y, color='blue', label='Training Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.scatter(x_test, y_pred, color='green', label='Predictions')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()