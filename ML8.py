import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 2, 3, 5])

# Create Linear Regression model
model = LinearRegression()

# Train
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Results
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.2f}")
print(f"R-squared: {r2_score(y, y_pred):.2f}")

