import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Synthetic Future Sales Data (Ad spending vs Sales)
data = {
    'TV_Ad_Budget': np.random.randint(1000, 5000, 100),
    'Radio_Ad_Budget': np.random.randint(500, 2000, 100),
    'Social_Media_Ad_Budget': np.random.randint(2000, 8000, 100),
}
df = pd.DataFrame(data)

# Sales = f(Budgets)
df['Sales'] = (df['TV_Ad_Budget'] * 0.5) + (df['Radio_Ad_Budget'] * 1.2) + (df['Social_Media_Ad_Budget'] * 0.8) + np.random.normal(0, 100, 100)

X = df.drop('Sales', axis=1)
y = df['Sales']

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
future_budget = pd.DataFrame([[4000, 1500, 6000]], columns=X.columns)
predicted_sales = model.predict(future_budget)

print(f"Predicted Future Sales: ${predicted_sales[0]:,.2f}")
