import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Synthetic House Price Data
data = {
    'Size_sqft': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 3, 4, 4, 5],
    'Age': [10, 5, 8, 2, 1],
    'Price': [300000, 450000, 500000, 650000, 800000]
}
df = pd.DataFrame(data)

X = df.drop('Price', axis=1)
y = df['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Result
y_pred = model.predict(X_test)
print(f"Predicted Price: ${y_pred[0]:,.2f}")
print(f"Actual Price: ${y_test.iloc[0]:,.2f}")
