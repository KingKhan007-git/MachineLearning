import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Synthetic Car Price Data
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Kilometers_Driven': [50000, 40000, 30000, 20000, 10000, 5000],
    'Fuel_Type': [0, 1, 0, 1, 0, 1], # 0: Petrol, 1: Diesel
    'Price': [500000, 600000, 700000, 850000, 1000000, 1200000]
}
df = pd.DataFrame(data)

X = df.drop('Price', axis=1)
y = df['Price']

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction for a new car
new_car = pd.DataFrame([[2021, 2000, 1]], columns=X.columns)
predicted_price = model.predict(new_car)

print(f"Predicted Price for 2021 car: ${predicted_price[0]:,.2f}")
