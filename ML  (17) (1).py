import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Synthetic Mobile Data
data = {
    'RAM': np.random.choice([2, 4, 6, 8, 12], 500),
    'Storage': np.random.choice([32, 64, 128, 256], 500),
    'Battery': np.random.randint(3000, 6000, 500),
    'ScreenSize': np.random.uniform(5.5, 7.0, 500),
}
df = pd.DataFrame(data)

# Target Price calculation based on features
df['Price'] = (df['RAM'] * 2000) + (df['Storage'] * 50) + (df['Battery'] * 2) + (df['ScreenSize'] * 100) + np.random.normal(0, 500, 500)

X = df.drop('Price', axis=1)
y = df['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print(f"Predicted Mobile Price: R{y_pred[0]:,.2f}")
print(f"Actual Mobile Price: R{y_test.iloc[0]:,.2f}")
