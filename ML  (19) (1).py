import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Synthetic Bank Loan Data
data = {
    'Income': np.random.randint(30000, 150000, 500),
    'Credit_Score': np.random.randint(300, 850, 500),
    'Loan_Amount': np.random.randint(5000, 100000, 500),
    'Debt_Ratio': np.random.uniform(0, 1, 500),
}
df = pd.DataFrame(data)

# Target: Loan Approved (1 for Yes, 0 for No)
df['Loan_Approved'] = ((df['Credit_Score'] > 600) & (df['Debt_Ratio'] < 0.5)).astype(int)

X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print("Bank Loan Prediction Report:")
print(classification_report(y_test, y_pred))
