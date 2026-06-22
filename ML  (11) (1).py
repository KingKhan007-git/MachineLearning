import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Create synthetic credit score data
data = {
    'Income': np.random.randint(20000, 100000, 1000),
    'Age': np.random.randint(18, 70, 1000),
    'LoanAmount': np.random.randint(1000, 50000, 1000),
    'CreditScore': np.random.randint(300, 850, 1000),
}
df = pd.DataFrame(data)

# Target: Credit Risk (0: Low, 1: Medium, 2: High)
df['CreditRisk'] = pd.cut(df['CreditScore'], bins=[300, 580, 740, 850], labels=[2, 1, 0], include_lowest=True)

X = df.drop('CreditRisk', axis=1)
y = df['CreditRisk']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Results
y_pred = clf.predict(X_test)
print("Credit Score Classification Report:")
print(classification_report(y_test, y_pred))
