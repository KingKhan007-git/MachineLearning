from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Result
y_pred = knn.predict(X_test)
print(f"Iris KNN Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
