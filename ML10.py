from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np

# Generate dummy cluster data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Create EM model (Gaussian Mixture)
gmm = GaussianMixture(n_components=3)

# Train
gmm.fit(X)

# Predict clusters
labels = gmm.predict(X)

# Results
print("Means of clusters:")
print(gmm.means_)
print("\nCovariances of clusters:")
print(gmm.covariances_)
print("\nFirst 10 cluster labels:", labels[:10])
