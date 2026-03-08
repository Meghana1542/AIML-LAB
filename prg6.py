# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# Step 2: Load Iris Dataset
iris = load_iris()

# Step 3: Select only two features for visualization
# Sepal Length (column 0) and Petal Length (column 2)
X = iris.data[:, [0, 2]]

# Convert to DataFrame (optional)
df = pd.DataFrame(X, columns=["Sepal Length", "Petal Length"])

print("Sample Dataset")
print(df.head())


# Step 4: Choose number of clusters (K = 3)
kmeans = KMeans(n_clusters=3, random_state=42)


# Step 5: Apply K-Means clustering
kmeans.fit(X)


# Step 6: Generate cluster labels
labels = kmeans.labels_

# Add cluster labels to dataset
df["Cluster"] = labels

print("\nClustered Data")
print(df)


# Step 7: Get centroid values
centroids = kmeans.cluster_centers_


# Step 8: Visualize clusters using scatter plot
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', label="Data Points")

# Plot centroids
plt.scatter(centroids[:,0], centroids[:,1],
            s=200, c='red', marker='X', label="Centroids")

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()

plt.show()