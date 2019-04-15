from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]]
X, labels_true = make_blobs(n_samples=800, centers=centers, cluster_std=0.4,
                            random_state=8)

X = StandardScaler().fit_transform(X)

np.savetxt('X.csv', X, delimiter=',')