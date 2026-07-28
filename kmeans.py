import numpy as np

class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _init_kmeans_plus_plus(self, X):
        n_samples, n_features = X.shape
        # Ensure k does not exceed sample count
        effective_k = min(self.k, n_samples)
        centroids = np.zeros((effective_k, n_features))

        # Randomly choose first centroid
        first_idx = np.random.choice(n_samples)
        centroids[0] = X[first_idx]

        for i in range(1, effective_k):
            # Distance squared to nearest existing centroid
            dists = np.array([min([np.sum((x - c)**2) for c in centroids[:i]]) for x in X])
            total_dist = np.sum(dists)

            # Prevent 0 / 0 division if all remaining points match centroids
            if total_dist == 0:
                probs = np.ones(n_samples) / n_samples
            else:
                probs = dists / total_dist

            next_idx = np.random.choice(n_samples, p=probs)
            centroids[i] = X[next_idx]

        return centroids

    def fit(self, X):
        n_samples = X.shape[0]
        # Cap k if dataset has fewer points than k
        if self.k > n_samples:
            self.k = n_samples

        self.centroids = self._init_kmeans_plus_plus(X)

        for _ in range(self.max_iters):
            # Compute distance matrix
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Re-calculate centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        # Calculate total inertia (WCSS)
        self.inertia_ = 0.0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)