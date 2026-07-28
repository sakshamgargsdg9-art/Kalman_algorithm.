import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Sample Covariance Matrix: (1 / (N - 1)) * X^T @ X
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigh is specifically designed for symmetric covariance matrices and returns real numbers
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Transpose so eigenvectors are rows for easier indexing
        eigenvectors = eigenvectors.T

        # Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[sorted_indices]

        # Keep top components and ensure real float values
        self.components = np.real(eigenvectors[:self.n_components])
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = np.real(eigenvalues[:self.n_components]) / total_variance

    def transform(self, X):
        X_centered = X - self.mean
        return np.real(np.dot(X_centered, self.components.T))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)