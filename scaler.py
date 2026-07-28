import numpy as np

class StandardScalerFromScratch:
    """
    Distance-based algorithms (like PCA and K-Means) are sensitive to feature scales.
    This class standardizes data to have Mean = 0 and Variance = 1 (Z-score normalization).
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # Calculate mean and standard deviation across each feature column
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # If a feature has zero variance (constant column), add a tiny epsilon to prevent division by zero
        self.std[self.std == 0] = 1e-8

    def transform(self, X):
        # Apply Z-score formula: (X - mu) / sigma
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)