import numpy as np

def silhouette_score_scratch(X, labels):
    """
    Evaluates cluster quality (-1 to +1 scale).
    Higher scores mean points are tightly packed inside their own cluster (cohesion)
    and well-separated from neighbor clusters (separation).
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0

    # Build full pairwise Euclidean distance matrix for all sample pairs
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    silhouette_vals = []
    for i in range(n_samples):
        own_cluster = labels[i]
        
        # a(i): Mean distance between current point i and all other points in SAME cluster (Cohesion)
        own_indices = np.where(labels == own_cluster)[0]
        if len(own_indices) > 1:
            a_i = np.sum(dist_matrix[i, own_indices]) / (len(own_indices) - 1)
        else:
            a_i = 0.0

        # b(i): Mean distance between current point i and points in NEAREST neighbor cluster (Separation)
        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_indices = np.where(labels == other_cluster)[0]
            mean_dist = np.mean(dist_matrix[i, other_indices])
            if mean_dist < b_i:
                b_i = mean_dist

        # Compute Silhouette index s(i) = (b(i) - a(i)) / max(a(i), b(i))
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_vals.append(s_i)

    # Return mean silhouette score across entire dataset
    return np.mean(silhouette_vals)