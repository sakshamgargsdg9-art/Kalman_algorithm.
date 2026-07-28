import csv
import numpy as np

def generate_blobs_scratch(n_samples=500, n_features=10, centers=4, cluster_std=1.5, seed=42):
    """
    Creates a synthetic high-dimensional dataset with distinct clusters.
    """
    np.random.seed(seed)
    samples_per_center = n_samples // centers
    
    center_coords = np.random.uniform(-10, 10, size=(centers, n_features))
    
    X_list, y_list = [], []
    for cluster_id, center in enumerate(center_coords):
        points = center + np.random.randn(samples_per_center, n_features) * cluster_std
        X_list.append(points)
        y_list.append(np.full(samples_per_center, cluster_id))
        
    return np.vstack(X_list), np.concatenate(y_list)


def load_csv_scratch(file_path, has_header=True, target_col=None):
    """
    Reads a CSV dataset from scratch using Python's standard csv library.
    Safely skips text headers and converts numerical values to floats.
    """
    data_rows = []
    
    # utf-8-sig handles UTF-8 files with or without a byte-order mark (BOM)
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if not row:  # Skip empty lines
                continue
            
            # Try converting row items to float; catch ValueError to skip header text
            try:
                numeric_row = [float(val.strip()) for val in row if val.strip() != '']
                if numeric_row:
                    data_rows.append(numeric_row)
            except ValueError:
                # Silently skip any non-numeric header rows (e.g., 'f1', 'f2', etc.)
                continue
            
    data_matrix = np.array(data_rows, dtype=np.float64)
    
    # Separate labels from feature columns if a target column index is provided
    if target_col is not None:
        y = data_matrix[:, target_col]
        X = np.delete(data_matrix, target_col, axis=1)
        return X, y
    
    return data_matrix, None