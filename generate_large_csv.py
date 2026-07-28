import numpy as np
import csv

def create_large_csv(filename="large_dataset.csv", n_samples=5000, n_features=10, centers=5, cluster_std=1.5, seed=42):
    np.random.seed(seed)
    samples_per_center = n_samples // centers
    center_coords = np.random.uniform(-15, 15, size=(centers, n_features))
    
    # Feature headers: f1, f2, ..., f10, label
    headers = [f"f{i+1}" for i in range(n_features)] + ["label"]
    
    rows = []
    for cluster_id, center in enumerate(center_coords):
        points = center + np.random.randn(samples_per_center, n_features) * cluster_std
        for pt in points:
            row = list(np.round(pt, 4)) + [cluster_id]
            rows.append(row)
            
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"Successfully generated '{filename}' with {len(rows)} data points and {n_features} features!")

if __name__ == "__main__":
    create_large_csv()