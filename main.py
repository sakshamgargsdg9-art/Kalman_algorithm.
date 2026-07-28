import argparse
import matplotlib.pyplot as plt
import numpy as np

# Import custom scratch implementations
from dataset import generate_blobs_scratch, load_csv_scratch
from scaler import StandardScalerFromScratch
from pca import PCAFromScratch
from kmeans import KMeansFromScratch
from metrics import silhouette_score_scratch

def main():
    # --- Step 0: Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Run PCA & K-Means pipeline with CSV or synthetic data.")
    
    parser.add_argument("--csv", type=str, default=None, help="Path to your CSV file (optional)")
    parser.add_argument("--target_col", type=int, default=None, help="Index of label column if present")
    
    parser.add_argument("--samples", type=int, default=600, help="Total synthetic data points")
    parser.add_argument("--features", type=int, default=10, help="Number of synthetic features")
    parser.add_argument("--centers", type=int, default=4, help="True number of clusters/centers")
    parser.add_argument("--std", type=float, default=1.2, help="Cluster standard deviation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--k", type=int, default=None, help="Number of clusters for final model")
    parser.add_argument("--k_max", type=int, default=7, help="Maximum K to evaluate in Elbow plot")
    
    args = parser.parse_args()

    print("=" * 50)
    
    # --- Step 1: Load CSV or Generate Synthetic Data ---
    if args.csv is not None:
        print(f"LOADING DATASET FROM CSV: '{args.csv}'")
        X_raw, y_true = load_csv_scratch(args.csv, has_header=True, target_col=args.target_col)
        print(f"  Shape Loaded: {X_raw.shape[0]} rows, {X_raw.shape[1]} features")
        optimal_k = args.k if args.k is not None else min(3, X_raw.shape[0])
    else:
        print("GENERATING SYNTHETIC DATASET:")
        print(f"  Samples: {args.samples}, Features: {args.features}, Centers: {args.centers}")
        X_raw, y_true = generate_blobs_scratch(
            n_samples=args.samples, 
            n_features=args.features, 
            centers=args.centers, 
            cluster_std=args.std, 
            seed=args.seed
        )
        optimal_k = args.k if args.k is not None else args.centers

    print("=" * 50)

    # Automatically cap k_max to not exceed total samples
    max_k_eval = min(args.k_max, X_raw.shape[0])

    # --- Step 2: Scale data ---
    print("\nStandardizing features...")
    scaler = StandardScalerFromScratch()
    X_scaled = scaler.fit_transform(X_raw)

    # --- Step 3: PCA Reduction ---
    print("Applying PCA dimensionality reduction...")
    pca = PCAFromScratch(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    expl_var_pcent = pca.explained_variance_ratio_ * 100

    # --- Step 4: Elbow Method ---
    print(f"Evaluating Elbow Method across k=1 to {max_k_eval}...")
    k_range = range(1, max_k_eval + 1)
    inertias = []
    for k in k_range:
        km = KMeansFromScratch(k=k, max_iters=50)
        km.fit(X_pca)
        inertias.append(km.inertia_)

    # --- Step 5: Final K-Means ---
    print(f"Fitting final K-Means model with k={optimal_k}...")
    kmeans_final = KMeansFromScratch(k=optimal_k, max_iters=100)
    kmeans_final.fit(X_pca)

    score = silhouette_score_scratch(X_pca, kmeans_final.labels)
    print(f"Clustering finished! Silhouette Score: {score:.3f}")

    # --- Step 6: Render Visual Dashboard ---
    print("Plotting dashboard results...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100, layout='constrained')

    # Subplot 1: PCA Scree Plot
    axes[0].bar([1, 2], expl_var_pcent, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[0].plot([1, 2], np.cumsum(expl_var_pcent), color='firebrick', marker='o', linewidth=2)
    axes[0].set_title("PCA Scree Plot", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Principal Component", fontsize=10)
    axes[0].set_ylabel("Variance Explained (%)", fontsize=10)
    axes[0].set_xticks([1, 2])
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Elbow Curve
    axes[1].plot(k_range, inertias, color='purple', marker='s', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Chosen k={optimal_k}')
    axes[1].set_title("Elbow Method (Inertia)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=10)
    axes[1].set_ylabel("Inertia (WCSS)", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Final Clusters
    scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_final.labels, cmap='viridis', alpha=0.7, edgecolors='k', s=50)
    centroids = kmeans_final.centroids
    
    # Centroids marked using a large red Star ('*') with black outline
    axes[2].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=350, edgecolors='black', linewidths=1.5, label='Centroids')
    axes[2].set_title(f"K-Means (k={optimal_k})\nSilhouette Score: {score:.3f}", fontsize=12, fontweight='bold')
    axes[2].set_xlabel(f"PC1 ({expl_var_pcent[0]:.1f}%)", fontsize=10)
    axes[2].set_ylabel(f"PC2 ({expl_var_pcent[1]:.1f}%)", fontsize=10)
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)

    title_source = args.csv if args.csv else "Synthetic Dataset"
    fig.suptitle(f"Scratch PCA and K Means ({title_source})", fontsize=16, fontweight='bold')

    plt.show()

if __name__ == "__main__":
    main()