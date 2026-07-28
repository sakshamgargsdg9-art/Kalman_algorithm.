# PCA & K-Means from Scratch 

A lightweight Python implementation of **Principal Component Analysis (PCA)** and **K-Means Clustering** built using only `NumPy` and `Matplotlib` — no `scikit-learn` or `pandas`.

---

## Core Concepts & Definitions

**Principal Component Analysis (PCA):** A dimensionality reduction technique that transforms high-dimensional data into fewer principal components (e.g., 2D) while retaining as much variance (information) as possible.
**K-Means Clustering:** An unsupervised learning algorithm that partitions data into $k$ distinct clusters by minimizing the distance between each data point and its assigned cluster center (centroid).
**K-Means++ Initialization:** A smart centroid selection strategy that spreads out initial cluster centers probabilistically, speeding up convergence and avoiding poor local minima.

---

## Project Structure

* **`dataset.py`** — Custom standard-library CSV loader & synthetic data generator
* **`scaler.py`** — Z-score feature standardizer (`StandardScaler`)
* **`pca.py`** — Covariance matrix & eigendecomposition solver
* **`kmeans.py`** — K-Means algorithm with $k$-means++ initialization
* **`metrics.py`** — Silhouette score & distance matrix calculations
* **`generate_large_csv.py`** — Script to generate a 5,000-row benchmark dataset
* **`main.py`** — CLI pipeline driver & Matplotlib visual dashboard

---

-----------------------------------------HOW TO RUN THIS CODE FOR DIFFERENT INPUTS-------------------------------------------

pip install numpy matplotlib

# Default run
python main.py

# Clean, well-separated clusters
python main.py --samples 800 --features 10 --centers 3 --std 0.8 --k 3

# Noisy, overlapping clusters
python main.py --samples 800 --features 10 --centers 4 --std 3.8 --k 4

# Generate 5,000-row benchmark dataset
python generate_large_csv.py

# Run on CSV files
python main.py --csv clean_data.csv --target_col 4 --k 3
python main.py --csv noisy_data.csv --target_col 4 --k 2
python main.py --csv large_dataset.csv --target_col 10 --k 5
