
import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# --- Configuration ---
OUTPUT_DIR = "output"
FIGURES_DIR = "report/figures"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting style to academic standards
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'scatter.edgecolors': 'black'
})

def load_mall_customers():
    """
    Download and load the Mall Customers dataset.
    Reliably handles the file path from kagglehub.
    """
    print("Downloading Mall Customers dataset...")
    try:
        path = kagglehub.dataset_download("simtoor/mall-customers")
        file_path = os.path.join(path, "Mall Customers.xlsx")
        if not os.path.exists(file_path):
             # Fallback if filename changes in future versions
             files = os.listdir(path)
             file_path = os.path.join(path, files[0])
        
        df = pd.read_excel(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def preprocess_data(df):
    """
    Select features and scale data using StandardScaler.
    Focuses on 'Annual Income' and 'Spending Score'.
    """
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, features

def run_dbscan_experiment(X_scaled, eps_values, min_samples_values):
    """
    Run DBSCAN over a grid of parameters and record metrics.
    Deterministic operation (DBSCAN is deterministic).
    """
    results = []
    
    print("Running DBSCAN parameter sweep...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            
            # Metrics
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)
            
            sil = -1.0
            if n_clusters > 1:
                sil = silhouette_score(X_scaled, labels)
            
            results.append({
                'eps': float(eps),
                'min_samples': int(min_samples),
                'n_clusters': int(n_clusters),
                'n_noise': int(n_noise),
                'noise_ratio': float(noise_ratio),
                'silhouette': float(sil)
            })
            
    return results

def plot_clusters(X, labels, title, filename, xlabel, ylabel):
    """
    Visualize clustering results with professional formatting.
    """
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    
    # Generate distinct colors
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
            label = "Noise"
            marker = 'X'
            alpha = 0.5
            size = 50
        else:
            label = f"Cluster {k}"
            marker = 'o'
            alpha = 1.0
            size = 70

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        plt.scatter(
            xy[:, 0], xy[:, 1], 
            c=[col], 
            marker=marker, 
            label=label, 
            edgecolor='k', 
            s=size, 
            alpha=alpha
        )

    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()

def exercise_1_and_2_sensitivity(metrics_df):
    """
    Plot eps vs n_clusters and noise_ratio for MULTIPLE min_samples.
    Covers Exercises 1 and 2 fully.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique min_samples to iterate
    min_samples_list = sorted(metrics_df['min_samples'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(min_samples_list)))
    
    # Plot 1: Epsilon vs Number of Clusters
    for min_samp, color in zip(min_samples_list, colors):
        subset = metrics_df[metrics_df['min_samples'] == min_samp].sort_values(by='eps')
        ax1.plot(subset['eps'], subset['n_clusters'], marker='o', label=f'Min Samples={min_samp}', color=color)
    
    ax1.set_xlabel('Epsilon (eps)')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Effect of eps & min_samples on Cluster Count')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Epsilon vs Noise Ratio
    for min_samp, color in zip(min_samples_list, colors):
        subset = metrics_df[metrics_df['min_samples'] == min_samp].sort_values(by='eps')
        ax2.plot(subset['eps'], subset['noise_ratio'], marker='s', linestyle='--', label=f'Min Samples={min_samp}', color=color)

    ax2.set_xlabel('Epsilon (eps)')
    ax2.set_ylabel('Noise Ratio')
    ax2.set_title('Effect of eps & min_samples on Noise Ratio')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("Sensitivity Analysis (Exercises 1 & 2)", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "exercise1_sensitivity.png"), dpi=300)
    plt.close()

def exercise_3_kmeans_comparison(X_scaled, n_clusters, xlabel, ylabel):
    """
    Compare DBSCAN with KMeans (k=n_clusters).
    Uses RANDOM_STATE for reproducibility.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    plot_clusters(X_scaled, labels, 
                  f'Exercise 3: KMeans Clustering (k={n_clusters})', 
                  "exercise3_kmeans_comparison.png",
                  xlabel, ylabel)

def exercise_4_varying_density():
    """
    Demonstrate DBSCAN on varying density data.
    Uses RANDOM_STATE for reproducibility.
    """
    # Generate data with different densities
    centers = [(-5, -5), (0, 0), (5, 5)]
    X1, _ = make_blobs(n_samples=200, centers=[centers[0]], cluster_std=0.5, random_state=RANDOM_STATE)
    X2, _ = make_blobs(n_samples=400, centers=[centers[1]], cluster_std=1.5, random_state=RANDOM_STATE) # Lower density
    X3, _ = make_blobs(n_samples=200, centers=[centers[2]], cluster_std=0.5, random_state=RANDOM_STATE)
    X_density = np.vstack([X1, X2, X3])
    
    scaler = StandardScaler()
    X_density_scaled = scaler.fit_transform(X_density)
    
    # Run DBSCAN with parameters that struggle with varying density
    db = DBSCAN(eps=0.3, min_samples=10)
    labels = db.fit_predict(X_density_scaled)
    
    plot_clusters(X_density_scaled, labels,
                  "Exercise 4: DBSCAN on Varying Densities",
                  "exercise4_varying_density.png",
                  "Feature 1", "Feature 2")

def main():
    print("--- Starting Part 1: DBSCAN Analysis ---")
    
    # 1. Load & Preprocess
    df = load_mall_customers()
    X_orig, X_scaled, feature_names = preprocess_data(df)
    
    # 2. Parameter Sweep
    eps_range = np.arange(0.1, 1.05, 0.05)
    min_samples_range = [3, 5, 10, 15, 20]
    
    metrics = run_dbscan_experiment(X_scaled, eps_range, min_samples_range)
    
    # Save Metrics
    metrics_file = os.path.join(OUTPUT_DIR, "dbscan_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    
    # Convert to DF for analysis
    metrics_df = pd.DataFrame(metrics)
    
    # 3. Best Model Visualization (Maximize Silhouette)
    # Filter for valid clusterings (>1 cluster, <all clusters)
    valid_metrics = metrics_df[(metrics_df['n_clusters'] > 1) & (metrics_df['n_clusters'] < len(X_scaled))]
    
    if not valid_metrics.empty:
        best_run = valid_metrics.loc[valid_metrics['silhouette'].idxmax()]
        print(f"Best Parameters Found: eps={best_run['eps']}, min_samples={best_run['min_samples']} (Score: {best_run['silhouette']:.3f})")
        
        # Run Best DBSCAN
        best_db = DBSCAN(eps=best_run['eps'], min_samples=int(best_run['min_samples']))
        best_labels = best_db.fit_predict(X_scaled)
        
        plot_clusters(
            X_scaled, best_labels, 
            f"Best DBSCAN Model (eps={best_run['eps']}, min_samples={int(best_run['min_samples'])})", 
            "best_dbscan_clusters.png",
            f"{feature_names[0]} (Scaled)", f"{feature_names[1]} (Scaled)"
        )
        
        # Exercise 3: Compare with KMeans
        exercise_3_kmeans_comparison(X_scaled, int(best_run['n_clusters']),
                                     f"{feature_names[0]} (Scaled)", f"{feature_names[1]} (Scaled)")
    else:
        print("Warning: No valid clustering configuration found.")

    # 4. Exercises
    exercise_1_and_2_sensitivity(metrics_df)
    exercise_4_varying_density()
    print("Part 1 Completed.")

if __name__ == "__main__":
    main()
