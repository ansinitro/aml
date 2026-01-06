
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
OUTPUT_DIR = "output"
FIGURES_DIR = "report/figures"
SAMPLE_SIZE = 5000  # Subset size for t-SNE speed
RANDOM_STATE = 42   # ENSURES REPRODUCIBILITY

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 8),
    'scatter.edgecolors': 'black'
})

def load_mnist_data():
    """
    Load MNIST dataset and subset it.
    Uses fixed RANDOM_STATE for subsampling.
    """
    print("Loading MNIST dataset (this may take a moment)...")
    try:
        # cache=True by default in fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', cache=True)
        X, y = mnist.data, mnist.target.astype(int)
        
        print(f"Original shape: {X.shape}")
        
        # Subsampling
        if len(X) > SAMPLE_SIZE:
            rng = np.random.RandomState(RANDOM_STATE)
            indices = rng.choice(len(X), SAMPLE_SIZE, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Subsampled to {SAMPLE_SIZE} samples for efficiency.")
            
        # Scale Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        raise

def run_pca_analysis(X, y):
    """
    Run PCA with fixed random state.
    """
    print("Running PCA...")
    start_time = time.time()
    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    duration = time.time() - start_time
    print(f"PCA completed in {duration:.2f}s")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    # Save Metrics
    result_data = {
        'method': 'PCA',
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'duration_seconds': duration
    }
    
    with open(os.path.join(OUTPUT_DIR, "mnist_pca_results.json"), "w") as f:
        json.dump(result_data, f)
        
    return X_pca

def run_tsne_analysis(X, y):
    """
    Run t-SNE with fixed random state.
    """
    print("Running t-SNE (this is computationally expensive)...")
    start_time = time.time()
    
    # Ensure init='random' or 'pca' is consistent if needed, but random_state handles it.
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        max_iter=1000, 
        random_state=RANDOM_STATE, 
        n_jobs=-1,
        init='pca', # PCA initialization is often more stable than random
        learning_rate='auto'
    )
    X_tsne = tsne.fit_transform(X)
    
    duration = time.time() - start_time
    print(f"t-SNE completed in {duration:.2f}s")
    print(f"KL Divergence: {tsne.kl_divergence_}")
    
    # Save Metrics
    result_data = {
        'method': 't-SNE',
        'kl_divergence': float(tsne.kl_divergence_),
        'duration_seconds': duration
    }
    
    with open(os.path.join(OUTPUT_DIR, "mnist_tsne_results.json"), "w") as f:
        json.dump(result_data, f)
        
    return X_tsne

def visualize_embedding(X_emb, y, title, filename):
    """
    Plot 2D embedding with high-quality styling.
    """
    plt.figure(figsize=(12, 10))
    
    # Distinct colors for 10 digits
    classes = sorted(np.unique(y))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for cls, color in zip(classes, colors):
        mask = (y == cls)
        plt.scatter(
            X_emb[mask, 0], X_emb[mask, 1], 
            label=str(cls), 
            color=color,
            alpha=0.7, 
            s=20, 
            edgecolor='none'
        )
        
    plt.legend(title="Digit", markerscale=2, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(title, fontweight='bold', fontsize=16)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()

def main():
    print("--- Starting Part 2: MNIST Visualization ---")
    
    X_scaled, y = load_mnist_data()
    
    # PCA
    X_pca = run_pca_analysis(X_scaled, y)
    visualize_embedding(X_pca, y, "PCA on MNIST (Global Structure)", "mnist_pca.png")
    
    # t-SNE
    X_tsne = run_tsne_analysis(X_scaled, y)
    visualize_embedding(X_tsne, y, "t-SNE on MNIST (Local Manifold)", "mnist_tsne.png")
    
    print("Part 2 Completed.")

if __name__ == "__main__":
    main()
