import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

def visualize_results(data_dir='data', figures_dir='report/figures'):
    """Generates visualizations from data and metrics."""
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    # 1. Load Metrics
    metrics_path = os.path.join(data_dir, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    # Convert metrics to DataFrame for plotting
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    
    # 2. Plot RMSE Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=df_metrics, palette='viridis')
    plt.title('Model Root Mean Squared Error (Lower is Better)')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'rmse_comparison.png'))
    plt.close()
    
    # 3. Plot Precision@10 Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Precision@10', data=df_metrics, palette='magma')
    plt.title('Model Precision@10 (Higher is Better)')
    plt.ylabel('Precision@10')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'precision_comparison.png'))
    plt.close()
    
    # 3b. Plot Precision@K Curve
    plt.figure(figsize=(10, 6))
    k_values = [5, 10, 15, 20]
    
    for model in metrics:
        precs = [metrics[model].get(f'Precision@{k}', 0) for k in k_values]
        plt.plot(k_values, precs, marker='o', label=model)
        
    plt.title('Precision at K')
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'precision_k_curve.png'))
    plt.close()

    # 4. Rating Distribution
    # Load raw data for this
    ratings_path = os.path.join(data_dir, 'processed', 'train.csv')
    if os.path.exists(ratings_path):
        df = pd.read_csv(ratings_path)
        
        # Hist
        plt.figure(figsize=(10, 6))
        sns.histplot(df['rating'], bins=10, kde=False, color='skyblue')
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'rating_distribution.png'))
        plt.close()
        
        # 5. User-Item Heatmap (Subset)
        plt.figure(figsize=(12, 10))
        # Pivot a subset
        # Filter for top users and movies
        top_users = df['userId'].value_counts().head(50).index
        top_movies = df['movieId'].value_counts().head(50).index
        
        subset = df[df['userId'].isin(top_users) & df['movieId'].isin(top_movies)]
        matrix = subset.pivot(index='userId', columns='movieId', values='rating')
        
        sns.heatmap(matrix, cmap='viridis', robust=True, cbar_kws={'label': 'Rating'})
        plt.title('User-Item Interaction Heatmap (Top 50 Users/Movies)')
        plt.xlabel('Movie ID')
        plt.ylabel('User ID')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'interaction_heatmap.png'))
        plt.close()
        
    print("Visualizations generated and saved to", figures_dir)

if __name__ == "__main__":
    visualize_results('../data', '../report/figures')
