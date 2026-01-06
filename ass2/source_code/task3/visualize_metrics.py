import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

def plot_classification_report(json_path="metrics/final_metrics.json"):
    print(f"Loading metrics from {json_path}...")
    with open(json_path, "r") as f:
        metrics = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics).transpose()
    
    # Drop accuracy and averages for the heatmap to focus on classes
    df_classes = df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    # Select only metrics columns
    df_plot = df_classes[['precision', 'recall', 'f1-score']]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plot, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0.5, vmax=1.0)
    plt.title("Classification Metrics per Class")
    plt.xlabel("Metric")
    plt.ylabel("Attack Type")
    plt.tight_layout()
    plt.savefig("figures/class_report_heatmap.png")
    print("Saved figures/class_report_heatmap.png")

if __name__ == "__main__":
    plot_classification_report()
