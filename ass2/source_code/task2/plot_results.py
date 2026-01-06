import matplotlib.pyplot as plt
import json
import numpy as np
import os

def main():
    with open('results.json', 'r') as f:
        results = json.load(f)

    # Filter out models with missing metrics if any
    results = [r for r in results if r.get('accuracy') is not None]

    model_names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    # Handle missing F1 for LSTM/VADER if null
    f1_scores = [r.get('f1', 0) if r.get('f1') is not None else 0 for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='orange')
    
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    os.makedirs('paper/figures', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    plt.savefig('paper/figures/model_comparison.png')
    plt.savefig('figures/model_comparison.png')
    print("Model comparison plot generated.")

if __name__ == "__main__":
    main()
