import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import preprocessing
import eda
import models
import utils
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(f"figures/{os.path.basename(filename)}") # Save to both locs
    plt.close()

def plot_feature_importance(feature_names, coefficients, title, filename, top_n=20):
    indices = np.argsort(coefficients)
    top_indices = indices[-top_n:]
    bottom_indices = indices[:top_n]
    
    top_features = [feature_names[i] for i in top_indices]
    top_coeffs = coefficients[top_indices]
    
    bottom_features = [feature_names[i] for i in bottom_indices]
    bottom_coeffs = coefficients[bottom_indices]
    
    plt.figure(figsize=(12, 6))
    plt.barh(bottom_features, bottom_coeffs, color="red", label="Negative Influence")
    plt.barh(top_features, top_coeffs, color="green", label="Positive Influence")
    
    plt.title(title)
    plt.xlabel("Coefficient Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(f"figures/{os.path.basename(filename)}")
    plt.close()

def plot_model_comparison(results_list, filename):
    model_names = [r['name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    f1_scores = [r['f1'] for r in results_list]
    
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
    plt.savefig(filename)
    plt.savefig(f"figures/{os.path.basename(filename)}")
    plt.close()

def main():
    print("=== Starting Robust Sentiment Analysis Pipeline ===")
    
    # 1. Set Global Seed for Reproducibility
    utils.set_seed(42)
    
    os.makedirs('paper/figures', exist_ok=True)
    os.makedirs('figures', exist_ok=True) # Ensure presentation figures dir exists

    # 2. Load Data
    data_path = 'data/sentiment140_raw.csv'
    if not os.path.exists(data_path):
        print("Data file not found! Please run download_data.py first.")
        return
        
    print("Loading data...")
    full_df = pd.read_csv(data_path)
    df = full_df.sample(n=200000, random_state=42) 
    print(f"Data Loaded. Shape: {df.shape}")
    
    # 3. Preprocessing
    print("\n--- Preprocessing (Improved: Keeping Negations) ---")
    df = preprocessing.preprocess_dataframe(df)
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"Data Shape after cleaning: {df.shape}")

    # 4. Run EDA (Reproducible with cleaned text)
    print("\n--- Running EDA ---")
    eda.run_eda(df=df, output_dir='paper/figures')
    # Copy figures to presentation dir effectively by running for that dir too?
    # Or just copy. eda script above saves to provided dir. 
    # Let's run just once to paper/figures and copy manually or let eda do it if updated.
    # We will copy all figures at the end to be safe.
    
    # 5. Split Data
    print("\n--- Splitting Data ---")
    X = df['cleaned_text'].fillna('')
    y = df['sentiment'] # 0 and 4
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    model_results = []
    serializable_results = []

    # 6. Baseline Model (Logistic Regression)
    print("\n--- Training Logistic Regression ---")
    lr_res = models.train_baseline_model(X_train, y_train, X_test, y_test)
    lr_res['name'] = 'Logistic Regression'
    model_results.append(lr_res)
    serializable_results.append({
        'name': 'Logistic Regression',
        'accuracy': lr_res['accuracy'],
        'f1': lr_res['f1'],
        'roc_auc': lr_res['roc_auc']
    })
    
    plot_confusion_matrix(lr_res['conf_matrix'], 'Logistic Regression CM', 'paper/figures/lr_cm.png')
    plot_feature_importance(lr_res['feature_names'], lr_res['coefficients'], 
                            'Top Influential Words (Logistic Regression)', 
                            'paper/figures/feature_importance.png')

    # 7. Naive Bayes
    print("\n--- Training Naive Bayes ---")
    nb_res = models.train_naive_bayes_model(X_train, y_train, X_test, y_test)
    nb_res['name'] = 'Naive Bayes'
    model_results.append(nb_res)
    serializable_results.append({
        'name': 'Naive Bayes',
        'accuracy': nb_res['accuracy'],
        'f1': nb_res['f1'],
        'roc_auc': nb_res['roc_auc']
    })
    plot_confusion_matrix(nb_res['conf_matrix'], 'Naive Bayes CM', 'paper/figures/nb_cm.png')

    # 8. VADER (Lexicon)
    print("\n--- Evaluating VADER ---")
    v_scores, v_preds = models.get_vader_scores(X_test)
    v_acc = accuracy_score(y_test, v_preds)
    # Using pos_label=4 explicitly 
    v_f1 = f1_score(y_test, v_preds, pos_label=4)
    
    print(f"VADER Accuracy: {v_acc:.4f}")
    
    v_res = {
        'name': 'VADER',
        'accuracy': v_acc,
        'f1': v_f1
    }
    model_results.append(v_res)
    serializable_results.append(v_res)
    
    # 9. LSTM
    print("\n--- Training LSTM ---")
    lstm_res = models.train_lstm_model(X_train, y_train, X_test, y_test)
    lstm_res['name'] = 'LSTM'
    model_results.append(lstm_res)
    serializable_results.append({
        'name': 'LSTM',
        'accuracy': lstm_res['accuracy'],
        'f1': lstm_res['f1'],
        'roc_auc': lstm_res['roc_auc']
    })
    plot_confusion_matrix(lstm_res['conf_matrix'], 'LSTM CM', 'paper/figures/lstm_confusion_matrix.png')

    # 10. Comparison & Saving
    print("\n--- Generating Comparison Plots ---")
    plot_model_comparison(model_results, 'paper/figures/model_comparison.png')
    
    # Save Results
    utils.save_results(serializable_results, 'results.json')
    
    # Copy all figures to 'figures/' for presentation reference
    print("\n--- Syncing Figures ---")
    os.system("cp paper/figures/*.png figures/")
    
    print("\n=== Pipeline Complete ===")
    print("All results saved to results.json and figures updated in paper/figures and figures/")

if __name__ == "__main__":
    main()
