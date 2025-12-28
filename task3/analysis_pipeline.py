import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
import json

from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set style - INTEL/OPS THEME
plt.style.use('dark_background')
sns.set_theme(style="dark", rc={
    "axes.facecolor": "#0a0a0a",
    "figure.facecolor": "#0a0a0a",
    "grid.color": "#333333",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#00ff41", # Terminal Green
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#ff3333"  # Alert Red
})

os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def load_data(full_data=False):
    print("Loading dataset...")
    df = pd.read_csv("gtd_dataset.csv", encoding='ISO-8859-1', low_memory=False)
    
    # Filter for relevant columns
    cols = ['iyear', 'imonth', 'country_txt', 'region_txt', 'attacktype1_txt', 
            'targtype1_txt', 'weaptype1_txt', 'nkill', 'nwound', 'success']
    df = df[cols].copy()
    
    # Basic Cleaning for Target Variable
    df.dropna(subset=['attacktype1_txt'], inplace=True)
    
    # Focus on top 5 attack types for clearer classification task
    top_attacks = df['attacktype1_txt'].value_counts().nlargest(5).index
    df = df[df['attacktype1_txt'].isin(top_attacks)]
    
    # Sample data for model comparison (speed) if requested, else full
    if not full_data and len(df) > 50000:
        print("Sampling 50k rows for model selection...")
        df = df.sample(50000, random_state=42)
    else:
        print(f"Using full dataset: {len(df)} rows")
        
    return df

def get_preprocessor():
    # Numeric features
    numeric_features = ['iyear', 'nkill', 'nwound']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features
    categorical_features = ['region_txt', 'weaptype1_txt', 'targtype1_txt']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def compare_models(df):
    print("\n--- Comparing Models ---")
    
    X = df[['iyear', 'region_txt', 'weaptype1_txt', 'targtype1_txt', 'nkill', 'nwound', 'success']]
    y = df['attacktype1_txt']
    
    preprocessor = get_preprocessor()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # 5-fold CV
        cv_results = cross_validate(clf, X, y, cv=5, scoring=['accuracy', 'f1_weighted'], return_train_score=False)
        
        res = {
            'Model': name,
            'Accuracy': cv_results['test_accuracy'].mean(),
            'F1_Weighted': cv_results['test_f1_weighted'].mean(),
            'Std_Acc': cv_results['test_accuracy'].std()
        }
        results.append(res)
        print(f"  Accuracy: {res['Accuracy']:.4f} (+/- {res['Std_Acc']:.4f})")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("metrics/model_comparison.csv", index=False)
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False))
    plt.title("Model Comparison (5-Fold CV Accuracy)")
    plt.xlim(0.5, 1.0)
    plt.savefig("figures/model_comparison.png")
    plt.close()
    
    best_model_name = results_df.sort_values('F1_Weighted', ascending=False).iloc[0]['Model']
    print(f"\nBest Model identified: {best_model_name}")
    
    # Return best model instance (untrained pipeline structure)
    best_clf_base = models[best_model_name]
    return best_clf_base, best_model_name

def train_final_model(model_base, model_name):
    print(f"\n--- Training Final Model ({model_name}) on Full Dataset ---")
    
    # Load FULL dataset
    df = load_data(full_data=True)
    
    X = df[['iyear', 'region_txt', 'weaptype1_txt', 'targtype1_txt', 'nkill', 'nwound', 'success']]
    y = df['attacktype1_txt']
    
    # Perform a hold-out test split for final honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = get_preprocessor()
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', model_base)])
    
    print("Fitting model...")
    final_pipeline.fit(X_train, y_train)
    
    print("Evaluating on Test Set...")
    y_pred = final_pipeline.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Final Test Accuracy: {acc:.4f}")
    
    # Save Metrics
    with open("metrics/final_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Save Model
    print("Saving model to models/best_model.pkl...")
    joblib.dump(final_pipeline, "models/best_model.pkl")
    
    # Save Confusion Matrix figure
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    # Normalize for better readability
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get labels from the pipeline
    # The output of predict is strings (target), so we need unique labels from y
    labels = np.unique(y)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    # 1. Load sample for comparison
    df_sample = load_data(full_data=False)
    
    # 2. Compare and Select
    best_base, best_name = compare_models(df_sample)
    
    # 3. Train Final
    train_final_model(best_base, best_name)
