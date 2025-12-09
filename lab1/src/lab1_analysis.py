"""
Lab 1: Kazakhstan Basin Water Level Prediction
Advanced Machine Learning Course

This script:
1. Loads the dataset from Kaggle using kagglehub
2. Preprocesses data (handles Russian columns, missing values)
3. Performs Hyperparameter Tuning using GridSearchCV
4. Trains Decision Tree, Random Forest, KNN with optimal parameters
5. Evaluates each model with 5 different train/test splits
6. Implements a Hybrid Stacking model (Best Model + MLP)
7. Outputs results in Table 1 format
"""

import pandas as pd
import numpy as np
import kagglehub
import os
import glob
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def load_data():
    """Download and load the Kazakhstan Basin Water Level dataset."""
    print("=" * 60)
    print("STEP 1: Loading Dataset from Kaggle")
    print("=" * 60)
    
    path = kagglehub.dataset_download("sabinayesniyazova/kazakhstan-basin-water-level-kazhydromet")
    print(f"Dataset downloaded to: {path}")
    
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = []
    
    print(f"\nFound {len(all_files)} CSV files:")
    for filename in all_files:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        df_list.append(df)
        print(f"  - {os.path.basename(filename)}: {len(df)} rows")
        
    if not df_list:
        raise ValueError("No CSV files found in the dataset path.")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined_df)}")
    return combined_df


def preprocess_data(df):
    """Preprocess the dataset: rename columns, extract features, clean data."""
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)
    
    # Column mapping (Russian -> English)
    rename_map = {
        'Код поста': 'gid',
        'Дата': 'date',
        'Значение': 'water_level'
    }
    df = df.rename(columns=rename_map)
    print(f"Renamed columns: {list(rename_map.keys())} -> {list(rename_map.values())}")
    
    # Validate target column
    if 'water_level' not in df.columns:
        raise ValueError("Target 'water_level' not found in dataset.")
    
    # Convert target to numeric (handle '-' placeholders)
    original_count = len(df)
    df['water_level'] = pd.to_numeric(df['water_level'], errors='coerce')
    df = df.dropna(subset=['water_level'])
    print(f"Removed {original_count - len(df)} rows with invalid target values")
    
    # Extract temporal features from date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        print("Extracted 'year' and 'month' from 'date' column")
    
    # Prepare features and target
    feature_cols = ['year', 'month', 'gid']
    X = df[feature_cols].copy()
    y = df['water_level'].copy()
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Target: water_level")
    print(f"Final dataset size: {len(X)} samples")
    
    return X, y


def tune_hyperparameters(X, y):
    """Perform GridSearchCV to find optimal hyperparameters for each model."""
    print("\n" + "=" * 60)
    print("STEP 3: Hyperparameter Tuning (GridSearchCV)")
    print("=" * 60)
    
    # Use a subset for faster tuning if dataset is large
    if len(X) > 10000:
        X_tune, _, y_tune, _ = train_test_split(X, y, train_size=10000, random_state=42)
        print(f"Using {len(X_tune)} samples for hyperparameter tuning (subset)")
    else:
        X_tune, y_tune = X, y
    
    best_params = {}
    
    # Decision Tree parameter grid
    print("\n🔍 Tuning Decision Tree...")
    dt_param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        dt_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    dt_grid.fit(X_tune, y_tune)
    best_params['Decision Tree'] = dt_grid.best_params_
    print(f"   Best params: {dt_grid.best_params_}")
    print(f"   Best CV R²: {dt_grid.best_score_:.4f}")
    
    # Random Forest parameter grid
    print("\n🔍 Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    rf_grid.fit(X_tune, y_tune)
    best_params['Random Forest'] = rf_grid.best_params_
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   Best CV R²: {rf_grid.best_score_:.4f}")
    
    # KNN parameter grid
    print("\n🔍 Tuning KNN...")
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_grid = GridSearchCV(
        KNeighborsRegressor(),
        knn_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    knn_grid.fit(X_tune, y_tune)
    best_params['KNN'] = knn_grid.best_params_
    print(f"   Best params: {knn_grid.best_params_}")
    print(f"   Best CV R²: {knn_grid.best_score_:.4f}")
    
    return best_params


def evaluate_models(X, y, best_params):
    """Train and evaluate models with tuned hyperparameters."""
    print("\n" + "=" * 60)
    print("STEP 4: Training and Evaluating Models (with tuned params)")
    print("=" * 60)
    
    results = []
    
    # Create models with best parameters
    algorithms = {
        "Decision Tree": DecisionTreeRegressor(random_state=42, **best_params['Decision Tree']),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1, **best_params['Random Forest']),
        "KNN": KNeighborsRegressor(**best_params['KNN'])
    }
    
    # Define test sizes for 5 iterations
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for algo_name, model in algorithms.items():
        print(f"\nTraining {algo_name}...")
        
        for i, test_size in enumerate(test_sizes):
            iteration = i + 1
            train_size_pct = round((1.0 - test_size) * 100, 1)
            test_size_pct = round(test_size * 100, 1)
            
            # Split data with consistent random state
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42 + i
            )
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                "Algorithm": algo_name,
                "Iteration": iteration,
                "Num Features": X.shape[1],
                "Num Targets": 1,
                "Train Size %": train_size_pct,
                "Test Size %": test_size_pct,
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            })
            
            print(f"  Iteration {iteration} (Train {train_size_pct}%/Test {test_size_pct}%): "
                  f"RMSE={rmse:.2f}, R²={r2:.4f}")
    
    return pd.DataFrame(results), algorithms, best_params


def train_hybrid_model(X, y, best_model_name, best_params):
    """Train a Hybrid Stacking model combining the best base model with MLP."""
    print("\n" + "=" * 60)
    print("STEP 5: Training Hybrid Model")
    print("=" * 60)
    print(f"Architecture: {best_model_name} (tuned) + MLP (Stacking Ensemble)")
    
    # Get tuned base model
    if best_model_name == "Decision Tree":
        base_estimator = DecisionTreeRegressor(random_state=42, **best_params['Decision Tree'])
    elif best_model_name == "Random Forest":
        base_estimator = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params['Random Forest'])
    else:
        base_estimator = KNeighborsRegressor(**best_params['KNN'])
    
    results = []
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for i, test_size in enumerate(test_sizes):
        iteration = i + 1
        train_size_pct = round((1.0 - test_size) * 100, 1)
        test_size_pct = round(test_size * 100, 1)
        
        # Create fresh stacking model for each iteration
        hybrid_model = StackingRegressor(
            estimators=[('base', base_estimator)],
            final_estimator=MLPRegressor(
                hidden_layer_sizes=(50, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        )
        
        # Split with same random state as base models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42 + i
        )
        
        # Train and predict
        hybrid_model.fit(X_train, y_train)
        y_pred = hybrid_model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Algorithm": f"Hybrid ({best_model_name} + MLP)",
            "Iteration": iteration,
            "Num Features": X.shape[1],
            "Num Targets": 1,
            "Train Size %": train_size_pct,
            "Test Size %": test_size_pct,
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })
        
        print(f"  Iteration {iteration} (Train {train_size_pct}%/Test {test_size_pct}%): "
              f"RMSE={rmse:.2f}, R²={r2:.4f}")
    
    return results


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("LAB 1: WATER LEVEL PREDICTION (with Hyperparameter Tuning)")
    print("Advanced Machine Learning")
    print("=" * 60)
    
    # 1. Load Dataset
    df = load_data()
    
    # 2. Preprocess
    X, y = preprocess_data(df)
    
    # 3. Hyperparameter Tuning
    best_params = tune_hyperparameters(X, y)
    
    # 4. Evaluate Models with tuned parameters
    results_df, algorithms, best_params = evaluate_models(X, y, best_params)
    
    # 5. Find best model based on average R2
    print("\n" + "=" * 60)
    print("STEP 6: Model Comparison")
    print("=" * 60)
    
    avg_scores = results_df.groupby("Algorithm")["R2"].mean()
    print("\nAverage R² Scores:")
    for algo, score in avg_scores.sort_values(ascending=False).items():
        print(f"  {algo}: {score:.4f}")
    
    best_algo = avg_scores.idxmax()
    print(f"\n★ Best Algorithm: {best_algo} (Avg R²: {avg_scores[best_algo]:.4f})")
    
    # 6. Hybrid Model
    hybrid_results = train_hybrid_model(X, y, best_algo, best_params)
    results_df = pd.concat([results_df, pd.DataFrame(hybrid_results)], ignore_index=True)
    
    # 7. Output Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS: Table 1")
    print("=" * 60)
    print(results_df.to_markdown(index=False))
    
    # Save to CSV (output folder)
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'lab1_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Save best parameters
    params_df = pd.DataFrame([
        {"Algorithm": k, "Best Parameters": str(v)} for k, v in best_params.items()
    ])
    params_path = os.path.join(output_dir, 'best_parameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"✓ Best parameters saved to {params_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nBest Hyperparameters Found:")
    for algo, params in best_params.items():
        print(f"  {algo}: {params}")
    
    hybrid_avg = results_df[results_df["Algorithm"].str.contains("Hybrid")]["R2"].mean()
    print(f"\nHybrid Model Average R²: {hybrid_avg:.4f}")
    print(f"Recommendation: Deploy {best_algo} with tuned parameters.")


if __name__ == "__main__":
    main()
