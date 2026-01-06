import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time

# 1. Load Data
DATA_PATH = 'ass1/data/almaty_housing_clean.csv' 
print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Basic Cleanup (Handling real world data issues if present)
# Ensure numerical columns are actually numerical
numeric_cols = ['area', 'rooms', 'floor', 'total_floors']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Target
target = 'price'
if target not in df.columns:
    # Try to find target case-insensitive
    candidates = [c for c in df.columns if 'price' in c.lower()]
    if candidates:
        target = candidates[0]
        print(f"Target column 'price' not found, using '{target}' instead.")

# Drop rows where target is missing
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

# 2. Preprocessing Pipeline
# Identify categorical and numerical columns
categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
numerical_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]

print(f"Categorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Define Models
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    "KNN Regression": KNeighborsRegressor(),
    "Extra Trees Regression": ExtraTreesRegressor(n_jobs=-1, random_state=42),
    "Adaptive Boosting": AdaBoostRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_jobs=-1, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

# 4. Training & Evaluation
results = []
metrics = ['neg_root_mean_squared_error', 'r2']
cv = KFold(n_splits=10, shuffle=True, random_state=42)

print("\nStarting 10-fold Cross-Validation...")
print(f"{'Algorithm':<25} | {'RMSE':<15} | {'R2':<10} | {'Time (s)':<10}")
print("-" * 70)

for name, model in models.items():
    start_time = time.time()
    
    # Create full pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', model)])
    
    # Cross Validate
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=metrics, n_jobs=1) # n_jobs=1 to avoid race conditions in some envs
    
    elapsed = time.time() - start_time
    
    # Metrics (sklearn returns negative RMSE)
    rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    r2 = cv_results['test_r2'].mean()
    
    print(f"{name:<25} | {rmse:,.2f}       | {r2:.4f}     | {elapsed:.2f}")
    
    results.append({
        "Algorithm": name,
        "Number of features": len(X.columns), # Pre-encoding
        "Number of targets": 1,
        "k-fold validation": 10,
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

# 5. Save Results
results_df = pd.DataFrame(results)
results_df.to_csv('ass1/report/results_table.csv', index=False)
print("\nResults saved to ass1/report/results_table.csv")
