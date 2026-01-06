import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import os

# Set style for academic publication
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def generate_figures():
    print("Loading data...")
    df = pd.read_csv('ass1/data/almaty_housing_clean.csv')
    
    # 1. Correlation Heatmap (Numeric)
    plt.figure(figsize=(10, 8))
    numeric_df = df[['price', 'area', 'rooms']].copy()
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix: Price vs Structural Features')
    plt.tight_layout()
    plt.savefig('ass1/report/figures/fig1_correlation.png', dpi=300)
    print("Generated Fig 1: Correlation")

    # Prepare Data for Model-Based Plots
    X = df.drop(columns=['price'])
    y = df['price']
    
    numeric_cols = ['area', 'rooms']
    categorical_cols = ['city', 'district']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # Train Ridge (Best Performer)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', Ridge())])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 2. Actual vs Predicted Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#2c3e50', s=10)
    
    # Perfect fit line
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    
    plt.xlabel('Actual Price (KZT)')
    plt.ylabel('Predicted Price (KZT)')
    plt.title('Ridge Regression: Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ass1/report/figures/fig2_actual_vs_pred.png', dpi=300)
    print("Generated Fig 2: Actual vs Predicted")
    
    # 3. Feature Importance (Ridge Coefficients)
    feature_names = (numeric_cols + 
                     list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)))
    
    coefs = model.named_steps['regressor'].coef_
    
    # Create DataFrame
    impt_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    impt_df['Abs_Coef'] = impt_df['Coefficient'].abs()
    impt_df = impt_df.sort_values(by='Abs_Coef', ascending=False).head(10) # Top 10
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Coefficient', y='Feature', data=impt_df, palette='viridis')
    plt.title('Top 10 Feature Coefficients (Ridge Regression)')
    plt.xlabel('Coefficient Magnitude (Impact on Price)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('ass1/report/figures/fig3_feature_importance.png', dpi=300)
    print("Generated Fig 3: Feature Importance")
    
    # 4. Residual Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#e74c3c', bins=50)
    plt.title('Residuals Distribution (Checking Normality)')
    plt.xlabel('Prediction Error (KZT)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ass1/report/figures/fig4_residuals.png', dpi=300)
    print("Generated Fig 4: Residuals")

if __name__ == "__main__":
    generate_figures()
