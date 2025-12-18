import pandas as pd
import preprocessing
import eda
import os

def main():
    print("=== Regenerating EDA Figures Only ===")
    
    # 1. Load Data
    data_path = 'data/sentiment140_raw.csv'
    if not os.path.exists(data_path):
        print("Data file not found!")
        return
        
    print("Loading data...")
    full_df = pd.read_csv(data_path)
    # Scale: Use 200k samples to match the main pipeline
    df = full_df.sample(n=200000, random_state=42) 
    print(f"Data Loaded. Shape: {df.shape}")
    
    # 2. Preprocessing
    print("\n--- Preprocessing (Critical for correct Word Clouds) ---")
    df = preprocessing.preprocess_dataframe(df)
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"Data Shape after cleaning: {df.shape}")
    
    # 3. Run EDA
    print("\n--- Generating Figures ---")
    # Ensure output directories exist
    os.makedirs('paper/figures', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Run EDA - saving to paper/figures
    eda.run_eda(df=df, output_dir='paper/figures')
    
    # Run EDA - saving to figures/ (for presentation)
    eda.run_eda(df=df, output_dir='figures')
    
    print("\n=== EDA Regeneration Complete ===")

if __name__ == "__main__":
    main()
