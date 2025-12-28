import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

def download_dataset():
    print("Downloading dataset...")
    # Download the dataset files
    path = kagglehub.dataset_download("START-UMD/gtd")
    print(f"Dataset files downloaded to: {path}")
    
    # List files to find the correct one
    files = os.listdir(path)
    print("Files found:", files)
    
    # Find the CSV or Excel file
    data_file = None
    for f in files:
        if f.endswith('.csv') or f.endswith('.xlsx'):
            data_file = os.path.join(path, f)
            break
            
    if not data_file:
        raise FileNotFoundError("No CSV or Excel file found in the dataset.")
        
    print(f"Loading data from {data_file}...")
    # Use 'low_memory=False' for large files with mixed types
    df = pd.read_csv(data_file, encoding='ISO-8859-1', low_memory=False)
    
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("First 5 records:")
    print(df.head())
    
    output_path = "gtd_dataset.csv"
    print(f"Saving dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Dataset saved.")

if __name__ == "__main__":
    download_dataset()
