import kagglehub
import pandas as pd
import os
import shutil

# Download latest version using the handle provided by user
try:
    path = kagglehub.dataset_download("yrsltn/housing-price-prediction-dataset-kazakhstan-2025")
    print("Path to dataset files:", path)
    
    # Analyze column structure
    # Find the csv file
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
    
    if csv_file:
        print(f"Found CSV file: {csv_file}")
        # Move to ass1/data
        target = "ass1/data/real_housing_data.csv"
        shutil.copy(csv_file, target)
        print(f"Copied to {target}")

        df = pd.read_csv(target)
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
    else:
        print("No CSV file found in the downloaded dataset.")

except Exception as e:
    print(f"Error: {e}")
