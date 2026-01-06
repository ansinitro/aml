import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def download_data():
    print("Downloading Sentiment140 using kagglehub...")
    
    try:
        # File name in the dataset
        file_path = "training.1600000.processed.noemoticon.csv"

        # Load the dataset
        # Sentiment140 is latin-1 encoded and has no header.
        df = kagglehub.load_dataset(
          KaggleDatasetAdapter.PANDAS,
          "kazanova/sentiment140",
          file_path,
          pandas_kwargs={
              'encoding': 'latin-1',
              'header': None
          }
        )
        
        print("Dataset loaded successfully.")
        
        # Rename columns standard for Sentiment140
        column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        if len(df.columns) == 6:
            df.columns = column_names
        else:
            print(f"Warning: Unexpected number of columns: {len(df.columns)}")
            
        print("First 5 records:")
        print(df.head())

        # Create data directory if not exits
        os.makedirs('data', exist_ok=True)
        
        print("Saving to data/sentiment140_raw.csv...")
        # Save only necessary columns
        df[['sentiment', 'text']].to_csv('data/sentiment140_raw.csv', index=False)
        print("Done.")

    except Exception as e:
        print(f"Failed to load data: {e}")
        raise e

if __name__ == "__main__":
    download_data()
