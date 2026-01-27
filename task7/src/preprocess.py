import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess(data_dir='data'):
    """Loads raw data, merges, and creates train/test splits."""
    ratings_path = os.path.join(data_dir, 'ml-latest-small', 'ratings.csv')
    movies_path = os.path.join(data_dir, 'ml-latest-small', 'movies.csv')
    
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError("Raw data files not found. Run download_data.py first.")

    print("Loading data...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Merge ratings with movie metadata
    df = pd.merge(ratings, movies, on='movieId')
    
    # Simple EDA prints
    print(f"Total ratings: {len(df)}")
    print(f"Unique users: {df['userId'].nunique()}")
    print(f"Unique movies: {df['movieId'].nunique()}")
    
    # Split data
    # We use a simple random split here. 
    # For time-series sensitive tasks, we should split by time, but for general collaborative filtering, random is standard for this scope.
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    output_dir = os.path.join(data_dir, 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Saving processed splits...")
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    movies.to_csv(os.path.join(output_dir, 'movies.csv'), index=False) # Save movies separately for content-based
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    load_and_preprocess('../data')
