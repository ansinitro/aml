import os
import requests
import zipfile
import io

def download_data(output_dir='data'):
    """Downloads ml-latest-small.zip and extracts it."""
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Check if data already exists to avoid re-downloading
    if os.path.exists(os.path.join(output_dir, 'ml-latest-small', 'ratings.csv')):
        print("Dataset already exists. Skipping download.")
        return

    print(f"Downloading dataset from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(output_dir)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        raise

if __name__ == "__main__":
    download_data('../data')
