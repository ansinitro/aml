import os
import gzip
import json
import wget
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Config
REVIEWS_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MIN_INTERACTIONS = 5
SUBCATEGORY = ['Headphones', 'Earbuds', 'Over-Ear Headphones'] 

def download_file(url, filename):
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} from {url}...")
        try:
            wget.download(url, path)
            print("\nDownload complete.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"{filename} already exists.")
    return path

def get_headphone_asins(meta_path):
    print("Scanning metadata for Headphones...")
    target_asins = set()
    count = 0
    with gzip.open(meta_path, 'r') as f:
        for line in f:
            try:
                d = eval(line) # SNAP metadata is python-eval safe usually, or use json if strict json
                # SNAP metadata is sometimes single-quote python dict string, let's try strict json first then eval
            except:
                try:
                    d = json.loads(line)
                except:
                    continue
            
            categories = d.get('categories', []) # SNAP uses 'categories' list of lists
            # e.g. [['Electronics', 'Headphones', ...]]
            
            is_target = False
            for cat_chain in categories:
                for cat in cat_chain:
                    if any(k in cat for k in SUBCATEGORY):
                        is_target = True
                        break
                if is_target: break
            
            if is_target:
                target_asins.add(d['asin'])
            
            count += 1
            if count % 100000 == 0:
                print(f"Scanned {count} meta items... Found {len(target_asins)} targets.")
                
    print(f"Total Headphone ASINs identified: {len(target_asins)}")
    return target_asins

def process_data():
    reviews_path = download_file(REVIEWS_URL, "reviews_Electronics_5.json.gz")
    meta_path = download_file(META_URL, "meta_Electronics.json.gz")

    headphone_asins = get_headphone_asins(meta_path)
    
    if not headphone_asins:
        print("No headphone ASINs found! Checking logic...")
        # Fallback logic if eval/json failed?
        return

    print("Filtering reviews...")
    data = []
    count = 0
    with gzip.open(reviews_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            if d['asin'] in headphone_asins:
                data.append({
                    'reviewerID': d['reviewerID'],
                    'asin': d['asin'],
                    'unixReviewTime': d['unixReviewTime']
                })
            
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} reviews... Kept {len(data)}.")

    df = pd.DataFrame(data)
    print(f"Total Headerphone interactions: {len(df)}")
    
    if len(df) == 0:
        print("Error: No interactions found after filtering.")
        return

    # 2. Filter 5-core
    print("Filtering (5-core)...")
    def filter_k_core(df, k=5):
        while True:
            users = df.groupby('reviewerID').size()
            items = df.groupby('asin').size()
            
            valid_users = users[users >= k].index
            valid_items = items[items >= k].index
            
            new_df = df[df['reviewerID'].isin(valid_users) & df['asin'].isin(valid_items)]
            
            if len(new_df) == len(df):
                break
            df = new_df
        return df

    df = filter_k_core(df, k=MIN_INTERACTIONS)
    print(f"Shape after 5-core: {df.shape}")
    print(f"Sparsity: {1 - len(df) / (df['reviewerID'].nunique() * df['asin'].nunique()):.6f}")

    # 3. Time Split
    df = df.sort_values('unixReviewTime')
    times = df['unixReviewTime'].values
    train_thresh = np.percentile(times, 70)
    val_thresh = np.percentile(times, 85)
    
    train = df[df['unixReviewTime'] <= train_thresh].copy()
    val = df[(df['unixReviewTime'] > train_thresh) & (df['unixReviewTime'] <= val_thresh)].copy()
    test = df[df['unixReviewTime'] > val_thresh].copy()
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    train.to_parquet(os.path.join(PROCESSED_DIR, 'train.parquet'))
    val.to_parquet(os.path.join(PROCESSED_DIR, 'val.parquet'))
    test.to_parquet(os.path.join(PROCESSED_DIR, 'test.parquet'))
    print("Saved to data/processed/")

if __name__ == "__main__":
    process_data()
