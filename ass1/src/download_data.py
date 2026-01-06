import kagglehub
import os
import shutil

# Download latest version using the handle found in search results (yersultan) or user provided (yrsltn)
# Trying 'yersultan' based on search result link.
try:
    path = kagglehub.dataset_download("yersultan/housing-price-prediction-dataset-kazakhstan-2025")
    print("Path to dataset files:", path)
    
    # Move to our data directory
    target_dir = "ass1/data"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Find the csv file
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, "real_housing_data.csv")
                shutil.copy(src_path, dst_path)
                print(f"Moved {file} to {dst_path}")
                break
except Exception as e:
    print(f"Error downloading: {e}")
