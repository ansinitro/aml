import pandas as pd
import io
import os
import sys

# Add src to path to import process_real_data
sys.path.append(os.path.join(os.getcwd(), 'src'))
from process_real_data import parse_aqicn_file

filepath = 'data/raw/517420/Daily no2'
print(f"Testing parsing of {filepath}")

try:
    df = parse_aqicn_file(filepath)
    if df is not None:
        print("Parsing successful")
        print(f"Shape: {df.shape}")
        print("Head:")
        print(df.head())
        print("\nDescribe:")
        print(df['median'].describe())
        
        # Check if dates are parsed correctly
        print("\nIndex type:", df.index.dtype)
        print("First 5 dates:", df.index[:5])
    else:
        print("Parsing returned None")

except Exception as e:
    print(f"Error: {e}")
