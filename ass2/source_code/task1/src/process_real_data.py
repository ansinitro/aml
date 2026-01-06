import pandas as pd
import os
import glob
import re

import io

def parse_aqicn_file(filepath):
    """
    Parses an HTML-wrapped CSV file from AQICN.
    Handles <br> delimiters and finds the header line starting with 'date'.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Replace <br> with newlines to fix formatting
        content = content.replace('<br>', '\n')
        
        # Use StringIO to treat string as file
        f_io = io.StringIO(content)
        lines = f_io.readlines()
        
        header_row_index = -1
        for i, line in enumerate(lines):
            # Check for header, case-insensitive
            if line.strip().lower().startswith("date,"):
                header_row_index = i
                break
        
        if header_row_index == -1:
            print(f"Warning: No header found in {filepath}")
            return None
            
        # Reset pointer to start of header
        f_io.seek(0)
        
        # Read CSV from the identified header row
        df = pd.read_csv(f_io, skiprows=header_row_index, skipinitialspace=True)
        
        # Clean column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Set index to date
        df = df.set_index('date')
        
        return df
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def process_station_data(base_dir):
    all_data = []
    
    # Define station directories
    # Note: One directory has a leading space " 216661"
    station_dirs = {
        '216661': os.path.join(base_dir, ' 216661'),
        '517420': os.path.join(base_dir, '517420')
    }
    
    for station_id, station_path in station_dirs.items():
        if not os.path.exists(station_path):
            print(f"Warning: Station path not found: {station_path}")
            continue
            
        files = glob.glob(os.path.join(station_path, 'Daily *'))
        
        for filepath in files:
            filename = os.path.basename(filepath)
            # Extract parameter name (e.g., "Daily pm25" -> "pm25")
            param = filename.replace('Daily ', '').replace('.', '_').lower()
            
            # Map parameter names to standard names if needed
            # met.t -> temp, met.h -> humidity, met.p -> pressure
            param_map = {
                'met_t': 'temperature',
                'met_h': 'humidity',
                'met_p': 'pressure',
                'pm25': 'pm2_5',
                'pm1': 'pm1', # Keep as is
                'pm10': 'pm10',
                'no2': 'no2',
                'so2': 'so2',
                'co': 'co'
            }
            
            std_param = param_map.get(param, param)
            
            print(f"Processing {station_id} - {std_param} from {filename}...")
            
            df = parse_aqicn_file(filepath)
            if df is not None:
                # We'll use the 'median' value as the representative daily value
                # Rename 'median' to the parameter name
                if 'median' in df.columns:
                    df_subset = df[['median']].copy()
                    df_subset.rename(columns={'median': std_param}, inplace=True)
                    
                    # Add station identifier
                    df_subset['station'] = station_id
                    
                    # Reset index to make date a column for concatenation
                    df_subset = df_subset.reset_index()
                    
                    all_data.append(df_subset)
                else:
                    print(f"Warning: 'median' column not found in {filename}")

    if not all_data:
        print("No data found!")
        return

    # Concatenate all data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Pivot to wide format: Date as index, (parameter, station) as columns
    pivot_df = full_df.pivot_table(index='date', columns=['station'], values=['pm2_5', 'pm10', 'pm1', 'no2', 'so2', 'co', 'temperature', 'humidity', 'pressure'])
    
    # Flatten columns
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    
    # Create unified columns
    # List of parameters to unify
    params = ['pm2_5', 'pm10', 'pm1', 'no2', 'so2', 'co', 'temperature', 'humidity', 'pressure']
    
    final_df = pd.DataFrame(index=pivot_df.index)
    
    for param in params:
        cols = [c for c in pivot_df.columns if c.startswith(param)]
        if not cols:
            continue
            
        # If multiple stations have data, take the mean
        # This handles overlapping periods by averaging, and non-overlapping by taking the available one
        combined_series = pivot_df[cols].mean(axis=1)
        
        # For pollutants, 0 is likely missing data (based on inspection)
        # We'll replace 0 with NaN for specific columns
        if param in ['pm2_5', 'pm10', 'pm1', 'no2', 'so2', 'co']:
            combined_series = combined_series.replace(0, float('nan'))
            
        final_df[param] = combined_series
        
        if param == 'no2':
            print(f"DEBUG: NO2 combined series stats:")
            print(combined_series.describe())
            print("DEBUG: NO2 head:")
            print(combined_series.head())
            print("DEBUG: NO2 tail:")
            print(combined_series.tail())
            
    # Rename columns to match config.POLLUTANTS (uppercase)
    # pm2_5 -> PM2.5, pm10 -> PM10, etc.
    rename_map = {
        'pm2_5': 'PM2.5',
        'pm10': 'PM10',
        'pm1': 'PM1',
        'no2': 'NO2',
        'so2': 'SO2',
        'co': 'CO',
        'o3': 'O3',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'pressure': 'Pressure'
    }
    final_df.rename(columns=rename_map, inplace=True)
            
    # Sort by date
    final_df = final_df.sort_index()
    
    # Filter for reasonable years (e.g., 2023-2025) or keep all? 
    # The user mentioned "fragmented" and "inconsistent coverage".
    # Let's keep all data but maybe flag the "valid" range for analysis.
    
    # Save to CSV
    output_path = os.path.join(base_dir, '../processed/aktobe_real_merged.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path)
    print(f"Saved merged data to {output_path}")
    print(final_df.describe())
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")

if __name__ == "__main__":
    # Use dynamic path relative to this script
    # Script is in src/, so data/raw is at ../data/raw
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, 'data', 'raw')
    process_station_data(base_dir)
