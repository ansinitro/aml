"""
Data Preprocessing Module
Handles data cleaning, missing values, outlier detection, and AQI calculation
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(filename, data_type='raw'):
    """Load data from file"""
    filepath = os.path.join(config.DATA_DIRS[data_type], filename)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    return df


def handle_missing_values(df, method='interpolate'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    method : str
        Method to handle missing values: 'interpolate', 'ffill', 'drop'
    """
    df = df.copy()
    
    print(f"Missing values before cleaning:")
    print(df.isnull().sum())
    print()
    
    if method == 'interpolate':
    # Linear interpolation for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # limit_direction='forward' prevents backfilling years of missing data with a single future value
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='forward')
    elif method == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'drop':
        df = df.dropna()
    
    print(f"Missing values after cleaning:")
    print(df.isnull().sum())
    
    return df


def remove_outliers(df, columns=None, method='iqr', threshold=3):
    """
    Remove outliers from specified columns
    
    Parameters:
    -----------
    df : DataFrame
    Input dataframe
    columns : list
    Columns to check for outliers (default: all pollutant columns)
    method : str
    'iqr' for IQR method, 'zscore' for Z-score method
    threshold : float
    Threshold for outlier detection (IQR multiplier or Z-score)
    """
    # SKIP OUTLIER REMOVAL for this dataset
    # Air pollution data naturally has spikes (high values) which are real and important.
    # Also, sparse data can skew IQR calculations, causing valid data to be removed.
    print("Skipping outlier removal to preserve real pollution spikes.")
    return df

    # Original code commented out below:
    """
    df = df.copy()
    
    if columns is None:
        columns = config.POLLUTANTS
    
    initial_rows = len(df)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df = df[z_scores < threshold]
    
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
    
    return df
    """


def calculate_aqi_single(pollutant, concentration):
    """
    Calculate AQI for a single pollutant using US EPA standard
    
    Parameters:
    -----------
    pollutant : str
        Pollutant name (PM2.5, PM10, etc.)
    concentration : float
        Pollutant concentration
    """
    if pollutant not in config.AQI_BREAKPOINTS:
        return None
    
    breakpoints = config.AQI_BREAKPOINTS[pollutant]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            # Linear interpolation formula
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
            return round(aqi)
    
    # If concentration exceeds all breakpoints
    if concentration > breakpoints[-1][1]:
        return 500  # Hazardous
    
    return 0


def calculate_aqi(df):
    """
    Calculate AQI for the entire dataset
    Takes the maximum AQI among all pollutants for each row
    """
    df = df.copy()
    
    aqi_columns = []
    
    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in df.columns:
            col_name = f'AQI_{pollutant}'
            df[col_name] = df[pollutant].apply(lambda x: calculate_aqi_single(pollutant, x))
            aqi_columns.append(col_name)
    
    # Overall AQI is the maximum of individual pollutant AQIs
    if aqi_columns:
        df['AQI'] = df[aqi_columns].max(axis=1)
    
    return df


def categorize_aqi(aqi):
    """Categorize AQI into health concern levels"""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


def add_temporal_features(df, date_column='date'):
    """
    Add temporal features for analysis
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    date_column : str
        Name of the date column
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract temporal features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_name'] = df[date_column].dt.day_name()
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        df['quarter'] = df[date_column].dt.quarter
        
        # Add season
        df['season'] = df['month'].apply(get_season)
        
        # Add heating season flag
        df['is_heating_season'] = df['month'].isin(config.HEATING_SEASON_MONTHS)
        
        # Weekend flag
        df['is_weekend'] = df['day_of_week'] >= 5
    
    return df


def get_season(month):
    """Get season name from month number"""
    for season, months in config.SEASONS.items():
        if month in months:
            return season
    return 'Unknown'


def compare_to_who_standards(df):
    """
    Compare pollutant levels to WHO standards
    Add flags for exceedance
    """
    df = df.copy()
    
    for pollutant in config.POLLUTANTS:
        if pollutant not in df.columns:
            continue
        
        if pollutant in config.WHO_STANDARDS:
            standards = config.WHO_STANDARDS[pollutant]
            
            # Check daily standard
            if 'daily' in standards:
                col_name = f'{pollutant}_exceeds_WHO_daily'
                df[col_name] = df[pollutant] > standards['daily']
            
            # Check annual standard (will be calculated separately)
            if 'annual' in standards:
                annual_mean = df[pollutant].mean()
                print(f"{pollutant} annual mean: {annual_mean:.2f} {standards['unit']}")
                print(f"WHO annual standard: {standards['annual']} {standards['unit']}")
                print(f"Exceeds WHO annual standard: {annual_mean > standards['annual']}")
                print()
    
    return df


def preprocess_pipeline(input_file, output_file='aktobe_processed.csv'):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    input_file : str
        Input filename in data/raw/
    output_file : str
        Output filename for data/processed/
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    # Check if file exists in raw or processed
    if os.path.exists(os.path.join(config.DATA_DIRS['processed'], input_file)):
        df = load_data(input_file, data_type='processed')
    else:
        df = load_data(input_file, data_type='raw')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Handle missing values
    print("\n2. Handling missing values...")
    df = handle_missing_values(df, method='interpolate')
    
    # Remove outliers
    print("\n3. Removing outliers...")
    df = remove_outliers(df, method='iqr', threshold=3)
    
    # Add temporal features
    print("\n4. Adding temporal features...")
    df = add_temporal_features(df)
    
    # Calculate AQI
    print("\n5. Calculating AQI...")
    df = calculate_aqi(df)
    if 'AQI' in df.columns:
        df['AQI_category'] = df['AQI'].apply(categorize_aqi)
        print(f"Average AQI: {df['AQI'].mean():.2f}")
        print("\nAQI Category Distribution:")
        print(df['AQI_category'].value_counts())
    
    # Compare to WHO standards
    print("\n6. Comparing to WHO standards...")
    df = compare_to_who_standards(df)
    
    # Save processed data
    print("\n7. Saving processed data...")
    output_path = os.path.join(config.DATA_DIRS['processed'], output_file)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Example usage
    preprocess_pipeline('aktobe_real_merged.csv', 'aktobe_real_processed.csv')
