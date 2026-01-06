import pandas as pd
import re
import numpy as np

def clean_price(price_str):
    if pd.isna(price_str): return np.nan
    # Remove all non-digits
    clean = re.sub(r'[^\d]', '', str(price_str))
    try:
        return int(clean)
    except:
        return np.nan

def parse_title(title_str):
    # Format: "4-комнатная квартира · 130.27 м²" or "Дом · ... м²"
    rooms = np.nan
    area = np.nan
    
    if pd.isna(title_str): return rooms, area
    
    # Extract Rooms
    room_match = re.search(r'(\d+)-комнатная', title_str)
    if room_match:
        rooms = int(room_match.group(1))
    
    # Extract Area
    area_match = re.search(r'(\d+[.,]?\d*)\s*м²', title_str)
    if area_match:
        area_str = area_match.group(1).replace(',', '.')
        try:
            area = float(area_str)
        except:
            pass
            
    return rooms, area

def parse_district(subtitle_str):
    # Format: "Бостандыкский р-н, Абая 62"
    if pd.isna(subtitle_str): return 'Unknown', 'Unknown'
    
    parts = subtitle_str.split(',')
    district = 'Unknown'
    
    if len(parts) > 0 and 'р-н' in parts[0]:
        district = parts[0].strip()
    
    return district

def preprocess():
    input_path = 'ass1/data/real_housing_data.csv'
    output_path = 'ass1/data/almaty_housing_clean.csv'
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Parsing features...")
    # Price
    df['price_clean'] = df['price'].apply(clean_price)
    
    # Title features
    title_features = df['title'].apply(parse_title)
    df['rooms'] = title_features.apply(lambda x: x[0])
    df['area'] = title_features.apply(lambda x: x[1])
    
    # Subtitle features (District)
    df['district'] = df['subtitle'].apply(parse_district)
    
    # Normalize City
    df['city'] = 'Almaty' # Dataset is mostly Almaty
    
    # Select final columns
    final_df = df[['city', 'district', 'rooms', 'area', 'price_clean']].rename(columns={'price_clean': 'price'})
    
    # Filter bad data
    final_df = final_df.dropna(subset=['price', 'area'])
    final_df = final_df[final_df['price'] > 1000000] # Basic filter
    
    print(f"Processed {len(final_df)} rows.")
    print(final_df.head())
    
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    preprocess()
