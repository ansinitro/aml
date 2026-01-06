import pandas as pd
import numpy as np
import random

# Simulating 'Housing Price Prediction Dataset Kazakhstan 2025' Schema
# Real file: krisha_50%.csv
# Columns: id, city, floor, total_floors, area, rooms, price, microdistrict, etc.
# We will generate a compatible CSV so the user can plug in the real one later.

def generate_placeholder_data(n_samples=5000):
    np.random.seed(42)
    random.seed(42)

    data = []
    cities = ['Almaty'] # Focusing on Almaty as per previous plan
    microdistricts = ['Samal-1', 'Samal-2', 'Aksay-1', 'Aksay-4', 'Koktem-1', 'Orbita-1', 'Medeu', 'Esentai']
    
    for i in range(n_samples):
        area = int(np.random.gamma(shape=2.5, scale=25)) + 30
        area = max(30, min(area, 400))
        
        # Logic: Rooms correlated with area
        if area < 45: rooms = 1
        elif area < 70: rooms = 2
        elif area < 100: rooms = 3
        elif area < 140: rooms = 4
        else: rooms = random.choice([5, 6])
        
        total_floors = random.choice([5, 9, 12, 16, 21])
        floor = random.randint(1, total_floors)
        
        # Price simulation (Almaty approximate: 400k - 800k KZT per sqm)
        base_sqm_price = 500000 
        mdist = random.choice(microdistricts)
        if mdist in ['Samal-1', 'Samal-2', 'Esentai']:
            base_sqm_price = 700000
        elif mdist in ['Orbita-1', 'Koktem-1']:
            base_sqm_price = 550000
            
        noise = np.random.normal(0, 50000)
        final_sqm_price = base_sqm_price + noise
        
        price = int(area * final_sqm_price)
        
        # Adding some messy data to mimic real scraping
        if random.random() < 0.01:
            price = int(price * 0.1) # Outlier text error

        data.append({
            'city': 'Almaty',
            'microdistrict': mdist,
            'floor': floor,
            'total_floors': total_floors,
            'area': area,
            'rooms': rooms,
            'price': price
        })

    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating schema-compliant placeholder dataset...")
    df = generate_placeholder_data(5000)
    # Save as 'almaty_housing.csv' to be consistent with our internal naming
    # The user can overwrite this with 'krisha_50%.csv' renamed
    df.to_csv('ass1/data/almaty_housing.csv', index=False)
    print("Dataset generated at ass1/data/almaty_housing.csv")
    print("Columns:", df.columns.tolist())
