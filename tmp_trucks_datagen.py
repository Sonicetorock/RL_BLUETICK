# gernerate trucks data based on restuarents
# truck should have unqiue id, longitude, latitude, capacity, speed, status, 
# longitde and latitude should be within delhi range
# capacity should be between 100 to 500
# speed should be between 30 to 70
# status should be either available or busy
# want to generate 2000 trucks

import pandas as pd
import numpy as np
import h3
import uuid

def generate_trucks(num_trucks=2000):
    # Delhi region boundaries
    delhi_boundaries = {
        "lat_min": 28.40, 
        "lat_max": 28.88, 
        "lng_min": 76.85, 
        "lng_max": 77.35
    }
    
    trucks = []
    
    for _ in range(num_trucks):
        truck_id = str(uuid.uuid4())  # Assign unique UUID
        
        # Generate random coordinates within Delhi boundaries
        truck_latitude = np.random.uniform(delhi_boundaries["lat_min"], delhi_boundaries["lat_max"])
        truck_longitude = np.random.uniform(delhi_boundaries["lng_min"], delhi_boundaries["lng_max"])
        
        # Generate h3 index for the location
        h3_index = h3.latlng_to_cell(truck_latitude, truck_longitude, 8)
        
        capacity = np.random.randint(100, 501)  # Capacity between 100-500
        speed = np.random.randint(20, 40)  # Speed between 30-70
        status = "available"  # Default status
        
        trucks.append([truck_id, h3_index, truck_latitude, truck_longitude, capacity, speed, status])
    
    trucks_df = pd.DataFrame(trucks, columns=['truck_id', 'h3_index', 'latitude', 'longitude', 'capacity', 'speed', 'status'])
    return trucks_df

# Generate trucks dataset - no longer depends on orders_df
trucks_df = generate_trucks(num_trucks=2000)
trucks_df.to_csv("data/jan1_trucks.csv", index=False)
