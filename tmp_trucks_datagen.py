# # gernerate trucks data based on restuarents
# # truck should have unqiue id, longitude, latitude, capacity, speed, status, 
# # longitde and latitude should be within delhi range
# # capacity should be between 100 to 500
# # speed should be between 30 to 70
# # status should be either available or busy
# # want to generate 2000 trucks

# import pandas as pd
# import numpy as np
# import h3
# import uuid

# def generate_trucks(num_trucks=2000):
#     # Delhi region boundaries
#     delhi_boundaries = {
#         "lat_min": 28.40, 
#         "lat_max": 28.88, 
#         "lng_min": 76.85, 
#         "lng_max": 77.35
#     }
    
#     trucks = []
    
#     for _ in range(num_trucks):
#         truck_id = str(uuid.uuid4())  # Assign unique UUID
        
#         # Generate random coordinates within Delhi boundaries
#         truck_latitude = np.random.uniform(delhi_boundaries["lat_min"], delhi_boundaries["lat_max"])
#         truck_longitude = np.random.uniform(delhi_boundaries["lng_min"], delhi_boundaries["lng_max"])
        
#         # Generate h3 index for the location
#         h3_index = h3.latlng_to_cell(truck_latitude, truck_longitude, 8)
        
#         capacity = np.random.randint(100, 501)  # Capacity between 100-500
#         speed = np.random.randint(20, 40)  # Speed between 30-70
#         status = "available"  # Default status
        
#         trucks.append([truck_id, h3_index, truck_latitude, truck_longitude, capacity, speed, status])
    
#     trucks_df = pd.DataFrame(trucks, columns=['truck_id', 'h3_index', 'latitude', 'longitude', 'capacity', 'speed', 'status'])
#     return trucks_df

# # Generate trucks dataset - no longer depends on orders_df
# trucks_df = generate_trucks(num_trucks=2000)
# trucks_df.to_csv("data/jan1_trucks.csv", index=False)


import pandas as pd
import numpy as np
import h3
import uuid
from collections import Counter

def generate_trucks(num_trucks=2000, restaurant_proximity_percent=70):
    # Load restaurants from orders data
    try:
        orders_df = pd.read_csv("data/jan1_orders.csv")
        
        # Count order frequency by restaurant_id and find top restaurants
        restaurant_counts = orders_df['restaurant_id'].value_counts()
        top_restaurants = restaurant_counts.head(100).index.tolist()
        
        # Get unique locations for top restaurants
        top_restaurant_locations = orders_df[orders_df['restaurant_id'].isin(top_restaurants)][['restaurant_id', 'Grid_ID']].drop_duplicates()
        
        # Convert Grid_ID to coordinates
        restaurant_coords = []
        for _, row in top_restaurant_locations.iterrows():
            h3_index = row['Grid_ID']
            lat, lng = h3.cell_to_latlng(h3_index)
            restaurant_coords.append((lat, lng))
        
        print(f"Loaded {len(top_restaurants)} most popular restaurants across {len(restaurant_coords)} unique locations")
        
        # Print top 5 restaurant IDs with their order counts
        top_5 = restaurant_counts.head(5)
        print("\nTop 5 most popular restaurants:")
        for rest_id, count in top_5.items():
            rest_name = orders_df[orders_df['restaurant_id'] == rest_id]['restaurant_name'].iloc[0]
            print(f"ID: {rest_id}, Name: {rest_name}, Orders: {count}")
            
    except FileNotFoundError:
        print("Warning: Could not find orders data. Using random distribution instead.")
        restaurant_coords = []
        restaurant_proximity_percent = 0
    
    # Delhi region boundaries
    delhi_boundaries = {
        "lat_min": 28.40, 
        "lat_max": 28.88, 
        "lng_min": 76.85, 
        "lng_max": 77.35
    }
    
    trucks = []
    
    # Calculate how many trucks to place near restaurants
    near_restaurant_count = int(num_trucks * restaurant_proximity_percent / 100)
    random_count = num_trucks - near_restaurant_count
    
    # Generate trucks near popular restaurants
    if restaurant_coords and near_restaurant_count > 0:
        print(f"\nGenerating {near_restaurant_count} trucks near popular restaurants")
        
        # Create enough coordinates by cycling through restaurant_coords if needed
        all_coords_needed = []
        while len(all_coords_needed) < near_restaurant_count:
            all_coords_needed.extend(restaurant_coords[:near_restaurant_count-len(all_coords_needed)])
        
        for i in range(near_restaurant_count):
            truck_id = str(uuid.uuid4())
            
            # Get restaurant location
            rest_lat, rest_lng = all_coords_needed[i]
            
            # Add small random offset (around 0.2 to 0.8 km)
            offset_lat = np.random.normal(0, 0.003)  # ~300m in latitude
            offset_lng = np.random.normal(0, 0.003)  # ~300m in longitude
            
            truck_latitude = rest_lat + offset_lat
            truck_longitude = rest_lng + offset_lng
            
            # Ensure coordinates are within Delhi boundaries
            truck_latitude = min(max(truck_latitude, delhi_boundaries["lat_min"]), delhi_boundaries["lat_max"])
            truck_longitude = min(max(truck_longitude, delhi_boundaries["lng_min"]), delhi_boundaries["lng_max"])
            
            # Generate h3 index for the location
            h3_index = h3.latlng_to_cell(truck_latitude, truck_longitude, 8)
            
            capacity = np.random.randint(100, 501)
            speed = np.random.randint(20, 40)
            status = "available"
            
            trucks.append([truck_id, h3_index, truck_latitude, truck_longitude, capacity, speed, status])
    
    # Generate random trucks
    print(f"Generating {random_count} randomly distributed trucks")
    for _ in range(random_count):
        truck_id = str(uuid.uuid4())
        
        # Generate random coordinates within Delhi boundaries
        truck_latitude = np.random.uniform(delhi_boundaries["lat_min"], delhi_boundaries["lat_max"])
        truck_longitude = np.random.uniform(delhi_boundaries["lng_min"], delhi_boundaries["lng_max"])
        
        # Generate h3 index for the location
        h3_index = h3.latlng_to_cell(truck_latitude, truck_longitude, 8)
        
        capacity = np.random.randint(100, 501)
        speed = np.random.randint(20, 40)
        status = "available"
        
        trucks.append([truck_id, h3_index, truck_latitude, truck_longitude, capacity, speed, status])
    
    trucks_df = pd.DataFrame(trucks, columns=['truck_id', 'h3_index', 'latitude', 'longitude', 'capacity', 'speed', 'status'])
    return trucks_df

# Generate trucks dataset
trucks_df = generate_trucks(num_trucks=2000, restaurant_proximity_percent=70)
trucks_df.to_csv("data/jan1_trucks.csv", index=False)
print(f"Generated {len(trucks_df)} trucks and saved to data/jan1_trucks.csv")