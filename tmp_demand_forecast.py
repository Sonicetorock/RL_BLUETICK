import pandas as pd
import numpy as np
import h3
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import uuid

# Set random seed
np.random.seed(42)

# Load restaurant dataset
df = pd.read_csv("zomato_timeline_orders_delhi.csv", encoding="ISO-8859-1")
df = df[df['City'] == 'New Delhi']

# Select top restaurants by votes
top_restaurants = df.nlargest(200
                              , 'Votes')

def generate_orders(df, total_orders=2000, days=1, start_hour=9, end_hour=22):
    """
    Generate synthetic orders with stronger temporal and spatial patterns.
    
    Args:
        df: DataFrame of restaurants
        total_orders: Total orders per day
        days: Number of days to generate data for
        start_hour: Starting hour (24-hour format)
        end_hour: Ending hour (24-hour format)
    
    Returns:
        DataFrame of generated orders
    """
    orders = []
    # Generate orders for multiple days
    for day_offset in range(days):
        date = datetime(2025, 1, 15) + timedelta(days=day_offset)
        start_time = date.replace(hour=start_hour, minute=0, second=0)
        end_time = date.replace(hour=end_hour, minute=0, second=0)
        time_span_seconds = (end_time - start_time).total_seconds()
        
        # Calculate weights based on restaurant popularity (votes)
        weights = df['Votes'].values
        weights = weights / weights.sum()
        
        # Track progress
        print(f"Generating {total_orders} orders for {date.strftime('%Y-%m-%d')}")
        
        # Create clusters of restaurants in similar areas
        # Group restaurants by H3 index to simulate popular areas
        restaurant_h3 = []
        for _, restaurant in df.iterrows():
            h3_index = h3.latlng_to_cell(restaurant['Latitude'], restaurant['Longitude'], 8)
            restaurant_h3.append((restaurant.name, h3_index))
        
        # Group by H3 and count restaurants in each area
        h3_counts = {}
        for _, h3_idx in restaurant_h3:
            h3_counts[h3_idx] = h3_counts.get(h3_idx, 0) + 1
        
        # Assign area popularity - areas with more restaurants are more popular
        area_weights = np.array(list(h3_counts.values()))
        area_weights = area_weights / area_weights.sum() * 1.3  # Boost effect
        area_indices = list(h3_counts.keys())
        
        # Generate restaurant indices based on weights
        restaurant_indices = np.random.choice(
            range(len(df)), 
            size=total_orders, 
            p=weights
        )
        
        # Define hourly patterns with stronger peaks
        hourly_weights = {
            9: 0.3,   # Morning start
            10: 0.5,  # Mid-morning
            11: 0.8,  # Pre-lunch
            12: 2.5,  # Lunch peak
            13: 2.2,  # Late lunch
            14: 0.7,  # Post-lunch
            15: 0.5,  # Mid-afternoon
            16: 0.6,  # Late afternoon
            17: 1.2,  # Pre-dinner
            18: 2.0,  # Early dinner
            19: 2.8,  # Dinner peak
            20: 2.3,  # Late dinner
            21: 1.5,  # Post-dinner
            22: 0.7   # Late evening
        }
        max_weight = max(hourly_weights.values())
        
        # Generate orders with clearer patterns
        for i in range(total_orders):
            # Get the restaurant
            restaurant_idx = restaurant_indices[i]
            restaurant = df.iloc[restaurant_idx]
            
            # Create H3 index for the restaurant location
            h3_index = h3.latlng_to_cell(restaurant['Latitude'], restaurant['Longitude'], 8)
            
            # Create order ID
            order_id = f"{date.strftime('%Y%m%d')}-{i+1:04d}"
            
            # Generate timestamp with enhanced time-of-day weighting
            order_time = None
            while True:
                # Target hour with more precision
                target_hour = np.random.choice(
                    range(start_hour, end_hour + 1),
                    p=[hourly_weights[h]/sum(hourly_weights.values()) for h in range(start_hour, end_hour + 1)]
                )
                
                # Random minutes within the hour
                minutes = np.random.randint(0, 60)
                
                # Create time
                order_time = date.replace(hour=target_hour, minute=minutes)
                
                # Add some daily variation - adjust weights slightly
                day_factor = 1.0
                if date.weekday() >= 5:  # Weekend
                    day_factor = 1.2  # More orders on weekends
                
                # Acceptance probability
                weight = hourly_weights.get(target_hour, 0.5) / max_weight * day_factor
                
                if np.random.random() < weight:
                    break
            
            # Generate preparation time with more variation by time of day
            # Busier times have longer prep times
            time_factor = hourly_weights.get(order_time.hour, 1.0) / max_weight
            base_prep_time = 8 + (restaurant['Votes'] / df['Votes'].max()) * 12 + time_factor * 5
            preparation_time = max(5, min(30, int(np.random.normal(base_prep_time, 4))))
            
            # Add to orders list
            orders.append([
                order_id,
                h3_index,
                order_time,
                preparation_time,
                restaurant['Restaurant ID'],
                restaurant['Restaurant Name']
            ])
            
        # Show progress
        print(f"Generated {total_orders} orders for {date.strftime('%Y-%m-%d')}")
    
    # Create DataFrame
    orders_df = pd.DataFrame(
        orders, 
        columns=[
            'order_id', 
            'Grid_ID',
            'Timestamp',
            'preparation_time',
            'restaurant_id',
            'restaurant_name'
        ]
    )
    
    # Add additional features
    orders_df['time_bin'] = orders_df['Timestamp'].dt.floor('15min')
    orders_df['day_of_week'] = orders_df['Timestamp'].dt.dayofweek
    orders_df['is_weekend'] = orders_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Sort by timestamp
    orders_df = orders_df.sort_values('Timestamp')
    
    return orders_df
    
# Generate synthetic order dataset with 5000 orders
orders_df = generate_orders(
    top_restaurants,
    total_orders=5000,  # Generate exactly 5000 orders
    # date=datetime(2025, 1, 1),
    start_hour=9,
    end_hour=22
)

# Print summary statistics
print("\nOrder Generation Summary:")
print(f"Total orders: {len(orders_df)}")
print(f"Time range: {orders_df['Timestamp'].min()} to {orders_df['Timestamp'].max()}")
print(f"Unique restaurants: {orders_df['restaurant_id'].nunique()}")
print(f"Unique grid cells: {orders_df['Grid_ID'].nunique()}")
print("\nOrders per hour:")
print(orders_df.groupby(orders_df['Timestamp'].dt.hour).size())

# Save to CSV
output_file = "data/current_day_orders.csv"
orders_df.to_csv(output_file, index=False)
print(f"\nOrders saved to {output_file}")

try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    orders_df.groupby(orders_df['Timestamp'].dt.hour).size().plot(
        kind='bar', 
        color='skyblue',
        edgecolor='black'
    )
    plt.title('Order Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Orders')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/order_distribution.png')
    print("Created visualization: data/order_distribution.png")
except Exception as e:
    print(f"Could not create visualization: {e}")