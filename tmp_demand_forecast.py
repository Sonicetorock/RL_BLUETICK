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
top_restaurants = df.nlargest(60, 'Votes')

def generate_orders(df, total_orders=2000, date=datetime(2025, 1, 2), start_hour=9, end_hour=22):
    """
    Generate synthetic orders distributed across restaurants.
    
    Args:
        df: DataFrame of restaurants
        total_orders: Total number of orders to generate
        date: Date to generate orders for
        start_hour: Starting hour (24-hour format)
        end_hour: Ending hour (24-hour format)
    
    Returns:
        DataFrame of generated orders
    """
    orders = []
    start_time = date.replace(hour=start_hour, minute=0, second=0)
    end_time = date.replace(hour=end_hour, minute=0, second=0)
    time_span_seconds = (end_time - start_time).total_seconds()
    
    # Calculate weights based on restaurant popularity (votes)
    # More popular restaurants get more orders
    weights = df['Votes'].values
    weights = weights / weights.sum()
    
    # Generate restaurant indices based on weights
    restaurant_indices = np.random.choice(
        range(len(df)), 
        size=total_orders, 
        p=weights
    )
    
    # Track progress
    print(f"Generating {total_orders} orders from {start_time} to {end_time}")
    
    # Generate orders
    for i in range(total_orders):
        # Get the restaurant
        restaurant_idx = restaurant_indices[i]
        restaurant = df.iloc[restaurant_idx]
        
        # Create H3 index for the restaurant location
        h3_index = h3.latlng_to_cell(restaurant['Latitude'], restaurant['Longitude'], 8)
        
        # Create order ID
        order_id = str(uuid.uuid4())
        
        # Generate timestamp with time-of-day weighting
        # More orders around lunch (12-2pm) and dinner (7-9pm) times
        while True:
            # Basic uniform distribution across the day
            random_offset = np.random.randint(0, int(time_span_seconds))
            order_time = start_time + timedelta(seconds=random_offset)
            
            # Apply time-of-day weighting
            hour = order_time.hour
            
            # Probability multipliers for different times of day
            if 11 <= hour < 14:  # Lunch rush
                prob_multiplier = 2.0
            elif 18 <= hour < 21:  # Dinner rush
                prob_multiplier = 2.5
            elif 14 <= hour < 18:  # Afternoon lull
                prob_multiplier = 0.7
            elif 9 <= hour < 11:  # Morning
                prob_multiplier = 0.5
            else:  # Late night
                prob_multiplier = 0.8
                
            # Accept or reject this time based on the multiplier
            if np.random.random() < (prob_multiplier / 2.5):  # Normalize by max multiplier
                break
        
        # Generate realistic preparation time - varies by restaurant popularity
        # More popular restaurants might have longer prep times due to volume
        base_prep_time = 10 + (restaurant['Votes'] / df['Votes'].max()) * 15
        preparation_time = max(10, min(30, int(np.random.normal(base_prep_time, 3))))
        
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
        if (i+1) % 500 == 0:
            print(f"Generated {i+1}/{total_orders} orders")
    
    # Create DataFrame
    orders_df = pd.DataFrame(
        orders, 
        columns=[
            'order_id', 
            'Grid_ID',  # Changed to match your environment naming
            'Timestamp',  # Changed to match your environment naming
            'preparation_time',
            'restaurant_id',
            'restaurant_name'
        ]
    )
    
    # Add additional features
    orders_df['time_bin'] = orders_df['Timestamp'].dt.floor('15min')
    
    # Sort by timestamp
    orders_df = orders_df.sort_values('Timestamp')
    
    return orders_df

# Generate synthetic order dataset with 2000 orders
orders_df = generate_orders(
    top_restaurants,
    total_orders=2000,  # Generate exactly 2000 orders
    date=datetime(2025, 1, 2),
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
output_file = "data/jan1_orders.csv"
orders_df.to_csv(output_file, index=False)
print(f"\nOrders saved to {output_file}")

# Optional: Create a visualization of order distribution by hour
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