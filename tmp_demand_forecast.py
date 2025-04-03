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

# Select top 80 restaurants by votes
top_restaurants = df.nlargest(60, 'Votes')

# Generate synthetic orders for 9 AM - 10 PM using Poisson process
def generate_orders(df):
    orders = []
    start_time = datetime(2025, 1, 2, 9, 0)
    end_time = datetime(2025, 1, 2, 22, 0)
    
    for _, row in df.iterrows():
        h3_index = h3.latlng_to_cell(row['Latitude'], row['Longitude'], 8)  # H3 resolution 8
        while True:
            num_orders = np.random.poisson(3)  # Poisson distributed orders
            for _ in range(num_orders):
                order_id = str(uuid.uuid4())
                random_offset = np.random.randint(0, (end_time - start_time).seconds + 1)
                order_time = start_time + timedelta(seconds=random_offset)
                if order_time >= end_time:
                    continue
                preparation_time = np.random.randint(15, 25)  # Random preparation time
                orders.append([order_id, h3_index, order_time, preparation_time])
            break  # Exit after generating orders for this restaurant
    
    orders_df = pd.DataFrame(orders, columns=['order_id', 'h3_index', 'timestamp', 'preparation_time'])

    orders_df['time_bin'] = orders_df['timestamp'].dt.floor('15min')
    return orders_df

# Generate synthetic order dataset
orders_df = generate_orders(top_restaurants)
orders_df.to_csv("data/jan2_orders.csv", index=False)


# # Aggregate orders per H3 hex and 15-min bin
# demand_df = orders_df.groupby(['h3_index', 'time_bin']).size().reset_index(name='demand')



# # Create lag features
# demand_df = demand_df.sort_values(['h3_index', 'time_bin'])
# for lag in range(1, 5):
#     demand_df[f'demand_lag_{lag}'] = demand_df.groupby('h3_index')['demand'].shift(lag)

# demand_df = demand_df.dropna()

# # Prepare data for training
# X = demand_df[['demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4']]
# y = demand_df['demand']

# # Train XGBRegressor model
# train_size = int(len(X) * 0.8)
# X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
# y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# model = XGBRegressor(objective='reg:squarederror', eta=0.05, max_depth=6, subsample=0.8, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# y_pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# mae_std = np.std(np.abs(y_test - y_pred))
# rmse_std = np.std((y_test - y_pred) ** 2)
# r2 = r2_score(y_test, y_pred)

# print(f"RMSE: {rmse:.4f} (std: {rmse_std:.4f})")
# print(f"MAE: {mae:.4f} (std: {mae_std:.4f})")
# print(f"R2 Score: {r2:.4f}")

# # Forecast next 15-minute demand
# latest_data = demand_df.groupby('h3_index').tail(1)[['h3_index', 'time_bin', 'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4']]
# latest_data['time_bin'] += timedelta(minutes=15)
# latest_data['predicted_demand'] = model.predict(latest_data[['demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4']])

# # Save predictions
# latest_data.to_csv("demand_forecast.csv", index=False)