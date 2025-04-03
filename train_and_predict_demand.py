# import pandas as pd
# import numpy as np
# import h3
# from xgboost import XGBRegressor
# from datetime import datetime, timedelta
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# import joblib
# import os
# import matplotlib.pyplot as plt

# # Set random seed for reproducibility
# np.random.seed(42)

# def prepare_features(orders_df):
#     """
#     Prepare features for the demand forecasting model
#     """
#     # Ensure timestamp is in datetime format
#     orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
#     orders_df['time_bin'] = pd.to_datetime(orders_df['time_bin'])
    
#     # Aggregate orders per H3 hex and 15-min bin
#     demand_df = orders_df.groupby(['h3_index', 'time_bin']).size().reset_index(name='demand')
    
#     # Add temporal features
#     demand_df['hour'] = demand_df['time_bin'].dt.hour
#     demand_df['day_of_week'] = demand_df['time_bin'].dt.dayofweek
    
#     # Sort by location and time
#     demand_df = demand_df.sort_values(['h3_index', 'time_bin'])
    
#     # Create lag features (previous 4 time windows = 1 hour of history)
#     for lag in range(1, 5):
#         demand_df[f'demand_lag_{lag}'] = demand_df.groupby('h3_index')['demand'].shift(lag)
    
#     # Drop rows with NaN values (first 4 time bins for each h3_index)
#     demand_df = demand_df.dropna()
    
#     return demand_df

# def train_and_evaluate_model(demand_df):
#     """
#     Train XGBoost model for demand forecasting and evaluate on test set
#     """
#     # Prepare data for training
#     feature_cols = ['hour', 'day_of_week', 'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4']
#     X = demand_df[feature_cols]
#     y = demand_df['demand']
    
#     # Split data into training and testing sets (80:20)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
#     # Train XGBRegressor model
#     model = XGBRegressor(
#         objective='reg:squarederror', 
#         eta=0.05, 
#         max_depth=6, 
#         subsample=0.8, 
#         random_state=42
#     )
#     model.fit(X_train, y_train)
    
#     # Evaluate model on training data
#     y_train_pred = model.predict(X_train)
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     train_mae = mean_absolute_error(y_train, y_train_pred)
#     train_r2 = r2_score(y_train, y_train_pred)
    
#     print(f"\nTraining Metrics:")
#     print(f"RMSE: {train_rmse:.4f}")
#     print(f"MAE: {train_mae:.4f}")
#     print(f"R2 Score: {train_r2:.4f}")
    
#     # Evaluate model on test data
#     y_test_pred = model.predict(X_test)
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#     test_mae = mean_absolute_error(y_test, y_test_pred)
#     test_r2 = r2_score(y_test, y_test_pred)
    
#     print(f"\nTest Metrics:")
#     print(f"RMSE: {test_rmse:.4f}")
#     print(f"MAE: {test_mae:.4f}")
#     print(f"R2 Score: {test_r2:.4f}")
    
#     # Feature importance
#     feature_importance = dict(zip(feature_cols, model.feature_importances_))
#     print("\nFeature Importance:")
#     for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
#         print(f"{feature}: {importance:.4f}")
    
#     # Plot actual vs predicted demand
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_test_pred, alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     plt.xlabel('Actual Demand')
#     plt.ylabel('Predicted Demand')
#     plt.title('Actual vs Predicted Demand')
#     plt.savefig('demand_prediction_scatter.png')
    
#     # Plot error distribution
#     plt.figure(figsize=(10, 6))
#     errors = y_test - y_test_pred
#     plt.hist(errors, bins=30)
#     plt.xlabel('Prediction Error')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Prediction Errors')
#     plt.savefig('demand_prediction_errors.png')
    
#     return model, test_rmse, test_mae, test_r2

# def predict_next_demand(model, feature_info, current_time, h3_index, demand_history):
#     """
#     Predict demand for the next 15 minutes
    
#     Parameters:
#         model: Trained XGBoost model
#         feature_info: Dictionary containing feature information
#         current_time: datetime object representing the current time
#         h3_index: H3 index of the location
#         demand_history: list of 4 values representing demand in the last 4 time bins
#                        [demand_t-1, demand_t-2, demand_t-3, demand_t-4]
    
#     Returns:
#         Predicted demand for the next 15 minute time bin
#     """
#     # Create feature row
#     features = {
#         'hour': current_time.hour,
#         'day_of_week': current_time.weekday(),
#     }
    
#     # Add lag features
#     for i, lag_val in enumerate(demand_history[:4], 1):
#         features[f'demand_lag_{i}'] = lag_val
    
#     # Ensure all required features are present
#     for feature in feature_info['feature_columns']:
#         if feature not in features:
#             features[feature] = 0
    
#     # Convert to DataFrame
#     X = pd.DataFrame([features])[feature_info['feature_columns']]
    
#     # Predict
#     prediction = model.predict(X)[0]
    
#     return max(0, round(prediction))

# def main():
#     # Load Jan 1 orders data
#     try:
#         orders_df = pd.read_csv("data/jan1_orders.csv")
#     except FileNotFoundError:
#         print("Error: Could not find jan1_orders.csv in the data directory")
#         return
    
#     print(f"Loaded {len(orders_df)} orders for processing")
    
#     # Prepare features
#     demand_df = prepare_features(orders_df)
#     print(f"Prepared {len(demand_df)} time-bin records for modeling")
    
#     # Train and evaluate the model
#     model, test_rmse, test_mae, test_r2 = train_and_evaluate_model(demand_df)
    
#     # Create models directory if it doesn't exist
#     os.makedirs("models", exist_ok=True)
    
#     # Save the model
#     model_path = "models/demand_forecast_xgb.joblib"
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")
    
#     # Save feature information for later use
#     feature_info = {
#         'feature_columns': ['hour', 'day_of_week', 'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4'],
#         'test_metrics': {
#             'rmse': test_rmse,
#             'mae': test_mae,
#             'r2': test_r2
#         }
#     }
#     joblib.dump(feature_info, "models/demand_forecast_features.joblib")
#     print("Feature information saved for later use")
    
#     # Example prediction
#     print("\nExample Prediction:")
#     current_time = datetime(2025, 1, 2, 10, 0)
#     h3_index = "883da1152bfffff"  # Example H3 index
#     demand_history = [5, 3, 4, 2]  # Last 4 time bins
    
#     prediction = predict_next_demand(model, feature_info, current_time, h3_index, demand_history)
#     print(f"Predicted demand for {h3_index} at {current_time + timedelta(minutes=15)}: {prediction} orders")

# if __name__ == "__main__":
#     main()
# -------------------------------------------------------------------------------------------------
# version 2
import pandas as pd
import numpy as np
import h3
from xgboost import XGBRegressor
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def prepare_features(orders_df):
    """
    Prepare features for the demand forecasting model with temporal and spatial adjustments.
    """
    # Ensure timestamp is in datetime format
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
    orders_df['time_bin'] = pd.to_datetime(orders_df['time_bin'])
    
    # Aggregate orders per H3 hex and 15-min bin
    demand_df = orders_df.groupby(['h3_index', 'time_bin']).size().reset_index(name='demand')
    
    # Add temporal features
    demand_df['hour'] = demand_df['time_bin'].dt.hour
    demand_df['day_of_week'] = demand_df['time_bin'].dt.dayofweek
    
    # Sort by location and time
    demand_df = demand_df.sort_values(['h3_index', 'time_bin'])
    
    # Create lag features (previous 4 time windows = 1 hour of history)
    for lag in range(1, 5):
        demand_df[f'demand_lag_{lag}'] = demand_df.groupby('h3_index')['demand'].shift(lag)
    
    # Add rolling demand features (e.g., demand over the last 2 hours)
    demand_df['rolling_2h'] = demand_df.groupby('h3_index')['demand'].rolling(window=8, min_periods=1).sum().reset_index(0, drop=True)
    
    # Add spatial features (demand from neighboring grids)
    demand_df['neighbor_demand'] = demand_df.apply(
        lambda row: calculate_neighbor_demand(row['h3_index'], row['time_bin'], demand_df), axis=1
    )
    
    # Drop rows with NaN values (first 4 time bins for each h3_index)
    demand_df = demand_df.dropna()
    
    return demand_df

def calculate_neighbor_demand(h3_index, time_bin, demand_df):
    """
    Calculate the average demand from neighboring grids.
    """
    try:
        # Get neighboring grids using H3
        neighbors = h3.grid_ring(h3_index, 1)
        neighbors = [n for n in neighbors if n != h3_index]
        
        # Filter demand_df for the same time_bin and neighboring grids
        neighbor_demand = demand_df[
            (demand_df['h3_index'].isin(neighbors)) & (demand_df['time_bin'] == time_bin)
        ]['demand'].mean()
        
        return neighbor_demand if not np.isnan(neighbor_demand) else 0
    except Exception as e:
        print(f"Error calculating neighbor demand for {h3_index}: {e}")
        return 0
def train_and_evaluate_model(demand_df):
    """
    Train XGBoost model for demand forecasting and evaluate on test set.
    """
    # Prepare data for training
    feature_cols = [
        'hour', 'day_of_week', 'demand_lag_1', 'demand_lag_2', 
        'demand_lag_3', 'demand_lag_4', 'rolling_2h', 'neighbor_demand'
    ]
    X = demand_df[feature_cols]
    y = demand_df['demand']
    
    # Split data into training and testing sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train XGBRegressor model
    model = XGBRegressor(
        objective='reg:squarederror', 
        eta=0.05, 
        max_depth=6, 
        subsample=0.8, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save feature columns for consistency during prediction
    feature_info = {
        'feature_columns': feature_cols,
        'test_metrics': {
            'rmse': np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
            'mae': mean_absolute_error(y_test, model.predict(X_test)),
            'r2': r2_score(y_test, model.predict(X_test))
        }
    }
    joblib.dump(feature_info, "models/demand_forecast_features.joblib")
    print("Feature information saved for later use")
    
    return model
def main():
    # Load Jan 1 orders data
    try:
        orders_df = pd.read_csv("data/jan1_orders.csv")
    except FileNotFoundError:
        print("Error: Could not find jan1_orders.csv in the data directory")
        return
    
    print(f"Loaded {len(orders_df)} orders for processing")
    
    # Prepare features
    demand_df = prepare_features(orders_df)
    print(f"Prepared {len(demand_df)} time-bin records for modeling")
    
    # Train and evaluate the model
    model = train_and_evaluate_model(demand_df)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model_path = "models/demand_forecast_xgb.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Load and display test metrics from the saved feature info
    feature_info = joblib.load("models/demand_forecast_features.joblib")
    test_metrics = feature_info['test_metrics']
    print("\nTest Metrics:")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R2 Score: {test_metrics['r2']:.4f}")
if __name__ == "__main__":
    main()