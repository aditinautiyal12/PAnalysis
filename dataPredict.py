import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import subprocess
import os

from mergeandtransform import read_and_transform_data

data_folder = 'Data'

def feature_engineering(df):
    # Create additional features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    return df

def train_and_predict_for_symbol(symbol, df_symbol):
    # Assuming 'date' column is in 'yyyy-mm-dd' format
    df_symbol['date'] = pd.to_datetime(df_symbol['date'])
    
    # Sort DataFrame by date if it's not already sorted
    df_symbol.sort_values('date', inplace=True)
    df_symbol.reset_index(drop=True, inplace=True)
    
    # Feature engineering
    df_symbol = feature_engineering(df_symbol)
    
    # Use 'lastPrice' and additional features as predictors
    X = df_symbol[['lastPrice', 'day_of_week', 'day_of_month', 'month']]
    y = df_symbol['lastPrice']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict prices for the next 2 days
    last_date = df_symbol['date'].max()
    next_dates = [last_date + timedelta(days=i) for i in range(1, 3)]  # Next 2 days
    
    # Create a DataFrame for future dates
    future_df = pd.DataFrame({'date': next_dates})
    future_df = feature_engineering(future_df)
    
    # Use the last known 'lastPrice' for predictions
    last_known_lastPrice = df_symbol.loc[df_symbol.index[-1], 'lastPrice']
    future_df['lastPrice'] = last_known_lastPrice
    
    # Predict using the trained model
    next_dates_df = future_df.copy()
    next_dates_df['predicted_price'] = model.predict(future_df[['lastPrice', 'day_of_week', 'day_of_month', 'month']])
    next_dates_df['symbol'] = symbol  # Include symbol in the DataFrame
    
    print(f"Predicted Prices for Symbol {symbol} for Next 2 Days:")
    print(next_dates_df[['date', 'predicted_price']])
    
    # Return the predicted results
    return next_dates_df[['symbol', 'date', 'predicted_price']]

def main():
    # Load combined DataFrame
    combined_df = read_and_transform_data(data_folder)
    
    if combined_df is not None:
        # Group by symbol and predict for each symbol
        predicted_dfs = []
        unique_symbols = combined_df['symbol'].unique()
        
        for symbol in unique_symbols:
            df_symbol = combined_df[combined_df['symbol'] == symbol]
            predicted_df = train_and_predict_for_symbol(symbol, df_symbol)
            if predicted_df is not None:
                predicted_dfs.append(predicted_df)
        
        # Combine all predicted DataFrames into a single DataFrame if needed
        if predicted_dfs:
            combined_predicted_df = pd.concat(predicted_dfs, ignore_index=True)
            combined_predicted_df.to_csv('predicted_prices_all_symbols.csv', index=False)
            print(f"All predicted prices saved to 'predicted_prices_all_symbols.csv'.")
        else:
            print("No predictions made.")
    else:
        print(f"No combined DataFrame returned from {data_folder}.")

if __name__ == "__main__":
    main()
