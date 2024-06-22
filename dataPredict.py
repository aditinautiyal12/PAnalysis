import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import subprocess
import os

from mergeandtransform import read_and_transform_data

data_folder = 'Data'

def train_and_predict_for_symbol(symbol, df_symbol):
    df_symbol['date'] = pd.to_datetime(df_symbol['date'])
    
    df_symbol.sort_values('date', inplace=True)
    df_symbol.reset_index(drop=True, inplace=True)
    
    X = df_symbol[['lastPrice']]  
    y = df_symbol['lastPrice']   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    last_date = df_symbol['date'].max()
    next_dates = [last_date + timedelta(days=i) for i in range(1, 3)]
    next_dates_df = pd.DataFrame({'date': next_dates})
    next_dates_df['date'] = pd.to_datetime(next_dates_df['date'])
    
    
    last_known_lastPrice = df_symbol.loc[df_symbol.index[-1], 'lastPrice']
    next_dates_df['lastPrice'] = last_known_lastPrice 

    if 'lastPrice' in next_dates_df.columns:  
        next_dates_df['predicted_price'] = model.predict(next_dates_df[['lastPrice']])
        next_dates_df['symbol'] = symbol  
        print(f"Predicted Prices for Symbol {symbol} for Next 2 Days:")
        print(next_dates_df[['date', 'predicted_price']])
        
        return next_dates_df[['symbol', 'date', 'predicted_price']]
    else:
        print(f"Error: 'lastPrice' column not found in symbol {symbol} data.")
        return None

def main():
    combined_df = read_and_transform_data(data_folder)
    
    if combined_df is not None:
        predicted_dfs = []
        unique_symbols = combined_df['symbol'].unique()
        
        for symbol in unique_symbols:
            df_symbol = combined_df[combined_df['symbol'] == symbol]
            predicted_df = train_and_predict_for_symbol(symbol, df_symbol)
            if predicted_df is not None:
                predicted_dfs.append(predicted_df)
        
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
