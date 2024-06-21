import requests
import os
import pandas as pd
from datetime import datetime

def fetch_crypto_data(api_url):
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            return df
        else:
            print(f"Failed to retrieve data from API. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {str(e)}")
        return None

def save_dataframe_to_csv(df, folder, filename):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath, index=False)
        print(f"Data frame stored successfully in {filepath}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {str(e)}")

def main():
    api_url = "https://api.wazirx.com/sapi/v1/tickers/24hr"
    df = fetch_crypto_data(api_url)

    if df is not None:
        try:
            current_date = datetime.now().strftime("%Y%m%d")
            data_folder = 'Data'
            filename = f"wazirx_data_{current_date}.csv"
            save_dataframe_to_csv(df, data_folder, filename)
        except Exception as e:
            print(f"Error processing data: {str(e)}")
    else:
        print("No data retrieved. Exiting.")

if __name__ == "__main__":
    main()
