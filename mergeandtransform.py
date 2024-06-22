import os
import pandas as pd

def read_and_transform_data(folder):
    all_files = os.listdir(folder)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    dfs = []
    
    for file in csv_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        
        if 'at' in df.columns:
            df['date'] = pd.to_datetime(df['at'], unit='ms').dt.strftime('%Y-%m-%d')
        else:
            print(f"No 'at' column found in {file}. Transformation skipped for this file.")
        
        dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(combined_df.head())
        return combined_df
    else:
        print(f"No CSV files found in {folder}. Returning None.")
        return None