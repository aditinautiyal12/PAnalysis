import requests
import pandas as pd
from datetime import datetime

response = requests.get("https://api.wazirx.com/sapi/v1/tickers/24hr");

data = response.json()
df = pd.DataFrame(data)

current_date = datetime.now().strftime("%Y%m%d")
filename = f"wazirx_data_{current_date}.csv"

df.to_csv(filename, index=False)

print("Data frame stored successfully")