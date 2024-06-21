import requests
import pandas as pd

response = requests.get("https://api.wazirx.com/sapi/v1/tickers/24hr");

data = response.json()
df = pd.DataFrame(data)