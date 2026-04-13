import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_stock_price(symbol):
    url = "xxx"
    params = {
        "access_key": os.getenv("xxx"),
        "symbols": symbol
    }

    response = requests.get(url, params=params)
    data = response.json()

    try:
        stock = data["data"][0]
        return f"Stock Symbol: {stock['symbol']}, Close Price: {stock['close']}"
    except:
        return "Stock data not available"
