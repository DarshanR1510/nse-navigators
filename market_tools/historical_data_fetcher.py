from datetime import datetime, timedelta
from typing import List
import time
import json
from market_tools.market import get_symbol_history_daily_data
from utils.redis_client import main_redis_client as r


def fetch_bulk_historical_data(symbols: List[str]) -> dict:
    """
    Fetches historical daily data for the given stock symbol between specified dates.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name.
    """

    today = datetime.today().strftime("%Y-%m-%d")
    one_year_ago = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    results = {}
    for symbol in symbols:
        try:
            data = get_symbol_history_daily_data(symbol, one_year_ago, today)
            results[symbol] = data
                
            # Store in Redis with 1 hour expiry (3600 seconds)
            r.set(f"historical:{symbol}", json.dumps(data))
            r.expire(f"historical:{symbol}", 3600)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        time.sleep(1.1)

    return results





