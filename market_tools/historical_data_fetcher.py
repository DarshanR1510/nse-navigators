from datetime import datetime, timedelta
from typing import List
import time
import json
from market_tools.market import get_symbol_history_daily_data
from utils.redis_client import main_redis_client as r



#====================#    PART OF MANUAL DATA FETCHING   #===================================#

# import redis
# import os
# from dhanhq import dhanhq
# from dotenv import load_dotenv
# load_dotenv(override=True)

# client_id = os.getenv("DHAN_CLIENT_ID")
# access_token = os.getenv("DHAN_ACCESS_TOKEN")
# dhan = dhanhq(client_id, access_token)

# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# main_redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
# r = main_redis_client

#=============================================================================================#



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
                
            # Store in Redis with 10 hour expiry (36000 seconds)
            r.set(f"historical:{symbol}", json.dumps(data))
            r.expire(f"historical:{symbol}", 36000)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        time.sleep(1.1)

    return results



#========#======#======#   TO MANUALLY FETCH DATA FOR SYMBOLS   #========#======#======#

# def get_symbol_history_daily_data(
#         symbol: str,         
#         from_date: str, 
#         to_date: str) -> dict:
    
#     """Fetches historical daily data form the given date range for a given symbol."""
    
#     data = dhan.historical_daily_data(
#         security_id=get_security_id(symbol),
#         exchange_segment="NSE_EQ",
#         instrument_type="EQUITY",
#         from_date=from_date,
#         to_date=to_date,
#         expiry_code=0
#     )
#     if not data:
#         return {}
#     return data


# def get_security_id(symbol: str) -> int:
#     """Fetches security_id for a given symbol, first from Redis, then DB as fallback."""    
#     sec_id = r.hget('symbol:' + symbol.lower(), 'security_id')
#     if sec_id:
#         try:
#             return int(sec_id.decode())
#         except Exception:
#             pass    


# if __name__ == "__main__":    
#     symbols = ["RELIANCE", "TATAMOTORS", "HDFCBANK", "TCS", "INFY", "BBOX", "TATAPOWER", "ASTERDM"]
#     historical_data = fetch_bulk_historical_data(symbols)
#     print(len(historical_data))