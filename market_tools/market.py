from dhanhq import dhanhq
import os
import time
from dotenv import load_dotenv
from data.database import DatabaseQueries
import requests
import threading
from utils.redis_client import main_redis_client as r

# Global cache for prices
_PRICE_CACHE = {
    "prices": {},
    "timestamp": 0
}
_CACHE_TTL = 300  # 5 minutes
_CACHE_LOCK = threading.Lock()

load_dotenv(override=True)

client_id = os.getenv("DHAN_CLIENT_ID")
access_token = os.getenv("DHAN_ACCESS_TOKEN")

dhan = dhanhq(client_id, access_token)


def is_market_open() -> bool:
    NSE_URL = "https://www.nseindia.com/api/marketStatus"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    try:
        response = requests.get(NSE_URL, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        for market in data.get("marketState", []):
            if market.get("marketStatus") == "Open":
                return True
        return False
    except Exception as e:        
        return False
    

def get_security_id(symbol: str) -> int:
    """Fetches security_id for a given symbol, first from Redis, then DB as fallback."""
    # Try Redis first
    sec_id = r.hget('security_id_index', symbol.upper())
    if sec_id:
        try:
            return int(sec_id)
        except Exception:
            pass
    # Fallback to DB
    return int(DatabaseQueries.get_security_id(symbol))


def resolve_symbol_impl(company_query: str) -> str:
    query = company_query.lower().strip()
    symbol = r.hget('symbol_index', query)
    if symbol:
        return symbol.decode()
    symbol = r.hget('name_index', query)
    if symbol:
        return symbol.decode()
    symbol = r.hget('display_index', query)
    if symbol:
        return symbol.decode()
    # Fallback: partial match
    for key in r.hkeys('symbol_index'):
        if query in key.decode():
            return r.hget('symbol_index', key).decode()
    for key in r.hkeys('name_index'):
        if query in key.decode():
            return r.hget('name_index', key).decode()
    for key in r.hkeys('display_index'):
        if query in key.decode():
            return r.hget('display_index', key).decode()
    return None


def get_symbol_ohlc(symbol: str) -> dict:
    """Fetches OHLC and last price data for a given symbol.
    Do not enter exchange name 'NSE_EQ'."""
    security_id = get_security_id(symbol)

    data = dhan.ohlc_data(
        securities={"NSE_EQ": [security_id]}
    )
    
    if not data:
        return {}
    return data


def get_symbol_price_impl(symbol: str) -> float:
    """Fetches the last traded price for a given symbol.
    Do not enter exchange name 'NSE_EQ'."""
    security_id = get_security_id(symbol)

    data = dhan.ticker_data(
        securities={"NSE_EQ": [security_id]}
    )

    if not data:
        return 0.0
    
    try:        
        return (
            data["data"]["data"]
            .get("NSE_EQ", {})
            .get(str(security_id), {})
            .get("last_price", 0.0)
        )
    except Exception as e:
        print(f"Error extracting last_price: {e} | Raw data: {data}", flush=True)
        return 0.0

    
def get_multiple_symbol_prices(symbols: list[str]) -> dict[str, float]:
    """
    Fetches the last traded prices for a list of symbols in a single batch API call.
    Returns a dict mapping symbol -> last_price (float).
    If a symbol's price is not found, its value will be 0.0.
    """        
    unique_symbols = tuple(symbols)

    # Map symbols to security_ids
    symbol_to_sec_id = {}
    for symbol in unique_symbols:
        try:
            sec_id = get_security_id(symbol)
            if sec_id:
                symbol_to_sec_id[symbol] = sec_id
        except Exception as e:
            print(f"Error getting security_id for {symbol}: {e}", flush=True)

    if not symbol_to_sec_id:
        return {symbol: 0.0 for symbol in unique_symbols}

    securities = {
        "NSE_EQ": list(symbol_to_sec_id.values())
        }
    data = dhan.ticker_data(securities=securities)
    
    prices = {}
    nse_data = data.get("data", {}).get("data", {}).get("NSE_EQ", {}) if data else {}

    # Reverse lookup: security_id -> symbol
    sec_id_to_symbol = {str(v): k for k, v in symbol_to_sec_id.items()}

    for sec_id, info in nse_data.items():
        symbol = sec_id_to_symbol.get(str(sec_id))
        price = info.get("last_price", 0.0)
        if symbol:
            prices[symbol] = price

    # Fill in 0.0 for any missing symbols
    for symbol in unique_symbols:
        if symbol not in prices:
            prices[symbol] = 0.0

    return prices


def get_prices_with_cache(symbols: list[str]) -> dict[str, float]:
    """
    Returns cached prices if cache is fresh, otherwise fetches new prices and updates cache.
    Shared across all agents.
    """    
    now = time.time()
    with _CACHE_LOCK:
        # Use cache if fresh and has all requested symbols
        if now - _PRICE_CACHE["timestamp"] < _CACHE_TTL:
            cached = {s: _PRICE_CACHE["prices"].get(s, 0.0) for s in symbols}
            # If all requested symbols are present in cache, return
            if all(s in _PRICE_CACHE["prices"] for s in symbols):
                return cached
        # Otherwise, fetch new prices
        prices = get_multiple_symbol_prices(symbols)
        _PRICE_CACHE["prices"].update(prices)
        _PRICE_CACHE["timestamp"] = now
        # Sleep 1 second to respect API rate limit
        time.sleep(1)
        return {s: _PRICE_CACHE["prices"].get(s, 0.0) for s in symbols}


def get_symbol_history_daily_data(
        symbol: str,         
        from_date: str, 
        to_date: str) -> dict:
    
    """Fetches historical daily data form the given date range for a given symbol."""
    
    data = dhan.historical_daily_data(
        security_id=get_security_id(symbol),
        exchange_segment="NSE_EQ",
        instrument_type="EQUITY",
        from_date=from_date,
        to_date=to_date,
        expiry_code=0
    )
    if not data:
        return {}
    return data


def get_symbol_history_intraday_data(
        symbol: str,        
        interval: str,
        from_date: str,
        to_date: str) -> dict:
    """Fetches historical intraday data from the given time period for a given symbol.
    Args:
        security_id (int): The security ID of the stock.
        exchange_segment (str): The exchange segment (e.g., "NSE_EQ").
        interval (str): The interval for intraday data ("1" for 1 minute) (1, 5, 10, 15, 30, 60).
        from_date (str): Start date in "YYYY-MM-DD HH:MM:SS" format.
        to_date (str): End date in "YYYY-MM-DD HH:MM:SS" format.
    Returns:
        dict: Historical intraday data for the specified security.
        Returns middle date data. (given from: 5th and to: 7th June, returns data for 6th June)
    """

    data = dhan.intraday_minute_data(
        security_id=get_security_id(symbol),
        exchange_segment="NSE_EQ",
        instrument_type="EQUITY",
        from_date=from_date,
        to_date=to_date,
        interval=interval        
    )
    return data





# Example usage
# security_id =   get_security_id("RELIANCE")
# ohlc_data =     (time.sleep(1) or get_symbol_ohlc("RELIANCE"))
# last_price =    (time.sleep(1) or get_symbol_price_impl("RELIANCE"))
# daily_data =    (time.sleep(1) or get_symbol_history_daily_data("RELIANCE", "2025-06-24", "2025-06-25"))
# intraday_data = (time.sleep(1) or get_symbol_history_intraday_data("BBOX", 1, "2025-06-27 09:59:00", "2025-06-27 10:01:00"))
# multiple_data = get_multiple_symbol_prices(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "TATAPOWER", "TATAMOTORS", "HINDUNILVR", "HCLTECH", "LT", "MARUTI", "ASIANPAINT", "BAJFINANCE", "AXISBANK", "ITC", "SBIN", "WIPRO", "ONGC", "ADANIGREEN", "ADANIPORTS"])


# print(f"Is Market Open: {is_market_open()}", flush=True)
# print(f"Security ID: {security_id}", flush=True)
# print(f"OHLC Data: {ohlc_data}", flush=True)
# print(f"Last Price: {last_price}", flush=True)
# print(f"Daily Data: {daily_data}", flush=True)
# print(f"Intraday Data: {intraday_data}", flush=True)
# print(f"Multiple Data: {multiple_data}", flush=True)
