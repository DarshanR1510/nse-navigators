from utils.dhan_client import dhan
import time
from dotenv import load_dotenv
from data.database import DatabaseQueries
import requests
import threading
from utils.redis_client import main_redis_client as r
import re
from typing import Optional, List, Tuple
from difflib import SequenceMatcher


# Global cache for prices
_PRICE_CACHE = {
    "prices": {},
    "timestamp": 0
}
_CACHE_TTL = 300  # 5 minutes
_CACHE_LOCK = threading.Lock()

load_dotenv(override=True)

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
    sec_id = r.hget('symbol:' + symbol.lower(), 'security_id')
    if sec_id:
        try:
            return int(sec_id.decode())
        except Exception:
            pass
    # Fallback to DB
    return int(DatabaseQueries.get_security_id(symbol))


# def resolve_symbol_impl(company_query: str) -> str:
#     """
#     Given a company query (symbol, name, or display), return the best matching UNDERLYING_SYMBOL.
#     """
#     query = company_query.lower().strip()
#     symbol_keys = r.keys('symbol:*')

#     # 1. Exact match on symbol
#     for key in symbol_keys:
#         symbol_data = r.hgetall(key)
#         symbol = symbol_data.get(b'symbol', b'').decode().lower()
#         if symbol == query:
#             return symbol_data.get(b'symbol', b'').decode()

#     # 2. Exact match on name or display
#     for key in symbol_keys:
#         symbol_data = r.hgetall(key)
#         name = symbol_data.get(b'name', b'').decode().lower()
#         display = symbol_data.get(b'display', b'').decode().lower()
#         if name == query or display == query:
#             return symbol_data.get(b'symbol', b'').decode()

#     # 3. Partial match on symbol, name, or display
#     for key in symbol_keys:
#         symbol_data = r.hgetall(key)
#         symbol = symbol_data.get(b'symbol', b'').decode().lower()
#         name = symbol_data.get(b'name', b'').decode().lower()
#         display = symbol_data.get(b'display', b'').decode().lower()
#         if query in symbol or query in name or query in display:
#             return symbol_data.get(b'symbol', b'').decode()

#     return None


def resolve_symbol_impl(company_query: str) -> Optional[str]:
    """
    Enhanced symbol resolution with improved accuracy and performance.
    Returns the best matching UNDERLYING_SYMBOL for a given company query.
    """
    if not company_query or not company_query.strip():
        return None
        
    query = company_query.strip()
    query_lower = query.lower()
    
    # Get all symbol keys once
    symbol_keys = r.keys('symbol:*')
    if not symbol_keys:
        return None
    
    # Store results for different match types with scores
    exact_matches = []
    fuzzy_matches = []
    partial_matches = []
    
    # Pre-compile regex for better performance on partial matching
    # Escape special regex characters in query
    escaped_query = re.escape(query_lower)
    partial_pattern = re.compile(escaped_query, re.IGNORECASE)
    
    for key in symbol_keys:
        try:
            symbol_data = r.hgetall(key)
            if not symbol_data:
                continue
                
            # Decode all fields once
            symbol = symbol_data.get(b'symbol', b'').decode().strip()
            name = symbol_data.get(b'name', b'').decode().strip()
            display = symbol_data.get(b'display', b'').decode().strip()
            
            if not symbol:  # Skip if no symbol
                continue
                
            symbol_lower = symbol.lower()
            name_lower = name.lower()
            display_lower = display.lower()
            
            # 1. EXACT MATCHES (highest priority)
            if (query_lower == symbol_lower or 
                query_lower == name_lower or 
                query_lower == display_lower):
                exact_matches.append((symbol, 1.0))
                continue
            
            # 2. FUZZY MATCHES (for typos and slight variations)
            # Check fuzzy matching with high threshold
            symbol_ratio = SequenceMatcher(None, query_lower, symbol_lower).ratio()
            name_ratio = SequenceMatcher(None, query_lower, name_lower).ratio()
            display_ratio = SequenceMatcher(None, query_lower, display_lower).ratio()
            
            max_fuzzy_ratio = max(symbol_ratio, name_ratio, display_ratio)
            if max_fuzzy_ratio >= 0.85:  # High similarity threshold
                fuzzy_matches.append((symbol, max_fuzzy_ratio))
                continue
            
            # 3. WORD-BASED MATCHING (for multi-word company names)
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            display_words = set(display_lower.split())
            
            # Check if all query words are present in name or display
            if query_words and (query_words.issubset(name_words) or query_words.issubset(display_words)):
                # Calculate word overlap score
                name_overlap = len(query_words.intersection(name_words)) / len(query_words)
                display_overlap = len(query_words.intersection(display_words)) / len(query_words)
                word_score = max(name_overlap, display_overlap)
                fuzzy_matches.append((symbol, word_score))
                continue
            
            # 4. PARTIAL MATCHES (lowest priority, more flexible)
            if (partial_pattern.search(symbol_lower) or 
                partial_pattern.search(name_lower) or 
                partial_pattern.search(display_lower)):
                
                # Calculate partial match score based on length ratio
                score = len(query) / max(len(symbol), len(name), len(display), 1)
                partial_matches.append((symbol, score))
                
        except (UnicodeDecodeError, AttributeError) as e:
            # Skip corrupted entries
            continue
    
    # Return best match based on priority and score
    if exact_matches:
        return exact_matches[0][0]  # Return first exact match
    
    if fuzzy_matches:
        # Sort by score descending and return best match
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        return fuzzy_matches[0][0]
    
    if partial_matches:
        # Sort by score descending and return best match
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return partial_matches[0][0]
    
    return None


def resolve_symbol_with_cache(company_query: str, use_cache: bool = True) -> Optional[str]:
    """
    Enhanced version with optional caching for frequently searched queries.
    """
    if not company_query or not company_query.strip():
        return None
    
    cache_key = f"symbol_cache:{company_query.lower().strip()}"
    
    if use_cache:
        # Check cache first
        cached_result = r.get(cache_key)
        if cached_result:
            return cached_result.decode() if cached_result != b'NULL' else None
    
    # Perform search
    result = resolve_symbol_impl(company_query)
    
    if use_cache:
        # Cache the result (including None results to avoid repeated searches)
        cache_value = result if result else 'NULL'
        r.setex(cache_key, 3600, cache_value)  # Cache for 1 hour
    
    return result


def resolve_symbol_batch(company_queries: List[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Batch processing for multiple queries - more efficient for bulk operations.
    """
    results = []
    
    # Get all symbol data once
    symbol_keys = r.keys('symbol:*')
    symbol_data_cache = {}
    
    for key in symbol_keys:
        try:
            data = r.hgetall(key)
            if data and b'symbol' in data:
                symbol = data.get(b'symbol', b'').decode().strip()
                if symbol:
                    symbol_data_cache[key] = {
                        'symbol': symbol,
                        'name': data.get(b'name', b'').decode().strip(),
                        'display': data.get(b'display', b'').decode().strip()
                    }
        except (UnicodeDecodeError, AttributeError):
            continue
    
    # Process each query
    for query in company_queries:
        result = resolve_symbol_with_preloaded_data(query, symbol_data_cache)
        results.append((query, result))
    
    return results


def resolve_symbol_with_preloaded_data(company_query: str, symbol_data_cache: dict) -> Optional[str]:
    """
    Helper function for batch processing with preloaded data.
    """
    if not company_query or not company_query.strip():
        return None
        
    query = company_query.strip()
    query_lower = query.lower()
    
    exact_matches = []
    fuzzy_matches = []
    partial_matches = []
    
    escaped_query = re.escape(query_lower)
    partial_pattern = re.compile(escaped_query, re.IGNORECASE)
    
    for key, data in symbol_data_cache.items():
        symbol = data['symbol']
        name = data['name']
        display = data['display']
        
        symbol_lower = symbol.lower()
        name_lower = name.lower()
        display_lower = display.lower()
        
        # Exact matches
        if (query_lower == symbol_lower or 
            query_lower == name_lower or 
            query_lower == display_lower):
            exact_matches.append((symbol, 1.0))
            continue
        
        # Fuzzy matches
        symbol_ratio = SequenceMatcher(None, query_lower, symbol_lower).ratio()
        name_ratio = SequenceMatcher(None, query_lower, name_lower).ratio()
        display_ratio = SequenceMatcher(None, query_lower, display_lower).ratio()
        
        max_fuzzy_ratio = max(symbol_ratio, name_ratio, display_ratio)
        if max_fuzzy_ratio >= 0.85:
            fuzzy_matches.append((symbol, max_fuzzy_ratio))
            continue
        
        # Word-based matching
        query_words = set(query_lower.split())
        name_words = set(name_lower.split())
        display_words = set(display_lower.split())
        
        if query_words and (query_words.issubset(name_words) or query_words.issubset(display_words)):
            name_overlap = len(query_words.intersection(name_words)) / len(query_words)
            display_overlap = len(query_words.intersection(display_words)) / len(query_words)
            word_score = max(name_overlap, display_overlap)
            fuzzy_matches.append((symbol, word_score))
            continue
        
        # Partial matches
        if (partial_pattern.search(symbol_lower) or 
            partial_pattern.search(name_lower) or 
            partial_pattern.search(display_lower)):
            score = len(query) / max(len(symbol), len(name), len(display), 1)
            partial_matches.append((symbol, score))
    
    # Return best match
    if exact_matches:
        return exact_matches[0][0]
    
    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        return fuzzy_matches[0][0]
    
    if partial_matches:
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return partial_matches[0][0]
    
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
