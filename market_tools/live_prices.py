import threading
from dhanhq import marketfeed
from utils.dhan_client import dhan_feed
import asyncio
import time
from market_tools.market import is_market_open
import pandas as pd


INSTRUMENTS = {}
live_prices = {}
listener_thread = None
data_lock = threading.Lock()


def update_instruments(new_instruments):
    global INSTRUMENTS
    INSTRUMENTS = new_instruments
    
    # Update live_prices in-place
    with data_lock:        
        for symbol in list(live_prices.keys()):
            if symbol not in new_instruments:
                del live_prices[symbol]

        for symbol in new_instruments:
            if symbol not in live_prices:
                live_prices[symbol] = {"LTP": 0.0}


def run_websocket_listener():
    print("Starting WebSocket listener...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Prepare instruments in the correct format (exchange_segment, security_id, subscription_type)
    instruments = [
        (marketfeed.NSE, str(security_id))
        for security_id in INSTRUMENTS.values()
    ]

    feed = dhan_feed(instruments)

    while True:
        feed.run_forever()
        response = feed.get_data()  
        # response example: {'type': 'Ticker Data', 'exchange_segment': 1, 'security_id': 1333, 'LTP': '1985.40', 'LTT': '13:02:12'}        
        SECID_TO_SYMBOL = {str(v): k for k, v in INSTRUMENTS.items()}

        if is_market_open():
            # Only process data if the market is open
            if response and isinstance(response, dict) and response.get("type") == "Ticker Data":
                security_id = str(response.get("security_id"))
                ltp = float(response.get("LTP", 0.0))

            symbol = SECID_TO_SYMBOL.get(security_id)
            
            if symbol:
                with data_lock:
                    prev_ltp = live_prices[symbol].get("LTP", 0.0)
                    live_prices[symbol]["prev_LTP"] = prev_ltp
                    live_prices[symbol]["LTP"] = ltp   
            time.sleep(1)  # Poll every second
        else:
            print("Market is closed. Waiting for market to open...")
            time.sleep(600)


def get_live_price_df():
    """
    This function retrieves the latest prices from the global dictionary
    and formats them into a Pandas DataFrame for Gradio to display.
    """
    with data_lock:
        # Create a list of dictionaries from our live_prices data
        data_for_df = [
            {"Symbol": symbol, "Live Price (LTP)": details["LTP"]}
            for symbol, details in live_prices.items()
        ]
    
    # Create and return the DataFrame
    df = pd.DataFrame(data_for_df)
    return df
