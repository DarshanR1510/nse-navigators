import os
from dhanhq import dhanhq
from dhanhq import marketfeed
from dotenv import load_dotenv

load_dotenv(override=True)

client_id = os.getenv("DHAN_CLIENT_ID")
access_token = os.getenv("DHAN_ACCESS_TOKEN")

dhan = dhanhq(client_id, access_token)

def dhan_feed(symbols: list[tuple[int, str]]) -> marketfeed.DhanFeed:
    """
    This function starts the Dhan WebSocket feed.
    """
    live_feed = marketfeed.DhanFeed(
        client_id,
        access_token,
        symbols,
        version='v2'
    )
    return live_feed