from mcp.server.fastmcp import FastMCP
from data.database import DatabaseQueries
from market_tools import market 
from market_tools.technical_tools import closing_bollinger_bands, closing_macd, closing_rsi, closing_sma_and_slope, closing_ema_and_slope, analyze_volume_patterns, calculate_relative_strength, detect_breakout_patterns, calculate_support_resistance_levels, momentum_indicators


mcp = FastMCP("technical_server")


symbol_index = {}
name_index = {}
display_index = {}
try:    
    rows = DatabaseQueries.get_scripts_name_symbols()
    print(f"Loaded {len(rows)} scripts from DatabaseQueries.")
    
    for row in rows:
        symbol_index[row["UNDERLYING_SYMBOL"].lower()] = row["UNDERLYING_SYMBOL"]
        name_index[row["SYMBOL_NAME"].lower()] = row["UNDERLYING_SYMBOL"]
        display_index[row["DISPLAY_NAME"].lower()] = row["UNDERLYING_SYMBOL"]

except Exception as e:
    print(f"Error during DB/index loading: {e}")
    raise



@mcp.tool()
async def check_is_market_open() -> bool:
    """
    Check if the Indian stock market is currently open for trading.
    Returns:
        True if the market is open, False otherwise.
    """
    return market.is_market_open()


@mcp.tool()
async def get_symbol_price(symbol: str) -> float:
    """
    Get the current market price for a given stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        The latest price as a float.
    """
    return market.get_symbol_price_impl(symbol)


@mcp.tool()
async def get_share_ohlc(symbol: str) -> dict:
    """
    Retrieve the OHLC (Open, High, Low, Close) data for a given stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with OHLC data for the latest available day.
    """
    return market.get_symbol_ohlc(symbol)


@mcp.tool()
async def get_share_history_daily_data(
    symbol: str, from_date: str, to_date: str
) -> dict:
    """
    Retrieve daily historical OHLCV data for a given stock symbol between two dates.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
        from_date: Start date in 'YYYY-MM-DD' format.
        to_date: End date in 'YYYY-MM-DD' format.
    Returns:
        A dictionary with daily historical data for the symbol.
    """
    return market.get_symbol_history_daily_data(symbol, from_date, to_date)


# @mcp.tool()
# async def get_shares_historical_data(
#     symbols: list[str]
# ) -> dict:
#     """
#     Fetch and cache 1-year daily historical data for a list of stock symbols in bulk.
#     This tool stores the results in Redis for 1 hour.
#     Args:
#         symbols: A list of valid trading symbols (must be resolved first).
#     Returns:
#         "success" if all data fetched and stored.
#     """
#     try:
#         fetch_bulk_historical_data(symbols)
#         return "success"
#     except Exception as e:
#         return f"error: {str(e)}"


# @mcp.tool()
# async def get_closing_sma_and_slope(
#     symbol: str) -> dict:
#     """
#     Calculate the Simple Moving Average (SMA) and its slope for the closing prices of a stock symbol.
#     Args:
#         symbol: The trading symbol of the stock (must be resolved first).
#     Returns:
#         The SMA value and its slope for the given date range.
#     """
#     return closing_sma_and_slope(symbol)


@mcp.tool()
async def get_closing_ema_and_slope(
    symbol: str) -> dict:
    """
    Calculate the Exponential Moving Average (EMA) and its slope for the closing prices of a stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        The EMA value and its slope for the given date range.
    """
    return closing_ema_and_slope(symbol)


@mcp.tool()
async def get_closing_macd(
    symbol: str) -> dict:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the closing prices of a stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with MACD line, signal line, histogram, and date offsets for the given date range.    
    """
    return closing_macd(symbol)


@mcp.tool()
async def get_closing_rsi(
    symbol: str) -> float:
    """
    Calculate the Relative Strength Index (RSI) for the closing prices of a stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        The RSI value for the given date range.    
    """
    return closing_rsi(symbol)


@mcp.tool()
async def get_closing_bollinger_bands(
    symbol: str) -> dict:
    """
    Calculate the Bollinger Bands for the closing prices of a stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with upper, lower, and middle bands for the given date range.
    """
    return closing_bollinger_bands(symbol)

# Very Advanced Market Tools

@mcp.tool()
async def get_analyze_volume_patterns(
    symbol: str) -> dict:
    """
    Analyze volume patterns for a given stock symbol over a year data and get 1-month trends.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with volume trends and significant changes of last one month.
    """
    return analyze_volume_patterns(symbol)


# @mcp.tool()
# async def get_relative_strength(
#     symbol: str) -> dict:
#     """
#     Calculate the Relative Strength (RS) of a stock compared to a NIFTY index.
#     Args:
#         symbol: The trading symbol of the stock (must be resolved first).
#     Returns:
#         A dictionary with RS values and their trends.
#     """
#     return calculate_relative_strength(symbol)


@mcp.tool()
async def get_detect_breakout_patterns(
    symbol: str) -> dict:
    """
    Detect breakout patterns for a given stock symbol over a last year data.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with breakout levels and their trends.
    """
    return detect_breakout_patterns(symbol)


@mcp.tool()
async def get_support_resistance_levels(
    symbol: str) -> dict:
    """
    Calculate support and resistance levels for a given stock symbol over a last year data.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with support and resistance levels.
    """
    return calculate_support_resistance_levels(symbol)


@mcp.tool()
async def get_momentum_indicators(
    symbol: str) -> dict:
    """
    Calculate various momentum indicators for a given stock symbol.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    Returns:
        A dictionary with momentum indicator values.
    """
    return momentum_indicators(symbol)


if __name__ == "__main__":
    mcp.run(transport='stdio')