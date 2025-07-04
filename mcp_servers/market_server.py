from mcp.server.fastmcp import FastMCP
from data.database import DatabaseQueries
from market_tools import market 
from market_tools.technical_tools import closing_sma, closing_ema, closing_macd, closing_rsi
from market_tools.fundamental_tools import get_latest_structured_financial

mcp = FastMCP("market_server")


symbol_index = {}
name_index = {}
display_index = {}
try:
    print("Database connection established.")
    rows = DatabaseQueries.get_scripts_name_symbols()
    print(f"Loaded {len(rows)} scripts from DatabaseQueries.")
    for row in rows:
        symbol_index[row["UNDERLYING_SYMBOL"].lower()] = row["UNDERLYING_SYMBOL"]
        name_index[row["SYMBOL_NAME"].lower()] = row["UNDERLYING_SYMBOL"]
        display_index[row["DISPLAY_NAME"].lower()] = row["UNDERLYING_SYMBOL"]
    print("Symbol indexes built.")
except Exception as e:
    print(f"Error during DB/index loading: {e}")
    raise


@mcp.tool()
async def check_is_market_open() -> bool:
    """This tool checks if the market is open for the given stock symbol."""
    return market.is_market_open()


@mcp.tool()
async def resolve_symbol(company_query: str) -> str:
    """This tool resolves a company name or symbol to its underlying symbol using Redis cache."""
    return market.resolve_symbol_impl(company_query)


@mcp.tool()
async def get_symbol_price(symbol: str) -> float:
    """This tool provides the current price of the given stock symbol.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name."""
    return market.get_symbol_price_impl(symbol)


@mcp.tool()
async def get_share_ohlc(symbol: str) -> dict:
    """This tool provides the OHLC (Open, High, Low, Close) data for the given stock symbol.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name."""
    return market.get_symbol_ohlc(symbol)


@mcp.tool()
async def get_share_history_daily_data(
    symbol: str, from_date: str, to_date: str
) -> dict:
    """This tool provides historical data for the given stock symbol between specified dates.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name."""
    return market.get_symbol_history_daily_data(symbol, from_date, to_date)


@mcp.tool()
async def get_share_history_intraday_data(
    symbol: str, interval: int, from_date: str, to_date: str
) -> dict:
    """This tool provides intraday historical data for the given stock symbol.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name."""
    return market.get_symbol_history_intraday_data(symbol, interval, from_date, to_date)


@mcp.tool()
async def get_closing_sma(
    symbol: str, from_date: str, to_date: str) -> float:
    """
    This tool is indicator and returns Simple Moving Average (SMA) of the closing prices for the given stock symbol.
    it gives SMA for given date range.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name.
    Use minimum of 80 working days data for SMA.
    """
    return closing_sma(symbol, from_date, to_date)


@mcp.tool()
async def get_closing_ema(
    symbol: str, from_date: str, to_date: str) -> float:
    """
    This tool is indicator and returns Exponential Moving Average (EMA) of the closing prices for the given stock symbol.
    it gives EMA for given date range.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name.
    Use minimum of 120 working days data for EMA.
    """
    return closing_ema(symbol, from_date, to_date)


@mcp.tool()
async def get_closing_macd(
    symbol: str, from_date: str, to_date: str) -> dict:
    """
    This tool is indicator and returns Moving Average Convergence Divergence (MACD) of the closing prices for the given stock symbol.
    it gives MACD_line, signal_line, histogram, and dates_offset for given date range.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name.
    Use minimum of 80 working days data for MACD.
    """
    return closing_macd(symbol, from_date, to_date)


@mcp.tool()
async def get_closing_rsi(
    symbol: str, from_date: str, to_date: str) -> float:
    """
    This tool is indicator and returns Relative Strength Index (RSI) of the closing prices for the given stock symbol.
    it gives RSI for given date range.
    This tool requires a valid symbol. Always use resolve_symbol first if you have a company name.
    Use minimum of 30 working days data for RSI.
    """
    return closing_rsi(symbol, from_date, to_date)


@mcp.tool()
async def get_financial_data(symbol: str) -> dict:
    """
    This tool retrieves financial data for the given stock symbol.
    It includes financial metrics such as symbol, company name, current price,
    market cap, financial ratios, and structured financial statements like balance sheet,
    income statement, cash flow statement, and shareholding pattern of the latest year.
    It also determines the latest fiscal year based on the available data.
    If no data is available, it returns an empty dictionary.
    """
    return get_latest_structured_financial(symbol)



if __name__ == "__main__":
    mcp.run(transport='stdio')