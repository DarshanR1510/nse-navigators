from mcp.server.fastmcp import FastMCP
from market_tools.fundamental_tools import get_company_financials
from market_tools import market
mcp = FastMCP("fundamental_server")

@mcp.tool()
async def get_financial_data(symbol: str) -> dict:
    """
    Retrieve comprehensive financial data for a given stock symbol, including:
      - Symbol, company name, and current price
      - Market capitalization and key financial ratios
      - Key financial metrics like P/E ratio, EPS, and dividend yield
      - Structured financial statements (balance sheet, income statement, cash flow statement)
      - Shareholding pattern for the latest year
    The tool automatically determines the latest fiscal year based on available data.
    If no data is available, it returns an empty dictionary.
    Args:
        symbol: The trading symbol of the stock (must be resolved first).
    """
    return get_company_financials(symbol)


@mcp.tool()
async def resolve_symbol(company_query: str) -> str:
    """
    Resolve a company name, display name, or symbol to its official trading symbol using the database and cache.
    Args:
        company_query: The company name, display name, or symbol to resolve.
    Returns:
        The resolved trading symbol as a string.
    """
    return market.resolve_symbol_impl(company_query)

if __name__ == "__main__":
    mcp.run(transport='stdio')