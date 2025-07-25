import logging
from mcp.server.fastmcp import FastMCP
from risk_management.risk_management_tools import t_calculate_position_size

mcp = FastMCP("decision_server")

logger = logging.getLogger(__name__)

@mcp.tool()
async def m_calculate_position_size(entry_price: float, stop_loss: float, portfolio_value: float, trader_name: str):
    """
    Calculate the optimal position size for a trade based on entry price, stop loss, and total portfolio value.
    Args:
        entry_price: The intended entry price for the trade.
        stop_loss: The stop loss price for the trade.
        portfolio_value: The total value of the agent's portfolio.
        trader_name: Optional name of the agent for logging purposes.
    Returns:
        The recommended position size (number of shares/contracts).
    """
    return await t_calculate_position_size(entry_price, stop_loss, portfolio_value, trader_name)

if __name__ == "__main__":
    mcp.run(transport='stdio')