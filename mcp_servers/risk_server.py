from mcp.server.fastmcp import FastMCP
from risk_management.risk_management_tools import t_check_stop_loss_triggers, t_calculate_portfolio_var, t_validate_trade_risk, t_check_portfolio_exposure, t_get_risk_metrics

mcp = FastMCP("risk_server")


@mcp.tool()
async def m_check_stop_loss_triggers(trader_name: str):
    """
    Check if any stop loss orders have been triggered for the agent's current positions.
    Args:
        trader_name: The name of the agent.
    Returns:
        A list or dictionary of triggered stop loss events, if any.
    """
    return await t_check_stop_loss_triggers(trader_name)


@mcp.tool()
async def m_calculate_portfolio_var(trader_name: str):
    """
    Calculate the Value at Risk (VaR) for the agent's portfolio using historical or parametric methods.
    Args:
        trader_name: The name of the agent.
    Returns:
        The VaR value and supporting details as a dictionary.
    """
    return await t_calculate_portfolio_var(trader_name)


@mcp.tool()
async def m_get_risk_metrics(trader_name: str):
    """
    Retrieve a summary of key risk metrics for the agent's portfolio, such as exposure, drawdown, and risk ratios.
    Args:
        trader_name: The name of the agent.
    Returns:
        A dictionary of risk metrics and their values.
    """
    return await t_get_risk_metrics(trader_name)


@mcp.tool()
async def m_validate_trade_risk(symbol: str, quantity: int, entry_price: float, stop_loss: float, trader_name: str):
    """
    Validate whether a proposed trade meets all risk management rules (e.g., position size, stop loss, exposure).
    Args:
        symbol: The trading symbol to be traded.
        quantity: The number of shares/contracts to buy or sell.
        entry_price: The intended entry price for the trade.
    Returns:
        A dictionary indicating if the trade is valid and any violations or warnings.
    """
    return await t_validate_trade_risk(symbol, quantity, entry_price, stop_loss, trader_name)



@mcp.tool()
async def m_check_portfolio_exposure(trader_name: str):
    """
    Check the current portfolio exposure for a given agent, including sector and position concentration.
    Args:
        trader_name: The name of the agent.
    Returns:
        A dictionary with exposure metrics and any warnings if limits are exceeded.
    """
    return await t_check_portfolio_exposure(trader_name)


if __name__ == "__main__":
    mcp.run(transport='stdio')