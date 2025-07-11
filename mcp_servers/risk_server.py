from mcp.server.fastmcp import FastMCP
from risk_management.risk_management_tools import t_calculate_position_size, t_set_stop_loss_order, t_check_stop_loss_triggers, t_calculate_portfolio_var, t_validate_trade_risk, t_check_portfolio_exposure, t_get_risk_metrics

mcp = FastMCP("risk_server")


@mcp.tool()
async def m_set_stop_loss_order(symbol: str, stop_price: float, quantity: int, agent_name: str):
    """
    Set a stop loss order for a specific stock symbol in the agent's portfolio.
    Args:
        symbol: The trading symbol for which to set the stop loss.
        stop_price: The stop loss price to trigger a sell.
        quantity: The number of shares/contracts to protect.
        agent_name: The name of the agent placing the stop loss order.
    Returns:
        Confirmation of stop loss order placement.
    """
    return await t_set_stop_loss_order(symbol, stop_price, quantity, agent_name)


@mcp.tool()
async def m_check_stop_loss_triggers(agent_name: str):
    """
    Check if any stop loss orders have been triggered for the agent's current positions.
    Args:
        agent_name: The name of the agent.
    Returns:
        A list or dictionary of triggered stop loss events, if any.
    """
    return await t_check_stop_loss_triggers(agent_name)


@mcp.tool()
async def m_calculate_portfolio_var(agent_name: str):
    """
    Calculate the Value at Risk (VaR) for the agent's portfolio using historical or parametric methods.
    Args:
        agent_name: The name of the agent.
    Returns:
        The VaR value and supporting details as a dictionary.
    """
    return await t_calculate_portfolio_var(agent_name)


@mcp.tool()
async def m_get_risk_metrics(agent_name: str):
    """
    Retrieve a summary of key risk metrics for the agent's portfolio, such as exposure, drawdown, and risk ratios.
    Args:
        agent_name: The name of the agent.
    Returns:
        A dictionary of risk metrics and their values.
    """
    return await t_get_risk_metrics(agent_name)


@mcp.tool()
async def m_validate_trade_risk(symbol: str, quantity: int, entry_price: float, stop_loss: float, agent_name: str):
    """
    Validate whether a proposed trade meets all risk management rules (e.g., position size, stop loss, exposure).
    Args:
        symbol: The trading symbol to be traded.
        quantity: The number of shares/contracts to buy or sell.
        entry_price: The intended entry price for the trade.
    Returns:
        A dictionary indicating if the trade is valid and any violations or warnings.
    """
    return await t_validate_trade_risk(symbol, quantity, entry_price, stop_loss, agent_name)


@mcp.tool()
async def m_calculate_position_size(entry_price: float, stop_loss: float, portfolio_value: float, agent_name: str = None):
    """
    Calculate the optimal position size for a trade based on entry price, stop loss, and total portfolio value.
    Args:
        entry_price: The intended entry price for the trade.
        stop_loss: The stop loss price for the trade.
        portfolio_value: The total value of the agent's portfolio.
    Returns:
        The recommended position size (number of shares/contracts).
    """
    return await t_calculate_position_size(entry_price, stop_loss, portfolio_value, agent_name=agent_name)



@mcp.tool()
async def m_check_portfolio_exposure(agent_name: str):
    """
    Check the current portfolio exposure for a given agent, including sector and position concentration.
    Args:
        agent_name: The name of the agent.
    Returns:
        A dictionary with exposure metrics and any warnings if limits are exceeded.
    """
    return await t_check_portfolio_exposure(agent_name)


if __name__ == "__main__":
    mcp.run(transport='stdio')