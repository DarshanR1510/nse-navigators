from mcp.server.fastmcp import FastMCP
from risk_management.risk_management_tools import t_calculate_position_size, t_set_stop_loss_order, t_check_stop_loss_triggers, t_calculate_portfolio_var, t_validate_trade_risk, t_check_portfolio_exposure, t_get_risk_metrics

mcp = FastMCP("risk_management_server")

@mcp.tool()
async def m_set_stop_loss_order(symbol: str, stop_price: float, 
                             quantity: int, agent_name: str):
    """Set a stop loss order for a given symbol."""
    return await t_set_stop_loss_order(symbol, stop_price, quantity, agent_name)


@mcp.tool()
async def m_check_stop_loss_triggers(agent_name: str):
    """Check if any stop loss orders have been triggered for the agent's positions."""
    return await t_check_stop_loss_triggers(agent_name)


@mcp.tool()
async def m_calculate_portfolio_var(agent_name: str):
    """Calculate the Value at Risk (VaR) for the agent's portfolio."""
    return await t_calculate_portfolio_var(agent_name)


@mcp.tool()
async def m_get_risk_metrics(agent_name: str):
    """Get a summary of key risk metrics for the agent's portfolio."""
    return await t_get_risk_metrics(agent_name)


@mcp.tool()
async def m_validate_trade_risk(symbol: str, quantity: int, entry_price: float):
    """Validate if a new trade meets all risk management rules."""
    return await t_validate_trade_risk(symbol, quantity, entry_price)


@mcp.tool()
async def m_calculate_position_size(entry_price: float, stop_loss: float, 
                                 portfolio_value: float):
    """Calculate position size based on entry price, stop loss, and portfolio value."""
    return await t_calculate_position_size(entry_price, stop_loss, portfolio_value)


@mcp.tool()
async def m_check_portfolio_exposure(agent_name: str):
    """Check the portfolio exposure for a given agent."""
    return await t_check_portfolio_exposure(agent_name)

