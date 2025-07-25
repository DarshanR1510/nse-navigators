from mcp.server.fastmcp import FastMCP
from memory.memory_tools import (
    get_overall_market_context,    
    get_positions,
    remove_position,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    log_trade_execution,
    get_performance_summary,
    get_agent_memory_status
)

mcp = FastMCP("memory_server")



#* DO NOT CHANGE IT'S ORDER (should be at 1st position) - being used in ResearcherAgent class.
@mcp.tool()
async def m_get_market_context():
    """
    Retrieve the today's market context.
    Returns:
        The market context dictionary for the specified agent and date.
    """
    return await get_overall_market_context()



@mcp.tool()
async def m_get_today_market_context():
    """
    Retrieve the latest overall market context, including market regime, sector performance, and volatility regime.
    This tool is useful for agents to make informed trading decisions based on the most up-to-date market conditions.
    Returns:
        A dictionary with the latest market context for all agents.
    """
    return get_overall_market_context()


@mcp.tool()
async def m_get_positions(trader_name: str):
    """
    Retrieve all active positions for a specific agent.
    Args:
        trader_name: The name of the agent.
    Returns:
        A dictionary of all current positions for the agent, keyed by symbol.
    """
    return await get_positions(trader_name)


@mcp.tool()
async def m_remove_position(trader_name: str, symbol: str):
    """
    Remove a specific position for an agent.
    Args:
        trader_name: The name of the agent.
        symbol: The stock symbol to remove from the agent's positions.
    Returns:
        Confirmation of removal.
    """
    return await remove_position(trader_name, symbol)


@mcp.tool()
async def m_add_to_watchlist(trader_name: str, symbol: str, details: str):
    """
    Add a stock to the agent's watchlist with optional details.
    Args:
        trader_name: The name of the agent.
        symbol: The stock symbol to add.
        details: Additional details or metadata about the symbol.
    Returns:
        Confirmation of addition.
    """
    return await add_to_watchlist(trader_name, symbol, details)


@mcp.tool()
async def m_remove_from_watchlist(trader_name: str, symbol: str):
    """
    Remove a stock from the agent's watchlist.
    Args:
        trader_name: The name of the agent.
        symbol: The stock symbol to remove.
    Returns:
        Confirmation of removal.
    """
    return await remove_from_watchlist(trader_name, symbol)


@mcp.tool()
async def m_get_watchlist(trader_name: str):
    """
    Retrieve the current watchlist for a specific agent.
    Args:
        trader_name: The name of the agent.
    Returns:
        A list or dictionary of stocks currently on the agent's watchlist.
    """
    return await get_watchlist(trader_name)


@mcp.tool()
async def m_log_trade_execution(trader_name: str, trade_details: dict):
    """
    Log the execution of a trade for a specific agent.
    Args:
        trader_name: The name of the agent.
        trade_details: A dictionary containing details of the executed trade (symbol, action, quantity, price, etc.).
    Returns:
        Confirmation of logging.
    """
    return await log_trade_execution(trader_name, trade_details)


@mcp.tool()
async def m_get_performance_summary(trader_name: str, days: int = 30):
    """
    Retrieve a performance summary for a specific agent over a given number of days.
    Args:
        trader_name: The name of the agent.
        days: The number of days to summarize (default: 30).
    Returns:
        A summary of the agent's trading performance, including P&L, win rate, and other metrics.
    """
    return await get_performance_summary(trader_name, days)


@mcp.tool()
async def m_get_agent_memory_status(trader_name: str):
    """
    Retrieve a summary of the agent's memory status, including stored positions, watchlist, and context.
    Args:
        trader_name: The name of the agent.
    Returns:
        A dictionary summarizing the agent's memory usage and contents.
    """
    return await get_agent_memory_status(trader_name)

if __name__ == "__main__":
    mcp.run(transport='stdio')
