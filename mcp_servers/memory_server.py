from mcp.server.fastmcp import FastMCP
from memory.memory_tools import (
    store_market_context,
    get_market_context,
    add_positions,
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

@mcp.tool()
async def m_store_market_context(agent_name: str, context: dict):
    """Store daily market context for the agent."""
    return await store_market_context(agent_name, context)

@mcp.tool()
async def m_get_market_context(agent_name: str, date: str = None):
    """Get daily market context for the agent."""
    return await get_market_context(agent_name, date)

@mcp.tool()
async def m_add_positions(agent_name: str, positions: dict):
    """Add or update active positions for the agent."""
    return await add_positions(agent_name, positions)

@mcp.tool()
async def m_get_positions(agent_name: str):
    """Get all active positions for the agent."""
    return await get_positions(agent_name)

@mcp.tool()
async def m_remove_position(agent_name: str, symbol: str):
    """Remove a position for the agent."""
    return await remove_position(agent_name, symbol)

@mcp.tool()
async def m_add_to_watchlist(agent_name: str, stock: str, details: dict):
    """Add a stock to the agent's watchlist."""
    return await add_to_watchlist(agent_name, stock, details)

@mcp.tool()
async def m_remove_from_watchlist(agent_name: str, stock: str):
    """Remove a stock from the agent's watchlist."""
    return await remove_from_watchlist(agent_name, stock)

@mcp.tool()
async def m_get_watchlist(agent_name: str):
    """Get the agent's watchlist."""
    return await get_watchlist(agent_name)

@mcp.tool()
async def m_log_trade_execution(agent_name: str, trade_details: dict):
    """Log a trade execution for the agent."""
    return await log_trade_execution(agent_name, trade_details)

@mcp.tool()
async def m_get_performance_summary(agent_name: str, days: int = 30):
    """Get a performance summary for the agent."""
    return await get_performance_summary(agent_name, days)

@mcp.tool()
async def m_get_agent_memory_status(agent_name: str):
    """Get a summary of the agent's memory status."""
    return await get_agent_memory_status(agent_name)

if __name__ == "__main__":
    mcp.run(transport='stdio')
