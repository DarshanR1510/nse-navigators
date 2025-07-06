from risk_management.stop_loss_manager import StopLossManager
from risk_management.position_manager import PositionManager
from utils.redis_client import main_redis_client
import numpy as np
from market_tools.market import get_prices_with_cache
from memory.agent_memory import AgentMemory


async def t_set_stop_loss_order(symbol: str, stop_price: float, quantity: int, agent_name: str):
    """
    Tool: t_set_stop_loss_order
    -----------------------------------
    Sets or updates a stop loss order for the given symbol and agent. Persists the stop loss in Redis for monitoring and execution.
    Args:
        symbol: Stock symbol
        stop_price: Stop loss price
        quantity: Number of shares
        agent_name: Name of the agent
    Returns: dict with status and details.
    """
    slm = StopLossManager(main_redis_client)
    slm.set_stop_loss(symbol, stop_price, quantity, agent_name)
    return {"status": "success", "symbol": symbol, "stop_price": stop_price, "quantity": quantity, "agent_name": agent_name}


async def t_check_stop_loss_triggers(agent_name: str):
    """
    Tool: t_check_stop_loss_triggers
    -----------------------------------
    Checks all active positions for the agent and returns a list of those where the stop loss has been triggered (i.e., price has crossed the stop).
    Args:
        agent_name: Name of the agent
    Returns: list of dicts with triggered stop loss details.
    """
    
    pm = PositionManager(agent_name=agent_name)
    symbols = list(pm.positions.keys())
    if not symbols:
        return []
    # Get current prices for all symbols
    current_prices = get_prices_with_cache(symbols)
    return pm.check_stop_loss_triggers(current_prices)


async def t_calculate_portfolio_var(agent_name: str, confidence: float = 0.95):
    """
    Tool: t_calculate_portfolio_var
    -----------------------------------
    Calculates the Value at Risk (VaR) for the agent's portfolio at the given confidence level using historical method.
    Args:
        agent_name: Name of the agent
        confidence: Confidence level (default 0.95)
    Returns: dict with VaR value and details.
    """
    
    mem = AgentMemory(agent_name)
    trades = mem.get_recent_trades(days=60)
    pnl_list = [t.get("realized_pnl", 0) for t in trades if t.get("type") in ("sell", "stop_loss")]
    
    if not pnl_list:
        return {"status": "error", "message": "No historical P&L data available for VaR calculation."}
    
    pnl_array = np.array(pnl_list)
    var = -np.percentile(pnl_array, (1 - confidence) * 100)
    
    return {"VaR": float(round(var, 2)), "confidence": confidence, "period_days": 60, "pnl_samples": len(pnl_list)}


async def t_get_risk_metrics(agent_name: str):
    """
    Tool: t_get_risk_metrics
    -----------------------------------
    Returns a summary of key risk metrics for the agent's portfolio, including portfolio risk, daily loss, VaR, and position limits.
    Args:
        agent_name: Name of the agent
    Returns: dict with risk metrics and status.
    """
    pm = PositionManager(agent_name=agent_name)
    exposure = pm.check_portfolio_limits()
    var_result = await t_calculate_portfolio_var(agent_name)
    return {"exposure": exposure, "VaR": var_result}



async def t_validate_trade_risk(symbol: str, quantity: int, entry_price: float, stop_loss: float, agent_name: str):
    """
    Tool: t_validate_trade_risk
    -----------------------------------
    Validates if a new trade (given symbol, quantity, entry, stop loss, agent_name) meets all risk management rules for the agent.
    Checks position limits, risk per trade, portfolio risk, daily loss, and other constraints.

    Returns: dict with 'is_valid' (bool) and 'reason' (str) explaining the result.
    """
    pm = PositionManager(agent_name=agent_name)
    is_valid, reason = pm.validate_new_position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        agent_name=agent_name
    )
    return {"is_valid": is_valid, "reason": reason}


async def t_calculate_position_size(entry_price: float, stop_loss: float, portfolio_value: float, agent_name: str) -> int:        
    """
    Tool: t_calculate_position_size
    -----------------------------------
    Calculates the optimal position size (number of shares) for a trade based on entry price, stop loss, portfolio value, and agent's risk settings.
    Returns: int (number of shares to buy)
    """
    pm = PositionManager(agent_name=agent_name)
    return pm.calculate_position_size(entry_price, stop_loss, portfolio_value)


async def t_check_portfolio_exposure(agent_name: str):
    """
    Tool: t_check_portfolio_exposure
    -----------------------------------
    Returns a summary of the agent's current portfolio exposure and risk limits.
    Includes number of positions, cash used, portfolio risk, daily P&L, and limit status.
    Returns: dict with exposure and risk status fields.
    """
    pm = PositionManager(agent_name=agent_name)
    return pm.check_portfolio_limits()