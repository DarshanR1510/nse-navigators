"""
trade_execution.py - Unified Trade Execution Orchestration

This module coordinates trade execution between Account (cash/holdings/transactions) and PositionManager (risk/positions),
ensuring all business rules, risk checks, and state updates are enforced in a single place.

Usage:
    - All agent buy/sell/close actions should go through this module.
    - This module should be imported and used by agent workflows (e.g., in traders.py).
    - PositionManager and Account should not directly execute trades; they only provide state and validation.

"""
from typing import Optional, Tuple
from data.accounts import Account
from risk_management.position_manager import PositionManager

class TradeExecutionError(Exception):
    pass

def execute_buy(
    agent_name: str,
    account: Account,
    position_manager: PositionManager,
    symbol: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    quantity: Optional[int] = None,
    rationale: str = "",
    position_type: str = "LONG",
    sector: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Orchestrate a buy trade: risk validation, account update, position creation.
    """
    # 1. Calculate position size if not provided
    if quantity is None:
        quantity = position_manager.calculate_position_size(entry_price, stop_loss, account.balance)
        if quantity <= 0:
            return False, f"Position size calculation failed for {symbol}."

    # 2. Validate position with PositionManager
    is_valid, reason = position_manager.validate_new_position(
        symbol, quantity, entry_price, stop_loss, agent_name, sector
    )
    if not is_valid:
        return False, f"Risk validation failed: {reason}"

    # 3. Check account funds
    total_cost = entry_price * quantity
    if total_cost > account.balance:
        return False, f"Insufficient funds: Need {total_cost}, have {account.balance}"

    # 4. Update Account (buy shares)
    try:
        account.buy_shares(symbol, quantity, rationale)
    except Exception as e:
        return False, f"Account buy failed: {e}"

    # 5. Add position to PositionManager
    success, msg = position_manager.add_position(
        symbol, quantity, entry_price, stop_loss, target, agent_name, rationale, position_type, sector
    )
    
    if not success:
        # Rollback account buy if needed (not implemented here)
        return False, f"PositionManager add failed: {msg}"

    return True, f"Buy executed: {quantity} {symbol} @ {entry_price} (stop: {stop_loss}, target: {target})"

def execute_sell(
    agent_name: str,
    account: Account,
    position_manager: PositionManager,
    symbol: str,
    quantity: int,
    rationale: str = "Sell order",
    execution_price: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Orchestrate a sell trade: update account, close/reduce position in PositionManager.
    """
    # 1. Check holdings
    if account.holdings.get(symbol, 0) < quantity:
        return False, f"Not enough shares to sell: {symbol}"

    # 2. Update Account (sell shares)
    try:
        account.sell_shares(symbol, quantity, rationale)
    except Exception as e:
        return False, f"Account sell failed: {e}"

    # 3. Update PositionManager (close/reduce position)
    # For now, we assume full close if quantity matches position, else partial
    position = position_manager.positions.get(symbol)
    if position:
        # If full close
        if quantity >= position.quantity:
            # Mark as closed
            position.status = "CLOSED"
            position_manager._save_closed_position(position)
            position_manager._remove_position(symbol)
            position_manager._remove_position_from_memory_file(symbol)
        else:
            # Reduce position size
            position.quantity -= quantity
            position_manager._save_positions()
    return True, f"Sell executed: {quantity} {symbol}"

def execute_stop_loss(
    agent_name: str,
    account: Account,
    position_manager: PositionManager,
    symbol: str,
    execution_price: float,
    rationale: str = "Stop loss triggered"
) -> Tuple[bool, str]:
    """
    Orchestrate a stop loss execution: sell shares, update position status.
    """
    # 1. Get position
    position = position_manager.positions.get(symbol)
    if not position:
        return False, f"No active position for {symbol}"
    quantity = position.quantity
    # 2. Sell shares in account
    try:
        account.sell_shares(symbol, quantity, rationale)
    except Exception as e:
        return False, f"Account sell failed: {e}"
    # 3. Mark position as stopped out
    position_manager.execute_stop_loss(symbol, execution_price)
    return True, f"Stop loss executed for {symbol} at {execution_price}"
