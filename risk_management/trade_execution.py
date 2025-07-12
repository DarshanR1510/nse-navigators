"""
trade_execution.py - Unified Trade Execution Orchestration

This module coordinates trade execution between Account (cash/holdings/transactions) and PositionManager (risk/positions),
ensuring all business rules, risk checks, and state updates are enforced in a single place.

Usage:
    - All agent buy/sell/close actions should go through this module.
    - This module should be imported and used by agent workflows (e.g., in traders.py or MCP servers).
    - PositionManager and Account should not directly execute trades; they only provide state and validation.

"""

from typing import Optional, Tuple
from data.accounts import Account, Transaction
from risk_management.position_manager import PositionManager
from datetime import datetime
from memory.agent_memory import AgentMemory
from market_tools.live_prices import update_instruments
from market_tools.market import get_security_id, get_symbol_price_impl
from data.database import DatabaseQueries

agent_names = ["warren", "george", "ray", "cathie"] 


def execute_buy(
    agent_name: str,
    account: Account,
    position_manager: PositionManager,
    symbol: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    quantity: int,
    rationale: str = "",
    position_type: str = "LONG",
    sector: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Orchestrate a buy trade: risk validation, account update, position creation.
    """

    try:
        entry_price = (get_symbol_price_impl(symbol) if float(entry_price) == 0.0 else float(entry_price))
        stop_loss = float(stop_loss)
        target = float(target)
        if quantity is not None:
            quantity = int(quantity)
        account.balance = float(account.balance)
    except Exception as e:
        return False, f"Type conversion error: {e}"    

    if position_manager is None:
        raise ValueError(f"PositionManager for agent '{agent_name}' is None!")
    
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

    # 4. Add position to PositionManager
    success, msg = position_manager.add_position(
        symbol, quantity, entry_price, stop_loss, target, agent_name, rationale, position_type, sector
    )

    # 5. Log trade in agent memory (optional)
    if success:
        try:
            # 1. Subtract cash from account
            total_cost = entry_price * quantity
            account.balance -= total_cost

            # 2. Add to holdings
            account.holdings[symbol] = account.holdings.get(symbol, 0) + quantity

            # 3. Add Transaction
            account.transactions.append(Transaction(
                symbol=symbol,
                quantity=quantity,
                price=entry_price,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                rationale=rationale
            ))

            # 4. Save account
            account.save()

            # 5. Update agent memory (active positions and trade log)
            agent_memory = AgentMemory(agent_name)
            agent_memory.store_active_positions({
                    symbol: {
                        "entry_price": entry_price,
                        "quantity": quantity,
                        "stop_loss": stop_loss,
                        "target": target,
                        "reason": rationale,
                        "entry_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                })
            trade_details = {
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "timestamp": datetime.now().timestamp(),
                "type": "buy",
                "rationale": rationale,
                "status": "executed"
            }
            agent_memory.log_trade(trade_details)
            DatabaseQueries.write_log(agent_name, "account", f"Bought {quantity} of {symbol}")

        except Exception as e:
            print(f"[WARN] Could not log buy trade to agent memory: {e}")

    # 6. Update live prices (if needed)for agent_name in agent_names:
    
    all_symbols = set()
    for agent_name in agent_names:
        account = Account.get(agent_name)
        all_symbols.update(account.get_holdings().keys())
    new_instruments = {symbol: get_security_id(symbol) for symbol in all_symbols}
    
    update_instruments(new_instruments)

    return True, f"Buy executed: {quantity} {symbol} @ {entry_price} (stop: {stop_loss}, target: {target})"



def execute_sell(
    agent_name: str,
    account: Account,
    position_manager: PositionManager,
    symbol: str,
    quantity: int,
    execution_price: float,
    rationale: str = "Sell order",
) -> Tuple[bool, str]:
    """
    Orchestrate a sell trade: update account, close/reduce position in PositionManager.
    """

    try:        
        if quantity is not None:
            quantity = int(quantity)
        account.balance = float(account.balance)
    except Exception as e:
        return False, f"Type conversion error: {e}"


    # 1. Check holdings
    if account.holdings.get(symbol, 0) < quantity:
        return False, f"Not enough shares to sell: {symbol}"

    # 2. Update PositionManager (close/reduce position)
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

    # 3. Log trade in agent memory
    try:
        # 1. Add cash to account
        sell_price = execution_price if execution_price is not None else 0.0
        total_proceeds = sell_price * quantity
        account.balance += total_proceeds

        # 2. Subtract from holdings
        account.holdings[symbol] = account.holdings.get(symbol, 0) - quantity
        if account.holdings[symbol] <= 0:
            del account.holdings[symbol]

        # 3. Update live prices (if needed)for agent_name in agent_names:
        all_symbols = set()
        for agent_name in agent_names:
            account = Account.get(agent_name)
            all_symbols.update(account.get_holdings().keys())
        new_instruments = {symbol: get_security_id(symbol) for symbol in all_symbols}
        
        update_instruments(new_instruments)

        # 4. Add Transaction
        account.transactions.append(Transaction(
            symbol=symbol,
            quantity=-quantity,
            price=sell_price,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rationale=rationale
        ))

        # 5. Save account
        account.save()

        # 6. Update agent memory (remove active positions and trade log)
        agent_memory = AgentMemory(agent_name)
        agent_memory.remove_active_position(symbol)
        trade_details = {
            "symbol": symbol,
            "quantity": quantity,
            "execution_price": execution_price,
            "timestamp": datetime.now().timestamp(),
            "type": "sell",
            "rationale": rationale,
            "status": "executed"
        }
        agent_memory.log_trade(trade_details)
        DatabaseQueries.write_log(agent_name, "account", f"Sold {quantity} of {symbol}")

        
    except Exception as e:
        print(f"[WARN] Could not log sell trade to agent memory: {e}")

    return True, f"Sell executed: {quantity} {symbol}"



def execute_stop_loss(
    agent_name: str,
    account: Account,
    position_manager: PositionManager,
    symbol: str,
    quantity: int,
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

    # 2. Mark position as stopped out
    position_manager.execute_stop_loss(symbol, execution_price)

    # 3. Log trade in agent memory (optional)
    try:
         # 1. Add cash to account
        total_proceeds = execution_price * quantity
        account.balance += total_proceeds

        # 2. Subtract from holdings
        account.holdings[symbol] = account.holdings.get(symbol, 0) - quantity
        if account.holdings[symbol] <= 0:
            del account.holdings[symbol]

        # 3. Update live prices (if needed)for agent_name in agent_names:
        all_symbols = set()
        for agent_name in agent_names:
            account = Account.get(agent_name)
            all_symbols.update(account.get_holdings().keys())
        new_instruments = {symbol: get_security_id(symbol) for symbol in all_symbols}
        
        update_instruments(new_instruments)

        # 4. Add Transaction
        account.transactions.append(Transaction(
            symbol=symbol,
            quantity=-quantity,
            price=execution_price,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rationale=rationale
        ))

        # 5. Save account
        account.save()

        # 6. Update agent memory (remove active position and log trade)
        agent_memory = AgentMemory(agent_name)
        agent_memory.remove_active_position(symbol)
        trade_details = {
            "symbol": symbol,
            "quantity": quantity,
            "execution_price": execution_price,
            "timestamp": datetime.now().timestamp(),
            "type": "stop_loss",
            "rationale": rationale,
            "status": "executed"
        }
        agent_memory.log_trade(trade_details)
    except Exception as e:
        print(f"[WARN] Could not log stop loss trade to agent memory: {e}")

    return True, f"Stop loss executed for {symbol}"