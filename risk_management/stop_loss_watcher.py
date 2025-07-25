import time
from risk_management.trade_execution import execute_stop_loss
from data.accounts import Account
from market_tools.market import get_prices_with_cache
from market_tools.market import is_market_open
from memory.agent_memory import AgentMemory
from trade_agents.trading_orchestrator import trader_names
from utils.util import bcolors
all_active_positions = {}


def stop_loss_watcher(stop_loss_manager, position_managers, poll_interval=300):
    while True:
        if not is_market_open():
            time.sleep(poll_interval)
            continue

        # 1. Get all active stop-losses
        get_all_active_positions = {
            trader_name: get_active_positions(trader_name)
            for trader_name in trader_names        
        }

        # Robust symbol extraction
        symbols = set()
        for positions in get_all_active_positions.values():
            if isinstance(positions, dict):
                for symbol in positions.keys():
                    symbols.add(symbol)
            elif isinstance(positions, list):
                for pos in positions:
                    if isinstance(pos, dict) and "symbol" in pos:
                        symbols.add(pos["symbol"])
        symbols = list(symbols)

        if not symbols:
            time.sleep(poll_interval)
            continue

        # 2. Fetch current prices for all symbols
        current_prices = get_prices_with_cache(symbols)
    
        # 3. Check for triggered stops
        triggered = stop_loss_manager.check_stop_losses(current_prices)

        print(f"{bcolors.WARNING}Triggered stop losses: {triggered}{bcolors.ENDC}")        

        for stop in triggered:
            symbol = stop["symbol"]
            trader_name = stop["trader_name"]
            price = stop["current_price"]
            
            account = Account.get(trader_name)
            position_manager = position_managers.get(trader_name)
            
            if not position_manager:
                print(f"No position manager found for agent {trader_name}, skipping stop loss execution for {symbol}")
                continue
            
            result = execute_stop_loss(
                trader_name=trader_name,
                account=account,
                position_manager=position_manager,
                symbol=symbol,
                quantity=stop["quantity"],
                execution_price=price,
                rationale="Stop loss triggered"
            )

            stop_loss_manager.execute_stop_loss(symbol, trader_name, reason="Triggered and executed")
            print(f"Stop loss executed for {trader_name} {symbol} at {price}: {result}")

        time.sleep(poll_interval)


def get_active_positions(trader_name):
    """
    Fetch active positions for a given agent from memory.
    """
    agent_memory = AgentMemory(trader_name)
    return agent_memory.get_active_positions()


# Usage (in your main app, or as a separate process/thread):
# from redis import Redis
# stop_loss_manager = StopLossManager(Redis(...))
# position_manager = PositionManager(...)  # per agent or global
# stop_loss_watcher(stop_loss_manager, position_manager)