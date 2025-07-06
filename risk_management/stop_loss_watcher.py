import time
from risk_management.trade_execution import execute_stop_loss
from data.accounts import Account
from market_tools.market import get_prices_with_cache
from market_tools.market import is_market_open
from memory.agent_memory import AgentMemory

agent_names = ["Warren", "George", "Ray", "Cathie"]
all_active_positions = {}


def stop_loss_watcher(stop_loss_manager, position_managers, poll_interval=300):
    while True:
        if not is_market_open():
            time.sleep(poll_interval)
            continue

        # 1. Get all active stop-losses
        get_all_active_positions = {
            agent_name: get_active_positions(agent_name)
            for agent_name in agent_names
        
        }
        symbols = list({s for (_, s) in get_all_active_positions.keys()})
        if not symbols:
            time.sleep(poll_interval)
            continue

        # 2. Fetch current prices for all symbols
        current_prices = get_prices_with_cache(symbols)
    
        # 3. Check for triggered stops
        triggered = stop_loss_manager.check_stop_losses(current_prices)
        
        for stop in triggered:
            symbol = stop["symbol"]
            agent_name = stop["agent_name"]
            price = stop["current_price"]
            
            account = Account.get(agent_name)
            position_manager = position_managers.get(agent_name)
            
            if not position_manager:
                print(f"No position manager found for agent {agent_name}, skipping stop loss execution for {symbol}")
                continue
            
            result = execute_stop_loss(
                agent_name=agent_name,
                account=account,
                position_manager=position_manager,
                symbol=symbol,
                execution_price=price,
                rationale="Stop loss triggered"
            )

            stop_loss_manager.execute_stop_loss(symbol, agent_name, reason="Triggered and executed")
            print(f"Stop loss executed for {agent_name} {symbol} at {price}: {result}")

        time.sleep(poll_interval)



def get_active_positions(agent_name):
    """
    Fetch active positions for a given agent from memory.
    """
    agent_memory = AgentMemory(agent_name)
    return agent_memory.get_active_positions()


# Usage (in your main app, or as a separate process/thread):
# from redis import Redis
# stop_loss_manager = StopLossManager(Redis(...))
# position_manager = PositionManager(...)  # per agent or global
# stop_loss_watcher(stop_loss_manager, position_manager)