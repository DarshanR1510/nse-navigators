from memory.agent_memory import AgentMemory
from datetime import datetime
from utils.redis_client import main_redis_client as r
import json
from data.schemas import MarketContext


def get_overall_market_context():
    today = datetime.now().strftime("%Y%m%d")
    key = f"market:daily_context:{today}"
    raw = r.get(key)
    if not raw:
        return None
    context_dict = json.loads(raw)
    context = MarketContext(**context_dict)
    return context


async def store_market_context(agent_name: str, context: dict):
    agent_memory = AgentMemory(agent_name)
    agent_memory.store_daily_context(context)

async def get_market_context(agent_name: str, date: str = None):
    agent_memory = AgentMemory(agent_name)
    return agent_memory.get_daily_context(date)


async def add_positions(agent_name: str, positions: dict):
    agent_memory = AgentMemory(agent_name)
    agent_memory.store_active_positions(positions)

async def get_positions(agent_name: str):
    agent_memory = AgentMemory(agent_name)
    return agent_memory.get_active_positions()

async def remove_position(agent_name: str, symbol: str):
    agent_memory = AgentMemory(agent_name)
    try:
        agent_memory.remove_active_position(symbol)
    except KeyError as e:
        print(f"Position for {symbol} not found: {e}")


async def add_to_watchlist(agent_name: str, stock: str, details: dict):
    agent_memory = AgentMemory(agent_name)
    watchlist = agent_memory.get_watchlist()
    watchlist[stock] = details
    agent_memory.store_watchlist(watchlist)

async def remove_from_watchlist(agent_name: str, stock: str):
    agent_memory = AgentMemory(agent_name)
    watchlist = agent_memory.get_watchlist()
    if stock in watchlist:
        del watchlist[stock]
        agent_memory.store_watchlist(watchlist)

async def get_watchlist(agent_name: str):
    agent_memory = AgentMemory(agent_name)
    return agent_memory.get_watchlist()


async def log_trade_execution(agent_name: str, trade_details: dict):
    agent_memory = AgentMemory(agent_name)
    if "timestamp" not in trade_details:
        trade_details["timestamp"] = datetime.now().timestamp()
    agent_memory.log_trade(trade_details)


#TODO : Implement a more comprehensive performance summary
async def get_performance_summary(agent_name: str, days: int = 30):    
    pass


async def get_agent_memory_status(agent_name: str):
    agent_memory = AgentMemory(agent_name)
    data = agent_memory._load()
    return {
        "dates_with_context": list(data["daily_context"].keys()),
        "positions_count": len(data["active_positions"]),
        "watchlist_count": len(data["watchlist"]),
        "trades_count": len(data["trades"])
    }