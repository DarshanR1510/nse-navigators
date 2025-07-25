from memory.agent_memory import AgentMemory
from datetime import datetime
from utils.redis_client import main_redis_client as r
import json
import pytz
from data.schemas import MarketContext, IST




def get_overall_market_context():
    today = datetime.now(IST).strftime("%Y-%m-%d")
    key = f"market:daily_context:{today}"
    raw = r.get(key)
    if not raw:
        return None
    context_dict = json.loads(raw)
    context = MarketContext(**context_dict)
    return context


def add_positions(trader_name: str, positions: dict):
    agent_memory = AgentMemory(trader_name)
    agent_memory.store_active_position(positions)

async def get_positions(trader_name: str):
    agent_memory = AgentMemory(trader_name)
    return agent_memory.get_active_positions()

async def remove_position(trader_name: str, symbol: str):
    agent_memory = AgentMemory(trader_name)
    try:
        agent_memory.remove_active_position(symbol)
    except KeyError as e:
        print(f"Position for {symbol} not found: {e}")


async def add_to_watchlist(trader_name: str, stock: str, details: str):
    agent_memory = AgentMemory(trader_name)
    watchlist = agent_memory.get_watchlist()
    watchlist[stock] = details
    agent_memory.store_watchlist(watchlist)

async def remove_from_watchlist(trader_name: str, stock: str):
    agent_memory = AgentMemory(trader_name)
    watchlist = agent_memory.get_watchlist()
    if stock in watchlist:
        del watchlist[stock]
        agent_memory.store_watchlist(watchlist)

async def get_watchlist(trader_name: str):
    agent_memory = AgentMemory(trader_name)
    return agent_memory.get_watchlist()


async def log_trade_execution(trader_name: str, trade_details: dict):
    agent_memory = AgentMemory(trader_name)
    if "timestamp" not in trade_details:
        trade_details["timestamp"] = datetime.now(IST).timestamp()
    agent_memory.log_trade(trade_details)


#TODO : Implement a more comprehensive performance summary
async def get_performance_summary(trader_name: str, days: int = 30):    
    pass


async def get_agent_memory_status(trader_name: str):
    agent_memory = AgentMemory(trader_name)
    data = agent_memory._load()
    return {
        "dates_with_context": list(data["daily_context"].keys()),
        "positions_count": len(data["active_positions"]),
        "watchlist_count": len(data["watchlist"]),
        "trades_count": len(data["trades"])
    }