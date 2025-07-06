import asyncio
from datetime import datetime, timedelta
import json
from utils.redis_client import redis_client as r
from data.schemas import MarketContext
from market_tools.advance_market import get_market_regime, get_sector_performance, get_volatility_regime

async def update_market_context():
    # Run your async market context functions
    regime = await get_market_regime()
    sector_perf = await get_sector_performance()
    volatility = await get_volatility_regime()

    # Combine results
    context = MarketContext(
        regime=regime.get("regime", "Unknown"),
        regime_confidence=regime.get("confidence", 0.0),
        sector_performance=sector_perf.get("performance", {}),
        volatility_regime=volatility.get("volatility_regime", "Unknown"),
        timestamp=datetime.now()
    )

    # Store in Redis with today's date as key
    today = datetime.now().strftime("%Y%m%d")
    
    # Remove previous day's context if exists
    prev_day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    prev_key = f"market:daily_context:{prev_day}"
    if r.exists(prev_key):
        r.delete(prev_key)
        print(f"Previous market context removed for {prev_day}")


    r.set(f"market:daily_context:{today}", json.dumps(context))
    print(f"Market context updated for {today}")

if __name__ == "__main__":
    asyncio.run(update_market_context())