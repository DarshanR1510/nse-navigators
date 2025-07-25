import json
from mcp_servers.accounts_client import read_accounts_resource, read_strategy_resource
from memory.agent_memory import AgentMemory
from utils.redis_client import main_redis_client as r
from trade_agents.trading_orchestrator import TradingOrchestrator
from memory import memory_tools
from market_tools import market_context_updater
from data.database import DatabaseQueries
import logging
from datetime import datetime
from data.schemas import IST

class Trader:

    """
    Trader class that manages trading workflow through orchestrator.
    Each trader has their own orchestrator, position manager and memory management.
    """

    def __init__(self, name: str, model_name="gpt-4o-mini", position_manager=None):
        
        self.name = name
        self.model_name = model_name        
        self.position_manager = position_manager                
        self.agent_memory = AgentMemory(self.name)       
        self.redis_client = r
        self.logger = logging.getLogger(f"trader.{name}")
        self._orchestrator = None
        self._strategy = None

    
    async def get_account_report(self) -> str:
        print(f"Fetching account report for {self.name}")
        account = await read_accounts_resource(self.name)
        account_json = json.loads(account)
        account_json.pop("portfolio_value_time_series", None)
        return json.dumps(account_json)
    

    async def ensure_strategy(self):
        """Ensure strategy is loaded before proceeding"""        
        if self._strategy is None:
            self._strategy = await read_strategy_resource(self.name)
        return self._strategy


    @property
    def orchestrator(self) -> TradingOrchestrator:
        """Non-async initialization of orchestrator"""
        
        if not self._orchestrator or self._strategy is None:  
            self._orchestrator = TradingOrchestrator(
                trader_name=self.name,
                model_name=self.model_name,
                strategy=self._strategy,
                position_manager=self.position_manager
            )
        return self._orchestrator


    async def run(self):
        """Execute trading workflow through orchestrator"""
        # try:
        self.logger.info(f"Starting trading run for {self.name}")

        # Get or store market and agent context
        overall_market_context = memory_tools.get_overall_market_context()
        today_date = datetime.now(IST).strftime("%Y-%m-%d")

        if not overall_market_context or overall_market_context.timestamp != today_date:
            overall_market_context = await market_context_updater.update_market_context()
        
        DatabaseQueries.write_log(self.name, "response", f"fetched overall market context")

        # Run orchestrator
        await self.ensure_strategy()
        results = await self.orchestrator.run()  
        
        return results

        # except Exception as e:
        #     self.logger.error(f"Error in trading run for {self.name}: {e}")
        #     raise
    

    async def get_memory_context(self):
            """Get trader's memory context"""
            return {
                "positions": (self.agent_memory.get_active_positions() or {}),
                "watchlist": (self.agent_memory.get_watchlist() or {})
            }
    

    
    