from trade_agents.traders import Trader
from typing import List, Dict, Optional
import asyncio
from risk_management.stop_loss_manager import StopLossManager
from risk_management.position_manager import PositionManager
from dotenv import load_dotenv
import os
from trade_agents.trading_orchestrator import TradingOrchestrator
from mcp_servers.accounts_client import read_strategy_resource
from data.schemas import IST
import threading
import time
from market_tools.market import is_market_open
import signal
from risk_management.stop_loss_watcher import stop_loss_watcher
from datetime import datetime
from trade_agents.trading_orchestrator import trader_names
from utils.redis_client import main_redis_client as r
import logging
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = os.getenv("RUN_EVEN_WHEN_MARKET_IS_CLOSED", "false").strip().lower() == "true"
USE_MANY_MODELS = os.getenv("USE_MANY_MODELS", "false").strip().lower() == "true"

lastnames = ["Patience", "Bold", "Systematic", "Growth"]
position_managers = {name: PositionManager(trader_name=name, redis_client=r) for name in trader_names}

class TradingFloor:

    _instance: Optional['TradingFloor'] = None
    stop_loss_manager: Optional[StopLossManager] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingFloor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.stop_event = threading.Event()
            self.watcher_thread = None

            # Configure models
            if USE_MANY_MODELS:
                self.model_names = ["gpt-4.1-mini", "deepseek-chat", "gemini-2.5-flash-preview-05-20", "gpt-4o-mini"]
                self.short_model_names = ["GPT 4.1", "DeepSeek R1", "Gemini 2.5", "GPT 4o"]
            else:
                self.model_names = ["gpt-4o-mini"] * 4
                self.short_model_names = ["GPT 4o"] * 4

            # Suppress warnings
            logging.getLogger("asyncio").setLevel(logging.CRITICAL)
            logging.getLogger("anyio").setLevel(logging.CRITICAL)

            self._initialized = True



    @classmethod
    def get_stop_loss_manager(cls) -> Optional[StopLossManager]:
        """Get the stop loss manager instance"""
        return cls.stop_loss_manager


    def create_traders(self) -> List[Trader]:
        return [
            Trader(name, model_name=model_name, position_manager=position_managers[name])
            for name, model_name in zip(trader_names, self.model_names)
        ]    

    async def main_workflow(self):
        traders = self.create_traders()
        orchestrators = []

        for trader in traders:
            try:
                strategy = await read_strategy_resource(trader.name)
                orchestrator = TradingOrchestrator(
                    trader_name=trader.name,
                    model_name=trader.model_name,
                    strategy=strategy,
                    position_manager=trader.position_manager
                )
                orchestrators.append(orchestrator)
            except Exception as e:
                logger.error(f"Failed to initialize orchestrator for {trader.name}: {e}")

        if not orchestrators:
            raise RuntimeError("No orchestrators could be initialized")

        try:
            results = await asyncio.gather(
                *(o.run() for o in orchestrators),
                return_exceptions=True
            )
            
            for orchestrator, result in zip(orchestrators, results):
                if isinstance(result, Exception):
                    logger.error(f"Orchestrator {orchestrator.trader_name} failed: {result}")
                    
        except Exception as e:
            logger.error(f"Orchestrator execution failed: {e}")

    async def periodic_main_workflow(self):
        while not self.stop_event.is_set():
            try:
                now = datetime.now(IST)
                
                if not is_market_open() and not RUN_EVEN_WHEN_MARKET_IS_CLOSED:
                    logger.info("Market closed, sleeping until next check...")
                    await asyncio.sleep(300)
                    continue
                    
                logger.info(f"Starting main workflow at {now.strftime('%H:%M:%S')}...")
                await self.main_workflow()
                logger.info(f"Main workflow complete at {datetime.now(IST).strftime('%H:%M:%S')}")
                logger.info(f"Sleeping for {RUN_EVERY_N_MINUTES} minutes...")

                await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)

            except Exception as e:
                logger.error(f"Error in main workflow: {e}")
                if not self.stop_event.is_set():
                    await asyncio.sleep(60)

    def _run_stop_loss_watcher(self):
        while not self.stop_event.is_set():
            now = datetime.now(IST)            
            if (now.hour > 9 or (now.hour == 9 and now.minute >= 20)) and (now.hour < 15 or (now.hour == 15 and now.minute < 30)):
                stop_loss_watcher(self.stop_loss_manager, position_managers, poll_interval=900)
            else:
                logger.info("Stop-loss watcher sleeping (market closed, IST)...")
                time.sleep(1800)

    async def shutdown(self):
        logger.info("Initiating graceful shutdown...")
        self.stop_event.set()
        
        if self.watcher_thread and self.watcher_thread.is_alive():
            self.watcher_thread.join(timeout=5)
            
        # Close connections
        for pm in position_managers.values():
            await pm.close()
        await self.stop_loss_manager.close()
        
        logger.info("Shutdown complete")

    def run(self):
        logger.info(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")

        self.watcher_thread = threading.Thread(
            target=self._run_stop_loss_watcher, 
            daemon=True
        )
        self.watcher_thread.start()

        def handle_exit(signum, frame):
            logger.info("Handling exit signal...")
            asyncio.run(self.shutdown())
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        try:
            asyncio.run(self.periodic_main_workflow())
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            logger.info("Shutting down gracefully...")
            asyncio.run(self.shutdown())

if __name__ == "__main__":
    trading_floor = TradingFloor()
    trading_floor.run()




