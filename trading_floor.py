from trade_agents.traders import Trader
from typing import List
import asyncio
import threading
import signal
import sys
import time
from risk_management.stop_loss_manager import StopLossManager
from risk_management.position_manager import PositionManager
from risk_management.stop_loss_watcher import stop_loss_watcher
from dotenv import load_dotenv
import os
import pytz
from datetime import datetime
from utils.redis_client import main_redis_client
from trade_agents.orchestrator import AgentOrchestrator, agent_names
from agents.mcp import MCPServerStdio
from contextlib import AsyncExitStack
from mcp_servers.mcp_params import trader_mcp_server_params, researcher_mcp_server_params
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
agent_names = ["warren", "george", "ray", "cathie"]

load_dotenv(override=True)

IST = pytz.timezone('Asia/Kolkata')
RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = os.getenv("RUN_EVEN_WHEN_MARKET_IS_CLOSED", "false").strip().lower() == "true"
USE_MANY_MODELS = os.getenv("USE_MANY_MODELS", "false").strip().lower() == "true"

lastnames = ["Patience", "Bold", "Systematic", "Growth"]

stop_loss_manager = StopLossManager(main_redis_client)
position_managers = {name: PositionManager(agent_name=name, redis_client=main_redis_client) for name in agent_names}

# GEMINI DOES NOT SUPPORT JSON RESPONSE FROM AGENT OR MCP SERVER


if USE_MANY_MODELS:
    model_names = ["gpt-4.1-mini", "deepseek-chat", "gemini-2.5-flash-preview-05-20", "gpt-4o-mini"]
    short_model_names = ["GPT 4.1", "DeepSeek V3", "Gemini 2.5", "GPT 4o"]
else:
    model_names = ["gpt-4o-mini"] * 4
    short_model_names = ["GPT 4o"] * 4


def create_traders() -> List[Trader]:
    traders = []
    for name, lastname, model_name in zip(agent_names, lastnames, model_names):        
        traders.append(Trader(name, lastname, model_name, position_manager=position_managers[name]))
    
    return traders

async def main_workflow():
    traders = create_traders()
    orchestrator = AgentOrchestrator(
        traders,         
        stop_loss_manager=stop_loss_manager,
        position_managers=position_managers
    )

    async with AsyncExitStack() as stack:
        market_mcp_servers = [
            await stack.enter_async_context(MCPServerStdio(params, client_session_timeout_seconds=30)) 
            for params in trader_mcp_server_params
        ]        

        researcher_mcp_servers = [
            await stack.enter_async_context(MCPServerStdio(params, client_session_timeout_seconds=30)) 
            for params in researcher_mcp_server_params
        ]

        await orchestrator.run_traders(market_mcp_servers, researcher_mcp_servers)



async def periodic_main_workflow():
    while True:
        try:
            logger.info("Starting main workflow...")
            await main_workflow()
            logger.info("Main workflow complete. Sleeping...")
        except Exception as e:
            logger.error(f"Error in main workflow: {e}")
        await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)
    

def run_with_stop_loss_watcher():
    logger.info(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")
    stop_event = threading.Event()

    def watcher_thread():
        # Only run between 9:20am and 3:30pm IST
        while not stop_event.is_set():
            now = datetime.now(IST)            
            if (now.hour > 9 or (now.hour == 9 and now.minute >= 20)) and (now.hour < 15 or (now.hour == 15 and now.minute < 30)):
                stop_loss_watcher(stop_loss_manager, position_managers, poll_interval=300)
            else:
                logger.info("Stop-loss watcher sleeping (market closed, IST)...")
                # Sleep until next check (e.g., 30 minutes)
                time.sleep(1800)

    watcher = threading.Thread(target=watcher_thread, daemon=True)
    watcher.start()

    def handle_exit(signum, frame):
        logger.info("Shutting down...")
        stop_event.set()
        watcher.join(timeout=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        asyncio.run(periodic_main_workflow())
    finally:
        stop_event.set()
        watcher.join(timeout=300)



if __name__ == "__main__":
    run_with_stop_loss_watcher()
