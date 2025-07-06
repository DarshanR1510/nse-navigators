from trade_agents.traders import Trader
from typing import List
import asyncio
import threading
import signal
import sys
import time
from redis import Redis
from risk_management.stop_loss_manager import StopLossManager
from risk_management.position_manager import PositionManager
from risk_management.stop_loss_watcher import stop_loss_watcher
from trade_agents.tracers import LogTracer
from agents import add_trace_processor
from market_tools.market import is_market_open
from dotenv import load_dotenv
import os
import pytz
from datetime import datetime
from utils.redis_client import main_redis_client

load_dotenv(override=True)

IST = pytz.timezone('Asia/Kolkata')
RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = os.getenv("RUN_EVEN_WHEN_MARKET_IS_CLOSED", "false").strip().lower() == "true"
USE_MANY_MODELS = os.getenv("USE_MANY_MODELS", "false").strip().lower() == "true"

names = ["Warren", "George", "Ray", "Cathie"]
lastnames = ["Patience", "Bold", "Systematic", "Growth"]

stop_loss_manager = StopLossManager(main_redis_client)
position_managers = {name: PositionManager(agent_name=name, redis_client=main_redis_client) for name in names}


if USE_MANY_MODELS:
    model_names = ["gpt-4.1-mini", "deepseek-chat", "gemini-2.5-flash-preview-05-20", "gpt-4o-mini"]
    short_model_names = ["GPT 4.1", "DeepSeek V3", "Gemini 2.5", "GPT 4o"]
else:
    model_names = ["gpt-4o-mini"] * 4
    short_model_names = ["GPT 4o"] * 4


def create_traders() -> List[Trader]:
    traders = []
    for name, lastname, model_name in zip(names, lastnames, model_names):        
        traders.append(Trader(name, lastname, model_name, position_manager=position_managers[name]))
    
    return traders


async def run_every_n_minutes_shared(traders: List[Trader]):
    add_trace_processor(LogTracer())
    while True:
        if RUN_EVEN_WHEN_MARKET_IS_CLOSED or is_market_open():
            print("Market is open, running traders")
            await asyncio.gather(*[trader.run() for trader in traders])
        else:
            print("Market is closed, skipping run")
        await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)


def run_with_stop_loss_watcher():
    print(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")        
    
    traders = create_traders()

    # Thread control
    stop_event = threading.Event()

    def watcher_thread():
        # Only run between 9:20am and 3:30pm IST
        while not stop_event.is_set():
            now = datetime.now(IST)            
            if (now.hour > 9 or (now.hour == 9 and now.minute >= 20)) and (now.hour < 15 or (now.hour == 15 and now.minute < 30)):
                stop_loss_watcher(stop_loss_manager, position_managers, poll_interval=300)
            else:
                print("Stop-loss watcher sleeping (market closed, IST)...")
                # Sleep until next check (e.g., 30 minutes)
                time.sleep(1800)

    watcher = threading.Thread(target=watcher_thread, daemon=True)
    watcher.start()

    def handle_exit(signum, frame):
        print("Shutting down...", flush=True)
        stop_event.set()
        watcher.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        asyncio.run(run_every_n_minutes_shared(traders))
    finally:
        stop_event.set()
        watcher.join(timeout=300)



if __name__ == "__main__":
    run_with_stop_loss_watcher()
