import os
import json
from datetime import datetime
from data.schemas import IST
class AgentMemory:    
    def __init__(self, trader_name: str, base_path: str = "trader_data"):        
        self.trader_name = trader_name
        self.base_path = base_path

        os.makedirs(self.base_path, exist_ok=True)
        self.memory_file = os.path.join(self.base_path, f"{(self.trader_name).lower()}.json")
        if not os.path.exists(self.memory_file):
            self._init_memory_file()

    def _init_memory_file(self):
        data = {
            "daily_context": {},
            "active_positions": {},
            "watchlist": {},
            "trades": []
        }
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)


    def _load(self):
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)


    def store_active_position(self, positions: dict):
        data = self._load()
        data["active_positions"].update(positions)
        self._save(data)

    def get_active_positions(self) -> dict:
        data = self._load()
        return data.get("active_positions", {})
    
    def remove_active_position(self, symbol: str):
        data = self._load()
        if symbol in data["active_positions"]:
            del data["active_positions"][symbol]
            self._save(data)
        else:
            f"Position for {symbol} not found in active positions."


    def store_watchlist(self, watchlist: dict):
        data = self._load()
        data["watchlist"] = watchlist
        self._save(data)

    def get_watchlist(self) -> dict:
        data = self._load()
        return data.get("watchlist", {})
    

    def add_trade_count(self):
        data = self._load()
        today = datetime.now(IST).strftime("%Y-%m-%d")
        today_key = f"{today}-trade-count"
        if today_key not in data:
            data[today_key] = 1
        else:
            data[today_key] += 1
        self._save(data)    

    def get_trade_count(self) -> int:
        data = self._load()
        today = datetime.now(IST).strftime("%Y-%m-%d")
        today_key = f"{today}-trade-count"
        return data.get(today_key, 0)
    

    def get_recent_trades(self, days: int = 30) -> list:
        data = self._load()
        cutoff = datetime.now(IST).timestamp() - days * 86400
        return [
            t for t in data["trades"]
            if "timestamp" in t and t["timestamp"] >= cutoff
        ]

    def log_trade(self, trade_details: dict):
        data = self._load()
        data["trades"].append(trade_details)
        self._save(data)


    def calculate_performance_metrics(self) -> dict:
        pass


    def backup_memory(self):
        pass

    def restore_memory(self, backup_date: str):
        pass

    def clear_memory(self):
        data = self._load()
        data["daily_context"] = {}
        data["active_positions"] = {}
        data["watchlist"] = {}
        data["trades"] = []
        self._save(data)

