import os
import json
from datetime import datetime

class AgentMemory:    
    def __init__(self, agent_name: str, base_path: str = "memory/agents_data"):        
        self.agent_name = agent_name
        self.base_path = base_path

        os.makedirs(self.base_path, exist_ok=True)
        self.memory_file = os.path.join(self.base_path, f"{(self.agent_name).lower()}.json")
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


    def store_daily_context(self, context: dict):
        data = self._load()
        today = datetime.now().strftime("%Y-%m-%d")
        data["daily_context"][today] = context
        self._save(data)

    def get_daily_context(self, date: str = None) -> dict:
        data = self._load()
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        return data["daily_context"].get(date, {})


    def store_active_positions(self, positions: dict):
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


    def get_recent_trades(self, days: int = 30) -> list:
        data = self._load()
        cutoff = datetime.now().timestamp() - days * 86400
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

