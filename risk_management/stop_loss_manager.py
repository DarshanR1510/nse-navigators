import json
from typing import Dict, List

class StopLossManager:
    """
    Manages stop loss orders for all agents using Redis for persistence and fast access.
    Each stop loss is stored as a Redis hash: key = f"stop_loss:{agent_name}:{symbol}"
    """
    def __init__(self, redis_client):
        self.redis = redis_client

    def set_stop_loss(self, 
            symbol: str, 
            stop_price: float, 
            quantity: int, 
            agent_name: str, 
            trailing: bool = False, 
            trail_percent: float = None):
        """
        Set or update a stop loss for a symbol/agent.
        """
        key = f"stop_loss:{agent_name}:{symbol.upper()}"
        stop_loss_data = {
            "symbol": symbol.upper(),
            "stop_price": stop_price,
            "quantity": quantity,
            "agent_name": agent_name,
            "trailing": trailing,
            "trail_percent": trail_percent,
            "active": True
        }
        self.redis.hmset(key, stop_loss_data)


    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check all active stop losses against current prices. Returns list of triggered stops."""
        triggered = []
        for key in self.redis.scan_iter(match="stop_loss:*"):
            data = self.redis.hgetall(key)
            if not data or not data.get(b"active") or data.get(b"active") == b"False":
                continue
            symbol = data[b"symbol"].decode()
            stop_price = float(data[b"stop_price"].decode())
            quantity = int(data[b"quantity"].decode())
            agent_name = data[b"agent_name"].decode()
            trailing = data.get(b"trailing", b"False") == b"True"
            trail_percent = float(data.get(b"trail_percent", b"0") or 0)
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue
            # Trigger for long only (can extend for short)
            if current_price <= stop_price:
                triggered.append({
                    "symbol": symbol,
                    "stop_price": stop_price,
                    "quantity": quantity,
                    "agent_name": agent_name,
                    "trailing": trailing,
                    "trail_percent": trail_percent,
                    "current_price": current_price
                })
        return triggered
    

    def update_trailing_stop(self, symbol: str, current_price: float, agent_name: str):
        """Update trailing stop price if current price exceeds previous high (for trailing stops)."""
        key = f"stop_loss:{agent_name}:{symbol.upper()}"
        data = self.redis.hgetall(key)
        if not data or data.get(b"trailing", b"False") != b"True":
            return
        trail_percent = float(data.get(b"trail_percent", b"0") or 0)
        stop_price = float(data[b"stop_price"].decode())
        # Only move stop up if price increases
        new_stop = current_price * (1 - trail_percent / 100)
        if new_stop > stop_price:
            self.redis.hset(key, "stop_price", new_stop)


    def stop_loss_executor(self, symbol: str, agent_name: str, reason: str = "Triggered"):  # reason for logging
        """Mark stop loss as executed/inactive. (You should also trigger trade_execution externally.)"""
        key = f"stop_loss:{agent_name}:{symbol.upper()}"
        if self.redis.exists(key):
            self.redis.hset(key, "active", False)
        

    def get_active_stop_losses(self) -> Dict[str, Dict]:
        """Return all active stop losses as a dict: (agent, symbol) -> stop loss data."""
        active = {}
        for key in self.redis.scan_iter(match="stop_loss:*"):
            data = self.redis.hgetall(key)
            if not data or not data.get(b"active") or data.get(b"active") == b"False":
                continue
            symbol = data[b"symbol"].decode()
            agent_name = data[b"agent_name"].decode()
            stop_price = float(data[b"stop_price"].decode())
            quantity = int(data[b"quantity"].decode())
            trailing = data.get(b"trailing", b"False") == b"True"
            trail_percent = float(data.get(b"trail_percent", b"0") or 0)
            active[(agent_name, symbol)] = {
                "symbol": symbol,
                "stop_price": stop_price,
                "quantity": quantity,
                "agent_name": agent_name,
                "trailing": trailing,
                "trail_percent": trail_percent
            }
        return active