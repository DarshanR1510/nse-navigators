import json
from typing import Dict, List
import logging
from data.database import DatabaseQueries

class StopLossManager:
    """
    Manages stop loss orders for all agents using Redis for persistence and fast access.
    Each stop loss is stored as a Redis hash: key = f"stop_loss:{trader_name}:{symbol}"
    """
    def __init__(self, redis_client):
        self.redis = redis_client

    def set_stop_loss(self, symbol: str, stop_price: float, quantity: int, trader_name: str, trailing: bool = False, trail_percent: float = None):
        """
        Set or update a stop loss for a symbol/agent.
        """
        key = f"stop_loss:{trader_name}:{symbol.upper()}"
        stop_loss_data = {
            "symbol": symbol.upper(),
            "stop_price": float(stop_price),
            "quantity": int(quantity),
            "trader_name": trader_name,
            "trailing": str(trailing),
            "trail_percent": str(trail_percent) if trail_percent is not None else "0",
            "active": str(True)
        }
        self.redis.hmset(key, stop_loss_data)
        
        DatabaseQueries.write_log(trader_name, "function", f"Stop loss has been set for {symbol} at {stop_price} with quantity {quantity}.")
        logging.info(f"Stop loss has been set for {symbol} at {stop_price} with quantity {quantity}.")


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
            trader_name = data[b"trader_name"].decode()
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
                    "trader_name": trader_name,
                    "trailing": trailing,
                    "trail_percent": trail_percent,
                    "current_price": current_price
                })
        return triggered
    

    def update_trailing_stop(self, symbol: str, current_price: float, trader_name: str):
        """Update trailing stop price if current price exceeds previous high (for trailing stops)."""
        key = f"stop_loss:{trader_name}:{symbol.upper()}"
        data = self.redis.hgetall(key)
        if not data or data.get(b"trailing", b"False") != b"True":
            return
        trail_percent = float(data.get(b"trail_percent", b"0") or 0)
        stop_price = float(data[b"stop_price"].decode())
        # Only move stop up if price increases
        new_stop = current_price * (1 - trail_percent / 100)
        if new_stop > stop_price:
            self.redis.hset(key, "stop_price", new_stop)


    def stop_loss_executor(self, symbol: str, trader_name: str, reason: str = "Triggered"):  # reason for logging
        """Mark stop loss as executed/inactive. (You should also trigger trade_execution externally.)"""
        key = f"stop_loss:{trader_name}:{symbol.upper()}"
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
            trader_name = data[b"trader_name"].decode()
            stop_price = float(data[b"stop_price"].decode())
            quantity = int(data[b"quantity"].decode())
            trailing = data.get(b"trailing", b"False") == b"True"
            trail_percent = float(data.get(b"trail_percent", b"0") or 0)
            active[(trader_name, symbol)] = {
                "symbol": symbol,
                "stop_price": stop_price,
                "quantity": quantity,
                "trader_name": trader_name,
                "trailing": trailing,
                "trail_percent": trail_percent
            }
        return active