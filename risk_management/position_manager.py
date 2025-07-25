import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from datetime import datetime
from enum import Enum, auto
import redis
import os
from dotenv import load_dotenv
from data.database import DatabaseQueries
from market_tools.market import get_multiple_symbol_prices
from memory import memory_tools
from utils.redis_client import main_redis_client as r
from data.schemas import Position, RiskLimits, IST

load_dotenv(override=True)

class PositionStatus(Enum):
    ACTIVE = auto()
    STOPPED_OUT = auto()
    CLOSED = auto()

class PositionManager:
    """
    Comprehensive position and risk manager.
    """
    
    # Constants
    REDIS_TTL = 3600  # 1 hour TTL for Redis cache
    
    def __init__(self, 
                 trader_name: str,
                 redis_client: redis.Redis = r,
                 portfolio_value: float = float(os.getenv("INITIAL_BALANCE", 500000.0)),
                 risk_limits: Optional[RiskLimits] = None):
        """
        Initialize PositionManager
        
        Args:
            trader_name: Name of the trader managing positions
            redis_client: Redis client for caching and real-time data
            portfolio_value: Total portfolio value
            risk_limits: Risk management configuration
        """
        self.trader_name = (trader_name).lower().strip()
        self.redis_client = redis_client
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits or RiskLimits()
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.total_portfolio_risk = 0.0
        self._load_positions()        
        self.logger = logging.getLogger(__name__)

    # Core Position Management Methods
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss: float, 
                              portfolio_value: Optional[float] = None,
                              trader_name: Optional[str] = None
                              ) -> int:
        """
        Calculate position size based on risk per trade
        
        Args:
            entry_price: Entry price of the stock
            stop_loss: Stop loss price
            portfolio_value: Current portfolio value (optional)
            
        Returns:
            Number of shares to buy (0 if error occurs)
        """
        try:
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            portfolio_value = float(portfolio_value if portfolio_value is not None else self.portfolio_value)
            trader_name = trader_name or self.trader_name
            
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share <= 0:
                self.logger.warning(f"Invalid risk per share for {trader_name}")
                return 0
                
            max_risk_amount = portfolio_value * float(self.risk_limits.risk_per_trade)
            position_size = int(max_risk_amount / risk_per_share)
            
            max_position_value = portfolio_value * float(self.risk_limits.max_position_size)
            max_shares_by_value = int(max_position_value / entry_price)
            
            final_position_size = min(position_size, max_shares_by_value)
            
            self.logger.info(
                f"[{trader_name}] Position size: Entry={entry_price}, Stop={stop_loss}, "
                f"Size={final_position_size}"
            )
            
            return final_position_size
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"[{trader_name}] Position size calculation error: {e}")
            return 0


    def validate_new_position(self, 
                         symbol: str, 
                         quantity: int, 
                         entry_price: float,
                         stop_loss: float,
                         sector: str = None) -> Tuple[bool, str]:
        """
        Validate if a new position can be opened

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            sector: Sector of the stock (optional)

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            quantity = int(quantity)
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            portfolio_value = float(self.portfolio_value)
        except Exception as e:            
            return False, f"Type conversion error in validate new position: {e}"


        # Check position limits
        if len(self.positions) >= self.risk_limits.max_open_positions:            
            return False, f"Max open positions reached ({self.risk_limits.max_open_positions})"

        # Check position size
        position_value = quantity * entry_price
        position_size_percent = position_value / portfolio_value
        if position_size_percent > self.risk_limits.max_position_size:      
            quantity = self.calculate_position_size(entry_price, stop_loss, trader_name=self.trader_name)      

        # Check risk-reward
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:            
            return False, "Invalid stop loss"

        # Check portfolio risk
        new_risk = (quantity * risk_per_share) / portfolio_value
        if self.total_portfolio_risk + new_risk > self.risk_limits.max_portfolio_risk:            
            return False, f"Portfolio risk would exceed limit"

        # Check daily loss limit
        if self.daily_pnl < -portfolio_value * self.risk_limits.max_daily_loss:
            return False, "Daily loss limit exceeded"

        return True, "Position validation passed"


    def add_position(self,
                    trader_name: str, 
                    symbol: str, 
                    quantity: int, 
                    entry_price: float,
                    stop_loss: float,
                    target: float,
                    reason: str,
                    position_type: str = "LONG",
                    sector: str = None) -> Tuple[bool, str]:
        """
        Add a new position to the portfolio
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            target: Target price
            reason: Reason for the trade
            position_type: LONG or SHORT
            sector: Sector of the stock
            
        Returns:
            Tuple of (success, message)
        """
        is_valid, validation_message = self.validate_new_position(
            symbol, quantity, entry_price, stop_loss, sector
        )
        if not is_valid:
            return False, validation_message
        
        position_value = quantity * entry_price
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            entry_date=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            trader_name=self.trader_name,
            position_type=position_type,
            reason=reason,
            position_size_percent=position_value / self.portfolio_value,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            status=PositionStatus.ACTIVE.name
        )
        
        self.positions[symbol] = position
        self._update_portfolio_metrics()
        self._save_position(symbol, position)
        
        self.logger.info(f"FROM POSITION MANAGER END [{self.trader_name}] Added position: {symbol} x{quantity} @ {entry_price}")
        return True, f"Position added successfully: {symbol}"


    def check_stop_loss_triggers(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check if any positions have triggered stop losses
        
        Args:
            current_prices: Dictionary of {symbol: current_price}
            
        Returns:
            List of positions that triggered stop losses
        """
        triggered_stops = []
        
        for symbol, position in self.positions.items():
            if position.status != PositionStatus.ACTIVE.name or symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            stop_triggered = (
                (position.position_type == "LONG" and current_price <= position.stop_loss) or
                (position.position_type == "SHORT" and current_price >= position.stop_loss)
            )
            
            if stop_triggered:
                triggered_stops.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "stop_loss": position.stop_loss,
                    "quantity": position.quantity,
                    "position_type": position.position_type,
                    "entry_price": position.entry_price,
                    "loss_amount": self._calculate_pnl(position, current_price)
                })
        
        return triggered_stops


    def execute_stop_loss(self, symbol: str, execution_price: float) -> Tuple[bool, str]:
        """
        Execute stop loss for a position
        
        Args:
            symbol: Stock symbol
            execution_price: Price at which stop loss was executed
            
        Returns:
            Tuple of (success, message)
        """
        if symbol not in self.positions:
            return False, f"Position {symbol} not found"
        
        position = self.positions[symbol]
        position.realized_pnl = self._calculate_pnl(position, execution_price)
        position.status = PositionStatus.STOPPED_OUT.name
        position.current_price = execution_price
        
        self._update_portfolio_metrics()
        self._remove_position(symbol)
        
        self.logger.info(f"[{self.trader_name}] Stop loss executed: {symbol} @ {execution_price}")
        return True, f"Stop loss executed for {symbol}"


    def get_active_positions(self) -> Dict[str, Position]:
        """
        Get all active positions
        
        Returns:
            Dictionary of active positions
        """
        return {symbol: pos for symbol, pos in self.positions.items() if pos.status == PositionStatus.ACTIVE.name}
    
    # Portfolio Management Methods
    def check_portfolio_limits(self) -> Dict[str, Any]:
        """
        Check all portfolio limits and return status
        
        Returns:
            Dictionary with limit status
        """
        current_positions = len(self.positions)
        total_portfolio_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        cash_used_percent = total_portfolio_value / self.portfolio_value
        daily_loss_percent = abs(self.daily_pnl) / self.portfolio_value if self.daily_pnl < 0 else 0
        
        return {
            "positions_count": current_positions,
            "max_positions": self.risk_limits.max_open_positions,
            "positions_limit_ok": current_positions <= self.risk_limits.max_open_positions,
            "cash_used_percent": cash_used_percent,
            "portfolio_risk_percent": self.total_portfolio_risk,
            "daily_pnl": self.daily_pnl,
            "daily_loss_percent": daily_loss_percent,
            "emergency_stop_triggered": daily_loss_percent >= self.risk_limits.emergency_stop_loss,
            "overall_status": "OK" if all([
                current_positions <= self.risk_limits.max_open_positions,
                self.total_portfolio_risk <= self.risk_limits.max_portfolio_risk,
                daily_loss_percent <= self.risk_limits.max_daily_loss,
                daily_loss_percent < self.risk_limits.emergency_stop_loss
            ]) else "RISK_BREACH"
        }


    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive position summary
        
        Returns:
            Dictionary with position summary
        """
        if not self.positions:
            return {
                "total_positions": 0,
                "total_invested": 0,
                "total_pnl": 0,
                "positions": []
            }
        
        total_invested = sum(pos.quantity * pos.entry_price for pos in self.positions.values())
        total_current_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        total_pnl = total_current_value - total_invested
        
        positions_list = [{
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "unrealized_pnl": pos.unrealized_pnl,
            "pnl_percent": (pos.unrealized_pnl / (pos.quantity * pos.entry_price)) * 100,
            "days_held": (datetime.now(IST) - datetime.strptime(pos.entry_date, "%Y-%m-%d %H:%M:%S")).days
        } for pos in self.positions.values()]
        
        return {
            "total_positions": len(self.positions),
            "total_invested": total_invested,
            "total_current_value": total_current_value,
            "total_pnl": total_pnl,
            "positions": positions_list
        }


    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value and recalculate metrics
        
        Args:
            new_value: New portfolio value
        """
        self.portfolio_value = new_value
        self._update_portfolio_metrics()
        self.logger.info(f"[{self.trader_name}] Portfolio value updated to: {new_value}")


    # Monitoring and Analytics
    def monitor_positions(self) -> Dict[str, Any]:
        """
        Monitor all positions for risk alerts and triggers
        
        Returns:
            Dictionary with monitoring results
        """
        now = datetime.now(IST)
        results = {
            "positions_monitored": len(self.positions),
            "risk_alerts": [],
            "stop_losses_triggered": [],
            "analytics": []
        }
        
        current_prices = get_multiple_symbol_prices(list(self.positions.keys()))
        
        for symbol, pos in self.positions.items():
            try:
                ltp = current_prices.get(symbol, 0)
                if ltp == 0:
                    continue
                    
                pnl_percent = ((ltp - pos.entry_price) / pos.entry_price) * 100
                analytics_data = {
                    "symbol": symbol,
                    "current_price": ltp,
                    "pnl_percent": pnl_percent,
                    "risk_level": "High" if pnl_percent < -8 else "Medium" if pnl_percent < -4 else "Low",
                    "days_held": (now - datetime.strptime(pos.entry_date, "%Y-%m-%d %H:%M:%S")).days
                }
                
                if pnl_percent <= -5:
                    results["risk_alerts"].append(analytics_data)
                if ltp <= pos.stop_loss:
                    results["stop_losses_triggered"].append(analytics_data)
                
                results["analytics"].append(analytics_data)
                
            except Exception as e:
                self.logger.error(f"[{self.trader_name}] Monitoring error for {symbol}: {e}")
                
        return results


    # Private Helper Methods
    def _update_portfolio_metrics(self):
        """Recalculate all portfolio metrics"""
        self.total_portfolio_risk = self._calculate_total_portfolio_risk()
        self.daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())


    def _calculate_total_portfolio_risk(self) -> float:
        """Calculate total portfolio risk percentage"""
        return sum(
            (pos.quantity * abs(pos.entry_price - pos.stop_loss)) / self.portfolio_value
            for pos in self.positions.values()
        )


    def _save_position(self, symbol: str, position: Position):
        """Persist position to all storage layers"""
        try:
            # Save to database
            DatabaseQueries.save_position(asdict(position), trader_name=self.trader_name)
            
            # Save to memory
            memory_tools.add_positions(
                trader_name=self.trader_name,
                positions={
                    symbol: {
                        "entry_price": position.entry_price,
                        "quantity": position.quantity,
                        "stop_loss": position.stop_loss,
                        "target": position.target,
                        "entry_date": position.entry_date
                    }
                }
            )
            
            # Cache in Redis
            self._cache_position_in_redis(symbol, position)
            
        except Exception as e:
            self.logger.error(f"[{self.trader_name}] Position persistence error: {e}")



    def _remove_position(self, symbol: str):
        """Remove position from all storage layers"""
        try:
            position = self.positions.get(symbol)
            # Remove from database
            DatabaseQueries.remove_position(self.trader_name, symbol)

            # Save closed position to database
            DatabaseQueries.save_closed_position(asdict(position), trader_name=self.trader_name)
            
            # Remove from memory
            memory_tools.remove_position(trader_name=self.trader_name, symbol=symbol)
            
            # Remove from Redis
            self.redis_client.delete(self._get_redis_key(symbol))
            
            # Remove from local cache
            self.positions.pop(symbol, None)
            
        except Exception as e:
            self.logger.error(f"[{self.trader_name}] Position removal error: {e}")


    def _load_positions(self):
        """Load positions from database"""
        try:
            positions_data = DatabaseQueries.load_positions(trader_name=self.trader_name)
            valid_keys = set(Position.__dataclass_fields__.keys())
            self.positions = {
                data['symbol']: Position(**{k: v for k, v in data.items() if k in valid_keys})
                for data in positions_data
                if 'symbol' in data and data.get('status') == PositionStatus.ACTIVE.name
            }

            self._update_portfolio_metrics()
            
        except Exception as e:
            self.logger.error(f"[{self.trader_name}] Error loading positions: {e}")
            self.positions = {}


    def _cache_position_in_redis(self, symbol: str, position: Position):
        """Cache position in Redis"""
        try:
            self.redis_client.setex(
                self._get_redis_key(symbol),
                self.REDIS_TTL,
                json.dumps(asdict(position))
            )
        except Exception as e:
            self.logger.error(f"[{self.trader_name}] Redis caching error: {e}")


    def _get_redis_key(self, symbol: str) -> str:
        """Generate Redis key for position"""
        return f"position:{self.trader_name}:{symbol}"


    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Calculate P&L for a position"""
        if position.position_type == "LONG":
            return position.quantity * (current_price - position.entry_price)
        return position.quantity * (position.entry_price - current_price)