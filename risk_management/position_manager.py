"""
PositionManager - Complete Risk Management System for NSE Navigators

This class handles all position sizing, risk management, and portfolio limits
for the NSE Navigators trading system.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import redis
import os
from dotenv import load_dotenv
from data.database import DatabaseQueries
from memory import memory_tools

load_dotenv(override=True)

@dataclass
class Position:
    """Individual position data structure"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float
    target: float
    entry_date: str
    agent_name: str
    position_type: str  # "LONG" or "SHORT"
    reason: str
    position_size_percent: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "ACTIVE"  # ACTIVE, CLOSED, STOPPED_OUT


@dataclass
class RiskLimits:
    """Risk management configuration"""
    max_position_size: float = 0.08          # 8% of portfolio per position
    max_sector_exposure: float = 0.25        # 25% in any sector
    max_daily_loss: float = 0.02             # 2% daily loss limit
    max_portfolio_risk: float = 0.15         # 15% total portfolio at risk
    min_risk_reward: float = 1.5             # Minimum 1:1.5 risk-reward
    max_correlation: float = 0.7             # Maximum correlation between positions
    max_open_positions: int = 15             # Maximum open positions
    max_new_positions_per_day: int = 3       # Maximum new positions per day
    risk_per_trade: float = 0.02             # 2% portfolio risk per trade
    emergency_stop_loss: float = 0.05        # 5% daily loss = emergency stop


class PositionManager:
    """
    Comprehensive position and risk manager.
    """
    
    def __init__(self, 
                 agent_name: str,
                 redis_client: redis.Redis,
                 portfolio_value: float = os.getenv("INITIAL_BALANCE", 500000.0),
                 risk_limits: Optional[RiskLimits] = None):
        """
        Initialize PositionManager
        
        Args:
            redis_client: Redis client for caching and real-time data
            portfolio_value: Total portfolio value
            risk_limits: Risk management configuration
            data_path: Path to store position data
        """
        self.agent_name = agent_name
        self.redis_client = redis_client
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits or RiskLimits()
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.total_portfolio_risk = 0.0
        self._load_positions()
        self.logger = logging.getLogger(__name__)
        

    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss: float, 
                              portfolio_value: Optional[float] = None) -> int:
        """
        Calculate position size based on risk per trade
        
        Args:
            entry_price: Entry price of the stock
            stop_loss: Stop loss price
            portfolio_value: Current portfolio value (optional)
            
        Returns:
            Number of shares to buy
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
            
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            self.logger.warning(f"Invalid risk per share: {risk_per_share}")
            return 0
            
        # Calculate maximum risk amount
        max_risk_amount = portfolio_value * self.risk_limits.risk_per_trade
        
        # Calculate position size
        position_size = int(max_risk_amount / risk_per_share)
        
        # Apply maximum position size limit
        max_position_value = portfolio_value * self.risk_limits.max_position_size
        max_shares_by_value = int(max_position_value / entry_price)
        
        final_position_size = min(position_size, max_shares_by_value)
        
        self.logger.info(f"Position size calculation: Entry={entry_price}, "
                        f"Stop={stop_loss}, Risk/share={risk_per_share}, "
                        f"Max risk=${max_risk_amount}, Size={final_position_size}")
        
        return final_position_size
    
    
    def validate_new_position(self, 
                            symbol: str, 
                            quantity: int, 
                            entry_price: float,
                            stop_loss: float,
                            agent_name: str,
                            sector: str = None) -> Tuple[bool, str]:
        """
        Validate if a new position can be opened
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            agent_name: Name of the agent
            sector: Sector of the stock (optional)
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if position already exists
        if symbol in self.positions and self.positions[symbol].status == "ACTIVE":
            return False, f"Position in {symbol} already exists"
        
        # Check maximum open positions
        if len(self.positions) >= self.risk_limits.max_open_positions:
            return False, f"Maximum open positions limit reached ({self.risk_limits.max_open_positions})"
        
        # Check daily position limit
        today = datetime.now().strftime("%Y-%m-%d")
        daily_positions = self._get_daily_new_positions(today)
        if len(daily_positions) >= self.risk_limits.max_new_positions_per_day:
            return False, f"Daily new position limit reached ({self.risk_limits.max_new_positions_per_day})"
        
        # Check position size limits
        position_value = quantity * entry_price
        position_size_percent = position_value / self.portfolio_value
        
        if position_size_percent > self.risk_limits.max_position_size:
            return False, f"Position size ({position_size_percent:.2%}) exceeds limit ({self.risk_limits.max_position_size:.2%})"
        
        # Check risk-reward ratio
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return False, "Invalid stop loss - must be different from entry price"
        
        # Check sector exposure (if sector provided)
        # if sector:
        #     sector_exposure = self._calculate_sector_exposure(sector)
        #     if sector_exposure > self.risk_limits.max_sector_exposure:
        #         return False, f"Sector exposure ({sector_exposure:.2%}) exceeds limit ({self.risk_limits.max_sector_exposure:.2%})"
        
        # Check portfolio risk
        new_risk = (quantity * risk_per_share) / self.portfolio_value
        if self.total_portfolio_risk + new_risk > self.risk_limits.max_portfolio_risk:
            return False, f"Portfolio risk would exceed limit ({self.risk_limits.max_portfolio_risk:.2%})"
        
        # Check daily loss limit
        if self.daily_pnl < -self.portfolio_value * self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit exceeded ({self.risk_limits.max_daily_loss:.2%})"
        
        return True, "Position validation passed"


    def add_position(self, 
                    symbol: str, 
                    quantity: int, 
                    entry_price: float,
                    stop_loss: float,
                    target: float,
                    agent_name: str,
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
            agent_name: Name of the agent
            reason: Reason for the trade
            position_type: LONG
            sector: Sector of the stock
            
        Returns:
            Tuple of (success, message)
        """
        # Validate position
        is_valid, validation_message = self.validate_new_position(
            symbol, quantity, entry_price, stop_loss, agent_name, sector
        )
        
        if not is_valid:
            return False, validation_message
        
        # Calculate position metrics
        position_value = quantity * entry_price
        position_size_percent = position_value / self.portfolio_value
        
        # Create position object
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            entry_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agent_name=agent_name,
            position_type=position_type,
            reason=reason,
            position_size_percent=position_size_percent,
            unrealized_pnl=0.0
        )        
        # Add to positions
        self.positions[symbol] = position
        
        # Update portfolio risk
        risk_per_share = abs(entry_price - stop_loss)
        position_risk = (quantity * risk_per_share) / self.portfolio_value
        self.total_portfolio_risk += position_risk
        
        # Save positions
        self._save_positions()
        self._remove_position_from_memory_file(symbol)        
        
        # Cache in Redis
        self._cache_position_in_redis(symbol, position)
        
        self.logger.info(f"Added position: {symbol} x{quantity} @ {entry_price} "
                        f"(Agent: {agent_name}, Risk: {position_risk:.2%})")
        
        return True, f"Position added successfully: {symbol}"


    def update_position_prices(self, price_data: Dict[str, float]) -> Dict[str, float]:
        """
        Update current prices for all positions and calculate P&L
        
        Args:
            price_data: Dictionary of {symbol: current_price}
            
        Returns:
            Dictionary of {symbol: unrealized_pnl}
        """
        updated_pnl = {}
        
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]
                
                # Update current price
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.position_type == "LONG":
                    position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
                else:  # SHORT
                    position.unrealized_pnl = position.quantity * (position.entry_price - current_price)
                
                updated_pnl[symbol] = position.unrealized_pnl
                
                # Update in Redis
                self._cache_position_in_redis(symbol, position)
        
        # Calculate total daily P&L
        self.daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Save updated positions
        self._save_positions()
        
        return updated_pnl


    def check_stop_loss_triggers(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check if any positions have triggered stop losses
        
        Args:
            current_prices: Dictionary of {symbol: current_price}
            
        Returns:
            List of positions that triggered stop losses
        """
        triggered_stops = []
        
        for symbol, position in self.positions.items():
            if position.status != "ACTIVE":
                continue
                
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Check stop loss trigger
            stop_triggered = False
            if position.position_type == "LONG":
                stop_triggered = current_price <= position.stop_loss
            else:  # SHORT
                stop_triggered = current_price >= position.stop_loss
            
            if stop_triggered:
                triggered_stops.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "stop_loss": position.stop_loss,
                    "quantity": position.quantity,
                    "agent_name": position.agent_name,
                    "position_type": position.position_type,
                    "entry_price": position.entry_price,
                    "loss_amount": self._calculate_loss_amount(position, current_price)
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
        
        # Calculate realized P&L
        if position.position_type == "LONG":
            realized_pnl = position.quantity * (execution_price - position.entry_price)
        else:  # SHORT
            realized_pnl = position.quantity * (position.entry_price - execution_price)
        
        # Update position
        position.status = "STOPPED_OUT"
        position.realized_pnl = realized_pnl
        position.current_price = execution_price
        
        # Update portfolio risk
        risk_per_share = abs(position.entry_price - position.stop_loss)
        position_risk = (position.quantity * risk_per_share) / self.portfolio_value
        self.total_portfolio_risk -= position_risk
        
        # Save transaction        
        self._save_closed_position(position)

        # Remove from active positions        
        self._remove_position(symbol)
        self._remove_position_from_memory_file(symbol)

        # Update Redis
        self.redis_client.delete(f"position:{symbol}")
        
        self.logger.info(f"Stop loss executed: {symbol} @ {execution_price} "
                        f"(P&L: {realized_pnl:.2f})")
        
        return True, f"Stop loss executed for {symbol}"
    

    ## Portfolio Management Methods

    def check_portfolio_limits(self) -> Dict[str, any]:
        """
        Check all portfolio limits and return status
        
        Returns:
            Dictionary with limit status
        """
        current_positions = len(self.positions)
        total_portfolio_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        cash_used_percent = total_portfolio_value / self.portfolio_value
        
        # Calculate sector exposures
        # sector_exposures = self._calculate_all_sector_exposures()
        
        # Check daily P&L
        daily_loss_percent = abs(self.daily_pnl) / self.portfolio_value if self.daily_pnl < 0 else 0
        
        status = {
            "positions_count": current_positions,
            "max_positions": self.risk_limits.max_open_positions,
            "positions_limit_ok": current_positions <= self.risk_limits.max_open_positions,
            
            "cash_used_percent": cash_used_percent,
            "portfolio_risk_percent": self.total_portfolio_risk,
            "max_portfolio_risk": self.risk_limits.max_portfolio_risk,
            "portfolio_risk_ok": self.total_portfolio_risk <= self.risk_limits.max_portfolio_risk,
            
            "daily_pnl": self.daily_pnl,
            "daily_loss_percent": daily_loss_percent,
            "max_daily_loss": self.risk_limits.max_daily_loss,
            "daily_loss_ok": daily_loss_percent <= self.risk_limits.max_daily_loss,
            
            # "sector_exposures": sector_exposures,
            "max_sector_exposure": self.risk_limits.max_sector_exposure,
            "emergency_stop_triggered": daily_loss_percent >= self.risk_limits.emergency_stop_loss,
            
            "overall_status": "OK" if all([
                current_positions <= self.risk_limits.max_open_positions,
                self.total_portfolio_risk <= self.risk_limits.max_portfolio_risk,
                daily_loss_percent <= self.risk_limits.max_daily_loss,
                daily_loss_percent < self.risk_limits.emergency_stop_loss
            ]) else "RISK_BREACH"
        }
        
        return status
    
    def calculate_portfolio_risk(self) -> float:
        """
        Calculate total portfolio risk as percentage
        
        Returns:
            Total portfolio risk percentage
        """
        total_risk = 0.0
        
        for position in self.positions.values():
            risk_per_share = abs(position.entry_price - position.stop_loss)
            position_risk = (position.quantity * risk_per_share) / self.portfolio_value
            total_risk += position_risk
        
        return total_risk
    
    def get_max_position_size(self, price: float) -> int:
        """
        Get maximum allowable position size for a symbol
        
        Args:
            symbol: Stock symbol
            price: Current price
            
        Returns:
            Maximum number of shares
        """
        # Maximum by portfolio percentage
        max_by_percent = int((self.portfolio_value * self.risk_limits.max_position_size) / price)
        
        # Maximum by available cash (assuming 75% cash deployment)
        available_cash = self.portfolio_value * 0.75
        current_investment = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        remaining_cash = available_cash - current_investment
        max_by_cash = int(remaining_cash / price) if remaining_cash > 0 else 0
        
        return min(max_by_percent, max_by_cash)
    
    def get_position_summary(self) -> Dict[str, any]:
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
        
        positions_list = []
        for symbol, pos in self.positions.items():
            positions_list.append({
                "symbol": symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "stop_loss": pos.stop_loss,
                "target": pos.target,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_percent": (pos.unrealized_pnl / (pos.quantity * pos.entry_price)) * 100,
                "agent_name": pos.agent_name,
                "entry_date": pos.entry_date,
                "days_held": (datetime.now() - datetime.strptime(pos.entry_date, "%Y-%m-%d %H:%M:%S")).days
            })
        
        return {
            "total_positions": len(self.positions),
            "total_invested": total_invested,
            "total_current_value": total_current_value,
            "total_pnl": total_pnl,
            "pnl_percent": (total_pnl / total_invested) * 100 if total_invested > 0 else 0,
            "portfolio_utilization": (total_current_value / self.portfolio_value) * 100,
            "positions": positions_list
        }
    
    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value (usually after realized P&L)
        
        Args:
            new_value: New portfolio value
        """
        self.portfolio_value = new_value
        self.logger.info(f"Portfolio value updated to: {new_value}")
    
    def get_agent_performance(self) -> Dict[str, Dict]:
        """
        Get performance metrics by agent
        
        Returns:
            Dictionary with agent performance
        """
        agent_performance = {}
        
        # Get closed positions from file
        closed_positions = self._load_closed_positions()
        
        for position in list(self.positions.values()) + closed_positions:
            agent = position.agent_name
            if agent not in agent_performance:
                agent_performance[agent] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pnl": 0,
                    "active_positions": 0
                }
            
            agent_performance[agent]["total_trades"] += 1
            
            if position.status == "ACTIVE":
                agent_performance[agent]["active_positions"] += 1
                pnl = position.unrealized_pnl
            else:
                pnl = position.realized_pnl
            
            agent_performance[agent]["total_pnl"] += pnl
            
            if pnl > 0:
                agent_performance[agent]["winning_trades"] += 1
            elif pnl < 0:
                agent_performance[agent]["losing_trades"] += 1
        
        # Calculate win rates
        for agent, data in agent_performance.items():
            total_closed = data["total_trades"] - data["active_positions"]
            if total_closed > 0:
                data["win_rate"] = data["winning_trades"] / total_closed
            else:
                data["win_rate"] = 0
        
        return agent_performance
    


    # Private helper methods
    def _save_positions(self):
        """Save current positions to database"""        
        for symbol, position in self.positions.items():
            DatabaseQueries.save_position(asdict(position), agent_name=self.agent_name)

    def _save_positions_to_memory_file(self):
        """Save current positions to memory"""
        for symbol, position in self.positions.items():
            memory_tools.add_positions(
                agent_name=self.agent_name,
                positions={
                    symbol: {
                        "entry_price": position.entry_price,
                        "quantity": position.quantity,                        
                        "stop_loss": position.stop_loss,
                        "target": position.target,
                        "reason": position.reason,
                        "entry_date": position.entry_date
                    }
                }
            )


    def _load_positions(self):
        """Load positions from db"""
        positions_data = DatabaseQueries.load_positions(agent_name=self.agent_name)
        self.positions = {
            symbol: Position(**data) for symbol, data in positions_data
        }

        self.total_portfolio_risk = self.calculate_portfolio_risk()
        for position in self.positions.values():            
            self._cache_position_in_redis(position.symbol, position)        


    def _remove_position(self, symbol: str):
        """Remove position from database"""
        for symbol, position in self.positions.items():
            DatabaseQueries.remove_position(self.agent_name, symbol)            

    def _remove_position_from_memory_file(self, symbol: str):
        """Remove position from memory"""
        memory_tools.remove_position(agent_name=self.agent_name, symbol=self.positions[symbol])
        self.positions.pop(symbol, None)



    def _save_closed_position(self, position: Position):
        """Save closed position to trade log"""
        DatabaseQueries.save_closed_position(asdict(position), agent_name=self.agent_name)

    def _load_closed_positions(self) -> List[Position]:
        """Load closed positions from trade log"""
        closed_positions_data = DatabaseQueries.load_closed_positions(agent_name=self.agent_name)
        return [Position(**data) for data in closed_positions_data]



    def _cache_position_in_redis(self, symbol: str, position: Position):
        """Cache position in Redis"""
        try:
            self.redis_client.setex(
                f"position:{self.agent_name}:{symbol}",
                3600,  # 1 hour TTL
                json.dumps(asdict(position))
            )
        except Exception as e:
            self.logger.error(f"Error caching position in Redis: {e}")

    
    def _get_daily_new_positions(self, date: str) -> List[Position]:
        """Get positions opened on a specific date"""
        return [pos for pos in self.positions.values() if pos.entry_date.startswith(date)]


    def _calculate_loss_amount(self, position: Position, current_price: float) -> float:
        """Calculate loss amount for stop loss"""
        if position.position_type == "LONG":
            return position.quantity * (position.entry_price - current_price)
        else:  # SHORT
            return position.quantity * (current_price - position.entry_price)
        

    # def _calculate_sector_exposure(self, sector: str) -> float:
    #     """Calculate exposure to a specific sector"""
    #     # This would need sector data for each position
    #     # For now, return 0 as a placeholder
    #     return 0.0
    
    # def _calculate_all_sector_exposures(self) -> Dict[str, float]:
        # """Calculate exposure to all sectors"""
        # # This would need sector data for each position
        # # For now, return empty dict as a placeholder
        # return {}
    
