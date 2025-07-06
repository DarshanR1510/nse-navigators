from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

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
class MarketContext:
    """Market context data structure"""
    regime: str  # Bull/Bear/Sideways
    regime_confidence: float
    sector_performance: Dict[str, float]
    volatility_regime: str  # High/Medium/Low        
    timestamp: datetime


@dataclass
class PositionMonitoringResult:
    """Position monitoring result structure"""
    symbol: str
    current_price: float
    entry_price: float
    pnl_percent: float
    pnl_amount: float
    risk_level: str  # Low/Medium/High
    stop_loss_price: Optional[float]
    target_price: Optional[float]
    days_held: int
    action_required: Optional[str]
    reason: Optional[str]


@dataclass
class RiskLimits:
    """Risk management configuration"""
    max_position_size: float = 0.08          # 8% of portfolio per position
    max_sector_exposure: float = 0.25        # 25% in any sector
    max_daily_loss: float = 0.03             # 3% daily loss limit
    max_portfolio_risk: float = 0.15         # 15% total portfolio at risk
    min_risk_reward: float = 1.5             # Minimum 1:1.5 risk-reward
    max_correlation: float = 0.7             # Maximum correlation between positions
    max_open_positions: int = 15             # Maximum open positions
    max_new_positions_per_day: int = 3       # Maximum new positions per day
    risk_per_trade: float = 0.02             # 2% portfolio risk per trade
    emergency_stop_loss: float = 0.08        # 8% daily loss = emergency stop


@dataclass
class RiskReview:
    """Risk review data structure"""
    portfolio_risk: float
    sector_exposure: Dict[str, float]
    position_concentration: Dict[str, float]
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    var_95: float
    risk_violations: List[Dict]
    recommendations: List[str]
    overall_risk_level: str