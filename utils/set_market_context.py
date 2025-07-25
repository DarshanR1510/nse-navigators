import json
from datetime import datetime
import argparse
import os
import redis
from dataclasses import asdict, dataclass
from typing import Dict
import pytz

IST = pytz.timezone('Asia/Kolkata')

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

@dataclass
class MarketContext:
    """Market context data structure"""
    regime: str  # Bull/Bear/Sideways
    regime_confidence: float
    sector_performance: Dict[str, float]
    volatility_regime: str  # High/Medium/Low        
    timestamp: datetime


r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

def get_example_market_context() -> MarketContext:
    """Returns a sample MarketContext object with example data"""
    return MarketContext(
        regime="Bull",  # Bull/Bear/Sideways
        regime_confidence=0.85,
        sector_performance={
            "IT": 2.5,
            "Banking": 1.8,
            "FMCG": -0.5,
            "Auto": 1.2,
            "Pharma": 0.8
        },
        volatility_regime="Medium",  # High/Medium/Low
        timestamp=datetime.now(IST)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set market context in Redis.")
    parser.add_argument("--date", type=str, default=datetime.now(IST).strftime("%Y-%m-%d"), 
                       help="Date in YYYY-MM-DD format")
    parser.add_argument("--regime", type=str, choices=["Bull", "Bear", "Sideways"],
                       help="Market regime")
    parser.add_argument("--confidence", type=float, 
                       help="Regime confidence (0-1)")
    parser.add_argument("--volatility", type=str, choices=["High", "Medium", "Low"],
                       help="Volatility regime")
    parser.add_argument("--sectors", type=str,
                       help="JSON string of sector performance, e.g., '{\"IT\": 2.5, \"Banking\": 1.8}'")
    args = parser.parse_args()

    try:
        if all([args.regime, args.confidence, args.volatility, args.sectors]):
            # Create context from provided arguments
            context = MarketContext(
                regime=args.regime,
                regime_confidence=args.confidence,
                sector_performance=json.loads(args.sectors),
                volatility_regime=args.volatility,
                timestamp=datetime.now(IST)
            )
        else:
            # Use example context
            context = get_example_market_context()

        # Convert to dict and store in Redis
        key = f"market:daily_context:{args.date}"
        context_dict = asdict(context)
        # Convert datetime to string for JSON serialization
        context_dict['timestamp'] = context.timestamp.strftime("%Y-%m-%d")

        r.set(key, json.dumps(context_dict))
        print(f"Market context set for {args.date} in Redis under key '{key}'.")
        print(json.dumps(context_dict, indent=2))

    except Exception as e:
        print(f"Error setting market context: {e}")
        exit(1)