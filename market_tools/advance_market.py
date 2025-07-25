import time
from dotenv import load_dotenv
from typing import Dict, List
from datetime import datetime, timedelta
import talib as ta
import numpy as np
import pandas as pd
from utils.dhan_client import dhan


load_dotenv(override=True)

nifty50_security_id = 13
india_vix_security_id = 21

sector_names = {
    25: "BANKNIFTY",
    27: "NIFTY FIN SERVICE",
    28: "NIFTY FMCG",
    29: "NIFTY IT",
    30: "NIFTY MEDIA",
    31: "NIFTY METAL",
    32: "NIFTY PHARMA",
    33: "NIFTY PSU BANK",
    34: "NIFTY REALTY",
    37: "NIFTY MIDCAP 100",
    5024: "GIFT NIFTY"
}

def get_nifty_data() -> Dict:
    """
    Fetches Nifty 50 data from Dhan API.
    Returns: Dictionary with Nifty 50 OHLC data.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    one_year_ago = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    nifty_data = dhan.historical_daily_data(
        security_id=nifty50_security_id, 
        exchange_segment="IDX_I", 
        instrument_type="INDEX", 
        from_date=one_year_ago, 
        to_date=today,
        expiry_code=0
    )

    if not nifty_data or 'data' not in nifty_data:
        return {}
    return nifty_data


def refined_ohlc_data(data_json: dict) -> pd.DataFrame:
    """
    Refine OHLC data from the API response into a DataFrame.
    
    Args:
        data_json (dict): JSON response from the API containing OHLC data.
    
    Returns:
        pd.DataFrame: Descending Sorted DataFrame 
    """
    if not data_json or 'data' not in data_json:
        return pd.DataFrame(columns=['date', 'close'])
    
    df = pd.DataFrame(data_json['data'])
    df['date'] = df['timestamp'].apply(datetime.fromtimestamp)
    df.drop(columns=['timestamp'], inplace=True)

    df_sorted = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df_sorted.reset_index(drop=True, inplace=True)

    return df_sorted


async def get_market_regime() -> Dict:
    """
    Analyzes Nifty vs 20/50/200 EMAs
    Returns: Bull/Bear/Sideways with confidence
    """
    nifty_data = get_nifty_data()
    df_nifty_data = refined_ohlc_data(nifty_data)
    
    # Calculate EMAs and Slopes
    df_nifty_data['ema20'] = ta.EMA(df_nifty_data['close'], 20)    
    df_nifty_data['ema50'] = ta.EMA(df_nifty_data['close'], 50)
    df_nifty_data['ema200'] = ta.EMA(df_nifty_data['close'], 200)

    # Use the most recent day's data
    recent = df_nifty_data.iloc[-1]

    close = recent["close"]
    ema20 = recent["ema20"]
    ema50 = recent["ema50"]
    ema200 = recent["ema200"]
    signals = []

    # Rule 1: Close vs EMAs
    if close > ema20 and ema20 > ema50 and ema50 > ema200:        
        signals.append("Bull")
    elif close < ema20 and ema20 < ema50 and ema50 < ema200:        
        signals.append("Bear")
    else:
        signals.append("Sideways")

    # Rule 2: EMA Slope - simple approximation over last 5 days
    ema20_trend = ema20 - df_nifty_data.iloc[-5]["ema20"]
    ema50_trend = ema50 - df_nifty_data.iloc[-5]["ema50"]
    ema200_trend = ema200 - df_nifty_data.iloc[-5]["ema200"]

    if ema20_trend > 0 and ema50_trend > 0 and ema200_trend > 0:
        signals.append("Bull")
    elif ema20_trend < 0 and ema50_trend < 0 and ema200_trend < 0:
        signals.append("Bear")
    else:
        signals.append("Sideways")

    # Rule 3: Distance from EMA
    ema_gap = (close - ema200) / ema200
    if ema_gap > 0.05:        
        signals.append("Bull")
    elif ema_gap < -0.05:        
        signals.append("Bear")
    else:        
        signals.append("Sideways")
    
    # Majority voting
    bull_votes = signals.count("Bull")
    bear_votes = signals.count("Bear")
    sideways_votes = signals.count("Sideways")

    if bull_votes >= 2:
        regime = "Bull"
        confidence = bull_votes / 3
    elif bear_votes >= 2:
        regime = "Bear"
        confidence = bear_votes / 3
    else:
        regime = "Sideways"
        confidence = sideways_votes / 3    

    return {"regime": regime, "confidence": round(confidence, 2)}


async def get_sector_performance(smooth_days: int = 3) -> Dict:
    """
    Advanced sector performance: percent change, slope, z-score normalized momentum.
    """
    sectors: List[Dict] = get_sector_data().get("sectors", [])        
    if not sectors:
        return {"rankings": []}

    sector_metrics = []

    for sector in sectors:
        name = sector.get("name")
        prices = sector.get("prices", [])        

        if not prices or len(prices) < smooth_days + 1:
            continue

        # Smooth last few days
        prices_np = np.array(prices)
        start_avg = np.mean(prices_np[:smooth_days])
        end_avg = np.mean(prices_np[-smooth_days:])        

        if start_avg <= 0:
            continue

        change_pct = ((end_avg - start_avg) / start_avg) * 100

        # Approximate slope: linear regression over price trend
        x = np.arange(len(prices_np))
        slope = np.polyfit(x, prices_np, 1)[0]

        sector_metrics.append({
            "sector": name,
            "change_pct": round(float(change_pct), 2),
            "slope": round(float(slope), 4)
        })    
    
    if not sector_metrics:
        return {"rankings": []}

    # Z-score normalization on change_pct
    change_array = np.array([s["change_pct"] for s in sector_metrics])
    mean = np.mean(change_array)
    std = np.std(change_array) if np.std(change_array) != 0 else 1.0

    for s in sector_metrics:
        z = float((s["change_pct"] - mean) / std)
        s["z_score"] = round(z, 2)

    # Normalize z-score to 0–1 (min-max)
    z_scores = np.array([s["z_score"] for s in sector_metrics])
    z_min, z_max = np.min(z_scores), np.max(z_scores)

    for s in sector_metrics:
        if z_max == z_min:
            s["momentum_score"] = 1.0
        else:
            norm = float((s["z_score"] - z_min) / (z_max - z_min))
            s["momentum_score"] = round(norm, 2)

    # Sort descending by momentum_score
    sector_metrics.sort(key=lambda x: x["momentum_score"], reverse=True)

    return {"rankings": sector_metrics}


async def get_volatility_regime() -> Dict:
    """
    VIX equivalent analysis for Indian markets.
    Classifies regime as High, Medium, or Low.
    """

    values: List[float] = get_vix_data().get("values", [])

    if not values or len(values) < 30:
        return {
            "volatility_regime": "Unknown",
            "reason": "Insufficient data (<30 entries)"
        }

    # Calculate recent (short-term) average
    recent_window = values[:5]
    recent_avg = float(sum(recent_window) / len(recent_window))

    # Calculate long-term average
    long_term_window = values[:30]
    long_term_avg = float(sum(long_term_window) / len(long_term_window))

    if long_term_avg == 0:
        return {
            "volatility_regime": "Unknown",
            "reason": "Invalid long-term average (0)"
        }

    ratio = round(recent_avg / long_term_avg, 2)

    # Classify regime
    if ratio > 1.1:
        regime = "High"
    elif ratio < 0.9:
        regime = "Low"
    else:
        regime = "Medium"

    return {
        "volatility_regime": regime,
        "recent_avg": round(recent_avg, 2),
        "long_term_avg": round(long_term_avg, 2),
        "ratio": ratio,
        "classification_thresholds": {
            "Low": "<0.9",
            "Medium": "0.9-1.1",
            "High": ">1.1"
        }
    }


# Helper functions

def get_sector_data() -> Dict:
    """
    Fetches sector data from Dhan API and returns it in a structured format.
    """
    sectors = []
    three_months_ago = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")    

    for security_id in sector_names.keys():
        sector_data = dhan.historical_daily_data(
            security_id=security_id, 
            exchange_segment="IDX_I", 
            instrument_type="INDEX", 
            from_date=three_months_ago, 
            to_date=datetime.today().strftime("%Y-%m-%d")
        )
        time.sleep(1.1)  # Avoid hitting API rate limits
        if not sector_data:
            continue
        
        df_sector = refined_ohlc_data(sector_data)        
        if df_sector.empty:
            continue

        sector_name = sector_names.get(security_id, "Unknown Sector")
        prices = df_sector['close'].tolist()
        
        sectors.append({
            "name": sector_name,
            "prices": prices
        })
        
    return {"sectors": sectors}


def get_vix_data() -> Dict:
    """
    Fetches India VIX data from Dhan API.
    Returns: Dictionary with VIX values.
    """

    two_months_ago = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
    vix_data = dhan.historical_daily_data(
        security_id=india_vix_security_id, 
        exchange_segment="IDX_I", 
        instrument_type="INDEX", 
        from_date=two_months_ago, 
        to_date=datetime.today().strftime("%Y-%m-%d"),
        expiry_code=0
    )
    
    if not vix_data or 'data' not in vix_data:
        return {"values": []}
    
    df_vix = refined_ohlc_data(vix_data)
    return {"values": df_vix['close'].tolist()}


