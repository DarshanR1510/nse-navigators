from typing import Dict, Tuple, List
from market_tools.market import get_symbol_history_daily_data
from datetime import datetime, timedelta
from scipy.signal import find_peaks, argrelextrema
from market_tools.advance_market import refined_ohlc_data, get_nifty_data
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import pandas as pd
import talib as ta
import numpy as np
import time



def closing_sma_and_slope(symbol: str, from_date: str, to_date: str):
    """ Calculate Simple Moving Averages (SMA) for a given symbol.
        To get bigger SMA values, you need to provide a longer date range.
        This function calculates SMA and it's slope for 20, 50, and 200 days.
    Args:
        symbol (str): Stock symbol
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
    Returns:
        dict: A dictionary containing the SMA values for 20, 50, and 200 days, and their slopes.
        Returns None if no data is available.
    """
    data = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    df['sma_20'] = ta.SMA(df['close'], 20)
    df['sma_50'] = ta.SMA(df['close'], 50)
    df['sma_200'] = ta.SMA(df['close'], 200)

    df['sma_20_slope'] = df['sma_20'].diff()
    df['sma_50_slope'] = df['sma_50'].diff()
    df['sma_200_slope'] = df['sma_200'].diff()

    return {
        "sma": [df['sma_20'], df['sma_50'], df['sma_200']],
        "slope": [df['sma_20_slope'], df['sma_50_slope'], df['sma_200_slope']]
    }


def closing_ema_and_slope(symbol: str, from_date: str, to_date: str) -> float:
    """ Calculate Exponential Moving Averages (EMA) for a given symbol.
        To get bigger EMA values, you need to provide a longer date range.
        This function calculates EMA and it's slope for 20, 50, and 200 days.
    Args:
        symbol (str): Stock symbol
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
    Returns:
        dict: A dictionary containing the EMA values for the specified window, and their slopes.
        Returns None if no data is available.
    """
    data = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    df['ema_20'] = ta.EMA(df['close'], window=20)
    df['ema_50'] = ta.EMA(df['close'], window=50)
    df['ema_200'] = ta.EMA(df['close'], window=200)

    df['ema_20_slope'] = df['ema_20'].diff()
    df['ema_50_slope'] = df['ema_50'].diff()
    df['ema_200_slope'] = df['ema_200'].diff()

    return {
        "ema": [df['ema_20'], df['ema_50'], df['ema_200']],
        "slope": [df['ema_20_slope'], df['ema_50_slope'], df['ema_200_slope']]
    }


def closing_macd(symbol: str, from_date: str, to_date: str, 
                         fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Calculate MACD using symbol data from your existing functions.
    
    Args:
        symbol (str): Stock symbol
        exchange (str): Exchange name
        from_date (str): Start date
        to_date (str): End date
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)
        signal_period (int): Signal line EMA period (default: 9)
    
    Returns:
        dict: MACD results or None if insufficient data
    """
    # Get OHLC data using your existing function
    ohlc_data = get_symbol_history_daily_data(symbol, from_date, to_date)
    data = refined_ohlc_data(ohlc_data)

    if data.empty:
        return None

    macd_data = ta.MACD(data['close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    return macd_data


def closing_rsi(symbol: str, from_date: str, to_date: str, period: int = 14) -> float:
    """
    Calculate RSI using symbol data from your existing data functions.
    
    Args:
        symbol (str): Stock symbol
        exchange (str): Exchange name  
        from_date (str): Start date
        to_date (str): End date
        period (int): RSI period (default: 14)
    
    Returns:
        float: RSI value (0-100), or None if insufficient data
    """
    # Get OHLC data using your existing function
    ohlc_data = get_symbol_history_daily_data(symbol, from_date, to_date)
    data = refined_ohlc_data(ohlc_data)

    if data.empty:
        return None

    rsi_data = ta.RSI(data['close'], timeperiod=period)
    return rsi_data


def closing_bollinger_bands(symbol: str, from_date: str, to_date: str, period: int = 20, std_dev: float = 2.0) -> dict:
    """
    Calculate Bollinger Bands using symbol data from your existing functions.
    
    Args:
        symbol (str): Stock symbol
        from_date (str): Start date
        to_date (str): End date
        period (int): Bollinger Bands period (default: 20)
        std_dev (float): Standard deviation multiplier (default: 2.0)
    
    Returns:
        dict: Bollinger Bands results or None if insufficient data
    """
    # Get OHLC data using your existing function
    ohlc_data = get_symbol_history_daily_data(symbol, from_date, to_date)
    data = refined_ohlc_data(ohlc_data)

    if data.empty:
        return None

    upper_band, middle_band, lower_band = ta.BBANDS(data['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    
    return {
        "upper_band": upper_band,
        "middle_band": middle_band,
        "lower_band": lower_band
    }


### ADVANCED TOOLS ###

def analyze_volume_patterns(symbol: str, from_date: str, to_date: str) -> Dict:
    """
    Performs advanced, multi-factor analysis of volume and price action to detect key volume-related signals and their strength.

    Analysis includes:
      - Volume moving averages (5, 10, 20, 50 days)
      - Price change and volatility (ATR, rolling std)
      - Volume ratio and velocity
      - Breakout confirmation (volume and price surge)
      - Volume dry-up (declining volume and volatility)
      - Climax volume (unusually high volume spikes)
      - Accumulation/Distribution (A/D line with slope and consistency)
      - On-Balance Volume (OBV) trend and slope
      - Volume-price correlation (Pearson correlation)
      - Volume momentum oscillator
      - Volume breakout pattern recognition (peaks with price action)
      - Weighted scoring and confidence estimation

    Returns:
        dict: {
            "symbol": str,
            "signals": List[Dict],              # Each signal with type, score, details, and weight
            "overall_volume_score": float,      # Weighted average score (0-1)
            "confidence_percentage": float,     # Confidence in the analysis (0-100)
            "volume_trend": str,                # "Increasing" or "Decreasing"
            "volume_strength": str,             # "High", "Normal", or "Low"
            "current_volume": int,
            "average_volume_20d": int,
            "volume_ratio": float,              # Current/average volume ratio
            "details": str                      # Human-readable summary
        }
    Notes:
        - Requires at least 50 data points for robust analysis.
        - Combines technical, statistical, and price/volume action methods for high-confidence volume signal detection.
        - Each signal is scored and weighted for overall assessment.
    """
     
    df = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(df)

    if df is None or len(df) < 50:  # Increased minimum data requirement
        return {"symbol": symbol, "signals": [], "overall_volume_score": 0.0, "details": "Insufficient data (minimum 50 periods required)"}

    df = df.copy()
    
    # Enhanced Volume Moving Averages
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma10'] = df['volume'].rolling(10).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ma50'] = df['volume'].rolling(50).mean()
    
    # Price changes and volatility
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = abs(df['price_change'])
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volatility'] = df['price_change_abs'].rolling(20).std()
    
    # Volume indicators
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    df['volume_velocity'] = df['volume'].pct_change()
    
    # Enhanced calculations
    last_vol = df['volume'].iloc[-1]
    avg_vol_20 = df['volume_ma20'].iloc[-1]
    avg_vol_50 = df['volume_ma50'].iloc[-1]
    
    # 1. Enhanced Breakout Confirmation Signal
    price_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
    breakout_volume_ratio = last_vol / avg_vol_20 if avg_vol_20 else 0
    
    # Check for sustained volume increase over 3 days
    recent_vol_sustained = df['volume'][-3:].mean() / avg_vol_20
    atr_normalized_move = abs(price_change_pct) / (df['atr'].iloc[-1] / df['close'].iloc[-1] * 100)
    
    breakout_signal = (breakout_volume_ratio > 1.5 and 
                      abs(price_change_pct) > 1.0 and 
                      recent_vol_sustained > 1.2 and
                      atr_normalized_move > 0.8)
    
    breakout_score = min((breakout_volume_ratio * atr_normalized_move * recent_vol_sustained) / 4.0, 1.0) if breakout_signal else 0.0
    breakout_details = f"Price: {price_change_pct:.2f}%; Vol ratio: {breakout_volume_ratio:.2f}; Sustained: {recent_vol_sustained:.2f}"

    # 2. Volume Dry-Up Detection (Enhanced)
    recent_vol_5 = df['volume'][-5:].mean()
    recent_vol_10 = df['volume'][-10:].mean()
    vol_decline_trend = recent_vol_5 / recent_vol_10 if recent_vol_10 else 1.0
    
    vdu_signal = (recent_vol_5 < 0.7 * avg_vol_20 and 
                  vol_decline_trend < 0.9 and 
                  df['volatility'].iloc[-1] < df['volatility'].iloc[-10])
    
    vdu_intensity = max(0, 1 - (recent_vol_5 / (0.7 * avg_vol_20)))
    vdu_score = min(vdu_intensity * (1 - vol_decline_trend), 1.0) if vdu_signal else 0.0
    vdu_details = f"Recent 5d avg: {recent_vol_5:.0f}, Threshold: {0.7*avg_vol_20:.0f}, Decline: {vol_decline_trend:.2f}"

    # 3. Climax Volume Detection (Enhanced)
    volume_percentile_90 = np.percentile(df['volume'][-50:], 90)
    volume_percentile_95 = np.percentile(df['volume'][-50:], 95)
    
    climax_signal = (last_vol >= volume_percentile_90 and 
                    abs(price_change_pct) > 2.0)
    
    climax_intensity = last_vol / volume_percentile_95 if volume_percentile_95 else 0
    climax_score = min(climax_intensity, 1.0) if climax_signal else 0.0
    climax_details = f"Volume: {last_vol:.0f}, 90th percentile: {volume_percentile_90:.0f}, Intensity: {climax_intensity:.2f}"

    # 4. Enhanced Accumulation/Distribution with Volume
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-6)
    df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
    
    # A/D Line with exponential smoothing
    ad_line = df['money_flow_volume'].ewm(span=14).mean().cumsum()
    ad_slope_20 = np.polyfit(range(20), ad_line[-20:], 1)[0]
    ad_slope_10 = np.polyfit(range(10), ad_line[-10:], 1)[0]
    
    ad_consistency = 1.0 if ad_slope_20 > 0 and ad_slope_10 > 0 else 0.5 if ad_slope_10 > 0 else 0.0
    ad_score = min(abs(ad_slope_20) / 1000000, 1.0) * ad_consistency  # Normalized slope
    ad_details = f"AD 20d slope: {ad_slope_20:.0f}, 10d slope: {ad_slope_10:.0f}, Consistency: {ad_consistency:.1f}"

    # 5. Enhanced OBV with Trend Strength
    obv = ta.OBV(df['close'], df['volume'])
    obv_ma = pd.Series(obv).rolling(20).mean()
    obv_slope_20 = np.polyfit(range(20), obv[-20:], 1)[0]
    obv_slope_10 = np.polyfit(range(10), obv[-10:], 1)[0]
    
    # OBV trend consistency
    obv_consistency = 1.0 if obv_slope_20 > 0 and obv_slope_10 > 0 else 0.5 if obv_slope_10 > 0 else 0.0
    obv_score = min(abs(obv_slope_20) / 1000000, 1.0) * obv_consistency
    obv_details = f"OBV 20d slope: {obv_slope_20:.0f}, 10d slope: {obv_slope_10:.0f}, Consistency: {obv_consistency:.1f}"

    # 6. Volume-Price Correlation Analysis
    correlation, p_value = pearsonr(df['volume'][-20:], df['price_change_abs'][-20:])
    correlation_strength = abs(correlation) if p_value < 0.05 else 0.0
    correlation_score = correlation_strength if correlation > 0 else 0.0
    correlation_details = f"Correlation: {correlation:.3f}, P-value: {p_value:.3f}, Strength: {correlation_strength:.3f}"

    # 7. Volume Momentum Oscillator
    volume_momentum = (df['volume_ma5'].iloc[-1] - df['volume_ma20'].iloc[-1]) / df['volume_ma20'].iloc[-1]
    momentum_signal = volume_momentum > 0.1
    momentum_score = min(abs(volume_momentum) * 2, 1.0) if momentum_signal else 0.0
    momentum_details = f"Volume momentum: {volume_momentum:.3f}, Signal: {momentum_signal}"

    # 8. Volume Breakout Pattern Recognition
    # Find volume peaks and validate with price action
    volume_peaks, _ = find_peaks(df['volume'][-30:], height=avg_vol_20 * 1.5)
    recent_peaks = len(volume_peaks) > 0 and volume_peaks[-1] >= 25  # Recent peak
    
    pattern_score = 0.0
    if recent_peaks:
        peak_volume = df['volume'].iloc[-(30-volume_peaks[-1])]
        peak_price_change = abs(df['price_change'].iloc[-(30-volume_peaks[-1])] * 100)
        pattern_score = min((peak_volume / avg_vol_20) * (peak_price_change / 2.0) / 4.0, 1.0)
    
    pattern_details = f"Recent volume peaks: {len(volume_peaks)}, Pattern strength: {pattern_score:.3f}"

    # Combine all signals with enhanced weighting
    signals = [
        {"type": "Breakout Confirmation", "score": float(round(breakout_score, 3)), "details": breakout_details, "weight": 0.20},
        {"type": "Volume Dry-Up", "score": float(round(vdu_score, 3)), "details": vdu_details, "weight": 0.15},
        {"type": "Climax Volume", "score": float(round(climax_score, 3)), "details": climax_details, "weight": 0.15},
        {"type": "Accumulation/Distribution", "score": float(round(ad_score, 3)), "details": ad_details, "weight": 0.15},
        {"type": "OBV Trend", "score": float(round(obv_score, 3)), "details": obv_details, "weight": 0.15},
        {"type": "Volume-Price Correlation", "score": float(round(correlation_score, 3)), "details": correlation_details, "weight": 0.10},
        {"type": "Volume Momentum", "score": float(round(momentum_score, 3)), "details": momentum_details, "weight": 0.05},
        {"type": "Volume Pattern", "score": float(round(pattern_score, 3)), "details": pattern_details, "weight": 0.05}
    ]

    # Weighted average score
    weighted_score = sum(signal["score"] * signal["weight"] for signal in signals)
    
    # Calculate confidence based on data quality and signal consistency
    signal_consistency = len([s for s in signals if s["score"] > 0.3]) / len(signals)
    data_quality = min(len(df) / 100, 1.0)  # Normalize data length
    confidence = (signal_consistency * 0.7 + data_quality * 0.3) * 100
    
    # Volume trend classification
    volume_trend = "Increasing" if df['volume_ma5'].iloc[-1] > df['volume_ma20'].iloc[-1] else "Decreasing"
    volume_strength = "High" if last_vol > avg_vol_20 * 1.5 else "Normal" if last_vol > avg_vol_20 * 0.8 else "Low"
    
    return {
        "symbol": symbol,
        "signals": signals,
        "overall_volume_score": float(round(weighted_score, 3)),
        "confidence_percentage": float(round(confidence, 1)),
        "volume_trend": volume_trend,
        "volume_strength": volume_strength,
        "current_volume": int(last_vol),
        "average_volume_20d": int(avg_vol_20),
        "volume_ratio": float(round(last_vol / avg_vol_20, 2)),
        "details": f"Enhanced volume analysis over {len(df)} periods with {len(signals)} indicators and {confidence:.1f}% confidence."
    }


def calculate_relative_strength(symbol: str) -> Dict:
    """
    Calculate comprehensive relative strength of a stock vs benchmark index.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'SWIGGY', 'NTPCGREEN')
    benchmark : str
        Benchmark index symbol (default: NIFTY)
        Other options: 'BANKNIFTY' (Bank Nifty)
    
    Returns:
    --------
    Dict: Comprehensive relative strength analysis
    """

    
    try:
        # Download data for multiple timeframes
        one_year_ago = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        today = datetime.today().strftime("%Y-%m-%d")
        
        nifty_data = get_nifty_data()

        stock_df = get_symbol_history_daily_data(symbol, one_year_ago, today)
        stock_df = refined_ohlc_data(stock_df)

        benchmark_df = refined_ohlc_data(nifty_data)

        if stock_df.empty or benchmark_df.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Align data by dates
        common_dates = stock_df.index.intersection(benchmark_df.index)
        if len(common_dates) < 50:
            return {"error": "Insufficient data for reliable calculation"}

        stock_prices = stock_df.loc[common_dates, 'close']
        benchmark_prices = benchmark_df.loc[common_dates, 'close']
        
        # Calculate returns for different periods
        periods = {
            '1D': 1,
            '1W': 5,
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '1Y': 252
        }
        
        results = {
            'symbol': symbol,
            'benchmark': 'NIFTY',
            'current_date': today,
            'data_points': len(common_dates),
            'period_returns': {},
            'relative_performance': {},
            'rs_rating': 0,
            'rs_trend': 'Neutral',
            'volatility_analysis': {},
            'momentum_indicators': {},
            'summary': {}
        }
        
        # Calculate period returns and relative performance
        for period, days in periods.items():
            if len(stock_prices) >= days:
                # Stock returns
                stock_return = ((stock_prices.iloc[-1] / stock_prices.iloc[-days]) - 1) * 100
                benchmark_return = ((benchmark_prices.iloc[-1] / benchmark_prices.iloc[-days]) - 1) * 100
                
                results['period_returns'][period] = {
                    'stock_return': float(round(stock_return, 2)),
                    'benchmark_return': float(round(benchmark_return, 2)),
                    'outperformance': float(round(stock_return - benchmark_return, 2))
                }
                
                # Relative performance ratio
                relative_perf = stock_return / benchmark_return if benchmark_return != 0 else 0
                results['relative_performance'][period] = float(round(relative_perf, 3))
        
        # Calculate RS Rating (0-100 scale based on relative performance)
        rs_scores = []
        weights = {'1M': 0.4, '3M': 0.3, '6M': 0.2, '1Y': 0.1}  # More weight to recent performance
        
        for period, weight in weights.items():
            if period in results['relative_performance']:
                rel_perf = results['relative_performance'][period]
                # Convert to percentile-like score
                if rel_perf > 1.5:  # 50% outperformance
                    score = 95
                elif rel_perf > 1.2:  # 20% outperformance
                    score = 80
                elif rel_perf > 1.1:  # 10% outperformance
                    score = 70
                elif rel_perf > 1.0:  # Any outperformance
                    score = 60
                elif rel_perf > 0.9:  # Less than 10% underperformance
                    score = 40
                elif rel_perf > 0.8:  # Less than 20% underperformance
                    score = 25
                else:  # Significant underperformance
                    score = 10
                
                rs_scores.append(score * weight)
        
        results['rs_rating'] = float(round(sum(rs_scores), 1))
        
        # Determine RS Trend
        if results['rs_rating'] >= 80:
            results['rs_trend'] = 'Very Strong'
        elif results['rs_rating'] >= 60:
            results['rs_trend'] = 'Strong'
        elif results['rs_rating'] >= 40:
            results['rs_trend'] = 'Moderate'
        elif results['rs_rating'] >= 20:
            results['rs_trend'] = 'Weak'
        else:
            results['rs_trend'] = 'Very Weak'
        
        # Calculate volatility metrics
        stock_daily_returns = stock_prices.pct_change().dropna()
        benchmark_daily_returns = benchmark_prices.pct_change().dropna()
        
        results['volatility_analysis'] = {
            'stock_volatility': float(round(stock_daily_returns.std() * np.sqrt(252) * 100, 2)),
            'benchmark_volatility': float(round(benchmark_daily_returns.std() * np.sqrt(252) * 100, 2)),
            'beta': float(round(np.cov(stock_daily_returns, benchmark_daily_returns)[0][1] / 
                            np.var(benchmark_daily_returns), 3)),
            'correlation': float(round(np.corrcoef(stock_daily_returns, benchmark_daily_returns)[0][1], 3))
        }
        
        # Calculate momentum indicators
        # Price momentum (rate of change)
        price_momentum_20 = ((stock_prices.iloc[-1] / stock_prices.iloc[-20]) - 1) * 100
        price_momentum_50 = ((stock_prices.iloc[-1] / stock_prices.iloc[-50]) - 1) * 100
        
        # Simple moving averages
        sma_20 = stock_prices.rolling(20).mean().iloc[-1]
        sma_50 = stock_prices.rolling(50).mean().iloc[-1]
        current_price = stock_prices.iloc[-1]
        
        results['momentum_indicators'] = {
            'price_momentum_20d': float(round(price_momentum_20, 2)),
            'price_momentum_50d': float(round(price_momentum_50, 2)),
            'price_vs_sma20': float(round(((current_price / sma_20) - 1) * 100, 2)),
            'price_vs_sma50': float(round(((current_price / sma_50) - 1) * 100, 2)),
            'sma20_vs_sma50': float(round(((sma_20 / sma_50) - 1) * 100, 2))
        }
        
        # Generate summary
        recent_outperformance = results['period_returns'].get('1M', {}).get('outperformance', 0)
        results['summary'] = {
            'overall_assessment': results['rs_trend'],
            'recent_performance': 'Outperforming' if recent_outperformance > 0 else 'Underperforming',
            'volatility_profile': 'High' if results['volatility_analysis']['stock_volatility'] > 30 else 
                                    'Medium' if results['volatility_analysis']['stock_volatility'] > 15 else 'Low',
            'market_sensitivity': 'High' if results['volatility_analysis']['beta'] > 1.2 else 
                                    'Medium' if results['volatility_analysis']['beta'] > 0.8 else 'Low'
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Error calculating relative strength: {str(e)}"}


def detect_breakout_patterns(symbol: str, from_date: str, to_date: str) -> Dict:
    """
    Detects advanced breakout patterns (flags, pennants, triangles, squeezes, channels, etc.) for a given symbol over a specified date range using a multi-factor, multi-method approach.

    Analysis includes:
      - Multi-timeframe moving averages (SMA/EMA)
      - Volatility indicators (ATR, Bollinger Bands, BB squeeze, ATR%)
      - Momentum indicators (RSI, MACD, Stochastic, Williams %R)
      - Volume indicators (OBV, AD, volume spikes, volume ratios)
      - Price action (body %, shadows, true range)
      - Support/resistance detection (rolling highs/lows)
      - Consolidation pattern analysis (range, coefficient of variation, BB squeeze, volume/volatility compression)
      - Breakout candle analysis (volume, body %, momentum, direction)
      - Trend context (short/medium/long-term, RSI/MACD confirmation)
      - Pattern probability scoring (weighted by consolidation, breakout, trend, volume)
      - Pattern identification (flag, pennant, triangle, channel, squeeze, continuation, or none)
      - Risk assessment (based on probability)
      - Detailed summary and key statistics

    Returns:
        dict: {
            "symbol": str,
            "current_price": float, 
            "pattern_type": str,           # Detected pattern (e.g., "Bullish Flag/Pennant Breakout")
            "breakout_probability": float, # Probability score (0-1)
            "direction": str,              # "Bullish", "Bearish", or "Neutral"
            "risk_level": str,             # "High", "Medium", "Low"
            "trend_strength": int,         # 0-3 (multi-timeframe trend)
            "consolidation_confirmed": bool,
            "breakout_confirmed": bool,
            "volume_confirmation": bool,
            "details": dict,               # Key stats and sub-scores
            "summary": str                 # Human-readable summary
        }
    Notes:
        - Requires at least 100 data points for robust pattern detection.
        - Uses technical, statistical, and price/volume action methods for high-confidence breakout analysis.
        - Probability and risk are composite scores based on multiple factors.
    """
    
    # Get data (assuming these functions exist)
    df = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(df)
    
    if df is None or len(df) < 100:  # Increased minimum data requirement
        return {"pattern_type": "Unknown", "breakout_probability": 0.0, "details": "Insufficient data"}
    
    df = df.copy()
    
    # ===== ENHANCED TECHNICAL INDICATORS =====
    
    # Moving Averages (Multiple timeframes)
    df['sma10'] = ta.SMA(df['close'], timeperiod=10)
    df['sma20'] = ta.SMA(df['close'], timeperiod=20)
    df['sma50'] = ta.SMA(df['close'], timeperiod=50)
    df['ema9'] = ta.EMA(df['close'], timeperiod=9)
    df['ema20'] = ta.EMA(df['close'], timeperiod=20)
    df['ema50'] = ta.EMA(df['close'], timeperiod=50)
    

    # Volatility indicators
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle * 100
    df['bb_position'] = (df['close'] - lower) / (upper - lower)
    

    # Momentum indicators
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'])
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
    df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    

    # Volume indicators
    df['volume_sma10'] = df['volume'].rolling(10).mean()
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_sma50'] = df['volume'].rolling(50).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    df['obv'] = ta.OBV(df['close'], df['volume'])
    df['ad'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    

    # Price action indicators
    df['body'] = abs(df['close'] - df['open'])
    df['body_pct'] = (df['body'] / df['open']) * 100
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['total_range'] = df['high'] - df['low']
    df['true_range'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                          abs(df['low'] - df['close'].shift(1))))
    
    # ===== PATTERN DETECTION LOGIC =====
    
    def detect_support_resistance_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[List[float], List[float]]:
        """Detect dynamic support and resistance levels"""
        highs = df['high'].rolling(window=lookback, center=True).max()
        lows = df['low'].rolling(window=lookback, center=True).min()
        
        # Find peaks and troughs
        resistance_levels = []
        support_levels = []
        
        for i in range(lookback, len(df) - lookback):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(float(x) for x in resistance_levels)), reverse=True)
        support_levels = sorted(list(set(float(x) for x in support_levels)))        
        
        return resistance_levels[:5], support_levels[:5]  # Top 5 levels
    
    def analyze_consolidation_pattern(df: pd.DataFrame, period: int = 20) -> Dict:
        """Advanced consolidation analysis"""
        recent_df = df.iloc[-period:]
        
        # Statistical measures
        price_std = recent_df['close'].std()
        price_mean = recent_df['close'].mean()
        coefficient_of_variation = price_std / price_mean
        
        # Range analysis
        range_high = recent_df['high'].max()
        range_low = recent_df['low'].min()
        range_spread = (range_high - range_low) / range_low
        
        # Bollinger Band squeeze
        bb_squeeze = recent_df['bb_width'].mean() < df['bb_width'].rolling(50).mean().iloc[-1]
        
        # Volume during consolidation
        volume_decline = recent_df['volume'].mean() < df['volume'].rolling(50).mean().iloc[-1]
        
        # Volatility compression
        volatility_compression = recent_df['atr_pct'].mean() < df['atr_pct'].rolling(50).mean().iloc[-1]
        
        return {
            'is_consolidating': range_spread < 0.08 and coefficient_of_variation < 0.05,
            'consolidation_strength': 1 - min(range_spread / 0.08, 1.0),
            'bb_squeeze': bb_squeeze,
            'volume_decline': volume_decline,
            'volatility_compression': volatility_compression,
            'range_spread': range_spread,
            'coefficient_of_variation': coefficient_of_variation
        }
    
    def detect_breakout_candle(df: pd.DataFrame, resistance_levels: List[float], support_levels: List[float]) -> Dict:
        """Enhanced breakout detection"""
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Volume confirmation
        volume_spike = last_candle['volume'] > 1.5 * last_candle['volume_sma20']
        volume_above_average = last_candle['volume'] > df['volume'].rolling(50).mean().iloc[-1]
        
        # Price action confirmation
        strong_body = last_candle['body_pct'] > df['body_pct'].rolling(20).mean().iloc[-1] * 1.5
        momentum_breakout = abs(last_candle['close'] - prev_candle['close']) / prev_candle['close'] > 0.02
        
        # Breakout direction and strength
        breakout_up = False
        breakout_down = False
        breakout_strength = 0
        
        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - last_candle['close']))
            if last_candle['close'] > nearest_resistance and last_candle['high'] > nearest_resistance:
                breakout_up = True
                breakout_strength = (last_candle['close'] - nearest_resistance) / nearest_resistance
        
        if support_levels:
            nearest_support = min(support_levels, key=lambda x: abs(x - last_candle['close']))
            if last_candle['close'] < nearest_support and last_candle['low'] < nearest_support:
                breakout_down = True
                breakout_strength = (nearest_support - last_candle['close']) / nearest_support
        
        return {
            'breakout_up': breakout_up,
            'breakout_down': breakout_down,
            'breakout_strength': breakout_strength,
            'volume_spike': volume_spike,
            'volume_above_average': volume_above_average,
            'strong_body': strong_body,
            'momentum_breakout': momentum_breakout
        }
    
    def analyze_trend_context(df: pd.DataFrame) -> Dict:
        """Comprehensive trend analysis"""
        last_close = df['close'].iloc[-1]
        
        # Multi-timeframe trend
        short_term_trend = last_close > df['ema9'].iloc[-1] and df['ema9'].iloc[-1] > df['ema20'].iloc[-1]
        medium_term_trend = df['ema20'].iloc[-1] > df['ema50'].iloc[-1]
        long_term_trend = df['sma20'].iloc[-1] > df['sma50'].iloc[-1]
        
        # Trend strength
        trend_strength = 0
        if short_term_trend and medium_term_trend and long_term_trend:
            trend_strength = 3
        elif (short_term_trend and medium_term_trend) or (medium_term_trend and long_term_trend):
            trend_strength = 2
        elif short_term_trend or medium_term_trend or long_term_trend:
            trend_strength = 1
        
        # Momentum confirmation
        rsi_bullish = 45 < df['rsi'].iloc[-1] < 75
        macd_bullish = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
        
        return {
            'trend_strength': trend_strength,
            'short_term_bullish': short_term_trend,
            'medium_term_bullish': medium_term_trend,
            'long_term_bullish': long_term_trend,
            'rsi_favorable': rsi_bullish,
            'macd_bullish': macd_bullish
        }
    
    def calculate_pattern_probability(consolidation: Dict, breakout: Dict, trend: Dict) -> float:
        """Advanced probability calculation using weighted scoring"""
        
        # Consolidation score (0-25 points)
        consolidation_score = 0
        if consolidation['is_consolidating']:
            consolidation_score += 10
        if consolidation['bb_squeeze']:
            consolidation_score += 5
        if consolidation['volume_decline']:
            consolidation_score += 5
        if consolidation['volatility_compression']:
            consolidation_score += 5
        
        # Breakout score (0-35 points)
        breakout_score = 0
        if breakout['breakout_up'] or breakout['breakout_down']:
            breakout_score += 15
        if breakout['volume_spike']:
            breakout_score += 10
        if breakout['strong_body']:
            breakout_score += 5
        if breakout['momentum_breakout']:
            breakout_score += 5
        
        # Trend score (0-25 points)
        trend_score = trend['trend_strength'] * 5
        if trend['rsi_favorable']:
            trend_score += 5
        if trend['macd_bullish']:
            trend_score += 5
        
        # Volume score (0-15 points)
        volume_score = 0
        if breakout['volume_above_average']:
            volume_score += 8
        if breakout['volume_spike']:
            volume_score += 7
        
        total_score = consolidation_score + breakout_score + trend_score + volume_score
        probability = min(total_score / 100, 1.0)
        
        return probability
    
    def identify_specific_pattern(df: pd.DataFrame, consolidation: Dict, breakout: Dict, trend: Dict) -> str:
        """Identify specific chart patterns"""
        recent_df = df.iloc[-30:]
        
        # Flag/Pennant patterns
        if consolidation['is_consolidating'] and trend['trend_strength'] >= 2:
            if breakout['breakout_up']:
                return "Bullish Flag/Pennant Breakout"
            elif breakout['breakout_down']:
                return "Bearish Flag/Pennant Breakdown"
            else:
                return "Flag/Pennant Formation"
        
        # Triangle patterns
        if consolidation['volatility_compression'] and consolidation['bb_squeeze']:
            if breakout['breakout_up']:
                return "Ascending Triangle Breakout"
            elif breakout['breakout_down']:
                return "Descending Triangle Breakdown"
            else:
                return "Triangle Formation"
        
        # Rectangle/Channel patterns
        if consolidation['is_consolidating'] and not consolidation['volatility_compression']:
            if breakout['breakout_up']:
                return "Rectangle/Channel Breakout"
            elif breakout['breakout_down']:
                return "Rectangle/Channel Breakdown"
            else:
                return "Rectangle/Channel Formation"
        
        # Squeeze patterns
        if consolidation['bb_squeeze'] and consolidation['volume_decline']:
            return "Bollinger Band Squeeze"
        
        # Continuation patterns
        if trend['trend_strength'] >= 2 and consolidation['is_consolidating']:
            return "Continuation Pattern"
        
        return "No Clear Pattern"
    
    # ===== MAIN ANALYSIS =====
    
    # Get support/resistance levels
    resistance_levels, support_levels = detect_support_resistance_levels(df)
    
    # Analyze consolidation
    consolidation_analysis = analyze_consolidation_pattern(df)
    
    # Detect breakout
    breakout_analysis = detect_breakout_candle(df, resistance_levels, support_levels)
    
    # Analyze trend context
    trend_analysis = analyze_trend_context(df)
    
    # Calculate probability
    probability = calculate_pattern_probability(consolidation_analysis, breakout_analysis, trend_analysis)
    
    # Identify pattern
    pattern_type = identify_specific_pattern(df, consolidation_analysis, breakout_analysis, trend_analysis)
    
    # Determine direction
    direction = "Bullish" if breakout_analysis['breakout_up'] else "Bearish" if breakout_analysis['breakout_down'] else "Neutral"
    
    # Risk assessment
    risk_level = "High" if probability < 0.3 else "Medium" if probability < 0.7 else "Low"
    
    # Generate detailed analysis
    details = {
        'consolidation_strength': float(round(consolidation_analysis['consolidation_strength'], 3)),
        'range_spread_pct': float(round(consolidation_analysis['range_spread'] * 100, 2)),
        'bb_squeeze': consolidation_analysis['bb_squeeze'],
        'volume_spike': breakout_analysis['volume_spike'],
        'breakout_strength': float(round(breakout_analysis['breakout_strength'], 3)),
        'trend_strength': trend_analysis['trend_strength'],
        'support_levels': support_levels[:3],
        'resistance_levels': resistance_levels[:3],
        'rsi_current': float(round(df['rsi'].iloc[-1], 2)),
        'volume_ratio': float(round(df['volume_ratio'].iloc[-1], 2))
    }
    
    return {
        "symbol": symbol,
        "current_price": round(float(df['close'].iloc[-1])),
        "pattern_type": pattern_type,
        "breakout_probability": float(round(probability, 3)),
        "direction": direction,
        "risk_level": risk_level,
        "trend_strength": trend_analysis['trend_strength'],
        "consolidation_confirmed": consolidation_analysis['is_consolidating'],
        "breakout_confirmed": breakout_analysis['breakout_up'] or breakout_analysis['breakout_down'],
        "volume_confirmation": breakout_analysis['volume_spike'],
        "details": details,
        "summary": f"Pattern: {pattern_type} | Probability: {float(round(probability*100, 1))}% | Direction: {direction} | Risk: {risk_level}"
    }


def calculate_support_resistance_levels(symbol: str, from_date: str, to_date: str, bins: int = 15) -> Dict:
    """
    Detects and ranks support and resistance levels for a given symbol within a specified date range using multiple advanced methods.

    Methods used:
      - Pivot Points (multiple timeframes: daily, weekly, bi-weekly, monthly)
      - Camarilla Pivot Points
      - Fractal and Swing Point Analysis (local maxima/minima)
      - Volume Profile & Clustering (high/low volume nodes)
      - Fibonacci Retracements and Extensions
      - Moving Average Confluence (SMA/EMA/Bollinger Bands)
      - Price Density Analysis (KMeans clustering)
      - Psychological Levels (round numbers)
      - Level Validation and Clustering (removes duplicates, clusters similar levels)

    Returns:
        dict: {
            "symbol": str,
            "current_price": float,
            "levels": List[Dict],           # All detected key levels, sorted by strength
            "support_levels": List[Dict],   # Strongest support levels (below current price)
            "resistance_levels": List[Dict],# Strongest resistance levels (above current price)
            "neutral_levels": List[Dict],   # Levels very close to current price
            "statistics": dict,             # Stats about analysis and methods used
            "details": str                  # Human-readable summary
        }
    Notes:
        - Each level dict contains: level (float), type (str), strength (float), source (str), and possibly other metadata.
        - Requires at least 50 data points for robust analysis.
        - Combines technical, statistical, and machine learning methods for robust level detection.
    """
    
    # Get data
    df = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(df)
    
    if df is None or len(df) < 50:  # Increased minimum requirement
        return {"symbol": symbol, "levels": [], "details": "Insufficient data"}
    
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    
    # Add technical indicators for confluence
    df['sma20'] = ta.SMA(df['close'], timeperiod=20)
    df['sma50'] = ta.SMA(df['close'], timeperiod=50)
    df['sma200'] = ta.SMA(df['close'], timeperiod=200)
    df['ema20'] = ta.EMA(df['close'], timeperiod=20)
    df['ema50'] = ta.EMA(df['close'], timeperiod=50)
    upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    
    # ===== 1. ADVANCED PIVOT POINT DETECTION =====
    
    def detect_pivot_points(df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Enhanced pivot point detection with multiple timeframes"""
        pivot_levels = []
        
        # Classic pivot points (multiple periods)
        for period in [1, 5, 10, 20]:  # Daily, weekly, bi-weekly, monthly approximation
            if len(df) >= period:
                recent_data = df.iloc[-period:]
                high_val = recent_data['high'].max()
                low_val = recent_data['low'].min()
                close_val = recent_data['close'].iloc[-1]

                pivot = round(float((high_val + low_val + close_val) / 3), 2)
                r1 = round(float(2 * pivot - low_val), 2)
                s1 = round(float(2 * pivot - high_val), 2)
                r2 = round(float(pivot + (high_val - low_val)), 2)
                s2 = round(float(pivot - (high_val - low_val)), 2)
                r3 = round(float(high_val + 2 * (pivot - low_val)), 2)
                s3 = round(float(low_val - 2 * (high_val - pivot)), 2)

                strength_multiplier = 0.8 if period == 1 else 0.7 if period == 5 else 0.6 if period == 10 else 0.5
                
                pivot_levels.extend([
                    {"level": pivot, "type": "Pivot", "strength": round(0.8 * strength_multiplier, 2), "source": f"Pivot_{period}d"},
                    {"level": r1, "type": "Resistance", "strength": round(0.7 * strength_multiplier, 2), "source": f"R1_{period}d"},
                    {"level": r2, "type": "Resistance", "strength": round(0.6 * strength_multiplier, 2), "source": f"R2_{period}d"},
                    {"level": r3, "type": "Resistance", "strength": round(0.5 * strength_multiplier, 2), "source": f"R3_{period}d"},
                    {"level": s1, "type": "Support", "strength": round(0.7 * strength_multiplier, 2), "source": f"S1_{period}d"},
                    {"level": s2, "type": "Support", "strength": round(0.6 * strength_multiplier, 2), "source": f"S2_{period}d"},
                    {"level": s3, "type": "Support", "strength": round(0.5 * strength_multiplier, 2), "source": f"S3_{period}d"},
                ])
        
        # Camarilla pivot points
        if len(df) >= 1:
            last_data = df.iloc[-1]
            high_val = last_data['high']
            low_val = last_data['low']
            close_val = last_data['close']
            
            diff = high_val - low_val
            
            # Camarilla levels
            r4 = round(float(close_val + (diff * 1.1) / 2), 2)
            r3 = round(float(close_val + (diff * 1.1) / 4), 2)
            r2 = round(float(close_val + (diff * 1.1) / 6), 2)
            r1 = round(float(close_val + (diff * 1.1) / 12), 2)

            s1 = round(float(close_val - (diff * 1.1) / 12), 2)
            s2 = round(float(close_val - (diff * 1.1) / 6), 2)
            s3 = round(float(close_val - (diff * 1.1) / 4), 2)
            s4 = round(float(close_val - (diff * 1.1) / 2), 2)

            camarilla_levels = [
                {"level": r4, "type": "Resistance", "strength": 0.9, "source": "Camarilla_R4"},
                {"level": r3, "type": "Resistance", "strength": 0.8, "source": "Camarilla_R3"},
                {"level": r2, "type": "Resistance", "strength": 0.7, "source": "Camarilla_R2"},
                {"level": r1, "type": "Resistance", "strength": 0.6, "source": "Camarilla_R1"},
                {"level": s1, "type": "Support", "strength": 0.6, "source": "Camarilla_S1"},
                {"level": s2, "type": "Support", "strength": 0.7, "source": "Camarilla_S2"},
                {"level": s3, "type": "Support", "strength": 0.8, "source": "Camarilla_S3"},
                {"level": s4, "type": "Support", "strength": 0.9, "source": "Camarilla_S4"},
            ]
            
            pivot_levels.extend(camarilla_levels)
        
        return pivot_levels
    
    # ===== 2. FRACTAL AND SWING ANALYSIS =====
    
    def detect_fractals_and_swings(df: pd.DataFrame) -> List[Dict]:
        """Advanced fractal and swing point detection"""
        fractal_levels = []
        
        # Multiple window sizes for different timeframes
        for window in [3, 5, 7, 10, 15]:
            if len(df) >= window * 2:
                # Find local maxima (resistance)
                highs = df['high'].values
                high_peaks = argrelextrema(highs, np.greater, order=window)[0]
                
                # Find local minima (support)
                lows = df['low'].values
                low_peaks = argrelextrema(lows, np.less, order=window)[0]
                
                # Weight by recency and significance
                for peak in high_peaks:
                    if peak < len(df) - 1:  # Not the last point
                        level = round(float(df['high'].iloc[peak]), 2)
                        # Recency weight (more recent = higher weight)
                        recency_weight = 1.0 - (len(df) - peak) / len(df)
                        # Significance weight (higher volume = higher weight)
                        volume_weight = min(df['volume'].iloc[peak] / df['volume'].mean(), 2.0)

                        strength = round(float(0.7 + (recency_weight * 0.2) + (volume_weight * 0.1)), 2)
                        strength = min(strength, 1.0)
                        
                        fractal_levels.append({
                            "level": level,
                            "type": "Resistance",
                            "strength": strength,
                            "source": f"Fractal_High_{window}",
                            "index": peak
                        })
                
                for valley in low_peaks:
                    if valley < len(df) - 1:  # Not the last point
                        level = round(float(df['low'].iloc[valley]), 2)
                        # Recency weight
                        recency_weight = 1.0 - (len(df) - valley) / len(df)
                        # Volume weight
                        volume_weight = min(df['volume'].iloc[valley] / df['volume'].mean(), 2.0)

                        strength = round(float(0.7 + (recency_weight * 0.2) + (volume_weight * 0.1)), 2)
                        strength = min(strength, 1.0)
                        
                        fractal_levels.append({
                            "level": level,
                            "type": "Support",
                            "strength": strength,
                            "source": f"Fractal_Low_{window}",
                            "index": valley
                        })
        
        return fractal_levels
    
    # ===== 3. ENHANCED VOLUME PROFILE =====
    
    def create_volume_profile(df: pd.DataFrame, bins: int = 20) -> List[Dict]:
        """Enhanced volume profile analysis with clustering"""
        volume_levels = []
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        # Dynamic binning based on price range
        if price_range > 0:
            bin_size = price_range / bins
            price_bins = np.arange(price_min, price_max + bin_size, bin_size)
            
            # Volume at Price (VAP) calculation
            vap_data = []
            for i in range(len(price_bins) - 1):
                bin_min = price_bins[i]
                bin_max = price_bins[i + 1]
                bin_center = (bin_min + bin_max) / 2
                
                # Calculate volume in this price range
                mask = (df['low'] <= bin_max) & (df['high'] >= bin_min)
                volume_in_bin = df.loc[mask, 'volume'].sum()
                
                if volume_in_bin > 0:
                    vap_data.append({
                        'price': bin_center,
                        'volume': volume_in_bin
                    })
            
            if vap_data:
                vap_df = pd.DataFrame(vap_data)
                
                # Find high volume nodes (HVN)
                volume_threshold = vap_df['volume'].quantile(0.7)  # Top 30% volume
                hvn_levels = vap_df[vap_df['volume'] >= volume_threshold]
                
                # Find low volume nodes (LVN) - potential breakout levels
                volume_low_threshold = vap_df['volume'].quantile(0.3)  # Bottom 30% volume
                lvn_levels = vap_df[vap_df['volume'] <= volume_low_threshold]
                
                # Add HVN levels (strong support/resistance)
                for _, row in hvn_levels.iterrows():
                    strength = min(row['volume'] / vap_df['volume'].max(), 1.0)
                    volume_levels.append({
                        "level": round(float(row['price']), 2),
                        "type": "Volume Cluster",
                        "strength": round(float(0.8 + (strength * 0.2)), 2),
                        "source": "HVN",
                        "volume": round(float(row['volume']), 3)
                    })
                
                # Add LVN levels (breakout levels)
                for _, row in lvn_levels.iterrows():
                    volume_levels.append({
                        "level": round(float(row['price']), 2),
                        "type": "Volume Gap",
                        "strength": 0.5,
                        "source": "LVN",
                        "volume": round(float(row['volume']), 3)
                    })
        
        return volume_levels
    
    # ===== 4. FIBONACCI RETRACEMENTS =====
    
    def calculate_fibonacci_levels(df: pd.DataFrame) -> List[Dict]:
        """Calculate Fibonacci retracement levels"""
        fib_levels = []
        
        # Find significant swings (last 50 periods)
        recent_df = df.iloc[-50:] if len(df) >= 50 else df
        
        swing_high = recent_df['high'].max()
        swing_low = recent_df['low'].min()
        
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            # Uptrend retracement
            fib_level = swing_high - (swing_high - swing_low) * ratio
            fib_levels.append({
                "level": round(float(fib_level), 2),
                "type": "Fibonacci",
                "strength": round(float(0.6 + (0.4 * (1 - abs(ratio - 0.5) * 2))), 2),  # 0.5 level gets highest weight
                "source": f"Fib_{ratio}",
                "ratio": ratio
            })
        
        # Add key Fibonacci extensions
        fib_extensions = [1.272, 1.618, 2.618]
        for ext in fib_extensions:
            ext_level = swing_high + (swing_high - swing_low) * (ext - 1)
            fib_levels.append({
                "level": round(float(ext_level), 2),
                "type": "Fibonacci Extension",
                "strength": 0.7,
                "source": f"Fib_Ext_{ext}",
                "ratio": ext
            })
        
        return fib_levels
    
    # ===== 5. MOVING AVERAGE CONFLUENCE =====
    
    def detect_ma_confluence(df: pd.DataFrame) -> List[Dict]:
        """Detect moving average confluence zones"""
        ma_levels = []
        
        # Current MA values
        current_sma20 = round(float(df['sma20'].iloc[-1]), 2)
        current_sma50 = round(float(df['sma50'].iloc[-1]), 2)
        current_sma200 = round(float(df['sma200'].iloc[-1]), 2)
        current_ema20 = round(float(df['ema20'].iloc[-1]), 2)
        current_ema50 = round(float(df['ema50'].iloc[-1]), 2)

        ma_values = [
            ("SMA20", current_sma20, 0.7),
            ("SMA50", current_sma50, 0.8),
            ("SMA200", current_sma200, 0.9),
            ("EMA20", current_ema20, 0.7),
            ("EMA50", current_ema50, 0.8),
        ]
        
        for ma_name, ma_value, strength in ma_values:
            if not np.isnan(ma_value):
                ma_levels.append({
                    "level": ma_value,
                    "type": "Moving Average",
                    "strength": strength,
                    "source": ma_name
                })
        
        # Bollinger Bands
        bb_upper = round(float(df['bb_upper'].iloc[-1]), 2)
        bb_middle = round(float(df['bb_middle'].iloc[-1]), 2)
        bb_lower = round(float(df['bb_lower'].iloc[-1]), 2)

        if not np.isnan(bb_upper):
            ma_levels.extend([
                {"level": bb_upper, "type": "Bollinger Upper", "strength": 0.8, "source": "BB_Upper"},
                {"level": bb_middle, "type": "Bollinger Middle", "strength": 0.7, "source": "BB_Middle"},
                {"level": bb_lower, "type": "Bollinger Lower", "strength": 0.8, "source": "BB_Lower"},
            ])
        
        return ma_levels
    
    # ===== 6. PRICE DENSITY ANALYSIS =====
    
    def analyze_price_density(df: pd.DataFrame) -> List[Dict]:
        """Analyze price density for consolidation zones"""
        density_levels = []
        
        # Create price clusters using machine learning
        prices = df['close'].values.reshape(-1, 1)
        
        # Determine optimal number of clusters
        n_clusters = min(8, len(df) // 10)  # Dynamic cluster count
        
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(prices)
            
            # Analyze each cluster
            for i in range(n_clusters):
                cluster_prices = prices[clusters == i]
                if len(cluster_prices) > 0:
                    cluster_center = np.mean(cluster_prices)
                    cluster_std = np.std(cluster_prices)
                    cluster_count = len(cluster_prices)
                    
                    # Strength based on cluster size and tightness
                    strength = min(0.4 + (cluster_count / len(df) * 0.4) + (1 / (cluster_std + 0.01) * 0.2), 1.0)
                    
                    density_levels.append({
                        "level": round(float(cluster_center), 2),
                        "type": "Price Cluster",
                        "strength": round(float(strength), 2),
                        "source": f"Cluster_{i}",
                        "count": cluster_count,
                        "std": round(float(cluster_std), 2)
                    })
        
        return density_levels
    
    # ===== 7. PSYCHOLOGICAL LEVELS =====
    
    def detect_psychological_levels(df: pd.DataFrame) -> List[Dict]:
        """Detect round number psychological levels"""
        psychological_levels = []
        
        current_price = df['close'].iloc[-1]
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        # Determine the appropriate round number base
        if current_price >= 10000:
            round_base = 1000
        elif current_price >= 1000:
            round_base = 100
        elif current_price >= 100:
            round_base = 50
        elif current_price >= 10:
            round_base = 10
        else:
            round_base = 5
        
        # Generate psychological levels
        start_level = int(price_min / round_base) * round_base
        end_level = int(price_max / round_base + 1) * round_base
        
        for level in range(start_level, end_level + round_base, round_base):
            if price_min <= level <= price_max:
                # Check if price has interacted with this level
                interactions = ((df['low'] <= level) & (df['high'] >= level)).sum()
                
                if interactions > 0:
                    strength = round(float(0.3 + min(interactions / len(df) * 5, 0.7)), 2)  # Max 0.7 strength
                    psychological_levels.append({
                        "level": float(level),
                        "type": "Psychological",
                        "strength": strength,
                        "source": f"Round_{round_base}",
                        "interactions": interactions
                    })
        
        # Add repeated digit levels (e.g., 111, 222, etc.)
        max_digits = len(str(int(price_max)))
        for d in range(1, max_digits + 1):
            for digit in range(1, 10):
                level = int(str(digit) * d)
                if price_min <= level <= price_max:
                    interactions = ((df['low'] <= level) & (df['high'] >= level)).sum()
                    if interactions > 0:
                        # Assign a strength, e.g., up to 0.8 (tune as needed)
                        strength = round(float(0.2 + min(interactions / len(df) * 4, 1.0)), 2)
                        psychological_levels.append({
                            "level": float(level),
                            "type": "Psychological",
                            "strength": strength,
                            "source": f"Repeat_{digit}_{d}d",
                            "interactions": interactions
                        })
        
        return psychological_levels
    
    # ===== 8. LEVEL VALIDATION AND CLUSTERING =====
    
    def validate_and_cluster_levels(all_levels: List[Dict], current_price: float) -> List[Dict]:
        """Validate and cluster similar levels"""
        if not all_levels:
            return []
        
        # Remove levels too far from current price (outside 2 standard deviations)
        price_std = np.std([lvl['level'] for lvl in all_levels])
        valid_levels = [
            lvl for lvl in all_levels 
            if abs(lvl['level'] - current_price) <= 1 * price_std
        ]
        
        # Cluster similar levels
        tolerance = current_price * 0.03  # 3% tolerance
        clustered_levels = []
        
        # Sort by level value
        valid_levels.sort(key=lambda x: x['level'])
        
        i = 0
        while i < len(valid_levels):
            current_level = valid_levels[i]
            cluster_levels = [current_level]
            
            # Find all levels within tolerance
            j = i + 1
            while j < len(valid_levels) and abs(valid_levels[j]['level'] - current_level['level']) <= tolerance:
                cluster_levels.append(valid_levels[j])
                j += 1
            
            # Create consolidated level
            if len(cluster_levels) == 1:
                clustered_levels.append(current_level)
            else:
                # Weighted average level
                total_weight = sum(lvl['strength'] for lvl in cluster_levels)
                weighted_level = sum(lvl['level'] * lvl['strength'] for lvl in cluster_levels) / total_weight
                
                # Combined strength (with diminishing returns)
                combined_strength = min(sum(lvl['strength'] for lvl in cluster_levels) * 0.8, 1.0)
                
                # Determine type based on strongest level
                strongest_level = max(cluster_levels, key=lambda x: x['strength'])
                
                clustered_levels.append({
                    "level": float(round(weighted_level, 2)),
                    "type": strongest_level['type'],
                    "strength": float(round(combined_strength, 3)),
                    "source": f"Cluster_{len(cluster_levels)}",
                    "sources": [lvl['source'] for lvl in cluster_levels],
                    "count": len(cluster_levels)
                })
            
            i = j
        
        return clustered_levels
    
    # ===== MAIN EXECUTION =====
    
    current_price = df['close'].iloc[-1]
    
    # Collect all levels
    all_levels = []
    
    # 1. Pivot points
    pivot_levels = detect_pivot_points(df)
    all_levels.extend(pivot_levels)           
    
    # 2. Fractals and swings
    fractal_levels = detect_fractals_and_swings(df)
    all_levels.extend(fractal_levels)

    # 3. Volume profile
    volume_levels = create_volume_profile(df, bins)
    all_levels.extend(volume_levels)

    # 4. Fibonacci levels
    fib_levels = calculate_fibonacci_levels(df)
    all_levels.extend(fib_levels)

    # 5. Moving average confluence
    ma_levels = detect_ma_confluence(df)
    all_levels.extend(ma_levels)


    # 6. Price density analysis
    density_levels = analyze_price_density(df)
    all_levels.extend(density_levels)


    # 7. Psychological levels
    psychological_levels = detect_psychological_levels(df)
    all_levels.extend(psychological_levels)

    # 8. Validate and cluster
    final_levels = validate_and_cluster_levels(all_levels, current_price)

    # Sort by strength (strongest first)
    final_levels.sort(key=lambda x: x['strength'], reverse=True)

    
    # Limit to top levels and add distance from current price
    top_levels = final_levels[:25]  # Top 25 levels
    
    for level in top_levels:
        level['distance_pct'] = float(round(((level['level'] - current_price) / current_price) * 100, 2))
        level['level'] = float(level['level'])
        level['strength'] = float(level['strength'])
    
    # Separate into support and resistance
    support_levels = [lvl for lvl in top_levels if lvl['level'] < current_price][:8]
    resistance_levels = [lvl for lvl in top_levels if lvl['level'] > current_price][:8]


    return {
        "symbol": symbol,
        "current_price": float(round(current_price, 2)),
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "details": f"After analysis {len(all_levels)} total levels identified, clustered into {len(top_levels)} key levels using 8 different methods."
    }


def momentum_indicators(symbol: str) -> Dict:
    """
    Performs advanced momentum analysis for a given symbol using multiple technical indicators.

    Analysis includes:
      - Rate of Change (ROC) over multiple timeframes (5, 10, 20, 50 days)
      - Williams %R over multiple timeframes (14, 21, 50 days)
      - Stochastic Oscillator (various configurations)
      - Relative Strength Index (RSI)
      - Momentum (MOM)
      - Commodity Channel Index (CCI)
      - Money Flow Index (MFI)
      - Price velocity and acceleration

    The function computes current values for all indicators, analyzes their signals and alignment, and generates a comprehensive momentum assessment including:
      - Overall momentum score and recommendation (bullish, bearish, neutral)
      - Confidence level and signal strength
      - Key insights and trading implications

    Returns:
        dict: {
            "symbol": str,
            "status": str,
            "timestamp": str,
            "data_period": str,
            "current_values": dict,         # Latest values for all indicators
            "roc_analysis": dict,           # ROC signal analysis
            "williams_r_analysis": dict,    # Williams %R signal analysis
            "stochastic_analysis": dict,    # Stochastic Oscillator analysis
            "overall_assessment": dict,     # Overall momentum assessment
            "analysis_summary": str,        # Human-readable summary
            "recommendation": str,          # "bullish", "bearish", or "neutral"
            "confidence": float,            # Confidence score (0-100)
            "momentum_score": float,        # Overall momentum score
            "strength_assessment": str,     # "Strong", "Moderate", or "Weak"
            "key_insights": list,           # List of key findings
            "signals": dict                 # Trading signal details
        }
    Notes:
        - Requires at least 50 data points for robust analysis.
        - Combines multiple momentum indicators for a comprehensive assessment.
        - Designed for use in systematic and discretionary trading workflows.
    """
    # Calculate date range (using last 200 days for comprehensive analysis)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)  # Extra buffer for indicators
    
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Get data using the same method as volume analysis
    df = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(df)
    
    if df is None or len(df) < 50:
        return {
            "symbol": symbol,
            "status": "error",
            "message": "Insufficient data for momentum analysis (minimum 50 periods required)",
            "analysis_summary": "Unable to perform momentum analysis due to insufficient historical data.",
            "recommendation": "neutral",
            "confidence": 0.0
        }
    
    df = df.copy()
    
    # ==================== MOMENTUM INDICATORS CALCULATION ====================
    
    # 1. Rate of Change (ROC) - Multiple timeframes
    df['roc_5'] = ta.ROC(df['close'], timeperiod=5)
    df['roc_10'] = ta.ROC(df['close'], timeperiod=10)
    df['roc_20'] = ta.ROC(df['close'], timeperiod=20)
    df['roc_50'] = ta.ROC(df['close'], timeperiod=50)
    
    # 2. Williams %R - Multiple timeframes
    df['williams_r_14'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['williams_r_21'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=21)
    df['williams_r_50'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=50)
    
    # 3. Stochastic Oscillator - Multiple configurations
    df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'], 
                                           fastk_period=14, slowk_period=3, slowd_period=3)
    df['stoch_k_fast'], df['stoch_d_fast'] = ta.STOCH(df['high'], df['low'], df['close'], 
                                                      fastk_period=5, slowk_period=3, slowd_period=3)
    df['stoch_k_slow'], df['stoch_d_slow'] = ta.STOCH(df['high'], df['low'], df['close'], 
                                                      fastk_period=21, slowk_period=5, slowd_period=5)
    
    # 4. Additional Momentum Indicators
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    df['momentum'] = ta.MOM(df['close'], timeperiod=10)
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    
    # 5. Price momentum derivatives
    df['price_velocity'] = df['close'].pct_change()
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # ==================== CURRENT VALUES ====================
    
    current_values = {
        'roc_5': float(round(df['roc_5'].iloc[-1], 3)),
        'roc_10': float(round(df['roc_10'].iloc[-1], 3)),
        'roc_20': float(round(df['roc_20'].iloc[-1], 3)),
        'roc_50': float(round(df['roc_50'].iloc[-1], 3)),
        'williams_r_14': float(round(df['williams_r_14'].iloc[-1], 2)),
        'williams_r_21': float(round(df['williams_r_21'].iloc[-1], 2)),
        'williams_r_50': float(round(df['williams_r_50'].iloc[-1], 2)),
        'stoch_k': float(round(df['stoch_k'].iloc[-1], 2)),
        'stoch_d': float(round(df['stoch_d'].iloc[-1], 2)),
        'stoch_k_fast': float(round(df['stoch_k_fast'].iloc[-1], 2)),
        'stoch_d_fast': float(round(df['stoch_d_fast'].iloc[-1], 2)),
        'rsi': float(round(df['rsi'].iloc[-1], 2)),
        'momentum': float(round(df['momentum'].iloc[-1], 3)),
        'cci': float(round(df['cci'].iloc[-1], 2)),
        'mfi': float(round(df['mfi'].iloc[-1], 2))
    }
    
    # ==================== MOMENTUM ANALYSIS & SIGNALS ====================
    
    def analyze_roc_signals(df) -> Dict:
        """Analyze Rate of Change signals"""
        roc_5 = df['roc_5'].iloc[-1]
        roc_10 = df['roc_10'].iloc[-1]
        roc_20 = df['roc_20'].iloc[-1]
        roc_50 = df['roc_50'].iloc[-1]
        
        # Signal strength based on ROC alignment
        positive_rocs = sum([roc_5 > 0, roc_10 > 0, roc_20 > 0, roc_50 > 0])
        roc_alignment = positive_rocs / 4.0
        
        # Momentum acceleration
        roc_acceleration = "Accelerating" if roc_5 > roc_10 > roc_20 else "Decelerating"
        
        # Signal classification
        if roc_alignment >= 0.75:
            signal = "Strong Bullish"
            strength = min(abs(roc_5) / 5.0, 1.0)
        elif roc_alignment >= 0.5:
            signal = "Moderate Bullish"
            strength = min(abs(roc_5) / 3.0, 1.0)
        elif roc_alignment <= 0.25:
            signal = "Strong Bearish"
            strength = min(abs(roc_5) / 5.0, 1.0)
        else:
            signal = "Moderate Bearish"
            strength = min(abs(roc_5) / 3.0, 1.0)
        
        return {
            "signal": signal,
            "strength": float(round(strength, 3)),
            "alignment_score": float(round(roc_alignment, 3)),
            "momentum_direction": roc_acceleration,
            "interpretation": f"ROC shows {signal.lower()} momentum with {roc_acceleration.lower()} trend. "
                           f"Alignment score: {roc_alignment:.1%}"
        }
    
    def analyze_williams_r_signals(df) -> Dict:
        """Analyze Williams %R signals"""
        wr_14 = df['williams_r_14'].iloc[-1]
        wr_21 = df['williams_r_21'].iloc[-1]
        wr_50 = df['williams_r_50'].iloc[-1]
        
        # Williams %R interpretation
        if wr_14 <= -80:
            condition = "Oversold"
            signal = "Potential Bullish Reversal"
        elif wr_14 >= -20:
            condition = "Overbought"
            signal = "Potential Bearish Reversal"
        elif -50 <= wr_14 <= -30:
            condition = "Bullish Zone"
            signal = "Bullish Momentum"
        elif -70 <= wr_14 <= -50:
            condition = "Bearish Zone"
            signal = "Bearish Momentum"
        else:
            condition = "Neutral"
            signal = "No Clear Signal"
        
        # Cross-timeframe confirmation
        confirmation = "Confirmed" if all(x < -50 for x in [wr_14, wr_21, wr_50]) or all(x > -50 for x in [wr_14, wr_21, wr_50]) else "Divergent"
        
        return {
            "signal": signal,
            "condition": condition,
            "confirmation": confirmation,
            "strength": float(round(abs(wr_14 + 50) / 50, 3)),
            "interpretation": f"Williams %R indicates {condition.lower()} conditions with {signal.lower()}. "
                           f"Cross-timeframe analysis shows {confirmation.lower()} signals."
        }
    
    def analyze_stochastic_signals(df) -> Dict:
        """Analyze Stochastic Oscillator signals"""
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        stoch_k_prev = df['stoch_k'].iloc[-2]
        stoch_d_prev = df['stoch_d'].iloc[-2]
        
        # Stochastic conditions
        if stoch_k <= 20 and stoch_d <= 20:
            condition = "Oversold"
        elif stoch_k >= 80 and stoch_d >= 80:
            condition = "Overbought"
        else:
            condition = "Normal"
        
        # Crossover detection
        if stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev:
            crossover = "Bullish Crossover"
        elif stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev:
            crossover = "Bearish Crossover"
        else:
            crossover = "No Crossover"
        
        # Signal strength
        divergence = abs(stoch_k - stoch_d)
        strength = min(divergence / 20, 1.0)
        
        return {
            "condition": condition,
            "crossover": crossover,
            "strength": float(round(strength, 3)),
            "k_value": float(round(stoch_k, 2)),
            "d_value": float(round(stoch_d, 2)),
            "interpretation": f"Stochastic shows {condition.lower()} conditions with {crossover.lower()}. "
                           f"Current K: {stoch_k:.1f}, D: {stoch_d:.1f}"
        }
    
    def generate_overall_momentum_assessment(roc_analysis, williams_analysis, stoch_analysis, current_values) -> Dict:
        """Generate comprehensive momentum assessment"""
        
        # Collect all signals
        signals = []
        
        # ROC signals
        if "Bullish" in roc_analysis["signal"]:
            signals.append(1 * roc_analysis["strength"])
        elif "Bearish" in roc_analysis["signal"]:
            signals.append(-1 * roc_analysis["strength"])
        else:
            signals.append(0)
        
        # Williams %R signals
        if "Bullish" in williams_analysis["signal"]:
            signals.append(1 * williams_analysis["strength"])
        elif "Bearish" in williams_analysis["signal"]:
            signals.append(-1 * williams_analysis["strength"])
        else:
            signals.append(0)
        
        # Stochastic signals
        if "Bullish" in stoch_analysis["crossover"]:
            signals.append(1 * stoch_analysis["strength"])
        elif "Bearish" in stoch_analysis["crossover"]:
            signals.append(-1 * stoch_analysis["strength"])
        else:
            signals.append(0)
        
        # RSI signal
        rsi_val = current_values['rsi']
        if rsi_val < 30:
            signals.append(0.8)  # Oversold bullish
        elif rsi_val > 70:
            signals.append(-0.8)  # Overbought bearish
        else:
            signals.append(0)
        
        # Calculate overall momentum score
        momentum_score = sum(signals) / len(signals)
        
        # Determine recommendation
        if momentum_score > 0.3:
            recommendation = "bullish"
            confidence = min(abs(momentum_score) * 100, 95)
        elif momentum_score < -0.3:
            recommendation = "bearish"
            confidence = min(abs(momentum_score) * 100, 95)
        else:
            recommendation = "neutral"
            confidence = 50
        
        return {
            "momentum_score": float(round(momentum_score, 3)),
            "recommendation": recommendation,
            "confidence": float(round(confidence, 1)),
            "signal_count": len([s for s in signals if abs(s) > 0.1]),
            "strength_assessment": "Strong" if abs(momentum_score) > 0.6 else "Moderate" if abs(momentum_score) > 0.3 else "Weak"
        }
    
    # ==================== EXECUTE ANALYSIS ====================
    
    roc_analysis = analyze_roc_signals(df)
    williams_analysis = analyze_williams_r_signals(df)
    stoch_analysis = analyze_stochastic_signals(df)
    overall_assessment = generate_overall_momentum_assessment(roc_analysis, williams_analysis, stoch_analysis, current_values)
    
    # ==================== LLM-FRIENDLY SUMMARY ====================
    
    # Create detailed analysis summary for LLM
    analysis_summary = f"""
        MOMENTUM ANALYSIS SUMMARY for {symbol}:

        OVERALL ASSESSMENT:
        - Momentum Score: {overall_assessment['momentum_score']:.3f} ({overall_assessment['strength_assessment']} {overall_assessment['recommendation'].upper()})
        - Confidence Level: {overall_assessment['confidence']:.1f}%
        - Active Signals: {overall_assessment['signal_count']} out of 4 indicators

        RATE OF CHANGE (ROC) ANALYSIS:
        - Signal: {roc_analysis['signal']}
        - Momentum Direction: {roc_analysis['momentum_direction']}
        - Strength: {roc_analysis['strength']:.3f}
        - 5-day ROC: {current_values['roc_5']:.2f}%
        - 20-day ROC: {current_values['roc_20']:.2f}%
        - Interpretation: {roc_analysis['interpretation']}

        WILLIAMS %R ANALYSIS:
        - Condition: {williams_analysis['condition']}
        - Signal: {williams_analysis['signal']}
        - Cross-timeframe Confirmation: {williams_analysis['confirmation']}
        - 14-day Williams %R: {current_values['williams_r_14']:.1f}
        - Interpretation: {williams_analysis['interpretation']}

        STOCHASTIC OSCILLATOR ANALYSIS:
        - Condition: {stoch_analysis['condition']}
        - Crossover Status: {stoch_analysis['crossover']}
        - Stochastic K: {stoch_analysis['k_value']:.1f}
        - Stochastic D: {stoch_analysis['d_value']:.1f}
        - Interpretation: {stoch_analysis['interpretation']}

        SUPPORTING INDICATORS:
        - RSI (14): {current_values['rsi']:.1f} ({'Oversold' if current_values['rsi'] < 30 else 'Overbought' if current_values['rsi'] > 70 else 'Neutral'})
        - Money Flow Index: {current_values['mfi']:.1f}
        - Commodity Channel Index: {current_values['cci']:.1f}

        TRADING IMPLICATIONS:
        - Primary Trend: {overall_assessment['recommendation'].upper()}
        - Risk Level: {'Low' if overall_assessment['confidence'] > 70 else 'Medium' if overall_assessment['confidence'] > 50 else 'High'}
        - Signal Reliability: {'High' if overall_assessment['signal_count'] >= 3 else 'Medium' if overall_assessment['signal_count'] >= 2 else 'Low'}
        """.strip()
    
    # ==================== RETURN COMPREHENSIVE RESULTS ====================
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "data_period": f"{len(df)} trading days",
        
        # Current indicator values
        "current_values": current_values,
        
        # Detailed analysis
        "roc_analysis": roc_analysis,
        "williams_r_analysis": williams_analysis,
        "stochastic_analysis": stoch_analysis,
        "overall_assessment": overall_assessment,
        
        # LLM-friendly outputs
        "analysis_summary": analysis_summary,
        "recommendation": overall_assessment['recommendation'],
        "confidence": overall_assessment['confidence'],
        "momentum_score": overall_assessment['momentum_score'],
        "strength_assessment": overall_assessment['strength_assessment'],
        
        # Key insights for LLM
        "key_insights": [
            f"Momentum is {overall_assessment['strength_assessment'].lower()} {overall_assessment['recommendation']} with {overall_assessment['confidence']:.1f}% confidence",
            f"ROC shows {roc_analysis['signal'].lower()} with {roc_analysis['momentum_direction'].lower()} trend",
            f"Williams %R indicates {williams_analysis['condition'].lower()} conditions",
            f"Stochastic is in {stoch_analysis['condition'].lower()} territory with {stoch_analysis['crossover'].lower()}",
            f"RSI at {current_values['rsi']:.1f} suggests {'oversold' if current_values['rsi'] < 30 else 'overbought' if current_values['rsi'] > 70 else 'neutral'} conditions"
        ],
        
        # Trading signals
        "signals": {
            "primary_signal": overall_assessment['recommendation'],
            "signal_strength": overall_assessment['strength_assessment'],
            "risk_level": "Low" if overall_assessment['confidence'] > 70 else "Medium" if overall_assessment['confidence'] > 50 else "High",
            "time_horizon": "Short to Medium term (1-4 weeks)",            
        }
    }


# ===== ADDITIONAL UTILITY FUNCTIONS =====

def get_nearest_levels(symbol: str, from_date: str, to_date: str, distance_pct: float = 5.0) -> Dict:
    """Get support/resistance levels within specified distance from current price"""
    result = calculate_support_resistance_levels(symbol, from_date, to_date)
    
    if not result.get('levels'):
        return result
    
    # Filter levels within distance
    nearby_levels = [
        level for level in result['levels']
        if abs(level['distance_pct']) <= distance_pct
    ]
    
    result['nearby_levels'] = nearby_levels
    result['nearby_count'] = len(nearby_levels)
    
    return result


def analyze_level_quality(symbol: str, from_date: str, to_date: str) -> Dict:
    """Analyze the quality and reliability of detected levels"""
    result = calculate_support_resistance_levels(symbol, from_date, to_date)

    if not result.get('levels'):
        return result
    
    # Get historical data for testing
    df = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(df)
    
    # Test each level for historical accuracy
    for level in result['levels']:
        level_price = level['level']
        
        # Count touches/rejections
        touches = 0
        rejections = 0
        
        for i in range(len(df)):
            candle = df.iloc[i]
            
            # Check if price touched this level (within 1% tolerance)
            tolerance = level_price * 0.01
            if candle['low'] <= level_price + tolerance and candle['high'] >= level_price - tolerance:
                touches += 1
                
                # Check for rejection
                if level['type'] == 'Resistance' and candle['close'] < level_price:
                    rejections += 1
                elif level['type'] == 'Support' and candle['close'] > level_price:
                    rejections += 1
        
        level['touches'] = touches
        level['rejections'] = rejections
        level['rejection_rate'] = rejections / touches if touches > 0 else 0
        level['reliability_score'] = level['rejection_rate'] * level['strength']
    
    return result


def validate_breakout_quality(symbol: str, from_date: str, to_date: str) -> Dict:
    """
    Validate the quality of detected breakouts with additional filters
    """
    result = detect_breakout_patterns(symbol, from_date, to_date)
    
    if result['breakout_probability'] < 0.5:
        return result
    
    # Additional quality checks
    df = get_symbol_history_daily_data(symbol, from_date, to_date)
    df = refined_ohlc_data(df)
    
    # Check for false breakout signs
    last_3_candles = df.iloc[-3:]
    
    # Volume consistency check
    volume_consistent = all(candle['volume'] > df['volume'].rolling(20).mean().iloc[-1] 
                          for _, candle in last_3_candles.iterrows())
    
    # Price follow-through check
    if result['direction'] == 'Bullish':
        follow_through = last_3_candles['close'].iloc[-1] > last_3_candles['close'].iloc[0]
    else:
        follow_through = last_3_candles['close'].iloc[-1] < last_3_candles['close'].iloc[0]
    
    # Adjust probability based on quality
    quality_multiplier = 1.0
    if not volume_consistent:
        quality_multiplier *= 0.8
    if not follow_through:
        quality_multiplier *= 0.7
    
    result['breakout_probability'] = float(round(result['breakout_probability'] * quality_multiplier, 3))
    result['quality_validated'] = True
    result['volume_consistent'] = volume_consistent
    result['price_follow_through'] = follow_through
    
    return result


def batch_level_analysis(symbols: List[str], from_date: str, to_date: str) -> Dict:
    """Analyze support/resistance levels for multiple symbols"""
    results = {}
    
    for symbol in symbols:
        try:
            result = calculate_support_resistance_levels(symbol, from_date, to_date)
            results[symbol] = result
        except Exception as e:
            results[symbol] = {"error": str(e)}
    
    return results


def batch_pattern_analysis(symbols: List[str], from_date: str, to_date: str) -> List[Dict]:
    """
    Analyze multiple symbols for breakout patterns
    """
    results = []
    
    for symbol in symbols:
        try:
            result = validate_breakout_quality(symbol, from_date, to_date)
            results.append(result)
            time.sleep(1)  
        except Exception as e:
            results.append({
                'symbol': symbol,
                'error': str(e),
                'pattern_type': 'Error',
                'breakout_probability': 0.0
            })
    
    # Sort by probability (highest first)
    results.sort(key=lambda x: x.get('breakout_probability', 0), reverse=True)
    
    return results






company_with_sc_id = {
    "RELIANCE": 2885,
    "BBOX": 8164,
    "TATAMOTORS": 3456,
    "TATAPOWER": 3426,
    "ASTERDM": 1508,
    "HDFCBANK": 1333,
    "TCS": 11536,
    "INFY": 1594,
} 
