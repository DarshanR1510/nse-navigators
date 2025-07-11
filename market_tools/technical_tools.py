from curses import raw
import json
from typing import Dict, Tuple, List
from datetime import datetime
from scipy.signal import find_peaks, argrelextrema
from market_tools.advance_market import refined_ohlc_data, get_nifty_data
from scipy.stats import pearsonr
import pandas as pd
import talib as ta
import numpy as np
import time
from utils.redis_client import main_redis_client as r
from sklearn.cluster import DBSCAN


def closing_sma_and_slope(symbol: str) -> dict:
    """ Calculate Simple Moving Averages (SMA) for a given symbol.
        To get bigger SMA values, you need to provide a longer date range.
        This function calculates SMA and it's slope for 20, 50, and 200 days.
    Args:
        symbol (str): Stock symbol        
    Returns:
        dict: A dictionary containing the SMA and it's slope values for specified windows.
        Returns None if no data is available.
    """
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    df['sma_20'] = ta.SMA(df['close'], 20)
    df['sma_50'] = ta.SMA(df['close'], 50)
    df['sma_200'] = ta.SMA(df['close'], 200)

    df['sma_20_slope'] = df['sma_20'].diff()
    df['sma_50_slope'] = df['sma_50'].diff()
    df['sma_200_slope'] = df['sma_200'].diff()

    lookback = 60
    df_recent = df.tail(lookback).iloc[::-1]    

    return {
        "sma": [
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['sma_20'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['sma_50'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['sma_200'].tolist()],
        ],
        "slope": [
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['sma_20_slope'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['sma_50_slope'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['sma_200_slope'].tolist()],
        ]
    }


def closing_ema_and_slope(symbol: str) -> dict:
    """ Calculate Exponential Moving Averages (EMA) for a given symbol.
        To get bigger EMA values, you need to provide a longer date range.
        This function calculates EMA and it's slope for 20, 50, and 200 days.
    Args:
        symbol (str): Stock symbol
    Returns:
        dict: A dictionary containing the EMA and its slope values for the specified window.
        Returns None if no data is available.
    """
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    df['ema_20'] = ta.EMA(df['close'], 20)
    df['ema_50'] = ta.EMA(df['close'], 50)
    df['ema_200'] = ta.EMA(df['close'], 200)

    df['ema_20_slope'] = df['ema_20'].diff()
    df['ema_50_slope'] = df['ema_50'].diff()
    df['ema_200_slope'] = df['ema_200'].diff()

    lookback = 60
    df_recent = df.tail(lookback).iloc[::-1]

    return {
        "ema": [
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['ema_20'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['ema_50'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['ema_200'].tolist()],
        ],
        "slope": [
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['ema_20_slope'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['ema_50_slope'].tolist()],
            [round(x, 2) if not pd.isna(x) else None for x in df_recent['ema_200_slope'].tolist()],
        ]
    }


def closing_macd(symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Calculate MACD using symbol data from your existing functions.
    
    Args:
        symbol (str): Stock symbol
        exchange (str): Exchange name
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)
        signal_period (int): Signal line EMA period (default: 9)
    
    Returns:
        dict: MACD results or None if insufficient data
    """
    # Get OHLC data using your existing function
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    macd, signal, hist = ta.MACD(df['close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    lookback = 60

    macd_recent = macd.tail(lookback).iloc[::-1]
    signal_recent = signal.tail(lookback).iloc[::-1]
    hist_recent = hist.tail(lookback).iloc[::-1]

    
    return {
        "macd": [round(x, 3) if not pd.isna(x) else None for x in macd_recent.tolist()],
        "signal": [round(x, 3) if not pd.isna(x) else None for x in signal_recent.tolist()],
        "histogram": [round(x, 3) if not pd.isna(x) else None for x in hist_recent.tolist()]
    }


def closing_rsi(symbol: str, period: int = 14) -> float:
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
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    rsi_series = ta.RSI(df['close'], timeperiod=period)
    rsi_value = rsi_series.iloc[-1]  # Get the most recent RSI value
    return round(float(rsi_value), 3)


def closing_bollinger_bands(symbol: str, period: int = 20, std_dev: float = 2.0) -> dict:
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
    
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)

    if df.empty:
        return None

    upper_band, middle_band, lower_band = ta.BBANDS(df['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    lookback = 60

    upper_band_recent = upper_band.tail(lookback).iloc[::-1]
    middle_band_recent = middle_band.tail(lookback).iloc[::-1]
    lower_band_recent = lower_band.tail(lookback).iloc[::-1]

    return {
        "upper_band": [round(x, 3) if not pd.isna(x) else None for x in upper_band_recent.tolist()],
        "middle_band": [round(x, 3) if not pd.isna(x) else None for x in middle_band_recent.tolist()],
        "lower_band": [round(x, 3) if not pd.isna(x) else None for x in lower_band_recent.tolist()]
    }


### ADVANCED TOOLS ###

def analyze_volume_patterns(symbol: str) -> dict:
    """
    Robust, multi-factor analysis of volume and price action to detect key volume-related signals and their strength.
    Returns a dict with signals, scores, and a human-readable summary.
    """
    try:
        row = r.get(f"historical:{symbol}")
        data = json.loads(row) if row else None
        df = refined_ohlc_data(data)
        if df is None or df.empty or len(df) < 50:
            return {
                "symbol": symbol,
                "status": "error",
                "signals": [],
                "overall_volume_score": 0.0,
                "confidence_percentage": 0.0,
                "details": "Insufficient data (minimum 50 periods required)"
            }
        df = df.copy()

        # Fill missing values to avoid calculation errors
        df['volume'] = df['volume'].fillna(0)
        df['close'] = df['close'].fillna(method='ffill').fillna(0)
        df['high'] = df['high'].fillna(df['close'])
        df['low'] = df['low'].fillna(df['close'])

        # Volume Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'volume_ma{window}'] = df['volume'].rolling(window).mean().fillna(0)

        # Price changes and volatility
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['price_change_abs'] = df['price_change'].abs()
        try:
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14).fillna(0)
        except Exception:
            df['atr'] = 0
        df['volatility'] = df['price_change_abs'].rolling(20).std().fillna(0)

        # Volume indicators
        df['volume_ratio'] = df['volume'] / (df['volume_ma20'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
        df['volume_velocity'] = df['volume'].pct_change().fillna(0)

        # Enhanced calculations
        last_vol = float(df['volume'].iloc[-1])
        avg_vol_20 = float(df['volume_ma20'].iloc[-1]) if df['volume_ma20'].iloc[-1] != 0 else 1
        avg_vol_50 = float(df['volume_ma50'].iloc[-1]) if df['volume_ma50'].iloc[-1] != 0 else 1

        # 1. Breakout Confirmation Signal
        try:
            price_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        except Exception:
            price_change_pct = 0.0
        breakout_volume_ratio = last_vol / avg_vol_20 if avg_vol_20 else 0
        recent_vol_sustained = df['volume'].iloc[-3:].mean() / avg_vol_20 if avg_vol_20 else 0
        atr_val = df['atr'].iloc[-1] if 'atr' in df else 1
        atr_normalized_move = abs(price_change_pct) / ((atr_val / df['close'].iloc[-1] * 100) if df['close'].iloc[-1] != 0 else 1)
        breakout_signal = (breakout_volume_ratio > 1.5 and abs(price_change_pct) > 1.0 and recent_vol_sustained > 1.2 and atr_normalized_move > 0.8)
        breakout_score = min((breakout_volume_ratio * atr_normalized_move * recent_vol_sustained) / 4.0, 1.0) if breakout_signal else 0.0
        breakout_details = f"Price: {price_change_pct:.2f}%; Vol ratio: {breakout_volume_ratio:.2f}; Sustained: {recent_vol_sustained:.2f}"

        # 2. Volume Dry-Up Detection
        recent_vol_5 = df['volume'].iloc[-5:].mean()
        recent_vol_10 = df['volume'].iloc[-10:].mean()
        vol_decline_trend = recent_vol_5 / recent_vol_10 if recent_vol_10 else 1.0
        vdu_signal = (recent_vol_5 < 0.7 * avg_vol_20 and vol_decline_trend < 0.9 and df['volatility'].iloc[-1] < df['volatility'].iloc[-10])
        vdu_intensity = max(0, 1 - (recent_vol_5 / (0.7 * avg_vol_20))) if avg_vol_20 else 0
        vdu_score = min(vdu_intensity * (1 - vol_decline_trend), 1.0) if vdu_signal else 0.0
        vdu_details = f"Recent 5d avg: {recent_vol_5:.0f}, Threshold: {0.7*avg_vol_20:.0f}, Decline: {vol_decline_trend:.2f}"

        # 3. Climax Volume Detection
        try:
            volume_percentile_90 = np.percentile(df['volume'].iloc[-50:], 90)
            volume_percentile_95 = np.percentile(df['volume'].iloc[-50:], 95)
        except Exception:
            volume_percentile_90 = volume_percentile_95 = last_vol
        climax_signal = (last_vol >= volume_percentile_90 and abs(price_change_pct) > 2.0)
        climax_intensity = last_vol / volume_percentile_95 if volume_percentile_95 else 0
        climax_score = min(climax_intensity, 1.0) if climax_signal else 0.0
        climax_details = f"Volume: {last_vol:.0f}, 90th percentile: {volume_percentile_90:.0f}, Intensity: {climax_intensity:.2f}"

        # 4. Accumulation/Distribution
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-6)
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
        ad_line = df['money_flow_volume'].ewm(span=14).mean().cumsum().fillna(0)
        ad_slope_20 = np.polyfit(range(20), ad_line.iloc[-20:], 1)[0] if len(ad_line) >= 20 else 0
        ad_slope_10 = np.polyfit(range(10), ad_line.iloc[-10:], 1)[0] if len(ad_line) >= 10 else 0
        ad_consistency = 1.0 if ad_slope_20 > 0 and ad_slope_10 > 0 else 0.5 if ad_slope_10 > 0 else 0.0
        ad_score = min(abs(ad_slope_20) / 1e6, 1.0) * ad_consistency
        ad_details = f"AD 20d slope: {ad_slope_20:.0f}, 10d slope: {ad_slope_10:.0f}, Consistency: {ad_consistency:.1f}"

        # 5. OBV with Trend Strength
        try:
            obv = ta.OBV(df['close'], df['volume'])
            obv_slope_20 = np.polyfit(range(20), obv[-20:], 1)[0] if len(obv) >= 20 else 0
            obv_slope_10 = np.polyfit(range(10), obv[-10:], 1)[0] if len(obv) >= 10 else 0
        except Exception:
            obv_slope_20 = obv_slope_10 = 0
        obv_consistency = 1.0 if obv_slope_20 > 0 and obv_slope_10 > 0 else 0.5 if obv_slope_10 > 0 else 0.0
        obv_score = min(abs(obv_slope_20) / 1e6, 1.0) * obv_consistency
        obv_details = f"OBV 20d slope: {obv_slope_20:.0f}, 10d slope: {obv_slope_10:.0f}, Consistency: {obv_consistency:.1f}"

        # 6. Volume-Price Correlation
        try:
            correlation, p_value = pearsonr(df['volume'].iloc[-20:], df['price_change_abs'].iloc[-20:])
        except Exception:
            correlation, p_value = 0.0, 1.0
        correlation_strength = abs(correlation) if p_value < 0.05 else 0.0
        correlation_score = correlation_strength if correlation > 0 else 0.0
        correlation_details = f"Correlation: {correlation:.3f}, P-value: {p_value:.3f}, Strength: {correlation_strength:.3f}"

        # 7. Volume Momentum Oscillator
        volume_momentum = (df['volume_ma5'].iloc[-1] - df['volume_ma20'].iloc[-1]) / df['volume_ma20'].iloc[-1] if df['volume_ma20'].iloc[-1] != 0 else 0
        momentum_signal = volume_momentum > 0.1
        momentum_score = min(abs(volume_momentum) * 2, 1.0) if momentum_signal else 0.0
        momentum_details = f"Volume momentum: {volume_momentum:.3f}, Signal: {momentum_signal}"

        # 8. Volume Breakout Pattern Recognition
        try:
            volume_peaks, _ = find_peaks(df['volume'].iloc[-30:], height=avg_vol_20 * 1.5)
        except Exception:
            volume_peaks = []
        recent_peaks = len(volume_peaks) > 0 and volume_peaks[-1] >= 25
        pattern_score = 0.0
        if recent_peaks:
            peak_idx = -(30 - volume_peaks[-1])
            peak_volume = df['volume'].iloc[peak_idx]
            peak_price_change = abs(df['price_change'].iloc[peak_idx] * 100)
            pattern_score = min((peak_volume / avg_vol_20) * (peak_price_change / 2.0) / 4.0, 1.0)
        pattern_details = f"Recent volume peaks: {len(volume_peaks)}, Pattern strength: {pattern_score:.3f}"

        # Combine all signals
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
        # Calculate confidence
        signal_consistency = len([s for s in signals if s["score"] > 0.3]) / len(signals)
        data_quality = min(len(df) / 100, 1.0)
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
            "volume_ratio": float(round(last_vol / avg_vol_20, 2)) if avg_vol_20 else 0.0,
            "details": f"Enhanced volume analysis over {len(df)} periods with {len(signals)} indicators and {confidence:.1f}% confidence."
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "status": "error",
            "signals": [],
            "overall_volume_score": 0.0,
            "confidence_percentage": 0.0,
            "details": f"Exception during analysis: {str(e)}"
        }


def calculate_relative_strength(symbol: str, benchmark: str = "NIFTY") -> dict:
    """
    Calculate comprehensive relative strength of a stock vs benchmark index.
    Returns a dict with returns, relative performance, RS rating, trend, volatility, momentum, and summary.
    """
    try:
        today = datetime.today().strftime("%Y-%m-%d")
        nifty_data = get_nifty_data() if benchmark.upper() == "NIFTY" else None  # Extend for other benchmarks if needed

        row = r.get(f"historical:{symbol}")
        data = json.loads(row) if row else None
        stock_df = refined_ohlc_data(data)
        benchmark_df = refined_ohlc_data(nifty_data)

        # Validate data
        if stock_df is None or stock_df.empty:
            return {"symbol": symbol, "status": "error", "details": f"No data found for {symbol}"}
        if benchmark_df is None or benchmark_df.empty:
            return {"symbol": symbol, "status": "error", "details": f"No data found for benchmark {benchmark}"}

        # Ensure datetime index for alignment
        if not pd.api.types.is_datetime64_any_dtype(stock_df.index):
            stock_df.index = pd.to_datetime(stock_df.index)
        if not pd.api.types.is_datetime64_any_dtype(benchmark_df.index):
            benchmark_df.index = pd.to_datetime(benchmark_df.index)

        # Align data by dates
        common_dates = stock_df.index.intersection(benchmark_df.index)
        if len(common_dates) < 50:
            return {"symbol": symbol, "status": "error", "details": "Insufficient data for reliable calculation"}

        stock_prices = stock_df.loc[common_dates, 'close'].sort_index()
        benchmark_prices = benchmark_df.loc[common_dates, 'close'].sort_index()

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
            'benchmark': benchmark,
            'current_date': today,
            'data_points': int(len(common_dates)),
            'period_returns': {},
            'relative_performance': {},
            'rs_rating': 0,
            'rs_trend': 'Neutral',
            'volatility_analysis': {},
            'momentum_indicators': {},
            'summary': {},
            'status': "ok"
        }

        # Calculate period returns and relative performance
        for period, days in periods.items():
            if len(stock_prices) > days and len(benchmark_prices) > days:
                stock_return = ((stock_prices.iloc[-1] / stock_prices.iloc[-days]) - 1) * 100
                benchmark_return = ((benchmark_prices.iloc[-1] / benchmark_prices.iloc[-days]) - 1) * 100

                results['period_returns'][period] = {
                    'stock_return': float(round(stock_return, 2)),
                    'benchmark_return': float(round(benchmark_return, 2)),
                    'outperformance': float(round(stock_return - benchmark_return, 2))
                }

                relative_perf = stock_return / benchmark_return if benchmark_return != 0 else 0
                results['relative_performance'][period] = float(round(relative_perf, 3))

        # RS Rating (0-100 scale)
        rs_scores = []
        weights = {'1M': 0.4, '3M': 0.3, '6M': 0.2, '1Y': 0.1}
        total_weight = sum(weights.values())
        for period, weight in weights.items():
            if period in results['relative_performance']:
                rel_perf = results['relative_performance'][period]
                # Convert to percentile-like score
                if rel_perf > 1.5:
                    score = 95
                elif rel_perf > 1.2:
                    score = 80
                elif rel_perf > 1.1:
                    score = 70
                elif rel_perf > 1.0:
                    score = 60
                elif rel_perf > 0.9:
                    score = 40
                elif rel_perf > 0.8:
                    score = 25
                else:
                    score = 10
                rs_scores.append(score * (weight / total_weight))
        results['rs_rating'] = float(round(sum(rs_scores), 1))

        # RS Trend
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

        # Volatility metrics
        stock_daily_returns = stock_prices.pct_change().dropna()
        benchmark_daily_returns = benchmark_prices.pct_change().dropna()
        try:
            beta = float(round(np.cov(stock_daily_returns, benchmark_daily_returns)[0][1] / np.var(benchmark_daily_returns), 3))
        except Exception:
            beta = None
        try:
            correlation = float(round(np.corrcoef(stock_daily_returns, benchmark_daily_returns)[0][1], 3))
        except Exception:
            correlation = None

        results['volatility_analysis'] = {
            'stock_volatility': float(round(stock_daily_returns.std() * np.sqrt(252) * 100, 2)),
            'benchmark_volatility': float(round(benchmark_daily_returns.std() * np.sqrt(252) * 100, 2)),
            'beta': beta,
            'correlation': correlation
        }

        # Momentum indicators
        try:
            price_momentum_20 = ((stock_prices.iloc[-1] / stock_prices.iloc[-20]) - 1) * 100 if len(stock_prices) > 20 else None
            price_momentum_50 = ((stock_prices.iloc[-1] / stock_prices.iloc[-50]) - 1) * 100 if len(stock_prices) > 50 else None
            sma_20 = stock_prices.rolling(20).mean().iloc[-1] if len(stock_prices) >= 20 else None
            sma_50 = stock_prices.rolling(50).mean().iloc[-1] if len(stock_prices) >= 50 else None
            current_price = stock_prices.iloc[-1]
            results['momentum_indicators'] = {
                'price_momentum_20d': float(round(price_momentum_20, 2)) if price_momentum_20 is not None else None,
                'price_momentum_50d': float(round(price_momentum_50, 2)) if price_momentum_50 is not None else None,
                'price_vs_sma20': float(round(((current_price / sma_20) - 1) * 100, 2)) if sma_20 else None,
                'price_vs_sma50': float(round(((current_price / sma_50) - 1) * 100, 2)) if sma_50 else None,
                'sma20_vs_sma50': float(round(((sma_20 / sma_50) - 1) * 100, 2)) if sma_20 and sma_50 else None
            }
        except Exception:
            results['momentum_indicators'] = {}

        # Generate summary
        recent_outperformance = results['period_returns'].get('1M', {}).get('outperformance', 0)
        stock_vol = results['volatility_analysis'].get('stock_volatility', 0)
        beta_val = results['volatility_analysis'].get('beta', 0)
        results['summary'] = {
            'overall_assessment': results['rs_trend'],
            'recent_performance': 'Outperforming' if recent_outperformance > 0 else 'Underperforming',
            'volatility_profile': 'High' if stock_vol and stock_vol > 30 else 'Medium' if stock_vol and stock_vol > 15 else 'Low',
            'market_sensitivity': 'High' if beta_val and beta_val > 1.2 else 'Medium' if beta_val and beta_val > 0.8 else 'Low'
        }

        return results

    except Exception as e:
        return {"symbol": symbol, "status": "error", "details": f"Error calculating relative strength: {str(e)}"}


def detect_breakout_patterns(symbol: str) -> dict:
    """
    Enhanced breakout pattern detection with improved technical analysis and additional patterns.
    Returns comprehensive analysis including pattern type, probability, direction, risk assessment.
    """
    try:
        # Data retrieval and preparation
        row = r.get(f"historical:{symbol}")
        if not row:
            return {"symbol": symbol, "status": "error", "pattern_type": "Unknown", 
                    "breakout_probability": 0.0, "details": "No data available"}
        
        data = json.loads(row)
        df = refined_ohlc_data(data)
        
        if df.empty or len(df) < 100:
            return {"symbol": symbol, "status": "error", "pattern_type": "Unknown", 
                    "breakout_probability": 0.0, "details": "Insufficient data"}

        # === ENHANCED INDICATORS ===
        try:
            # Moving Averages
            df['sma10'] = ta.SMA(df['close'], 10)
            df['sma20'] = ta.SMA(df['close'], 20)
            df['sma50'] = ta.SMA(df['close'], 50)
            df['sma200'] = ta.SMA(df['close'], 200)
            df['ema9'] = ta.EMA(df['close'], 9)
            df['ema20'] = ta.EMA(df['close'], 20)
            df['ema50'] = ta.EMA(df['close'], 50)
            
            # Volatility
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], 14)
            df['atr_pct'] = (df['atr'] / df['close']).replace([np.inf, -np.inf], 0) * 100
            
            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(df['close'], 20, 2, 2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = ((upper - lower) / middle).replace([np.inf, -np.inf], 0) * 100
            df['bb_position'] = ((df['close'] - lower) / (upper - lower)).replace([np.inf, -np.inf], 0)
            
            # Momentum Indicators
            df['rsi'] = ta.RSI(df['close'], 14)
            macd, macd_signal, macd_hist = ta.MACD(df['close'])
            df['macd'], df['macd_signal'], df['macd_hist'] = macd, macd_signal, macd_hist
            df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
            df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'], 14)
            df['cci'] = ta.CCI(df['high'], df['low'], df['close'], 20)
            
            # Volume Analysis
            df['volume_sma10'] = df['volume'].rolling(10).mean()
            df['volume_sma20'] = df['volume'].rolling(20).mean()
            df['volume_sma50'] = df['volume'].rolling(50).mean()
            df['volume_ratio'] = (df['volume'] / df['volume_sma20']).replace([np.inf, -np.inf], 0)
            df['obv'] = ta.OBV(df['close'], df['volume'])
            df['ad'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # Candlestick Analysis
            df['body'] = abs(df['close'] - df['open'])
            df['body_pct'] = (df['body'] / df['open']).replace([np.inf, -np.inf], 0) * 100
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['total_range'] = df['high'] - df['low']
            df['true_range'] = np.maximum(df['high'] - df['low'],
                                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                                    abs(df['low'] - df['close'].shift(1))))
            
            # Additional Advanced Indicators
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], 14)
            df['dmi_plus'] = ta.PLUS_DI(df['high'], df['low'], df['close'], 14)
            df['dmi_minus'] = ta.MINUS_DI(df['high'], df['low'], df['close'], 14)
            
        except Exception as e:
            return {"symbol": symbol, "status": "error", "pattern_type": "Unknown", 
                    "breakout_probability": 0.0, "details": f"Indicator error: {str(e)}"}

        # === IMPROVED SUPPORT/RESISTANCE DETECTION ===
        def detect_support_resistance_levels(df: pd.DataFrame, lookback: int = 20, 
                                          min_distance: float = 0.02) -> Tuple[List[float], List[float]]:
            """Improved S/R detection using swing highs/lows and clustering"""
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # Find local maxima/minima
            highs_idx = argrelextrema(high_prices, np.greater, order=lookback)[0]
            lows_idx = argrelextrema(low_prices, np.less, order=lookback)[0]
            
            resistance = high_prices[highs_idx]
            support = low_prices[lows_idx]
            
            # Cluster nearby levels (within min_distance)
            def cluster_levels(levels: np.ndarray) -> List[float]:
                if len(levels) == 0:
                    return []
                levels = np.sort(levels)[::-1]  # Descending for resistance
                clusters = []
                current_cluster = [levels[0]]
                
                for price in levels[1:]:
                    if abs(price - np.mean(current_cluster)) / np.mean(current_cluster) < min_distance:
                        current_cluster.append(price)
                    else:
                        clusters.append(np.mean(current_cluster))
                        current_cluster = [price]
                
                if current_cluster:
                    clusters.append(np.mean(current_cluster))
                
                return clusters
            
            resistance_levels = cluster_levels(resistance)
            support_levels = cluster_levels(support)
            
            return resistance_levels[:5], support_levels[:5]

        # === ENHANCED CONSOLIDATION ANALYSIS ===
        def analyze_consolidation_pattern(df: pd.DataFrame, period: int = 20) -> Dict:
            recent_df = df.iloc[-period:]
            
            # Price consolidation metrics
            price_std = recent_df['close'].std()
            price_mean = recent_df['close'].mean()
            coefficient_of_variation = price_std / price_mean if price_mean != 0 else 0
            range_high = recent_df['high'].max()
            range_low = recent_df['low'].min()
            range_spread = (range_high - range_low) / range_low if range_low != 0 else 0
            
            # Volatility metrics
            bb_squeeze = recent_df['bb_width'].mean() < df['bb_width'].rolling(50).mean().iloc[-1] * 0.7
            volatility_compression = recent_df['atr_pct'].mean() < df['atr_pct'].rolling(50).mean().iloc[-1] * 0.7
            
            # Volume metrics
            volume_decline = recent_df['volume'].mean() < df['volume'].rolling(50).mean().iloc[-1] * 0.8
            volume_consistency = recent_df['volume'].std() / recent_df['volume'].mean() if recent_df['volume'].mean() != 0 else 0
            
            # Trend metrics within consolidation
            slope = np.polyfit(np.arange(len(recent_df)), recent_df['close'], 1)[0]
            slope_pct = slope / recent_df['close'].iloc[0] if recent_df['close'].iloc[0] != 0 else 0
            
            return {
                'is_consolidating': range_spread < 0.08 and coefficient_of_variation < 0.05,
                'consolidation_strength': 1 - min(range_spread / 0.08, 1.0) if range_spread < 0.08 else 0.0,
                'bb_squeeze': bb_squeeze,
                'volume_decline': volume_decline,
                'volatility_compression': volatility_compression,
                'range_spread': range_spread,
                'coefficient_of_variation': coefficient_of_variation,
                'volume_consistency': volume_consistency,
                'consolidation_slope': slope_pct,
                'consolidation_type': 'neutral' if abs(slope_pct) < 0.002 else 'ascending' if slope_pct > 0 else 'descending'
            }

        # === ENHANCED BREAKOUT DETECTION ===
        def detect_breakout_candle(df: pd.DataFrame, resistance_levels: List[float], 
                                 support_levels: List[float]) -> Dict:
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Volume analysis
            volume_spike = last_candle['volume'] > 2.0 * last_candle['volume_sma20']
            volume_above_average = last_candle['volume'] > df['volume'].rolling(50).mean().iloc[-1]
            
            # Price momentum
            strong_body = last_candle['body_pct'] > df['body_pct'].rolling(20).mean().iloc[-1] * 1.5
            momentum_breakout = abs(last_candle['close'] - prev_candle['close']) / prev_candle['close'] > 0.02 if prev_candle['close'] != 0 else False
            
            # Breakout confirmation
            breakout_up = False
            breakout_down = False
            breakout_strength = 0
            confirmation_candles = 0
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - last_candle['close']))
                resistance_distance = (last_candle['close'] - nearest_resistance) / nearest_resistance
                
                # Check if price closed above resistance with confirmation
                if last_candle['close'] > nearest_resistance and last_candle['high'] > nearest_resistance:
                    breakout_up = True
                    breakout_strength = resistance_distance
                    
                    # Look for confirmation in previous candles
                    for i in range(2, min(5, len(df))):
                        if df.iloc[-i]['close'] > nearest_resistance * 0.99:  # Within 1% of resistance
                            confirmation_candles += 1
            
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - last_candle['close']))
                support_distance = (nearest_support - last_candle['close']) / nearest_support
                
                if last_candle['close'] < nearest_support and last_candle['low'] < nearest_support:
                    breakout_down = True
                    breakout_strength = support_distance
                    
                    for i in range(2, min(5, len(df))):
                        if df.iloc[-i]['close'] < nearest_support * 1.01:  # Within 1% of support
                            confirmation_candles += 1
            
            return {
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'breakout_strength': breakout_strength,
                'volume_spike': volume_spike,
                'volume_above_average': volume_above_average,
                'strong_body': strong_body,
                'momentum_breakout': momentum_breakout,
                'confirmation_candles': confirmation_candles
            }

        # === IMPROVED TREND ANALYSIS ===
        def analyze_trend_context(df: pd.DataFrame) -> Dict:
            last_close = df['close'].iloc[-1]
            
            # Moving average alignment
            ma_alignment = 0
            if (last_close > df['ema9'].iloc[-1] > df['ema20'].iloc[-1] > df['ema50'].iloc[-1] > df['sma200'].iloc[-1]):
                ma_alignment = 4  # Strong uptrend
            elif (last_close < df['ema9'].iloc[-1] < df['ema20'].iloc[-1] < df['ema50'].iloc[-1] < df['sma200'].iloc[-1]):
                ma_alignment = -4  # Strong downtrend
            elif (last_close > df['ema9'].iloc[-1] > df['ema20'].iloc[-1] > df['ema50'].iloc[-1]):
                ma_alignment = 3
            elif (last_close < df['ema9'].iloc[-1] < df['ema20'].iloc[-1] < df['ema50'].iloc[-1]):
                ma_alignment = -3
            
            # Trend strength indicators
            adx_strength = df['adx'].iloc[-1] > 25
            dmi_bullish = df['dmi_plus'].iloc[-1] > df['dmi_minus'].iloc[-1]
            
            # Momentum indicators
            rsi_bullish = 45 < df['rsi'].iloc[-1] < 70
            rsi_overbought = df['rsi'].iloc[-1] >= 70
            rsi_oversold = df['rsi'].iloc[-1] <= 30
            macd_bullish = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
            cci_bullish = df['cci'].iloc[-1] > -100
            
            # Price position
            above_200ma = last_close > df['sma200'].iloc[-1]
            
            return {
                'trend_strength': abs(ma_alignment),
                'trend_direction': 'up' if ma_alignment > 0 else 'down' if ma_alignment < 0 else 'neutral',
                'adx_strength': adx_strength,
                'dmi_bullish': dmi_bullish,
                'rsi_favorable': rsi_bullish,
                'rsi_extreme': rsi_overbought or rsi_oversold,
                'macd_bullish': macd_bullish,
                'cci_bullish': cci_bullish,
                'above_200ma': above_200ma,
                'price_vs_200ma': (last_close - df['sma200'].iloc[-1]) / df['sma200'].iloc[-1] if df['sma200'].iloc[-1] != 0 else 0
            }

        # === ENHANCED PATTERN IDENTIFICATION ===
        def identify_specific_pattern(df: pd.DataFrame, consolidation: Dict, 
                                    breakout: Dict, trend: Dict) -> str:
            last_close = df['close'].iloc[-1]
            
            # Flag/Pennant detection
            if (consolidation['is_consolidating'] and 
                trend['trend_strength'] >= 2 and 
                consolidation['volume_decline']):
                
                if consolidation['consolidation_type'] == 'ascending' and breakout['breakout_up']:
                    return "Bullish Pennant Breakout"
                elif consolidation['consolidation_type'] == 'descending' and breakout['breakout_down']:
                    return "Bearish Pennant Breakdown"
                elif breakout['breakout_up']:
                    return "Bullish Flag Breakout"
                elif breakout['breakout_down']:
                    return "Bearish Flag Breakdown"
                else:
                    return "Flag/Pennant Formation"
            
            # Triangle patterns
            if consolidation['volatility_compression']:
                if (consolidation['consolidation_type'] == 'ascending' and 
                    breakout['breakout_up']):
                    return "Ascending Triangle Breakout"
                elif (consolidation['consolidation_type'] == 'descending' and 
                      breakout['breakout_down']):
                    return "Descending Triangle Breakdown"
                elif breakout['breakout_up']:
                    return "Symmetrical Triangle Breakout (Bullish)"
                elif breakout['breakout_down']:
                    return "Symmetrical Triangle Breakdown (Bearish)"
                else:
                    return "Triangle Formation"
            
            # Rectangle/Channel patterns
            if (consolidation['is_consolidating'] and 
                not consolidation['volatility_compression'] and 
                consolidation['volume_consistency'] < 0.5):
                
                if breakout['breakout_up']:
                    return "Rectangle Breakout (Bullish)"
                elif breakout['breakout_down']:
                    return "Rectangle Breakdown (Bearish)"
                else:
                    return "Rectangle/Channel Formation"
            
            # Bollinger Band Squeeze
            if (consolidation['bb_squeeze'] and 
                consolidation['volume_decline'] and 
                breakout['volume_spike']):
                
                if breakout['breakout_up']:
                    return "Bollinger Band Squeeze (Bullish)"
                elif breakout['breakout_down']:
                    return "Bollinger Band Squeeze (Bearish)"
                else:
                    return "Bollinger Band Squeeze Formation"
            
            # Cup and Handle (needs longer timeframe)
            if len(df) > 200:
                max_idx = df['high'].idxmax()
                if (max_idx < len(df) - 50 and  # Cup formed in past
                    df['high'].iloc[-20:].max() < df['high'].iloc[max_idx] * 0.95 and  # Handle formation
                    breakout['breakout_up'] and 
                    trend['trend_direction'] == 'up'):
                    return "Cup with Handle Breakout"
            
            # Head and Shoulders (needs specific price structure)
            if len(df) > 100:
                # Simplified detection - would need more sophisticated pattern recognition
                peaks = argrelextrema(df['high'].values, np.greater, order=10)[0]
                if len(peaks) >= 3:
                    left_shoulder = peaks[-3] if len(peaks) >= 3 else -1
                    head = peaks[-2] if len(peaks) >= 2 else -1
                    right_shoulder = peaks[-1] if len(peaks) >= 1 else -1
                    
                    if (left_shoulder != -1 and head != -1 and right_shoulder != -1 and
                        df['high'].iloc[head] > df['high'].iloc[left_shoulder] and
                        df['high'].iloc[head] > df['high'].iloc[right_shoulder] and
                        abs(df['high'].iloc[left_shoulder] - df['high'].iloc[right_shoulder]) / 
                        df['high'].iloc[left_shoulder] < 0.02 and  # Shoulders roughly equal
                        breakout['breakout_down']):
                        return "Head and Shoulders Breakdown"
            
            # Continuation patterns
            if trend['trend_strength'] >= 2 and consolidation['is_consolidating']:
                if trend['trend_direction'] == 'up' and breakout['breakout_up']:
                    return "Continuation Pattern (Bullish)"
                elif trend['trend_direction'] == 'down' and breakout['breakout_down']:
                    return "Continuation Pattern (Bearish)"
            
            return "No Clear Pattern"

        # === IMPROVED PROBABILITY CALCULATION ===
        def calculate_pattern_probability(consolidation: Dict, breakout: Dict, 
                                        trend: Dict) -> float:
            base_prob = 0.5  # Neutral base probability
            
            # Consolidation factors
            if consolidation['is_consolidating']:
                base_prob += 0.15
                if consolidation['bb_squeeze']:
                    base_prob += 0.05
                if consolidation['volume_decline']:
                    base_prob += 0.05
                if consolidation['volatility_compression']:
                    base_prob += 0.05
            
            # Breakout factors
            if breakout['breakout_up'] or breakout['breakout_down']:
                base_prob += 0.15
                if breakout['volume_spike']:
                    base_prob += 0.08
                if breakout['strong_body']:
                    base_prob += 0.05
                if breakout['momentum_breakout']:
                    base_prob += 0.05
                if breakout['confirmation_candles'] >= 1:
                    base_prob += 0.02 * breakout['confirmation_candles']
            
            # Trend factors
            if trend['trend_direction'] == 'up' and breakout['breakout_up']:
                base_prob += trend['trend_strength'] * 0.03
            elif trend['trend_direction'] == 'down' and breakout['breakout_down']:
                base_prob += trend['trend_strength'] * 0.03
            
            if trend['adx_strength']:
                base_prob += 0.05
            if trend['dmi_bullish'] and breakout['breakout_up']:
                base_prob += 0.03
            if not trend['dmi_bullish'] and breakout['breakout_down']:
                base_prob += 0.03
            
            # Volume factors
            if breakout['volume_above_average']:
                base_prob += 0.05
            if breakout['volume_spike']:
                base_prob += 0.05
            
            # RSI factors
            if not trend['rsi_extreme']:
                base_prob += 0.03
            elif (trend['rsi_extreme'] and 
                 ((trend['rsi_favorable'] and breakout['breakout_up']) or 
                 (not trend['rsi_favorable'] and breakout['breakout_down']))):
                base_prob += 0.02
            
            return min(max(base_prob, 0), 1)  # Clamp between 0 and 1

        # === MAIN ANALYSIS ===
        resistance_levels, support_levels = detect_support_resistance_levels(df)
        consolidation_analysis = analyze_consolidation_pattern(df)
        breakout_analysis = detect_breakout_candle(df, resistance_levels, support_levels)
        trend_analysis = analyze_trend_context(df)
        probability = calculate_pattern_probability(consolidation_analysis, breakout_analysis, trend_analysis)
        pattern_type = identify_specific_pattern(df, consolidation_analysis, breakout_analysis, trend_analysis)
        
        direction = "Bullish" if breakout_analysis['breakout_up'] else "Bearish" if breakout_analysis['breakout_down'] else "Neutral"
        risk_level = "High" if probability < 0.4 else "Medium" if probability < 0.7 else "Low"
        
        details = {
            'consolidation_strength': float(round(consolidation_analysis['consolidation_strength'], 3)),
            'range_spread_pct': float(round(consolidation_analysis['range_spread'] * 100, 2)),
            'bb_squeeze': consolidation_analysis['bb_squeeze'],
            'volume_spike': breakout_analysis['volume_spike'],
            'breakout_strength': float(round(breakout_analysis['breakout_strength'], 3)),
            'trend_strength': trend_analysis['trend_strength'],
            'trend_direction': trend_analysis['trend_direction'],
            'support_levels': support_levels[:3],
            'resistance_levels': resistance_levels[:3],
            'rsi_current': float(round(df['rsi'].iloc[-1], 2)),
            'volume_ratio': float(round(df['volume_ratio'].iloc[-1], 2)),
            'price_vs_200ma': float(round(trend_analysis['price_vs_200ma'] * 100, 2)),
            'adx_strength': float(round(df['adx'].iloc[-1], 2))
        }
        
        return {
            "symbol": symbol,
            "current_price": round(float(df['close'].iloc[-1]), 2),
            "pattern_type": pattern_type,
            "breakout_probability": float(round(probability, 3)),
            "direction": direction,
            "risk_level": risk_level,
            "trend_strength": trend_analysis['trend_strength'],
            "trend_direction": trend_analysis['trend_direction'],
            "consolidation_confirmed": consolidation_analysis['is_consolidating'],
            "breakout_confirmed": breakout_analysis['breakout_up'] or breakout_analysis['breakout_down'],
            "volume_confirmation": breakout_analysis['volume_spike'],
            "details": details,
            "summary": f"{pattern_type} | Prob: {float(round(probability*100, 1))}% | Dir: {direction} | Risk: {risk_level} | Trend: {trend_analysis['trend_direction'].title()}"
        }
        
    except Exception as e:
        return {"symbol": symbol, "status": "error", "pattern_type": "Unknown", 
                "breakout_probability": 0.0, "details": f"Exception: {str(e)}"}


def calculate_support_resistance_levels(symbol: str, bins: int = 15) -> dict:
    """
    Enhanced support/resistance level detection with improved technical analysis and machine learning.
    
    Major Improvements:
    1. Added VWAP (Volume Weighted Average Price) detection
    2. Improved Fibonacci calculations with multiple swing points
    3. Added Ichimoku Cloud levels
    4. Enhanced volume profile with dynamic binning
    5. Better level clustering with DBSCAN algorithm
    6. Added trend line detection
    7. Improved level strength calculation
    8. Added time-based weighting for recent levels
    9. Better handling of psychological levels
    10. Added confluence scoring for stronger levels
    
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
    """
    
    # Get data
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)
    
    if df is None or len(df) < 50:
        return {
            "symbol": symbol, 
            "levels": [], 
            "support_levels": [],
            "resistance_levels": [],
            "neutral_levels": [],
            "statistics": {},
            "details": "Insufficient data"
        }
    
    df = df.copy()
    current_price = float(df['close'].iloc[-1])
        
    df['sma20'] = ta.SMA(df['close'], timeperiod=20)
    df['sma50'] = ta.SMA(df['close'], timeperiod=50)
    df['sma200'] = ta.SMA(df['close'], timeperiod=200)
    df['ema20'] = ta.EMA(df['close'], timeperiod=20)
    df['ema50'] = ta.EMA(df['close'], timeperiod=50)
    
    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    
    # VWAP calculation
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cumulative_volume'] = df['volume'].cumsum()
    df['cumulative_vwap'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap'] = df['cumulative_vwap'] / df['cumulative_volume']
    
    # Ichimoku Cloud
    tenkan_sen = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    kijun_sen = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # ===== 1. ENHANCED PIVOT POINT DETECTION =====
    def detect_pivot_points(df: pd.DataFrame) -> List[Dict]:
        """Enhanced pivot point detection with multiple timeframes and VWAP"""
        pivot_levels = []
        
        # Timeframes: intraday, daily, weekly, monthly
        timeframes = [
            (1, '1D'), (3, '3D'), (5, '1W'), (10, '2W'), 
            (20, '1M'), (60, '3M'), (120, '6M')
        ]
        
        for period, label in timeframes:
            if len(df) >= period:
                recent = df.iloc[-period:]
                high = recent['high'].max()
                low = recent['low'].min()
                close = recent['close'].iloc[-1]
                vwap = recent['vwap'].iloc[-1]
                
                # Standard pivot points
                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                s1 = 2 * pivot - high
                
                # Fibonacci pivot points
                fib_r1 = pivot + 0.382 * (high - low)
                fib_s1 = pivot - 0.382 * (high - low)
                
                # Add all pivot levels with time-based weighting
                weight = min(0.5 + (1 / period), 0.9)  # Smaller periods get more weight
                
                pivot_levels.extend([
                    {"level": pivot, "type": "Pivot", "strength": weight * 0.9, "source": f"Pivot_{label}"},
                    {"level": r1, "type": "Resistance", "strength": weight * 0.8, "source": f"R1_{label}"},
                    {"level": s1, "type": "Support", "strength": weight * 0.8, "source": f"S1_{label}"},
                    {"level": fib_r1, "type": "Fibonacci", "strength": weight * 0.7, "source": f"Fib_R1_{label}"},
                    {"level": fib_s1, "type": "Fibonacci", "strength": weight * 0.7, "source": f"Fib_S1_{label}"},
                    {"level": vwap, "type": "VWAP", "strength": weight * 0.85, "source": f"VWAP_{label}"}
                ])
        
        # Add Ichimoku levels
        if len(df) > 52:
            pivot_levels.extend([
                {"level": tenkan_sen.iloc[-1], "type": "Ichimoku", "strength": 0.8, "source": "Tenkan"},
                {"level": kijun_sen.iloc[-1], "type": "Ichimoku", "strength": 0.85, "source": "Kijun"},
                {"level": senkou_span_a.iloc[-1], "type": "Ichimoku", "strength": 0.75, "source": "SpanA"},
                {"level": senkou_span_b.iloc[-1], "type": "Ichimoku", "strength": 0.75, "source": "SpanB"}
            ])
        
        return pivot_levels
    
    # ===== 2. IMPROVED FRACTAL AND SWING ANALYSIS =====
    def detect_fractals_and_swings(df: pd.DataFrame) -> List[Dict]:
        """Enhanced swing point detection with volume confirmation"""
        swing_levels = []
        
        # Find all significant swing points
        highs = df['high'].values
        lows = df['low'].values
        
        # Find peaks and troughs with multiple window sizes
        for window in [3, 5, 8, 13, 21]:
            if len(df) >= window * 2:
                high_peaks = argrelextrema(highs, np.greater, order=window)[0]
                low_peaks = argrelextrema(lows, np.less, order=window)[0]
                
                for peak in high_peaks:
                    if peak < len(df) - 1:  # Not the last candle
                        price = highs[peak]
                        # Calculate strength based on:
                        # 1. Recency (more recent = stronger)
                        # 2. Volume (higher volume = stronger)
                        # 3. Price rejection (longer wicks = stronger)
                        recency = 1 - (len(df) - peak) / len(df)
                        volume_strength = min(df['volume'].iloc[peak] / df['volume'].rolling(20).mean().iloc[peak], 2)
                        wick_ratio = (df['high'].iloc[peak] - max(df['open'].iloc[peak], df['close'].iloc[peak])) / \
                                    (df['high'].iloc[peak] - df['low'].iloc[peak]) if (df['high'].iloc[peak] != df['low'].iloc[peak]) else 0
                        
                        strength = 0.5 + (recency * 0.3) + (volume_strength * 0.1) + (wick_ratio * 0.1)
                        swing_levels.append({
                            "level": float(price),
                            "type": "Resistance",
                            "strength": min(strength, 1.0),
                            "source": f"Swing_High_{window}",
                            "index": int(peak)
                        })
                
                for trough in low_peaks:
                    if trough < len(df) - 1:
                        price = lows[trough]
                        recency = 1 - (len(df) - trough) / len(df)
                        volume_strength = min(df['volume'].iloc[trough] / df['volume'].rolling(20).mean().iloc[trough], 2)
                        wick_ratio = (min(df['open'].iloc[trough], df['close'].iloc[trough]) - df['low'].iloc[trough]) / \
                                    (df['high'].iloc[trough] - df['low'].iloc[trough]) if (df['high'].iloc[trough] != df['low'].iloc[trough]) else 0
                        
                        strength = 0.5 + (recency * 0.3) + (volume_strength * 0.1) + (wick_ratio * 0.1)
                        swing_levels.append({
                            "level": float(price),
                            "type": "Support",
                            "strength": min(strength, 1.0),
                            "source": f"Swing_Low_{window}",
                            "index": int(trough)
                        })
        
        return swing_levels
    
    # ===== 3. ADVANCED VOLUME PROFILE =====
    def create_volume_profile(df: pd.DataFrame, dynamic_bins: bool = True) -> List[Dict]:
        """Enhanced volume profile with dynamic binning and clustering"""
        profile_levels = []
        
        # Dynamic binning based on price volatility
        if dynamic_bins:
            price_range = df['high'].max() - df['low'].min()
            atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            avg_atr = atr.mean()
            bins = max(10, min(50, int(price_range / (avg_atr * 0.5))))
        else:
            bins = 20
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        bin_size = (price_max - price_min) / bins
        price_bins = np.arange(price_min, price_max + bin_size, bin_size)
        
        # Calculate volume at price
        vap = []
        for i in range(len(price_bins) - 1):
            low = price_bins[i]
            high = price_bins[i + 1]
            mask = (df['low'] <= high) & (df['high'] >= low)
            volume = df.loc[mask, 'volume'].sum()
            if volume > 0:
                vap.append({
                    'price': (low + high) / 2,
                    'volume': volume,
                    'count': mask.sum()
                })
        
        if not vap:
            return profile_levels
        
        vap_df = pd.DataFrame(vap)
        
        # Find high volume nodes (HVN) - potential support/resistance
        volume_threshold = vap_df['volume'].quantile(0.75)
        for _, row in vap_df[vap_df['volume'] >= volume_threshold].iterrows():
            strength = min(0.5 + (row['volume'] / vap_df['volume'].max() * 0.5), 1.0)
            profile_levels.append({
                "level": float(row['price']),
                "type": "Volume Node",
                "strength": strength,
                "source": "HVN",
                "volume": float(row['volume']),
                "count": int(row['count'])
            })
        
        # Find low volume nodes (LVN) - potential breakout areas
        low_volume_threshold = vap_df['volume'].quantile(0.25)
        for _, row in vap_df[vap_df['volume'] <= low_volume_threshold].iterrows():
            profile_levels.append({
                "level": float(row['price']),
                "type": "Volume Gap",
                "strength": 0.3,
                "source": "LVN",
                "volume": float(row['volume']),
                "count": int(row['count'])
            })
        
        return profile_levels
    
    # ===== 4. ENHANCED FIBONACCI LEVELS =====
    def calculate_fibonacci_levels(df: pd.DataFrame) -> List[Dict]:
        """Improved Fibonacci with multiple swing points"""
        fib_levels = []
        
        # Find major swings in different timeframes
        swing_windows = [20, 50, 100, 200]
        swing_pairs = []
        
        for window in swing_windows:
            if len(df) >= window:
                recent = df.iloc[-window:]
                swing_high = recent['high'].max()
                swing_low = recent['low'].min()
                swing_pairs.append((swing_high, swing_low))
        
        if not swing_pairs:
            return fib_levels
        
        # Calculate Fibonacci levels for each swing pair
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        ext_ratios = [1.272, 1.618, 2.0, 2.618]
        
        for i, (high, low) in enumerate(swing_pairs):
            diff = high - low
            weight = 0.5 + (0.5 * (i + 1) / len(swing_pairs))  # More weight to larger windows
            
            # Retracement levels
            for ratio in fib_ratios:
                level = high - diff * ratio
                strength = weight * (0.6 + (0.4 * (1 - abs(ratio - 0.5) * 2)))  # 0.5 is strongest
                fib_levels.append({
                    "level": float(level),
                    "type": "Fibonacci",
                    "strength": min(strength, 1.0),
                    "source": f"Fib_{ratio}_{swing_windows[i]}D",
                    "ratio": float(ratio)
                })
            
            # Extension levels
            for ratio in ext_ratios:
                level = high + diff * (ratio - 1)
                fib_levels.append({
                    "level": float(level),
                    "type": "Fibonacci Ext",
                    "strength": weight * 0.7,
                    "source": f"FibExt_{ratio}_{swing_windows[i]}D",
                    "ratio": float(ratio)
                })
        
        return fib_levels
    
    # ===== 5. MOVING AVERAGE CONFLUENCE =====
    def detect_ma_confluence(df: pd.DataFrame) -> List[Dict]:
        """Detect moving average confluence zones with strength calculation"""
        ma_levels = []
        
        # Current MA values
        ma_values = [
            ("SMA20", df['sma20'].iloc[-1], 0.8),
            ("SMA50", df['sma50'].iloc[-1], 0.85),
            ("SMA200", df['sma200'].iloc[-1], 0.9),
            ("EMA20", df['ema20'].iloc[-1], 0.8),
            ("EMA50", df['ema50'].iloc[-1], 0.85),
            ("BB_Upper", df['bb_upper'].iloc[-1], 0.75),
            ("BB_Middle", df['bb_middle'].iloc[-1], 0.7),
            ("BB_Lower", df['bb_lower'].iloc[-1], 0.75),
            ("VWAP", df['vwap'].iloc[-1], 0.85)
        ]
        
        for name, value, base_strength in ma_values:
            if not np.isnan(value):
                # Increase strength if MA is trending
                if name.startswith('SMA') or name.startswith('EMA'):
                    slope = value - (df[name.lower()].iloc[-5] if len(df) >= 5 else value)
                    slope_strength = min(0.1 + abs(slope) / (df['close'].std() or 1) * 0.2, 0.3)
                    if slope > 0:
                        base_strength += slope_strength
                
                ma_levels.append({
                    "level": float(value),
                    "type": "MA" if name.startswith(('SMA', 'EMA')) else name,
                    "strength": min(base_strength, 1.0),
                    "source": name
                })
        
        # Detect MA clusters (confluence zones)
        ma_values = [x[1] for x in ma_values if not np.isnan(x[1])]
        if len(ma_values) >= 3:
            ma_values = np.array(ma_values).reshape(-1, 1)
            
            clustering = DBSCAN(eps=0.03*current_price, min_samples=2).fit(ma_values)
            
            for label in set(clustering.labels_):
                if label != -1:  # -1 is noise
                    cluster_points = ma_values[clustering.labels_ == label]
                    if len(cluster_points) >= 2:
                        cluster_mean = float(cluster_points.mean())
                        strength = min(0.7 + (len(cluster_points) * 0.1), 1.0)
                        ma_levels.append({
                            "level": cluster_mean,
                            "type": "MA Cluster",
                            "strength": strength,
                            "source": f"MA_Confluence_{len(cluster_points)}",
                            "count": len(cluster_points)
                        })
        
        return ma_levels
    
    # ===== 6. TREND LINE DETECTION =====
    def detect_trend_lines(df: pd.DataFrame) -> List[Dict]:
        """Detect significant trend lines using swing points"""
        trend_levels = []
        
        if len(df) < 30:
            return trend_levels
        
        # Find significant swing highs and lows
        highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
        lows = argrelextrema(df['low'].values, np.less, order=5)[0]
        
        # We'll implement a simplified version - full trend line detection would be more complex
        # Here we just take the most recent swing points
        if len(highs) >= 2:
            last_two_highs = sorted(highs[-2:])
            x = np.array([last_two_highs[0], last_two_highs[1]])
            y = df['high'].iloc[last_two_highs].values
            slope, intercept = np.polyfit(x, y, 1)
            
            # Project the trend line to current position
            current_level = slope * (len(df) - 1) + intercept
            trend_levels.append({
                "level": float(current_level),
                "type": "Trend Resistance",
                "strength": 0.8,
                "source": "TrendLine_Highs",
                "slope": float(slope)
            })
        
        if len(lows) >= 2:
            last_two_lows = sorted(lows[-2:])
            x = np.array([last_two_lows[0], last_two_lows[1]])
            y = df['low'].iloc[last_two_lows].values
            slope, intercept = np.polyfit(x, y, 1)
            
            current_level = slope * (len(df) - 1) + intercept
            trend_levels.append({
                "level": float(current_level),
                "type": "Trend Support",
                "strength": 0.8,
                "source": "TrendLine_Lows",
                "slope": float(slope)
            })
        
        return trend_levels
    
    # ===== 7. PSYCHOLOGICAL LEVELS =====
    def detect_psychological_levels(df: pd.DataFrame) -> List[Dict]:
        """Enhanced psychological level detection"""
        psych_levels = []
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        # Determine round number base
        if current_price >= 1000:
            base = 100
        elif current_price >= 100:
            base = 10
        elif current_price >= 10:
            base = 5
        else:
            base = 1
        
        # Generate round numbers
        start = int(price_min / base) * base
        end = int(price_max / base + 1) * base
        
        for level in range(start, end + base, base):
            if price_min <= level <= price_max:
                # Check price interactions with this level
                touches = ((df['low'] <= level) & (df['high'] >= level)).sum()
                if touches > 0:
                    # Strength based on number of touches and recency
                    recent_touches = ((df['low'].iloc[-20:] <= level) & 
                                     (df['high'].iloc[-20:] >= level)).sum()
                    strength = 0.3 + min(touches / 10, 0.4) + min(recent_touches / 5, 0.3)
                    psych_levels.append({
                        "level": float(level),
                        "type": "Psychological",
                        "strength": min(strength, 1.0),
                        "source": f"Round_{base}",
                        "touches": touches,
                        "recent_touches": recent_touches
                    })
        
        # Special numbers (e.g., 1234, 1111, etc.)
        special_numbers = []
        max_digits = len(str(int(price_max)))
        
        for digits in range(1, min(4, max_digits + 1)):
            for num in range(1, 10):
                special_num = int(str(num) * digits)
                if price_min <= special_num <= price_max:
                    special_numbers.append(special_num)
        
        for num in special_numbers:
            touches = ((df['low'] <= num) & (df['high'] >= num)).sum()
            if touches > 0:
                strength = 0.4 + min(touches / 5, 0.6)
                psych_levels.append({
                    "level": float(num),
                    "type": "Psychological",
                    "strength": min(strength, 1.0),
                    "source": f"Special_{num}",
                    "touches": touches
                })
        
        return psych_levels
    
    # ===== 8. LEVEL VALIDATION AND CLUSTERING =====
    def process_levels(all_levels: List[Dict], current_price: float) -> Dict:
        """Validate, cluster, and score all levels"""
        if not all_levels:
            return {
                "levels": [],
                "support_levels": [],
                "resistance_levels": [],
                "neutral_levels": []
            }
        
        # Filter out levels too far from current price
        price_std = np.std([x['level'] for x in all_levels])
        valid_levels = [
            x for x in all_levels 
            if abs(x['level'] - current_price) <= 2 * price_std
        ]
        
        if not valid_levels:
            return {
                "levels": [],
                "support_levels": [],
                "resistance_levels": [],
                "neutral_levels": []
            }
        
        # Cluster similar levels using DBSCAN
        levels_array = np.array([x['level'] for x in valid_levels]).reshape(-1, 1)
        clustering = DBSCAN(
            eps=0.02 * current_price,  # 2% tolerance
            min_samples=1
        ).fit(levels_array)
        
        # Process clusters
        clustered_levels = []
        for label in set(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == label)[0]
            cluster_levels = [valid_levels[i] for i in cluster_indices]
            
            if len(cluster_levels) == 1:
                clustered_levels.append(cluster_levels[0])
            else:
                # Calculate weighted average level
                total_strength = sum(x['strength'] for x in cluster_levels)
                weighted_level = sum(x['level'] * x['strength'] for x in cluster_levels) / total_strength
                
                # Calculate combined strength with diminishing returns
                combined_strength = min(0.3 + 0.7 * (1 - np.exp(-total_strength / 0.7)), 1.0)
                
                # Determine most common type
                types = [x['type'] for x in cluster_levels]
                main_type = max(set(types), key=types.count)
                
                # Combine sources
                sources = list(set([x['source'] for x in cluster_levels]))
                
                clustered_levels.append({
                    "level": float(round(weighted_level, 4)),
                    "type": main_type,
                    "strength": float(round(combined_strength, 3)),
                    "source": f"Cluster_{len(cluster_levels)}",
                    "sources": sources,
                    "count": len(cluster_levels)
                })
        
        # Add confluence score (number of methods confirming this level)
        for level in clustered_levels:
            if 'count' in level:
                level['confluence'] = level['count']
            else:
                level['confluence'] = 1
        
        # Sort by strength * confluence
        clustered_levels.sort(
            key=lambda x: x['strength'] * (1 + 0.1 * x['confluence']), 
            reverse=True
        )
        
        # Limit to top levels
        top_levels = clustered_levels[:30]
        
        # Add distance from current price
        for level in top_levels:
            level['distance_pct'] = float(round(
                ((level['level'] - current_price) / current_price) * 100, 2
            ))
        
        # Separate into support, resistance, and neutral
        support = [x for x in top_levels if x['level'] < current_price * 0.995]
        resistance = [x for x in top_levels if x['level'] > current_price * 1.005]
        neutral = [x for x in top_levels if x not in support and x not in resistance]
        
        return {
            "levels": top_levels,
            "support_levels": support[:5],  # Top 5 support levels
            "resistance_levels": resistance[:5],  # Top 5 resistance levels
            "neutral_levels": neutral
        }
    
    # ===== MAIN EXECUTION =====
    
    # Collect all levels from different methods
    all_levels = []
    
    # 1. Pivot Points
    all_levels.extend(detect_pivot_points(df))
    
    # 2. Fractals and Swings
    all_levels.extend(detect_fractals_and_swings(df))
    
    # 3. Volume Profile
    all_levels.extend(create_volume_profile(df))
    
    # 4. Fibonacci Levels
    all_levels.extend(calculate_fibonacci_levels(df))
    
    # 5. Moving Average Confluence
    all_levels.extend(detect_ma_confluence(df))
    
    # 6. Trend Lines
    all_levels.extend(detect_trend_lines(df))
    
    # 7. Psychological Levels
    all_levels.extend(detect_psychological_levels(df))
    
    # Process and cluster all levels
    processed = process_levels(all_levels, current_price)
    
    # Add statistics
    method_counts = {}
    for level in all_levels:
        source = level['source'].split('_')[0]
        method_counts[source] = method_counts.get(source, 0) + 1
    
    statistics = {
        "total_levels_detected": len(all_levels),
        "clustered_levels": len(processed["levels"]),
        "method_counts": method_counts,
        "price_volatility": float(round(df['close'].pct_change().std() * 100, 2)),
        "average_true_range": float(round(ta.ATR(df['high'], df['low'], df['close'], 14).iloc[-1], 2))
    }
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "levels": processed["levels"],
        "support_levels": processed["support_levels"],
        "resistance_levels": processed["resistance_levels"],
        "neutral_levels": processed["neutral_levels"],
        "statistics": statistics,
        "details": (
            f"Detected {len(all_levels)} potential levels from {len(method_counts)} methods, "
            f"clustered into {len(processed['levels'])} key levels. "
            f"Current price volatility: {statistics['price_volatility']}%"
        )
    }


def momentum_indicators(symbol: str) -> dict:
    """
    Enhanced momentum analysis with additional indicators and improved signal processing.
    
    Major Improvements:
    1. Added MACD analysis with histogram momentum
    2. Implemented ADX for trend strength measurement
    3. Added Aroon oscillator for trend detection
    4. Improved RSI analysis with divergence detection
    5. Added Bollinger Band momentum analysis
    6. Enhanced signal confirmation with multiple timeframes
    7. Added volume confirmation for momentum signals
    8. Improved overall scoring system with weighted factors
    9. Added market regime detection (trending/range-bound)
    10. Better risk assessment and confidence calculation
    
    Returns:
        dict: {
            "symbol": str,
            "status": str,
            "timestamp": str,
            "data_period": str,
            "current_values": dict,         # Latest values for all indicators
            "indicator_analysis": dict,     # Detailed analysis by indicator type
            "overall_assessment": dict,     # Comprehensive momentum assessment
            "market_regime": str,           # "Trending" or "Range-bound"
            "trend_strength": float,        # 0-100 strength of current trend
            "analysis_summary": str,        # Human-readable summary
            "recommendation": str,          # "bullish", "bearish", or "neutral"
            "confidence": float,            # Confidence score (0-100)
            "momentum_score": float,        # Overall momentum score (-1 to 1)
            "strength_assessment": str,     # "Strong", "Moderate", or "Weak"
            "key_insights": list,           # List of key findings
            "signals": dict,                # Trading signal details
            "risk_metrics": dict            # Risk assessment metrics
        }
    """
    
    # Get data
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)
    
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
    current_price = df['close'].iloc[-1]
    
    # ==================== ENHANCED MOMENTUM INDICATORS ====================
    
    # 1. Rate of Change (ROC) - Multiple timeframes
    roc_periods = [3, 5, 10, 14, 20, 50]
    for period in roc_periods:
        df[f'roc_{period}'] = ta.ROC(df['close'], timeperiod=period)
    
    # 2. Williams %R - Multiple timeframes
    willr_periods = [7, 14, 21, 28, 50]
    for period in willr_periods:
        df[f'williams_r_{period}'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
    
    # 3. Stochastic Oscillator - Multiple configurations
    stoch_configs = {
        'fast': (5, 3, 3),
        'standard': (14, 3, 3),
        'slow': (21, 5, 5),
        'long_term': (50, 5, 5)
    }
    for name, (fastk, slowk, slowd) in stoch_configs.items():
        k, d = ta.STOCH(df['high'], df['low'], df['close'], 
                       fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)
        df[f'stoch_k_{name}'], df[f'stoch_d_{name}'] = k, d
    
    # 4. Core Momentum Indicators
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    df['momentum'] = ta.MOM(df['close'], timeperiod=10)
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    
    # 5. Trend Strength Indicators
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['dmi_plus'] = ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['dmi_minus'] = ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 6. Additional Momentum Indicators
    macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    df['macd_hist'] = macdhist
    
    aroon_down, aroon_up = ta.AROON(df['high'], df['low'], timeperiod=14)
    df['aroon_down'] = aroon_down
    df['aroon_up'] = aroon_up
    df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
    
    # 7. Price Momentum Derivatives
    df['price_velocity'] = df['close'].pct_change()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 8. Volume Momentum Indicators
    df['volume_roc'] = ta.ROC(df['volume'], timeperiod=5)
    df['obv'] = ta.OBV(df['close'], df['volume'])
    df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # 9. Bollinger Band Momentum
    upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_percent'] = (df['close'] - lower) / (upper - lower)
    df['bb_width'] = (upper - lower) / middle
    
    # ==================== CURRENT VALUES ====================
    
    current_values = {
        # ROC values
        **{f'roc_{p}': float(round(df[f'roc_{p}'].iloc[-1], 3)) for p in roc_periods},
        
        # Williams %R values
        **{f'williams_r_{p}': float(round(df[f'williams_r_{p}'].iloc[-1], 2)) for p in willr_periods},
        
        # Stochastic values
        **{f'stoch_k_{name}': float(round(df[f'stoch_k_{name}'].iloc[-1], 2)) for name in stoch_configs},
        **{f'stoch_d_{name}': float(round(df[f'stoch_d_{name}'].iloc[-1], 2)) for name in stoch_configs},
        
        # Core momentum indicators
        'rsi': float(round(df['rsi'].iloc[-1], 2)),
        'momentum': float(round(df['momentum'].iloc[-1], 3)),
        'cci': float(round(df['cci'].iloc[-1], 2)),
        'mfi': float(round(df['mfi'].iloc[-1], 2)),
        
        # Trend strength indicators
        'adx': float(round(df['adx'].iloc[-1], 2)),
        'dmi_plus': float(round(df['dmi_plus'].iloc[-1], 2)),
        'dmi_minus': float(round(df['dmi_minus'].iloc[-1], 2)),
        
        # Additional indicators
        'macd': float(round(df['macd'].iloc[-1], 3)),
        'macd_signal': float(round(df['macd_signal'].iloc[-1], 3)),
        'macd_hist': float(round(df['macd_hist'].iloc[-1], 3)),
        'aroon_up': float(round(df['aroon_up'].iloc[-1], 2)),
        'aroon_down': float(round(df['aroon_down'].iloc[-1], 2)),
        'aroon_osc': float(round(df['aroon_osc'].iloc[-1], 2)),
        
        # Price derivatives
        'price_velocity': float(round(df['price_velocity'].iloc[-1], 4)),
        'price_acceleration': float(round(df['price_acceleration'].iloc[-1], 4)),
        
        # Volume indicators
        'volume_roc': float(round(df['volume_roc'].iloc[-1], 2)),
        'obv': float(round(df['obv'].iloc[-1], 2)),
        'volume_ratio': float(round(df['volume_ratio'].iloc[-1], 2)),
        
        # Bollinger Bands
        'bb_percent': float(round(df['bb_percent'].iloc[-1], 3)),
        'bb_width': float(round(df['bb_width'].iloc[-1], 4))
    }
    
    # ==================== ENHANCED ANALYSIS FUNCTIONS ====================
    
    def analyze_roc_signals(df) -> Dict:
        """Enhanced ROC analysis with multiple timeframes"""
        signals = []
        strengths = []
        directions = []
        
        for period in roc_periods:
            roc = df[f'roc_{period}'].iloc[-1]
            roc_prev = df[f'roc_{period}'].iloc[-2]
            
            if roc > 0 and roc > roc_prev:
                signals.append(1)  # Bullish
                strengths.append(min(abs(roc)/10, 1))
                directions.append("accelerating")
            elif roc > 0:
                signals.append(0.5)  # Mild bullish
                strengths.append(min(abs(roc)/20, 0.5))
                directions.append("decelerating" if roc < roc_prev else "steady")
            elif roc < 0 and roc < roc_prev:
                signals.append(-1)  # Bearish
                strengths.append(min(abs(roc)/10, 1))
                directions.append("accelerating")
            else:
                signals.append(-0.5)  # Mild bearish
                strengths.append(min(abs(roc)/20, 0.5))
                directions.append("decelerating" if roc > roc_prev else "steady")
        
        # Calculate overall ROC score
        avg_signal = sum(signals) / len(signals)
        avg_strength = sum(strengths) / len(strengths)
        
        # Determine primary direction
        bull_count = sum(1 for s in signals if s > 0)
        bear_count = sum(1 for s in signals if s < 0)
        
        if bull_count > bear_count and bull_count >= len(roc_periods)//2:
            primary_signal = "Bullish"
        elif bear_count > bull_count and bear_count >= len(roc_periods)//2:
            primary_signal = "Bearish"
        else:
            primary_signal = "Neutral"
        
        # Check alignment
        alignment = sum(1 for d in directions if "accelerating" in d) / len(directions)
        
        return {
            "primary_signal": primary_signal,
            "signal_strength": float(round(avg_strength, 3)),
            "alignment_score": float(round(alignment, 3)),
            "trend_direction": max(set(directions), key=directions.count),
            "interpretation": f"ROC shows {primary_signal.lower()} momentum with {max(set(directions), key=directions.count)} trend. "
                           f"Alignment score: {alignment:.1%} across {len(roc_periods)} timeframes",
            "details": {f"roc_{p}": {
                "value": current_values[f'roc_{p}'],
                "signal": "Bullish" if signals[i] > 0 else "Bearish" if signals[i] < 0 else "Neutral",
                "strength": strengths[i],
                "direction": directions[i]
            } for i, p in enumerate(roc_periods)}
        }
    
    def analyze_williams_r_signals(df) -> Dict:
        """Enhanced Williams %R analysis with divergence detection"""
        signals = []
        conditions = []
        strengths = []
        
        for period in willr_periods:
            wr = df[f'williams_r_{period}'].iloc[-1]
            wr_prev = df[f'williams_r_{period}'].iloc[-2]
            price_curr = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-2]
            
            # Basic condition
            if wr <= -80:
                conditions.append("Oversold")
                signal_strength = min((abs(wr + 80) / 20, 1))
                if wr > wr_prev and price_curr > price_prev:
                    signals.append(1 * signal_strength)  # Bullish reversal
                else:
                    signals.append(0.5 * signal_strength)  # Potential reversal
            elif wr >= -20:
                conditions.append("Overbought")
                signal_strength = min((abs(wr + 20) / 20, 1))
                if wr < wr_prev and price_curr < price_prev:
                    signals.append(-1 * signal_strength)  # Bearish reversal
                else:
                    signals.append(-0.5 * signal_strength)  # Potential reversal
            elif -50 <= wr <= -30:
                conditions.append("Bullish")
                signals.append(0.7)
            elif -70 <= wr <= -50:
                conditions.append("Bearish")
                signals.append(-0.7)
            else:
                conditions.append("Neutral")
                signals.append(0)
            
            strengths.append(abs(signals[-1]))
        
        # Divergence detection
        div_signal = None
        if len(df) > 20:
            # Check for bullish divergence (price lower lows, W%R higher lows)
            recent_lows = df['close'].rolling(5).min().iloc[-5:]
            recent_wr = df[f'williams_r_{willr_periods[0]}'].rolling(5).min().iloc[-5:]
            if (recent_lows.is_monotonic_decreasing and 
                recent_wr.is_monotonic_increasing):
                div_signal = "Bullish Divergence"
                signals.append(0.8)
            
            # Check for bearish divergence (price higher highs, W%R lower highs)
            recent_highs = df['close'].rolling(5).max().iloc[-5:]
            recent_wr_highs = df[f'williams_r_{willr_periods[0]}'].rolling(5).max().iloc[-5:]
            if (recent_highs.is_monotonic_increasing and 
                recent_wr_highs.is_monotonic_decreasing):
                div_signal = "Bearish Divergence"
                signals.append(-0.8)
        
        avg_signal = sum(signals) / len(signals)
        avg_strength = sum(strengths) / len(strengths)
        
        # Determine primary condition
        primary_condition = max(set(conditions), key=conditions.count)
        
        return {
            "primary_condition": primary_condition,
            "signal_strength": float(round(avg_strength, 3)),
            "divergence": div_signal,
            "interpretation": f"Williams %R shows {primary_condition.lower()} conditions "
                           f"{'with ' + div_signal.lower() if div_signal else ''} "
                           f"across {len(willr_periods)} timeframes",
            "details": {f"williams_r_{p}": {
                "value": current_values[f'williams_r_{p}'],
                "condition": conditions[i],
                "strength": strengths[i]
            } for i, p in enumerate(willr_periods)}
        }
    
    def analyze_stochastic_signals(df) -> Dict:
        """Enhanced Stochastic analysis with multiple configurations"""
        signals = []
        conditions = []
        crossovers = []
        strengths = []
        
        for name in stoch_configs:
            k = df[f'stoch_k_{name}'].iloc[-1]
            d = df[f'stoch_d_{name}'].iloc[-1]
            k_prev = df[f'stoch_k_{name}'].iloc[-2]
            d_prev = df[f'stoch_d_{name}'].iloc[-2]
            
            # Condition
            if k <= 20 and d <= 20:
                conditions.append("Oversold")
                base_strength = min((20 - min(k, d)) / 20, 1)
            elif k >= 80 and d >= 80:
                conditions.append("Overbought")
                base_strength = min((max(k, d) - 80) / 20, 1)
            else:
                conditions.append("Neutral")
                base_strength = 0.3
            
            # Crossover
            if k > d and k_prev <= d_prev:
                crossovers.append("Bullish")
                signal = 1 * base_strength
            elif k < d and k_prev >= d_prev:
                crossovers.append("Bearish")
                signal = -1 * base_strength
            else:
                crossovers.append("None")
                signal = 0
            
            signals.append(signal)
            strengths.append(abs(signal))
        
        avg_signal = sum(signals) / len(signals)
        avg_strength = sum(strengths) / len(strengths)
        
        # Determine primary condition and crossover
        primary_condition = max(set(conditions), key=conditions.count)
        primary_crossover = max(set(crossovers), key=crossovers.count) if crossovers else "None"
        
        return {
            "primary_condition": primary_condition,
            "primary_crossover": primary_crossover,
            "signal_strength": float(round(avg_strength, 3)),
            "interpretation": f"Stochastic shows {primary_condition.lower()} conditions "
                           f"with {primary_crossover.lower()} crossover across {len(stoch_configs)} configurations",
            "details": {f"stoch_{name}": {
                "k_value": current_values[f'stoch_k_{name}'],
                "d_value": current_values[f'stoch_d_{name}'],
                "condition": conditions[i],
                "crossover": crossovers[i],
                "strength": strengths[i]
            } for i, name in enumerate(stoch_configs)}
        }
    
    def analyze_macd_signals(df) -> Dict:
        """Enhanced MACD analysis with histogram momentum"""
        macd = df['macd'].iloc[-1]
        signal = df['macd_signal'].iloc[-1]
        hist = df['macd_hist'].iloc[-1]
        hist_prev = df['macd_hist'].iloc[-2]
        
        # Crossover
        if macd > signal and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            crossover = "Bullish"
            crossover_strength = min(abs(macd - signal) / (abs(macd) + abs(signal) + 0.0001), 1)
        elif macd < signal and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
            crossover = "Bearish"
            crossover_strength = min(abs(macd - signal) / (abs(macd) + abs(signal) + 0.0001), 1)
        else:
            crossover = "None"
            crossover_strength = 0
        
        # Histogram momentum
        if hist > 0 and hist > hist_prev:
            hist_momentum = "Bullish Accelerating"
            hist_strength = min(abs(hist) * 10, 1)
        elif hist > 0:
            hist_momentum = "Bullish Decelerating"
            hist_strength = min(abs(hist) * 5, 0.7)
        elif hist < 0 and hist < hist_prev:
            hist_momentum = "Bearish Accelerating"
            hist_strength = min(abs(hist) * 10, 1)
        elif hist < 0:
            hist_momentum = "Bearish Decelerating"
            hist_strength = min(abs(hist) * 5, 0.7)
        else:
            hist_momentum = "Neutral"
            hist_strength = 0
        
        # Divergence detection
        div_signal = None
        if len(df) > 26:  # MACD minimum period
            # Check for bullish divergence (price lower lows, MACD higher lows)
            recent_lows = df['close'].rolling(5).min().iloc[-5:]
            recent_macd = df['macd'].rolling(5).min().iloc[-5:]
            if (recent_lows.is_monotonic_decreasing and 
                recent_macd.is_monotonic_increasing):
                div_signal = "Bullish Divergence"
            
            # Check for bearish divergence (price higher highs, MACD lower highs)
            recent_highs = df['close'].rolling(5).max().iloc[-5:]
            recent_macd_highs = df['macd'].rolling(5).max().iloc[-5:]
            if (recent_highs.is_monotonic_increasing and 
                recent_macd_highs.is_monotonic_decreasing):
                div_signal = "Bearish Divergence"
        
        # Overall signal
        if crossover != "None" or hist_momentum != "Neutral":
            if "Bullish" in crossover or "Bullish" in hist_momentum:
                overall_signal = "Bullish"
                signal_strength = max(crossover_strength, hist_strength)
            else:
                overall_signal = "Bearish"
                signal_strength = max(crossover_strength, hist_strength)
        else:
            overall_signal = "Neutral"
            signal_strength = 0
        
        return {
            "crossover": crossover,
            "crossover_strength": float(round(crossover_strength, 3)),
            "histogram_momentum": hist_momentum,
            "histogram_strength": float(round(hist_strength, 3)),
            "divergence": div_signal,
            "overall_signal": overall_signal,
            "signal_strength": float(round(signal_strength, 3)),
            "interpretation": f"MACD shows {overall_signal.lower()} signal with {hist_momentum.lower()} momentum "
                           f"{'and ' + div_signal.lower() if div_signal else ''}",
            "details": {
                "macd_line": current_values['macd'],
                "signal_line": current_values['macd_signal'],
                "histogram": current_values['macd_hist']
            }
        }
    
    def analyze_trend_strength(df) -> Dict:
        """Comprehensive trend strength analysis using ADX and other indicators"""
        adx = df['adx'].iloc[-1]
        dmi_plus = df['dmi_plus'].iloc[-1]
        dmi_minus = df['dmi_minus'].iloc[-1]
        
        # ADX interpretation
        if adx > 25:
            trend_strength = "Strong"
            strength_score = min((adx - 25) / 25, 1)
        elif adx > 20:
            trend_strength = "Moderate"
            strength_score = min((adx - 20) / 30, 0.8)
        else:
            trend_strength = "Weak"
            strength_score = adx / 20
        
        # Trend direction
        if dmi_plus > dmi_minus:
            trend_direction = "Up"
            direction_strength = min((dmi_plus - dmi_minus) / 25, 1)
        elif dmi_minus > dmi_plus:
            trend_direction = "Down"
            direction_strength = min((dmi_minus - dmi_plus) / 25, 1)
        else:
            trend_direction = "Neutral"
            direction_strength = 0
        
        # Market regime
        if adx > 25:
            market_regime = "Trending"
        else:
            market_regime = "Range-bound"
        
        # Aroon confirmation
        aroon_up = df['aroon_up'].iloc[-1]
        aroon_down = df['aroon_down'].iloc[-1]
        if aroon_up > 70 and aroon_down < 30:
            aroon_signal = "Strong Uptrend"
        elif aroon_down > 70 and aroon_up < 30:
            aroon_signal = "Strong Downtrend"
        elif aroon_up > aroon_down:
            aroon_signal = "Mild Uptrend"
        elif aroon_down > aroon_up:
            aroon_signal = "Mild Downtrend"
        else:
            aroon_signal = "No Clear Trend"
        
        return {
            "adx_value": float(round(adx, 2)),
            "trend_strength": trend_strength,
            "strength_score": float(round(strength_score, 3)),
            "trend_direction": trend_direction,
            "direction_strength": float(round(direction_strength, 3)),
            "market_regime": market_regime,
            "aroon_signal": aroon_signal,
            "interpretation": f"Market is {market_regime.lower()} with {trend_strength.lower()} "
                           f"{trend_direction.lower()} trend. Aroon indicates {aroon_signal.lower()}."
        }
    
    def analyze_rsi_signals(df) -> Dict:
        """Enhanced RSI analysis with divergence detection"""
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2]
        price_curr = df['close'].iloc[-1]
        price_prev = df['close'].iloc[-2]
        
        # Basic condition
        if rsi <= 30:
            condition = "Oversold"
            strength = min((30 - rsi) / 30, 1)
            if rsi > rsi_prev and price_curr > price_prev:
                signal = "Bullish Reversal"
                signal_strength = strength
            else:
                signal = "Potential Reversal"
                signal_strength = strength * 0.7
        elif rsi >= 70:
            condition = "Overbought"
            strength = min((rsi - 70) / 30, 1)
            if rsi < rsi_prev and price_curr < price_prev:
                signal = "Bearish Reversal"
                signal_strength = strength
            else:
                signal = "Potential Reversal"
                signal_strength = strength * 0.7
        elif 30 < rsi < 50:
            condition = "Bearish Zone"
            signal = "Bearish"
            signal_strength = min((50 - rsi) / 20, 0.7)
        elif 50 <= rsi < 70:
            condition = "Bullish Zone"
            signal = "Bullish"
            signal_strength = min((rsi - 50) / 20, 0.7)
        else:
            condition = "Neutral"
            signal = "Neutral"
            signal_strength = 0
        
        # Divergence detection
        div_signal = None
        if len(df) > 14:
            # Bullish divergence (price lower lows, RSI higher lows)
            recent_lows = df['close'].rolling(5).min().iloc[-5:]
            recent_rsi = df['rsi'].rolling(5).min().iloc[-5:]
            if (recent_lows.is_monotonic_decreasing and 
                recent_rsi.is_monotonic_increasing):
                div_signal = "Bullish Divergence"
                signal_strength = max(signal_strength, 0.8)
            
            # Bearish divergence (price higher highs, RSI lower highs)
            recent_highs = df['close'].rolling(5).max().iloc[-5:]
            recent_rsi_highs = df['rsi'].rolling(5).max().iloc[-5:]
            if (recent_highs.is_monotonic_increasing and 
                recent_rsi_highs.is_monotonic_decreasing):
                div_signal = "Bearish Divergence"
                signal_strength = max(signal_strength, 0.8)
        
        return {
            "condition": condition,
            "signal": signal,
            "signal_strength": float(round(signal_strength, 3)),
            "divergence": div_signal,
            "interpretation": f"RSI shows {condition.lower()} conditions with {signal.lower()} signal "
                           f"{'and ' + div_signal.lower() if div_signal else ''}",
            "details": {
                "rsi_value": current_values['rsi'],
                "previous_rsi": float(round(rsi_prev, 2))
            }
        }
    
    def analyze_volume_signals(df) -> Dict:
        """Volume confirmation analysis"""
        volume_ratio = df['volume_ratio'].iloc[-1]
        volume_roc = df['volume_roc'].iloc[-1]
        obv = df['obv'].iloc[-1]
        obv_prev = df['obv'].iloc[-2]
        
        # Volume spike
        if volume_ratio > 1.5:
            volume_spike = "High"
            spike_strength = min((volume_ratio - 1) / 2, 1)
        elif volume_ratio > 1.2:
            volume_spike = "Moderate"
            spike_strength = min((volume_ratio - 1) / 2, 0.7)
        else:
            volume_spike = "Normal"
            spike_strength = 0
        
        # Volume trend
        if volume_roc > 0:
            volume_trend = "Increasing"
            trend_strength = min(volume_roc / 50, 1)
        elif volume_roc < 0:
            volume_trend = "Decreasing"
            trend_strength = min(abs(volume_roc) / 50, 1)
        else:
            volume_trend = "Steady"
            trend_strength = 0
        
        # OBV signal
        if obv > obv_prev:
            obv_signal = "Bullish"
            obv_strength = min((obv - obv_prev) / (df['volume'].mean() or 1), 1)
        elif obv < obv_prev:
            obv_signal = "Bearish"
            obv_strength = min((obv_prev - obv) / (df['volume'].mean() or 1), 1)
        else:
            obv_signal = "Neutral"
            obv_strength = 0
        
        return {
            "volume_spike": volume_spike,
            "spike_strength": float(round(spike_strength, 3)),
            "volume_trend": volume_trend,
            "trend_strength": float(round(trend_strength, 3)),
            "obv_signal": obv_signal,
            "obv_strength": float(round(obv_strength, 3)),
            "interpretation": f"Volume shows {volume_spike.lower()} activity with {volume_trend.lower()} trend. "
                           f"OBV indicates {obv_signal.lower()} accumulation/distribution.",
            "details": {
                "volume_ratio": current_values['volume_ratio'],
                "volume_roc": current_values['volume_roc'],
                "obv": current_values['obv']
            }
        }
    
    def analyze_bollinger_bands(df) -> Dict:
        """Bollinger Band momentum analysis"""
        bb_percent = df['bb_percent'].iloc[-1]
        bb_width = df['bb_width'].iloc[-1]
        price = df['close'].iloc[-1]
        
        # Band position
        if bb_percent > 0.8:
            position = "Upper Band"
            position_strength = min((bb_percent - 0.8) / 0.2, 1)
            signal = "Overbought"
        elif bb_percent < 0.2:
            position = "Lower Band"
            position_strength = min((0.2 - bb_percent) / 0.2, 1)
            signal = "Oversold"
        else:
            position = "Middle Range"
            position_strength = 0
            signal = "Neutral"
        
        # Band width (volatility)
        bb_width_ma = df['bb_width'].rolling(20).mean().iloc[-1]
        if bb_width > bb_width_ma * 1.2:
            volatility = "High"
            vol_strength = min((bb_width - bb_width_ma) / bb_width_ma, 1)
        elif bb_width < bb_width_ma * 0.8:
            volatility = "Low"
            vol_strength = min((bb_width_ma - bb_width) / bb_width_ma, 1)
        else:
            volatility = "Normal"
            vol_strength = 0
        
        # Squeeze detection
        if bb_width < bb_width_ma * 0.5:
            squeeze = "Tight Squeeze"
            squeeze_strength = 1
        elif bb_width < bb_width_ma * 0.8:
            squeeze = "Moderate Squeeze"
            squeeze_strength = 0.7
        else:
            squeeze = "No Squeeze"
            squeeze_strength = 0
        
        # Overall signal
        if position in ["Upper Band", "Lower Band"] and volatility != "Low":
            if position == "Upper Band":
                overall_signal = "Bearish Reversal"
                signal_strength = position_strength * 0.8
            else:
                overall_signal = "Bullish Reversal"
                signal_strength = position_strength * 0.8
        elif squeeze != "No Squeeze":
            overall_signal = "Potential Breakout"
            signal_strength = squeeze_strength * 0.7
        else:
            overall_signal = "Neutral"
            signal_strength = 0
        
        return {
            "band_position": position,
            "position_strength": float(round(position_strength, 3)),
            "volatility": volatility,
            "volatility_strength": float(round(vol_strength, 3)),
            "squeeze_condition": squeeze,
            "squeeze_strength": float(round(squeeze_strength, 3)),
            "signal": signal,
            "overall_signal": overall_signal,
            "signal_strength": float(round(signal_strength, 3)),
            "interpretation": f"Price is at {position.lower()} with {volatility.lower()} volatility. "
                           f"{squeeze} detected. Overall signal: {overall_signal}.",
            "details": {
               "bb_percent": current_values['bb_percent'],
            "bb_width": current_values['bb_width'],
            "price_relative_to_bands": {
                "upper": current_values['bb_upper'],
                "middle": current_values['bb_middle'],
                "lower": current_values['bb_lower']
            }
            }
        }
    
    def analyze_mfi_signals(df) -> Dict:
        """Money Flow Index analysis"""
        mfi = df['mfi'].iloc[-1]
        mfi_prev = df['mfi'].iloc[-2]
        
        if mfi <= 20:
            condition = "Oversold"
            strength = min((20 - mfi) / 20, 1)
            signal = "Bullish" if mfi > mfi_prev else "Potential Reversal"
        elif mfi >= 80:
            condition = "Overbought"
            strength = min((mfi - 80) / 20, 1)
            signal = "Bearish" if mfi < mfi_prev else "Potential Reversal"
        elif 20 < mfi < 50:
            condition = "Bearish Zone"
            strength = min((50 - mfi) / 30, 0.7)
            signal = "Bearish"
        elif 50 <= mfi < 80:
            condition = "Bullish Zone"
            strength = min((mfi - 50) / 30, 0.7)
            signal = "Bullish"
        else:
            condition = "Neutral"
            strength = 0
            signal = "Neutral"
        
        return {
            "condition": condition,
            "signal": signal,
            "signal_strength": float(round(strength, 3)),
            "interpretation": f"MFI shows {condition.lower()} conditions with {signal.lower()} signal",
            "details": {
                "mfi_value": current_values['mfi'],
                "previous_mfi": float(round(mfi_prev, 2))
            }
        }
    
    def analyze_cci_signals(df) -> Dict:
        """Commodity Channel Index analysis"""
        cci = df['cci'].iloc[-1]
        cci_prev = df['cci'].iloc[-2]
        
        if cci > 100:
            condition = "Strong Bullish"
            strength = min((cci - 100) / 100, 1)
        elif cci > 0:
            condition = "Mild Bullish"
            strength = min(cci / 100, 0.7)
        elif cci < -100:
            condition = "Strong Bearish"
            strength = min((abs(cci) - 100) / 100, 1)
        elif cci < 0:
            condition = "Mild Bearish"
            strength = min(abs(cci) / 100, 0.7)
        else:
            condition = "Neutral"
            strength = 0
        
        # Momentum
        if cci > cci_prev:
            momentum = "Increasing"
        elif cci < cci_prev:
            momentum = "Decreasing"
        else:
            momentum = "Steady"
        
        return {
            "condition": condition,
            "signal_strength": float(round(strength, 3)),
            "momentum": momentum,
            "interpretation": f"CCI shows {condition.lower()} conditions with {momentum.lower()} momentum",
            "details": {
                "cci_value": current_values['cci'],
                "previous_cci": float(round(cci_prev, 2))
            }
        }
    
    def analyze_aroon_signals(df) -> Dict:
        """Aroon oscillator analysis"""
        aroon_up = df['aroon_up'].iloc[-1]
        aroon_down = df['aroon_down'].iloc[-1]
        osc = df['aroon_osc'].iloc[-1]
        
        if aroon_up > 70 and aroon_down < 30:
            condition = "Strong Uptrend"
            strength = min((aroon_up - 70) / 30, 1)
        elif aroon_down > 70 and aroon_up < 30:
            condition = "Strong Downtrend"
            strength = min((aroon_down - 70) / 30, 1)
        elif osc > 0:
            condition = "Mild Uptrend"
            strength = min(osc / 100, 0.7)
        elif osc < 0:
            condition = "Mild Downtrend"
            strength = min(abs(osc) / 100, 0.7)
        else:
            condition = "No Clear Trend"
            strength = 0
        
        return {
            "condition": condition,
            "signal_strength": float(round(strength, 3)),
            "interpretation": f"Aroon shows {condition.lower()}",
            "details": {
                "aroon_up": current_values['aroon_up'],
                "aroon_down": current_values['aroon_down'],
                "aroon_oscillator": current_values['aroon_osc']
            }
        }
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    # Perform all analyses
    indicator_analysis = {
        "roc": analyze_roc_signals(df),
        "williams_r": analyze_williams_r_signals(df),
        "stochastic": analyze_stochastic_signals(df),
        "macd": analyze_macd_signals(df),
        "trend_strength": analyze_trend_strength(df),
        "rsi": analyze_rsi_signals(df),
        "volume": analyze_volume_signals(df),
        "bollinger_bands": analyze_bollinger_bands(df),
        "mfi": analyze_mfi_signals(df),
        "cci": analyze_cci_signals(df),
        "aroon": analyze_aroon_signals(df)
    }
    
    # ==================== OVERALL ASSESSMENT ====================
    
    # Calculate momentum score (weighted average of all signals)
    weights = {
        "roc": 0.15,
        "williams_r": 0.1,
        "stochastic": 0.1,
        "macd": 0.15,
        "trend_strength": 0.15,
        "rsi": 0.1,
        "volume": 0.05,
        "bollinger_bands": 0.1,
        "mfi": 0.05,
        "cci": 0.05
    }
    
    momentum_score = 0
    for indicator, analysis in indicator_analysis.items():
        if "signal_strength" in analysis:
            signal = analysis.get("primary_signal", analysis.get("signal", "Neutral"))
            if "Bullish" in signal:
                multiplier = 1
            elif "Bearish" in signal:
                multiplier = -1
            else:
                multiplier = 0
            momentum_score += weights.get(indicator, 0) * analysis["signal_strength"] * multiplier
    
    # Normalize to -1 to 1 range
    momentum_score = max(min(momentum_score, 1), -1)
    
    # Determine recommendation
    if momentum_score > 0.5:
        recommendation = "strong bullish"
        confidence = min(90 + (momentum_score - 0.5) * 20, 100)
    elif momentum_score > 0.2:
        recommendation = "bullish"
        confidence = 70 + (momentum_score - 0.2) * 66
    elif momentum_score < -0.5:
        recommendation = "strong bearish"
        confidence = min(90 + (abs(momentum_score) - 0.5) * 20, 100)
    elif momentum_score < -0.2:
        recommendation = "bearish"
        confidence = 70 + (abs(momentum_score) - 0.2) * 66
    else:
        recommendation = "neutral"
        confidence = 50 + abs(momentum_score) * 40
    
    # Strength assessment
    abs_score = abs(momentum_score)
    if abs_score > 0.7:
        strength = "Strong"
    elif abs_score > 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    # Market regime
    market_regime = indicator_analysis["trend_strength"]["market_regime"]
    trend_strength = indicator_analysis["trend_strength"]["strength_score"]
    
    # Key insights
    key_insights = []
    if indicator_analysis["macd"]["divergence"]:
        key_insights.append(indicator_analysis["macd"]["divergence"] + " detected in MACD")
    if indicator_analysis["rsi"]["divergence"]:
        key_insights.append(indicator_analysis["rsi"]["divergence"] + " detected in RSI")
    if indicator_analysis["bollinger_bands"]["squeeze_condition"] != "No Squeeze":
        key_insights.append(indicator_analysis["bollinger_bands"]["squeeze_condition"] + " detected in Bollinger Bands")
    if indicator_analysis["volume"]["volume_spike"] != "Normal":
        key_insights.append(indicator_analysis["volume"]["volume_spike"] + " volume spike detected")
    
    # Signals dictionary
    signals = {
        "primary_signal": recommendation,
        "confirmed_by": [],
        "conflicting_indicators": []
    }
    
    # Check for confirmation
    bullish_indicators = []
    bearish_indicators = []
    for indicator, analysis in indicator_analysis.items():
        signal = analysis.get("primary_signal", analysis.get("signal", ""))
        if "Bullish" in signal:
            bullish_indicators.append(indicator)
        elif "Bearish" in signal:
            bearish_indicators.append(indicator)
    
    if recommendation.startswith("bullish"):
        signals["confirmed_by"] = bullish_indicators
        signals["conflicting_indicators"] = bearish_indicators
    elif recommendation.startswith("bearish"):
        signals["confirmed_by"] = bearish_indicators
        signals["conflicting_indicators"] = bullish_indicators
    
    # Risk metrics
    risk_metrics = {
        "volatility": indicator_analysis["bollinger_bands"]["volatility"],
        "trend_strength": trend_strength,
        "volume_confirmation": indicator_analysis["volume"]["volume_spike"],
        "overbought_oversold": {
            "rsi": indicator_analysis["rsi"]["condition"],
            "stochastic": indicator_analysis["stochastic"]["primary_condition"],
            "williams_r": indicator_analysis["williams_r"]["primary_condition"]
        }
    }
    
    # Analysis summary
    analysis_summary = (
        f"Momentum analysis for {symbol} shows {recommendation} conditions with {strength.lower()} strength. "
        f"The market is currently {market_regime.lower()}. "
    )
    
    if key_insights:
        analysis_summary += "Key observations: " + "; ".join(key_insights) + ". "
    
    analysis_summary += (
        f"Confidence level: {confidence:.0f}%. "
        f"Composite momentum score: {momentum_score:.2f} (-1 to 1 scale)."
    )
    
    # Final result
    return {
        "symbol": symbol,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "data_period": f"{len(df)} periods",
        "current_values": current_values,
        "indicator_analysis": indicator_analysis,
        "overall_assessment": {
            "momentum_score": float(round(momentum_score, 3)),
            "recommendation": recommendation,
            "confidence": float(round(confidence, 1)),
            "strength": strength,
            "trend_alignment": indicator_analysis["trend_strength"]["trend_direction"],
            "market_conditions": market_regime
        },
        "market_regime": market_regime,
        "trend_strength": float(round(trend_strength * 100, 1)),
        "analysis_summary": analysis_summary,
        "recommendation": recommendation.split()[0],  # Just "bullish", "bearish" or "neutral"
        "confidence": float(round(confidence, 1)),
        "momentum_score": float(round(momentum_score, 3)),
        "strength_assessment": strength,
        "key_insights": key_insights,
        "signals": signals,
        "risk_metrics": risk_metrics
    }



# ===== ADDITIONAL UTILITY FUNCTIONS =====

def get_nearest_levels(symbol: str, distance_pct: float = 5.0) -> Dict:
    """Get support/resistance levels within specified distance from current price"""
    result = calculate_support_resistance_levels(symbol)
    
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


def analyze_level_quality(symbol: str) -> Dict:
    """Analyze the quality and reliability of detected levels"""
    result = calculate_support_resistance_levels(symbol)

    if not result.get('levels'):
        return result
    
    # Get historical data for testing
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)
    
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


def validate_breakout_quality(symbol: str) -> Dict:
    """
    Validate the quality of detected breakouts with additional filters
    """
    result = detect_breakout_patterns(symbol)
    
    if result['breakout_probability'] < 0.5:
        return result
    
    # Additional quality checks
    row = r.get(f"historical:{symbol}")
    data = json.loads(row) if row else None
    df = refined_ohlc_data(data)
    
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


def batch_level_analysis(symbols: List[str]) -> Dict:
    """Analyze support/resistance levels for multiple symbols"""
    results = {}
    
    for symbol in symbols:
        try:
            result = calculate_support_resistance_levels(symbol)
            results[symbol] = result
        except Exception as e:
            results[symbol] = {"error": str(e)}
    
    return results


def batch_pattern_analysis(symbols: List[str]) -> List[Dict]:
    """
    Analyze multiple symbols for breakout patterns
    """
    results = []
    
    for symbol in symbols:
        try:
            result = validate_breakout_quality(symbol)
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
