from market import get_symbol_history_daily_data

def extract_close_list(data_json: dict) -> list:
    try:
        return data_json['data']['close']
    except (KeyError, TypeError):
        return []


def calculate_ema(prices: list, period: int) -> list:
    """ This is to use in MACD calculation """
    if len(prices) < period:
        return []
    ema_values = []
    multiplier = 2 / (period + 1)
    
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    for i in range(period, len(prices)):
        ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)
    
    return ema_values


def calculate_macd(closes: list, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.
    Returns the latest (most recent) values only.
    
    Args:
        closes (list): List of closing prices (chronological order, oldest first)
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)  
        signal_period (int): Signal line EMA period (default: 9)
    
    Returns:
        dict: {
            'macd_line': float (latest MACD value),
            'signal_line': float (latest signal line value),
            'histogram': float (latest histogram value)
        } or None if insufficient data
    """
    
    # Check if we have enough data
    min_required = slow_period + signal_period - 1
    if len(closes) < min_required:
        return None
    
    # Calculate Fast and Slow EMAs
    fast_ema = calculate_ema(closes, fast_period)
    slow_ema = calculate_ema(closes, slow_period)
    
    if not fast_ema or not slow_ema:
        return None
    
    # Calculate MACD line (Fast EMA - Slow EMA)
    # We need to align the EMAs since slow EMA starts later
    macd_line = []
    offset = slow_period - fast_period  # Difference in start points
    
    for i in range(len(slow_ema)):
        fast_index = i + offset
        if fast_index < len(fast_ema):
            macd_value = fast_ema[fast_index] - slow_ema[i]
            macd_line.append(macd_value)
    
    # Calculate Signal line (EMA of MACD line)
    signal_line = calculate_ema(macd_line, signal_period)
    
    if not signal_line:
        return None
    
    # Get the latest values
    latest_macd = macd_line[-1]
    latest_signal = signal_line[-1]
    latest_histogram = latest_macd - latest_signal
    total_offset = slow_period - 1 + signal_period - 1
    
    return {
        'macd_line': round(latest_macd, 4),
        'signal_line': round(latest_signal, 4),
        'histogram': round(latest_histogram, 4),
        'dates_offset': total_offset
    }


def calculate_rsi(closes: list, period: int = 14) -> float:
    """
    Simple RSI calculation function - suitable for API backend.
    
    Args:
        closes (list): List of closing prices in chronological order (oldest first)
        period (int): RSI period, default 14
    
    Returns:
        float: RSI value (0-100), rounded to 2 decimal places
        None: If insufficient data
    """
    
    # Validate input
    if not closes or len(closes) < period + 1:
        return None
    
    # Calculate price changes
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    
    # Separate gains and losses
    gains = [max(change, 0) for change in changes]
    losses = [abs(min(change, 0)) for change in changes]
    
    # Calculate average gain and loss using Wilder's smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Apply Wilder's smoothing for remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def closing_sma(symbol: str, from_date: str, to_date: str, window: int = 50) -> float:
    ohlc_data = get_symbol_history_daily_data(symbol, from_date, to_date)
    closes = extract_close_list(ohlc_data)
    
    if not closes or len(closes) < window:
        return None
    
    # Take only the last 'window' closes
    recent_closes = closes[-window:]
    sma = sum(recent_closes) / window
    return round(sma, 4)


def closing_ema(symbol: str, from_date: str, to_date: str, window: int = 50) -> float:
    ohlc_data = get_symbol_history_daily_data(symbol, from_date, to_date)
    closes = extract_close_list(ohlc_data)
    
    if not closes or len(closes) < window:
        return None
    
    k = 2 / (window + 1)
    ema = sum(closes[:window]) / window
    
    for price in closes[window:]:
        ema = price * k + ema * (1 - k)
    
    return round(ema, 4)


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
    closes = extract_close_list(ohlc_data)
    
    if not closes:
        return None
    
    return calculate_macd(closes, fast_period, slow_period, signal_period)


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
    closes = extract_close_list(ohlc_data)
    
    if not closes:
        return None
    
    return calculate_rsi(closes, period)


# closing_sma_value = closing_sma("RELIANCE", "NSE_EQ", "2025-01-01", "2025-06-28")
# closing_ema_value = closing_ema("RELIANCE", "NSE_EQ", "2025-01-01", "2025-06-28")
# closing_macd_value = closing_macd("RELIANCE", "NSE_EQ", "2025-01-01", "2025-06-28")
# closing_rsi_value = closing_rsi("RELIANCE", "NSE_EQ", "2025-01-01", "2025-06-28")
# print("SMA:", closing_sma_value)
# print("EMA:", closing_ema_value)
# print("MACD:", closing_macd_value)
# print("RSI:", closing_rsi_value)