import numpy as np
import pandas as pd
import talib


def add_talib_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a curated set of technical indicators using TA-Lib.
    Returns a new DataFrame with additional feature columns.
    """

    df = df.copy()

    close = np.asarray(df["close"].astype(float).values, dtype=float)
    high = np.asarray(df["high"].astype(float).values, dtype=float)
    low = np.asarray(df["low"].astype(float).values, dtype=float)
    volume = np.asarray(df["volume"].astype(float).values, dtype=float)

    # ------------------------
    # Trend indicators
    # ------------------------
    df["SMA_5"] = talib.SMA(close, timeperiod=5)
    df["SMA_20"] = talib.SMA(close, timeperiod=20)
    df["SMA_50"] = talib.SMA(close, timeperiod=50)

    df["EMA_12"] = talib.EMA(close, timeperiod=12)
    df["EMA_26"] = talib.EMA(close, timeperiod=26)

    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    df["ADX_14"] = talib.ADX(high, low, close, timeperiod=14)

    # ------------------------
    # Momentum indicators
    # ------------------------
    df["RSI_14"] = talib.RSI(close, timeperiod=14)
    df["Stoch_K"], df["Stoch_D"] = talib.STOCH(
        high,
        low,
        close,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    df["ROC_10"] = talib.ROC(close, timeperiod=10)

    # ------------------------
    # Volatility indicators
    # ------------------------
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)

    upper, middle, lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower

    # ------------------------
    # Volume indicators
    # ------------------------
    df["OBV"] = talib.OBV(close, volume)
    df["AD"] = talib.AD(high, low, close, volume)

    # ------------------------
    # Cleanup
    # ------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df
