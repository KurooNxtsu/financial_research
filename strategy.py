"""
strategy.py  --  AutoFin Strategy Definition
============================================
THIS FILE IS THE LLM's SOLE WRITABLE SURFACE.

Rules (enforced by backtest_harness.py):
  - You MAY change parameter values in *_PARAMS dicts.
  - You MAY change the logic inside generate_signals().
  - You MUST NOT rename any function.
  - You MUST NOT change any function signature.
  - You MUST NOT add new import statements.
  - You MUST keep all four indicator functions present and intact.
"""

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PARAMETER BLOCK -- LLM edits these values
# ---------------------------------------------------------------------------

ICHIMOKU_PARAMS = dict(
    conversion_period=9,
    base_period=26,
    lagging_span2_period=52,
    displacement=26,
)

RSI_PARAMS = dict(
    period=14,
    overbought=70,
    oversold=30,
)

ATR_PARAMS = dict(
    period=14,
    multiplier=2.5,
)

VWAP_PARAMS = dict(
    period=14,
)


# ---------------------------------------------------------------------------
# INDICATOR FUNCTIONS  (signatures are FIXED -- do not alter)
# ---------------------------------------------------------------------------

def ichimoku_cloud(
    df: pd.DataFrame,
    conversion_period: int,
    base_period: int,
    lagging_span2_period: int,
    displacement: int,
) -> pd.DataFrame:
    """
    Ichimoku Cloud indicator.

    Returns DataFrame with columns:
      tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
      cloud_top, cloud_bottom, bullish (bool), bearish (bool)

    bullish = price is above the cloud  (long bias)
    bearish = price is below the cloud  (short bias)
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    def midpoint(h, l, p):
        return (h.rolling(p).max() + l.rolling(p).min()) / 2

    tenkan  = midpoint(high, low, conversion_period)
    kijun   = midpoint(high, low, base_period)
    span_a  = ((tenkan + kijun) / 2).shift(displacement)
    span_b  = midpoint(high, low, lagging_span2_period).shift(displacement)

    cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    bullish = close > cloud_top
    bearish = close < cloud_bottom

    return pd.DataFrame({
        "tenkan_sen":    tenkan,
        "kijun_sen":     kijun,
        "senkou_span_a": span_a,
        "senkou_span_b": span_b,
        "cloud_top":     cloud_top,
        "cloud_bottom":  cloud_bottom,
        "bullish":       bullish,
        "bearish":       bearish,
    }, index=df.index)


def rsi(
    df: pd.DataFrame,
    period: int,
    overbought: float,
    oversold: float,
) -> pd.DataFrame:
    """
    Relative Strength Index.

    Returns DataFrame with columns:
      rsi_value      : 0-100 oscillator value
      buy_signal     : True on the bar RSI crosses UP through oversold
      sell_signal    : True on the bar RSI crosses DOWN through overbought
    """
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs      = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    prev = rsi_val.shift(1)
    buy_signal  = (prev <= oversold)   & (rsi_val > oversold)
    sell_signal = (prev >= overbought) & (rsi_val < overbought)

    return pd.DataFrame({
        "rsi_value":   rsi_val,
        "buy_signal":  buy_signal,
        "sell_signal": sell_signal,
    }, index=df.index)


def atr(
    df: pd.DataFrame,
    period: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Average True Range -- used for stop-loss placement.

    Returns DataFrame with columns:
      atr_value   : raw ATR
      long_stop   : close - multiplier * ATR  (stop for long positions)
      short_stop  : close + multiplier * ATR  (stop for short positions)
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_val = true_range.ewm(com=period - 1, min_periods=period).mean()

    long_stop = close - multiplier * atr_val
    short_stop = close + multiplier * atr_val

    return pd.DataFrame({
        "atr_value":   atr_val,
        "long_stop":   long_stop,
        "short_stop":  short_stop,
    }, index=df.index)


def vwap(
    df: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """
    Volume Weighted Average Price.

    Returns DataFrame with columns:
      vwap_value   : rolling VWAP
      above_vwap   : True if price > VWAP
      below_vwap   : True if price < VWAP
    """
    volume = df["Volume"]
    close  = df["Close"]

    typical_price = (close + df["High"] + df["Low"]) / 3
    pv = typical_price * volume
    vwap_val = pv.rolling(period).sum() / volume.rolling(period).sum()

    return pd.DataFrame({
        "vwap_value":   vwap_val,
        "above_vwap":   close > vwap_val,
        "below_vwap":   close < vwap_val,
    }, index=df.index)


# ---------------------------------------------------------------------------
# SIGNAL GENERATION -- LLM edits this logic
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry/exit signals using all indicators.

    Returns DataFrame with columns:
      signal (int: 1=long, -1=short, 0=flat)
      stop (float: stop price for current position)
    """
    ichi = ichimoku_cloud(df, **ICHIMOKU_PARAMS)
    rsi_ = rsi(df, **RSI_PARAMS)
    atr_ = atr(df, **ATR_PARAMS)
    vwap_ = vwap(df, **VWAP_PARAMS)

    # Loosen RSI condition: use zone check instead of crossover
    # RSI < oversold + 15 (45) instead of RSI crossover through oversold
    long_entry = ichi["bullish"] & (rsi_["rsi_value"] < RSI_PARAMS["oversold"] + 15)
    long_exit  = rsi_["sell_signal"] | (df["Close"] < atr_["long_stop"])

    short_entry = ichi["bearish"] & (rsi_["rsi_value"] > RSI_PARAMS["overbought"] - 15)
    short_exit  = rsi_["buy_signal"] | (df["Close"] > atr_["short_stop"])

    # Combine into single signal column
    signal = pd.Series(0, index=df.index)
    signal[long_entry & ~long_exit] = 1
    signal[short_entry & ~short_exit] = -1

    # Stop price depends on position
    stop = pd.Series(0.0, index=df.index)
    stop[long_entry & ~long_exit] = atr_["long_stop"].values[long_entry & ~long_exit]
    stop[short_entry & ~short_exit] = atr_["short_stop"].values[short_entry & ~short_exit]

    return pd.DataFrame({
        "signal": signal,
        "stop": stop,
    }, index=df.index)


# ---------------------------------------------------------------------------
# PARAMETER ACCESSOR
# ---------------------------------------------------------------------------

def get_params() -> dict:
    return {
        **ICHIMOKU_PARAMS,
        **RSI_PARAMS,
        **ATR_PARAMS,
        **VWAP_PARAMS,
    }