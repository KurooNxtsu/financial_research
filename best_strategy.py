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
    period=14,  # rolling anchor window in trading days
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
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    prev_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_value = true_range.rolling(period).mean()

    long_stop = close - multiplier * atr_value
    short_stop = close + multiplier * atr_value

    return pd.DataFrame({
        "atr_value":   atr_value,
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
      above_vwap   : True if price above VWAP
      below_vwap   : True if price below VWAP
    """
    close = df["Close"]
    volume = df["Volume"]

    typical_price = (df["High"] + df["Low"] + close) / 3
    tp_cum = typical_price.cumsum()
    vol_cum = volume.cumsum()

    vwap_value = tp_cum / vol_cum

    above_vwap = close > vwap_value
    below_vwap = close < vwap_value

    return pd.DataFrame({
        "vwap_value":   vwap_value,
        "above_vwap":   above_vwap,
        "below_vwap":   below_vwap,
    }, index=df.index)


# ---------------------------------------------------------------------------
# SIGNAL GENERATION LOGIC -- LLM edits this
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry/exit signals based on indicators.

    Returns DataFrame with columns:
      long_entry, long_exit, short_entry, short_exit (all bool)
    """
    ichi = ichimoku_cloud(df, **ICHIMOKU_PARAMS)
    rsi_ = rsi(df, **RSI_PARAMS)
    atr_ = atr(df, **ATR_PARAMS)
    vwap_ = vwap(df, **VWAP_PARAMS)

    # Adaptive stop based on ATR volatility
    atr_ma = atr_["atr_value"].rolling(20).mean()
    high_vol = atr_["atr_value"] > atr_ma * 1.5
    adaptive_mult = np.where(high_vol, ATR_PARAMS["multiplier"] * 0.7, ATR_PARAMS["multiplier"])
    long_stop = df["Close"] - adaptive_mult * atr_["atr_value"]
    short_stop = df["Close"] + adaptive_mult * atr_["atr_value"]

    # Volatility gate: skip entries when ATR > 2x its 20-day average
    atr_ma_20 = atr_["atr_value"].rolling(20).mean()
    high_vol_gate = atr_["atr_value"] > atr_ma_20 * 2.0

    # Long Entry: Ichimoku bullish AND tenkan > kijun AND RSI < 45 (mild pullback)
    long_entry = (
        (ichi["bullish"] == True) &
        (ichi["tenkan_sen"] > ichi["kijun_sen"]) &
        (rsi_["rsi_value"] < RSI_PARAMS["oversold"] + 15) &
        (~high_vol_gate)
    )

    # Long Exit: RSI sell signal OR price below long stop OR price re-enters cloud
    long_exit = (
        (rsi_["sell_signal"] == True) |
        (df["Close"] < long_stop) |
        (ichi["bearish"] == True)
    )

    # Short Entry: Ichimoku bearish AND tenkan < kijun AND RSI > 65 (mild rally)
    short_entry = (
        (ichi["bearish"] == True) &
        (ichi["tenkan_sen"] < ichi["kijun_sen"]) &
        (rsi_["rsi_value"] > RSI_PARAMS["overbought"] - 15) &
        (~high_vol_gate)
    )

    # Short Exit: RSI buy signal OR price above short stop OR price re-enters cloud
    short_exit = (
        (rsi_["buy_signal"] == True) |
        (df["Close"] > short_stop) |
        (ichi["bullish"] == True)
    )

    return pd.DataFrame({
        "long_entry":   long_entry,
        "long_exit":    long_exit,
        "short_entry":  short_entry,
        "short_exit":   short_exit,
    }, index=df.index)


def get_params() -> dict:
    """
    Return all parameter values as a flat dict.
    """
    return {
        **ICHIMOKU_PARAMS,
        **RSI_PARAMS,
        **ATR_PARAMS,
        **VWAP_PARAMS,
    }