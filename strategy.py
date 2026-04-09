"""
strategy.py  —  AutoFin Strategy Definition
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
# PARAMETER BLOCK — LLM edits these values
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
    multiplier=2.0,
)

VWAP_PARAMS = dict(
    period=14,  # rolling anchor window in trading days
)


# ---------------------------------------------------------------------------
# INDICATOR FUNCTIONS  (signatures are FIXED — do not alter)
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
      rsi_value      : 0–100 oscillator value
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
    Average True Range — used for stop-loss placement.

    Returns DataFrame with columns:
      atr_value   : raw ATR
      long_stop   : close - multiplier * ATR  (stop for long positions)
      short_stop  : close + multiplier * ATR  (stop for short positions)
    """
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_val    = true_range.ewm(span=period, min_periods=period).mean()
    long_stop  = df["Close"] - multiplier * atr_val
    short_stop = df["Close"] + multiplier * atr_val

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
    Rolling Volume-Weighted Average Price.

    Returns DataFrame with columns:
      vwap_value  : rolling VWAP over `period` days
      above_vwap  : True when close > VWAP
      below_vwap  : True when close < VWAP
    """
    typical  = (df["High"] + df["Low"] + df["Close"]) / 3
    vol      = df["Volume"].replace(0, np.nan)
    vwap_val = (typical * vol).rolling(period).sum() / vol.rolling(period).sum()

    return pd.DataFrame({
        "vwap_value": vwap_val,
        "above_vwap": df["Close"] > vwap_val,
        "below_vwap": df["Close"] < vwap_val,
    }, index=df.index)


# ---------------------------------------------------------------------------
# SIGNAL GENERATION — LLM may modify the logic inside this function
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all four indicators into a position signal.

    Returns a DataFrame with columns:
      signal : int   — 1 = long, -1 = short, 0 = flat
      stop   : float — active stop-loss price (NaN when flat)

    Called by backtest_harness.py — signature is FIXED.
    `df` must have columns: Open, High, Low, Close, Volume (DatetimeIndex).
    """
    ichi  = ichimoku_cloud(df, **ICHIMOKU_PARAMS)
    rsi_  = rsi(df,           **RSI_PARAMS)
    atr_  = atr(df,           **ATR_PARAMS)
    vwap_ = vwap(df,          **VWAP_PARAMS)

    n      = len(df)
    signal = pd.Series(0,      index=df.index, dtype=int)
    stop   = pd.Series(np.nan, index=df.index, dtype=float)

    position    = 0
    active_stop = np.nan

    for i in range(1, n):

        # ---- Stop-loss exit (checked first) ----
        if position == 1 and df["Low"].iloc[i] < active_stop:
            position    = 0
            active_stop = np.nan
        elif position == -1 and df["High"].iloc[i] > active_stop:
            position    = 0
            active_stop = np.nan

        # ---- RSI-based exit ----
        if position == 1  and rsi_["sell_signal"].iloc[i]:
            position    = 0
            active_stop = np.nan
        if position == -1 and rsi_["buy_signal"].iloc[i]:
            position    = 0
            active_stop = np.nan

        # ---- Entry (only when flat) ----
        if position == 0:
            # FIX: Use RSI zone check (value in zone) instead of crossover
            # (single-bar crossover is too rare in a ~21-bar monthly window).
            # FIX: Drop VWAP from entry — it contradicts the RSI zone condition.
            # Ichimoku provides the trend filter; RSI zone provides momentum.
            long_entry = (
                bool(ichi["bullish"].iloc[i])                   # price above cloud
                and rsi_["rsi_value"].iloc[i] < RSI_PARAMS["oversold"] + 15  # RSI < 45: mild weakness in uptrend
                and not np.isnan(atr_["long_stop"].iloc[i])
            )
            short_entry = (
                bool(ichi["bearish"].iloc[i])                   # price below cloud
                and rsi_["rsi_value"].iloc[i] > RSI_PARAMS["overbought"] - 15  # RSI > 55: mild strength in downtrend
                and not np.isnan(atr_["short_stop"].iloc[i])
            )

            if long_entry:
                position    = 1
                active_stop = float(atr_["long_stop"].iloc[i])
            elif short_entry:
                position    = -1
                active_stop = float(atr_["short_stop"].iloc[i])

        signal.iloc[i] = position
        stop.iloc[i]   = active_stop

    return pd.DataFrame({"signal": signal, "stop": stop}, index=df.index)
# ---------------------------------------------------------------------------
# METADATA — used by the harness for logging and the GitHub trigger payload
# ---------------------------------------------------------------------------

STRATEGY_NAME    = "AutoFin-v1"
STRATEGY_VERSION = "1.0.0"


def get_params() -> dict:
    """Return a flat dict of all current parameter values for logging."""
    return {
        **{f"ichi_{k}": v for k, v in ICHIMOKU_PARAMS.items()},
        **{f"rsi_{k}":  v for k, v in RSI_PARAMS.items()},
        **{f"atr_{k}":  v for k, v in ATR_PARAMS.items()},
        **{f"vwap_{k}": v for k, v in VWAP_PARAMS.items()},
        "strategy_name":    STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run directly to confirm the file loads cleanly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    np.random.seed(0)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx   = pd.date_range("2020-01-01", periods=n, freq="B")
    df_test = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)

    sigs = generate_signals(df_test)
    n_long  = (sigs["signal"] == 1).sum()
    n_short = (sigs["signal"] == -1).sum()
    print(f"Self-test passed: {n_long} long bars, {n_short} short bars over {n} days.")
    print(f"Params: {get_params()}")