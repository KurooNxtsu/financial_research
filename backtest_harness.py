"""
backtest_harness.py  —  AutoFin Evaluation Engine & LLM Orchestration
=======================================================================
THIS FILE IS READ-ONLY TO THE LLM.

Improvements vs original:
  - MULTI-TICKER SCORING: aggregate score is the mean across multiple tickers
    and shards, preventing overfitting to a single momentum stock.
  - ENSEMBLE VALIDATION GATE: before accepting a new best, the candidate is
    checked on out-of-sample shards across all tickers. If it fails badly
    on non-primary tickers, it is rejected even if primary-ticker score improved.
  - DIVERSITY FORCING: when no_improve >= 2, the prompt is augmented with a
    hard requirement to try a structurally different approach.
  - STRATEGY HASH TRACKING: detects when the LLM is submitting identical code
    repeatedly (due to repair-loop rollbacks) and injects a stronger diversity
    prompt to break the loop.
  - ATR FORMULA CONTRACT: validates that the LLM has not broken the ATR
    true-range formula (must use close.shift(1), not current close).
  - TEMPERATURE SCHEDULING: generation temperature decays from 0.55 to 0.20
    over iterations, so early runs explore broadly and later runs refine.
  - SHARD HEATMAP IN PROMPT: ASCII bar chart of per-shard scores makes weak
    shards visually salient for smaller models.
  - HTML DASHBOARD: writes a live results.html after every iteration for easy
    monitoring without tailing verbose.log.
  - Shards are YEARLY windows (keyed "YYYY").
  - Each shard gets a 100-day warm-up buffer.
  - Rollback only on aggregate < best_score; explore when flat.

Usage:
    # Single ticker (original behaviour)
    python backtest_harness.py --ticker "NVDA" --device cuda --forever

    # Multi-ticker (recommended — better generalisation)
    python backtest_harness.py --tickers "NVDA,^GSPC,MSFT,GLD" --device cuda --forever

    # Colab-friendly quick run
    python backtest_harness.py --tickers "NVDA,^GSPC" --device cuda --iterations 30
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import torch

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parent.resolve()
PROGRAM_FILE  = ROOT / "program.md"
STRATEGY_FILE = ROOT / "strategy.py"
BEST_FILE     = ROOT / "best_strategy.py"
LOG_FILE      = ROOT / "autofin_log.jsonl"
RESULTS_TSV   = ROOT / "results.tsv"
TRIGGER_FILE  = ROOT / "push_trigger.json"
VERBOSE_LOG   = ROOT / "verbose.log"
DASHBOARD_HTML = ROOT / "results.html"

WARMUP_DAYS = 100

# ---------------------------------------------------------------------------
# Multi-ticker defaults
# A diverse basket: large-cap tech, broad index, commodity, bond ETF, mid-cap.
# Keep this small enough to run on Colab — each ticker adds ~N_SHARDS evaluations.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = ["NVDA", "^GSPC", "MSFT", "GLD"]

# Ensemble tickers used for the acceptance gate (subset of DEFAULT_TICKERS,
# excluding the primary ticker so we check cross-asset generalisation).
ENSEMBLE_GATE_TICKERS = ["^GSPC", "MSFT", "GLD"]

# ---------------------------------------------------------------------------
# Get-params boilerplate for auto-repair
# ---------------------------------------------------------------------------
GET_PARAMS_BOILERPLATE = '''

STRATEGY_NAME    = "AutoFin-v2"
STRATEGY_VERSION = "2.0.0"


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
'''

# ---------------------------------------------------------------------------
# Structural idea bank — injected when diversity forcing triggers
# ---------------------------------------------------------------------------
DIVERSITY_IDEAS = [
    "Use Tenkan/Kijun cross (tenkan_sen > kijun_sen) as the primary entry trigger instead of price vs cloud.",
    "Add a trailing-stop mechanic: after N bars in position, tighten active_stop by 10% each bar.",
    "Use the Ichimoku displacement value to time exits: exit when price re-enters the cloud.",
    "Implement asymmetric RSI thresholds: long entry RSI < 40 but short entry RSI > 65.",
    "Combine VWAP and Tenkan: enter long only when close > vwap AND tenkan > kijun (drop cloud requirement).",
    "Use a volatility-regime gate: skip entries entirely when ATR > 2x its 20-day average (too choppy).",
    "Try a pure momentum approach: enter long on Tenkan crossover above Kijun, exit on reverse crossover.",
    "Add a look-back exit: if position has been open > 20 bars with no new high (long) / low (short), exit.",
    "Use RSI divergence pattern: enter long when price makes lower low but RSI makes higher low.",
    "Scale RSI thresholds with volatility: in high-vol regimes use oversold=20, in low-vol use oversold=40.",
]


# ---------------------------------------------------------------------------
# Verbose logger
# ---------------------------------------------------------------------------

def vlog(tag: str, content: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 70
    entry = f"\n{separator}\n[{timestamp}] [{tag}]\n{separator}\n{content}\n"
    with open(VERBOSE_LOG, "a") as fh:
        fh.write(entry)
    print(entry)


# ---------------------------------------------------------------------------
# TSV results log
# ---------------------------------------------------------------------------

TSV_HEADER = "iteration\taggregate_score\tstatus\tdescription\tparams_snapshot\n"


def init_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)
        vlog("INIT", f"Initialised {RESULTS_TSV}")


def append_results_tsv(
    iteration: int,
    score: float,
    status: str,
    description: str,
    params: dict,
) -> None:
    params_str = json.dumps(params).replace("\t", " ")
    desc       = description.replace("\t", " ").replace("\n", " ")
    row = f"{iteration}\t{score:.6f}\t{status}\t{desc}\t{params_str}\n"
    with open(RESULTS_TSV, "a") as fh:
        fh.write(row)


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def write_html_dashboard(history: list[dict], best_score: float) -> None:
    """Write a live HTML results dashboard after every iteration."""
    rows_html = ""
    for h in history:
        score  = h["aggregate_score"]
        status = h["status"]
        color  = {"keep": "#2ecc71", "baseline": "#3498db",
                  "explore": "#f39c12", "discard": "#e74c3c",
                  "crash": "#8e44ad"}.get(status, "#95a5a6")
        rows_html += (
            f"<tr>"
            f"<td style='padding:4px 12px'>{h['iteration']}</td>"
            f"<td style='padding:4px 12px'>{score:.5f}</td>"
            f"<td style='padding:4px 12px;color:{color};font-weight:bold'>{status}</td>"
            f"<td style='padding:4px 12px'>{h.get('description','')[:80]}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>AutoFin Dashboard</title>
  <meta http-equiv='refresh' content='30'>
  <style>
    body{{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px}}
    h2{{color:#00d4ff}}
    table{{border-collapse:collapse;width:100%}}
    th{{background:#16213e;padding:6px 12px;text-align:left;color:#00d4ff}}
    tr:nth-child(even){{background:#16213e}}
    tr:hover{{background:#0f3460}}
    .best{{font-size:1.4em;color:#2ecc71;margin:10px 0}}
  </style>
</head>
<body>
  <h2>AutoFin — Live Results Dashboard</h2>
  <p>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
     &nbsp;|&nbsp; Auto-refreshes every 30s</p>
  <div class='best'>Best aggregate score: {best_score:.5f}</div>
  <table>
    <tr>
      <th>Iter</th><th>Score</th><th>Status</th><th>Description</th>
    </tr>
    {rows_html}
  </table>
</body>
</html>"""
    try:
        DASHBOARD_HTML.write_text(html)
    except Exception:
        pass  # non-critical


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    vlog("DATA DOWNLOAD", f"Ticker: {ticker}  Range: {start} -> {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    vlog("DATA LOADED",
        f"Ticker   : {ticker}\n"
        f"Rows     : {len(df)}\n"
        f"From     : {df.index[0].date()}\n"
        f"To       : {df.index[-1].date()}\n"
    )
    return df


# ---------------------------------------------------------------------------
# Yearly sharding with warm-up buffer
# ---------------------------------------------------------------------------

def make_shards(df: pd.DataFrame, start_year: int = 2018) -> dict[str, pd.DataFrame]:
    shards: dict[str, pd.DataFrame] = {}
    years = sorted(set(df.index.year))

    for yr in years:
        if yr < start_year:
            continue

        year_mask = df.index.year == yr
        year_df   = df[year_mask]

        if len(year_df) < 50:
            continue

        year_start = year_df.index[0]
        pre_mask   = df.index < year_start
        pre_df     = df[pre_mask].iloc[-WARMUP_DAYS:]

        combined = pd.concat([pre_df, year_df])
        combined.attrs["eval_start"] = year_start
        shards[str(yr)] = combined

    return shards


def strip_warmup(series: pd.Series, eval_start: pd.Timestamp) -> pd.Series:
    return series[series.index >= eval_start]


def normalise_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if "signal" in signals.columns:
        return signals

    cols = set(signals.columns)

    if {"long_entry", "long_exit", "short_entry", "short_exit"}.issubset(cols):
        position, unified = 0, []
        for le, lx, se, sx in zip(
            signals["long_entry"], signals["long_exit"],
            signals["short_entry"], signals["short_exit"],
        ):
            if le:             position =  1
            elif se:           position = -1
            elif lx or sx:     position =  0
            unified.append(position)
        signals = signals.copy()
        signals["signal"] = unified

    elif {"long_entry", "long_exit"}.issubset(cols):
        position, unified = 0, []
        for le, lx in zip(signals["long_entry"], signals["long_exit"]):
            if le:   position = 1
            elif lx: position = 0
            unified.append(position)
        signals = signals.copy()
        signals["signal"] = unified

    else:
        raise ValueError(
            f"generate_signals must return a DataFrame with a 'signal' column. "
            f"Got columns: {list(signals.columns)}"
        )

    return signals


# ---------------------------------------------------------------------------
# Trade simulator
# ---------------------------------------------------------------------------

def simulate_trades(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    signals = normalise_signals(signals)

    pos     = signals["signal"].shift(1).fillna(0)
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)
    strat   = pos * log_ret

    return pd.DataFrame({
        "position":          pos,
        "strategy_log_ret":  strat,
        "daily_return":      np.exp(strat) - 1,
        "cumulative_return": strat.cumsum().apply(np.exp) - 1,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(daily_returns: pd.Series, periods: int = 252) -> float:
    std = daily_returns.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return float(daily_returns.mean() / std * math.sqrt(periods))


def profit_factor(daily_returns: pd.Series) -> float:
    gains  = daily_returns[daily_returns > 0].sum()
    losses = daily_returns[daily_returns < 0].sum()
    if losses == 0:
        return 10.0
    return float(min(gains / abs(losses), 10.0))


def max_drawdown(daily_returns: pd.Series) -> float:
    cum = (1 + daily_returns).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    return float(abs(dd.min()))


def composite_score(sharpe: float, pf: float, mdd: float) -> float:
    """score = Sharpe*0.6 + PF*0.2 - MDD*0.2  (fixed metric)."""
    return round(sharpe * 0.6 + pf * 0.2 - mdd * 0.2, 6)


# ---------------------------------------------------------------------------
# Shard evaluation
# ---------------------------------------------------------------------------

def evaluate_shard(
    shard: pd.DataFrame,
    generate_signals_fn,
    shard_key: str,
) -> dict:
    eval_start = shard.attrs.get("eval_start", shard.index[0])

    try:
        signals = generate_signals_fn(shard)
        signals = normalise_signals(signals)
        trades  = simulate_trades(shard, signals)

        dr_full = trades["daily_return"]
        dr      = strip_warmup(dr_full, eval_start)

        if len(dr) == 0:
            return _empty_result(shard, "no eval bars after warm-up strip")

        sh   = sharpe_ratio(dr)
        pf_  = profit_factor(dr)
        mdd  = max_drawdown(dr)
        sc   = composite_score(sh, pf_, mdd)

        sig_eval = strip_warmup(signals["signal"], eval_start)
        n_trades = int((sig_eval.diff().abs() > 0).sum())

        if n_trades == 0:
            sc = 0.0

        result = {
            "sharpe":        round(sh,  4),
            "profit_factor": round(pf_, 4),
            "max_drawdown":  round(mdd, 4),
            "score":         sc,
            "n_trades":      n_trades,
            "n_bars":        len(dr),
            "error":         None,
        }

        vlog(f"SHARD EVAL [{shard_key}]",
            f"eval_start    : {eval_start.date()}\n"
            f"eval_bars     : {len(dr)}\n"
            f"n_trades      : {n_trades}\n"
            f"sharpe        : {sh:.4f}\n"
            f"profit_factor : {pf_:.4f}\n"
            f"max_drawdown  : {mdd:.4f}\n"
            f"score         : {sc:.6f}\n"
        )
        return result

    except Exception as exc:
        vlog(f"SHARD EVAL ERROR [{shard_key}]",
            f"Exception : {exc}\n\n{traceback.format_exc()}"
        )
        return {
            "sharpe": 0.0, "profit_factor": 0.0,
            "max_drawdown": 1.0, "score": -99.0,
            "n_trades": 0, "n_bars": len(shard),
            "error": str(exc),
        }


def _empty_result(shard: pd.DataFrame, reason: str) -> dict:
    return {
        "sharpe": 0.0, "profit_factor": 0.0,
        "max_drawdown": 0.0, "score": 0.0,
        "n_trades": 0, "n_bars": len(shard),
        "error": reason,
    }


# ---------------------------------------------------------------------------
# Multi-ticker aggregate scoring
# ---------------------------------------------------------------------------

def evaluate_all_tickers(
    shards_by_ticker: dict[str, dict[str, pd.DataFrame]],
    generate_signals_fn,
) -> tuple[float, dict[str, dict[str, dict]]]:
    """
    Evaluate strategy across all tickers and shards.
    Returns (aggregate_score, results_by_ticker).
    """
    results_by_ticker: dict[str, dict[str, dict]] = {}
    all_valid_scores: list[float] = []

    for ticker, shards in shards_by_ticker.items():
        results_by_ticker[ticker] = {}
        for key, shard_df in shards.items():
            m = evaluate_shard(shard_df, generate_signals_fn, f"{ticker}_{key}")
            results_by_ticker[ticker][key] = m
            if m["score"] != -99.0:
                all_valid_scores.append(m["score"])

    aggregate = float(np.mean(all_valid_scores)) if all_valid_scores else -99.0
    return aggregate, results_by_ticker


# ---------------------------------------------------------------------------
# Ensemble validation gate
# ---------------------------------------------------------------------------

def ensemble_gate_check(
    candidate_path: Path,
    shards_by_ticker: dict[str, dict[str, pd.DataFrame]],
    gate_tickers: list[str],
    min_oos_score: float = -0.30,
) -> tuple[bool, float]:
    """
    Run candidate strategy on out-of-sample shards for gate_tickers only.
    Returns (passed, mean_oos_score).

    A strategy that fits NVDA perfectly but collapses on other assets is
    rejected here before being accepted as a new best.
    """
    try:
        strat = load_strategy(candidate_path)
    except Exception as exc:
        vlog("ENSEMBLE GATE ERROR", f"Failed to load candidate: {exc}")
        return False, -99.0

    oos_scores = []
    for ticker in gate_tickers:
        if ticker not in shards_by_ticker:
            continue
        shards = shards_by_ticker[ticker]
        for key in ["2023", "2024"]:
            if key in shards:
                m = evaluate_shard(
                    shards[key],
                    strat.generate_signals,
                    f"GATE_{ticker}_{key}",
                )
                if m["score"] != -99.0:
                    oos_scores.append(m["score"])

    if not oos_scores:
        vlog("ENSEMBLE GATE", "No OOS scores collected — passing gate by default.")
        return True, 0.0

    mean_oos = float(np.mean(oos_scores))
    passed   = mean_oos >= min_oos_score
    vlog("ENSEMBLE GATE",
        f"Gate tickers  : {gate_tickers}\n"
        f"OOS shards    : {len(oos_scores)}\n"
        f"Mean OOS score: {mean_oos:.4f}\n"
        f"Threshold     : {min_oos_score}\n"
        f"Passed        : {passed}\n"
        f"Individual    : {oos_scores}"
    )
    return passed, mean_oos


# ---------------------------------------------------------------------------
# Shard heatmap for prompt
# ---------------------------------------------------------------------------

def format_shard_heatmap(
    results_by_ticker: dict[str, dict[str, dict]],
) -> str:
    """
    Return an ASCII bar-chart summary of per-ticker per-shard scores.
    Weak shards are flagged with '<- FIX' to help the LLM prioritise.
    """
    lines = ["Per-ticker per-shard score heatmap:"]
    for ticker, shards in sorted(results_by_ticker.items()):
        lines.append(f"  {ticker}:")
        for key, m in sorted(shards.items()):
            score   = m["score"]
            bar_len = min(int(abs(score) * 8), 16)
            if score >= 0:
                bar = "█" * bar_len
                flag = ""
            else:
                bar = "░" * bar_len
                flag = "  <- FIX"
            lines.append(
                f"    {key} [{bar:<16}] {score:+.3f}"
                f"  Sh={m['sharpe']:+.2f} PF={m['profit_factor']:.2f}"
                f"  T={m['n_trades']}{flag}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy hash tracking
# ---------------------------------------------------------------------------

def strategy_hash(path: Path) -> str:
    """MD5 of just the generate_signals body + PARAM dicts."""
    content = path.read_text()
    gs_idx  = content.find("def generate_signals")
    pb_idx  = content.find("ICHIMOKU_PARAMS")
    segment = content[pb_idx:] if pb_idx != -1 else content[gs_idx:]
    return hashlib.md5(segment.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------

def load_strategy(path: Path):
    spec   = importlib.util.spec_from_file_location("strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Contract / syntax validators
# ---------------------------------------------------------------------------

def validate_syntax(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<strategy>", "exec")
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def validate_strategy_contract(code: str) -> tuple[bool, str]:
    if "def generate_signals" not in code:
        return False, "missing generate_signals function"
    if "def get_params" not in code:
        return False, "missing get_params function"
    for name in ["rsi", "atr", "vwap", "ichimoku_cloud"]:
        if f"\n    {name} = {name}(" in code:
            return False, (
                f"name-shadowing: '{name}' shadows module-level function. "
                f"Use '{name}_' alias instead."
            )
    return True, ""


def validate_atr_formula(code: str) -> tuple[bool, str]:
    """
    Check that the ATR function body still uses close.shift(1) for previous
    close, not the current close. The LLM often breaks this silently.
    """
    # Find the atr function body
    atr_start = code.find("def atr(")
    if atr_start == -1:
        return True, ""   # no atr func found — contract validator will catch

    # Extract up to the next top-level def or end of file
    next_def = code.find("\ndef ", atr_start + 10)
    atr_body = code[atr_start:next_def] if next_def != -1 else code[atr_start:]

    # Patterns that indicate the wrong formula (using current close)
    bad_patterns = [
        "(df[\"High\"] - df[\"Close\"])",
        "(df[\"Low\"] - df[\"Close\"])",
        "(high - close)",
        "(low - close)",
        "high - close)",
        "low - close)",
    ]
    for pat in bad_patterns:
        if pat in atr_body:
            return False, (
                f"ATR formula error: found '{pat}'. "
                "True range must use close.shift(1) (previous close), "
                "not current close. Use: (df['High'] - prev_close).abs() "
                "where prev_close = df['Close'].shift(1)."
            )
    return True, ""


def validate_strategy_output(
    path: Path,
    df_sample: pd.DataFrame,
) -> tuple[bool, str]:
    try:
        strat   = load_strategy(path)
        signals = strat.generate_signals(df_sample)
        signals = normalise_signals(signals)

        if "signal" not in signals.columns:
            return False, "no 'signal' column after normalisation"

        n_active = (signals["signal"] != 0).sum()
        if n_active == 0:
            return False, f"zero non-flat signals on validation sample ({len(df_sample)} bars)"

        n_long  = (signals["signal"] == 1).sum()
        n_short = (signals["signal"] == -1).sum()
        n_total = len(signals)
        if n_long / n_total > 0.95:
            return False, f"strategy is {n_long/n_total:.0%} long — likely stuck"
        if n_short / n_total > 0.95:
            return False, f"strategy is {n_short/n_total:.0%} short — likely stuck"

        transitions = (signals["signal"].diff().abs() > 0).sum()
        if transitions < 2:
            return False, f"only {transitions} signal transitions — not a real strategy"

        return True, f"ok — {n_active} active bars, {transitions} transitions"

    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# LLM pipeline
# ---------------------------------------------------------------------------

def build_llm_pipeline(model_id: str, device: str):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )

    vlog("LLM LOADING", f"Model : {model_id}\nDevice: {device}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # NOTE: temperature is set per-call in the generation step (scheduled).
    # We set a placeholder here; actual temperature is passed at inference time.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=6000,
        do_sample=True,
        top_p=0.8,
        return_full_text=False,
    )

    vlog("LLM LOADED", f"Model: {model_id}  |  4-bit NF4 double-quant")
    return pipe


# ---------------------------------------------------------------------------
# Temperature scheduler
# ---------------------------------------------------------------------------

def get_temperature(iteration: int, no_improve: int) -> float:
    """
    Decay temperature from 0.55 (early, high exploration) to 0.20 (late, refinement).
    When stuck (no_improve >= 2) bump temperature back up to encourage diversity.
    """
    base   = max(0.20, 0.55 - iteration * 0.015)
    if no_improve >= 2:
        base = min(0.55, base + 0.15)  # forced exploration boost
    return round(base, 3)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    program_source: str,
    strategy_source: str,
    results_by_ticker: dict[str, dict[str, dict]],
    aggregate_score: float,
    iteration: int,
    history: list[dict],
    no_improve: int,
    tried_hashes: list[str],
    last_failed_source: Optional[str] = None,
    last_failed_score:  Optional[float] = None,
) -> list[dict]:

    # ---- Per-ticker/shard results table ----
    primary_ticker = list(results_by_ticker.keys())[0] if results_by_ticker else "?"
    rows = []
    for ticker, shards in sorted(results_by_ticker.items()):
        for key, m in sorted(shards.items()):
            err = f"  ERR: {m['error'][:40]}" if m["error"] else ""
            rows.append(
                f"  {ticker:6s} {key} | Sharpe={m['sharpe']:>7.4f} | PF={m['profit_factor']:>5.2f} "
                f"| MDD={m['max_drawdown']:>5.3f} | Score={m['score']:>8.5f} "
                f"| Trades={m['n_trades']:>4}{err}"
            )
    shard_table = "\n".join(rows)

    # ---- ASCII heatmap ----
    heatmap = format_shard_heatmap(results_by_ticker)

    # ---- Score history ----
    hist_lines = [
        f"  iter {h['iteration']:>3}: aggregate={h['aggregate_score']:.5f}  status={h['status']}"
        for h in history[-5:]
    ] or ["  (no history yet — this is the baseline run)"]
    hist_str = "\n".join(hist_lines)

    # ---- Diversity forcing block ----
    diversity_block = ""
    if no_improve >= 2:
        import random
        idea_a = DIVERSITY_IDEAS[iteration % len(DIVERSITY_IDEAS)]
        idea_b = DIVERSITY_IDEAS[(iteration + 3) % len(DIVERSITY_IDEAS)]
        diversity_block = (
            f"\n{'='*60}\n"
            f"DIVERSITY REQUIREMENT (no_improve={no_improve}):\n"
            f"Your last {no_improve} attempts modified similar things (RSI thresholds,\n"
            f"VWAP filter, ATR multiplier). You MUST try something STRUCTURALLY\n"
            f"DIFFERENT. Here are two concrete ideas — pick one or invent your own:\n\n"
            f"  IDEA A: {idea_a}\n\n"
            f"  IDEA B: {idea_b}\n\n"
            f"Do NOT just re-tune a threshold. Change the signal logic.\n"
            f"{'='*60}\n"
        )

    # ---- Task line ----
    if iteration == 1:
        task_line = (
            "This is iteration 1 — the BASELINE run. "
            "Output the current strategy.py UNCHANGED."
        )
    else:
        task_line = (
            f"The CURRENT BEST aggregate score is {aggregate_score:.5f} "
            f"(averaged across ALL tickers and shards). "
            "Propose an improved strategy.py. /no_think\n\n"
            "YOU MUST RESPOND IN EXACTLY THIS FORMAT:\n"
            "  <reasoning> ... your analysis ... </reasoning>\n"
            "  <strategy> ... full python code ... </strategy>\n\n"
            "Do NOT write anything outside these two tags.\n"
            "The <strategy> block must be complete valid Python with NO markdown fences."
        )

    # ---- Failure context block ----
    failure_block = ""
    if last_failed_source is not None:
        MAX_FAILED_CHARS = 2000
        failed_display = last_failed_source
        if len(last_failed_source) > MAX_FAILED_CHARS:
            gs_idx = last_failed_source.find("def generate_signals")
            if gs_idx != -1:
                failed_display = (
                    "# [indicator functions omitted — unchanged from baseline]\n\n"
                    + last_failed_source[gs_idx:]
                )
            else:
                failed_display = last_failed_source[:MAX_FAILED_CHARS] + "\n# ... (truncated)"

        failure_block = (
            f"\n--- YOUR LAST ATTEMPT (score={last_failed_score:.5f}, NOT an improvement) ---\n"
            f"Study this carefully. Do NOT repeat the same changes.\n\n"
            f"{failed_display}\n"
            f"--- END OF LAST ATTEMPT ---\n"
        )

    MAX_STRATEGY_CHARS = 4000
    strategy_display = strategy_source
    if len(strategy_source) > MAX_STRATEGY_CHARS:
        strategy_display = (
            strategy_source[:MAX_STRATEGY_CHARS]
            + "\n\n# ... (indicator function bodies omitted — do not modify them)\n"
            + "# Include all four indicator functions with their ORIGINAL implementations in your output."
        )

    user_content = (
        f"=== ITERATION {iteration} ===\n\n"
        f"--- Per-ticker/shard results ---\n{shard_table}\n\n"
        f"{heatmap}\n\n"
        f"--- Aggregate score (mean across ALL tickers): {aggregate_score:.5f} ---\n\n"
        f"--- Score history (last 5) ---\n{hist_str}\n\n"
        f"{diversity_block}"
        f"{failure_block}"
        f"--- Current BEST strategy.py ---\n{strategy_display}\n\n"
        f"{task_line}"
    )

    return [
        {"role": "system", "content": program_source},
        {"role": "user",   "content": user_content},
    ]


def build_repair_prompt(
    program_source: str,
    failed_code: str,
    validation_error: str,
    error_type: str,
    attempt: int,
) -> list[dict]:
    user_content = (
        f"=== REPAIR ATTEMPT {attempt} ===\n\n"
        f"Your last strategy failed validation with this error:\n\n"
        f"  Error type : {error_type}\n"
        f"  Error      : {validation_error}\n\n"
        f"Common causes by error type:\n"
        f"  atr_formula  : ATR must use close.shift(1) for previous close.\n"
        f"                 Use (df['High'] - prev_close).abs() NOT (high - close).abs().\n"
        f"  output/zero  : Entry conditions too strict. Loosen RSI or Ichimoku filter.\n"
        f"  contract     : generate_signals and get_params must both be present.\n"
        f"  syntax       : Fix the Python syntax error shown above.\n\n"
        f"--- YOUR FAILING STRATEGY ---\n"
        f"{failed_code}\n"
        f"--- END OF FAILING STRATEGY ---\n\n"
        f"Fix ONLY the error above. Respond in EXACTLY this format:\n\n"
        f"<reasoning>\nWhat was wrong and what you fixed (2-3 sentences max)\n</reasoning>\n"
        f"<strategy>\nfull corrected Python code here\n</strategy>"
    )
    return [
        {"role": "system", "content": program_source},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> tuple[Optional[str], Optional[str]]:
    r_matches = re.findall(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    s_matches = re.findall(r"<strategy>(.*?)</strategy>",   text, re.DOTALL | re.IGNORECASE)

    reasoning = r_matches[-1].strip() if r_matches else None

    if s_matches:
        strategy = max(s_matches, key=len).strip()
    else:
        strategy = None

    if not strategy or len(strategy) < 100:
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        if blocks:
            strategy = max(blocks, key=len).strip()
            vlog("PARSE FALLBACK", "No valid <strategy> tag — fell back to ```python``` block")

    if strategy:
        strategy = re.sub(r"^```[a-zA-Z]*\n?", "", strategy)
        strategy = re.sub(r"\n?```$",           "", strategy)
        strategy = strategy.strip()

    return reasoning, strategy


# ---------------------------------------------------------------------------
# Auto-repair & sanitise
# ---------------------------------------------------------------------------

def auto_repair_strategy(code: str) -> tuple[str, bool]:
    if "def get_params" not in code:
        repaired = code.rstrip() + GET_PARAMS_BOILERPLATE
        return repaired, True
    return code, False


def sanitise_strategy_code(code: str) -> str:
    replacements = {
        "\u2018": "'",  "\u2019": "'",
        "\u201c": '"',  "\u201d": '"',
        "\u2014": "--", "\u2013": "-",
        "\u00a0": " ",  "\u2026": "...",
    }
    for bad, good in replacements.items():
        code = code.replace(bad, good)
    code = code.replace("\x00", "")
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.splitlines()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

def log_jsonl(record: dict) -> None:
    with open(LOG_FILE, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# GitHub push trigger
# ---------------------------------------------------------------------------

def write_trigger(score: float, iteration: int, params: dict) -> None:
    payload = {
        "score":     score,
        "iteration": iteration,
        "params":    params,
        "timestamp": datetime.utcnow().isoformat(),
    }
    TRIGGER_FILE.write_text(json.dumps(payload, indent=2))
    vlog("TRIGGER WRITTEN", f"push_trigger.json updated\n{json.dumps(payload, indent=2)}")


# ---------------------------------------------------------------------------
# LLM repair helper
# ---------------------------------------------------------------------------

def _run_llm_repair(pipe, messages: list[dict], temperature: float) -> Optional[str]:
    try:
        torch.cuda.empty_cache()
        formatted = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        outputs    = pipe(formatted, max_new_tokens=10000, temperature=temperature)
        llm_output = outputs[0]["generated_text"]
        vlog("REPAIR LLM OUTPUT", f"Length: {len(llm_output)} chars\n\n{llm_output[:2000]}")
    except Exception as exc:
        vlog("REPAIR LLM ERROR", f"Exception: {exc}\n\n{traceback.format_exc()}")
        return None

    _, strategy = parse_llm_response(llm_output)
    if strategy:
        strategy, _ = auto_repair_strategy(strategy)
        strategy    = sanitise_strategy_code(strategy)
    return strategy


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    tickers:    list[str],
    start:      str,
    end:        str,
    model_id:   str,
    device:     str,
    iterations: int,
    forever:    bool,
    patience:   int,
    start_year: int,
) -> None:

    init_results_tsv()
    program_source = PROGRAM_FILE.read_text()

    vlog("INIT",
        f"tickers    : {tickers}\n"
        f"start      : {start}\n"
        f"end        : {end}\n"
        f"model      : {model_id}\n"
        f"device     : {device}\n"
        f"iterations : {iterations}\n"
        f"forever    : {forever}\n"
        f"patience   : {patience}\n"
        f"start_year : {start_year}\n"
    )

    # ---- Download data for all tickers ----
    shards_by_ticker: dict[str, dict[str, pd.DataFrame]] = {}
    for ticker in tickers:
        try:
            df = download_data(ticker, start, end)
            shards_by_ticker[ticker] = make_shards(df, start_year=start_year)
            vlog("SHARDS",
                f"Ticker {ticker}: {list(shards_by_ticker[ticker].keys())}"
            )
        except Exception as exc:
            vlog("DATA ERROR", f"Failed to load {ticker}: {exc}. Skipping.")

    if not shards_by_ticker:
        raise RuntimeError("No tickers loaded successfully. Aborting.")

    # Ensemble gate uses all tickers except the primary (first) one
    primary_ticker   = tickers[0]
    gate_tickers     = [t for t in tickers if t != primary_ticker and t in shards_by_ticker]
    validation_shard = list(shards_by_ticker[primary_ticker].values())[-1]

    pipe = build_llm_pipeline(model_id, device)

    best_score          = -math.inf
    no_improve          = 0
    history: list[dict] = []
    iteration           = 0
    last_failed_source: Optional[str]  = None
    last_failed_score:  Optional[float] = None

    # Strategy hash tracking to detect repeated code
    tried_hashes: list[str] = []
    stale_count  = 0
    MAX_STALE    = 3
    last_results_by_ticker: Optional[dict] = None

    while True:
        iteration += 1
        if not forever and iteration > iterations:
            vlog("LOOP END", "Max iterations reached.")
            break

        print(f"\n{'='*62}")
        print(f"  AutoFin  Iteration {iteration}" + ("" if forever else f"/{iterations}"))
        print(f"  Tickers: {tickers}")
        print(f"{'='*62}")

        # ---- Load strategy ----
        try:
            strat               = load_strategy(STRATEGY_FILE)
            generate_signals_fn = strat.generate_signals
            current_params      = strat.get_params()
        except Exception as exc:
            vlog("STRATEGY CRASH", f"Failed to load strategy.py: {exc}\n{traceback.format_exc()}")
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                vlog("ROLLBACK", "Rolled back to best_strategy.py after load crash.")
            append_results_tsv(iteration, 0.0, "crash", f"load error: {exc}", {})
            continue

        # ---- Evaluate on all tickers and shards ----
        aggregate, results_by_ticker = evaluate_all_tickers(
            shards_by_ticker, generate_signals_fn
        )

        # Print summary
        for ticker, shards in sorted(results_by_ticker.items()):
            for key, m in sorted(shards.items()):
                err_str = f"  [ERR: {m['error']}]" if m["error"] else ""
                print(
                    f"  {ticker:6s} {key}: "
                    f"Sh={m['sharpe']:>6.3f}  PF={m['profit_factor']:>4.2f}  "
                    f"MDD={m['max_drawdown']:>5.3f}  Sc={m['score']:>7.4f}  "
                    f"T={m['n_trades']:>3}{err_str}"
                )

        any_crash = any(
            m["score"] == -99.0
            for shards in results_by_ticker.values()
            for m in shards.values()
        )

        print(f"\n  Aggregate (all tickers): {aggregate:.5f}   Best so far: {best_score:.5f}")

        # ---- Stale detection ----
        current_flat = {
            f"{tk}_{k}": v["score"]
            for tk, shards in results_by_ticker.items()
            for k, v in shards.items()
        }
        if last_results_by_ticker is not None and current_flat == last_results_by_ticker:
            stale_count += 1
        else:
            stale_count = 0
        last_results_by_ticker = current_flat

        # ---- Determine status ----
        if any_crash and aggregate == -99.0:
            status      = "crash"
            description = "all shards errored"
        elif aggregate > best_score:
            status      = "keep" if iteration > 1 else "baseline"
            description = f"iter {iteration}: aggregate={aggregate:.5f}"
        elif aggregate == best_score:
            status      = "explore"
            description = f"iter {iteration}: flat ({aggregate:.5f}), keeping for exploration"
        else:
            status      = "discard" if iteration > 1 else "baseline"
            description = f"iter {iteration}: regression ({aggregate:.5f} < {best_score:.5f})"

        # ---- Log ----
        record = {
            "iteration":       iteration,
            "aggregate_score": aggregate,
            "results_by_ticker": {
                tk: {k: {fk: fv for fk, fv in m.items() if fk != "error"}
                     for k, m in shards.items()}
                for tk, shards in results_by_ticker.items()
            },
            "params":      current_params,
            "status":      status,
            "description": description,
            "timestamp":   datetime.utcnow().isoformat(),
        }
        log_jsonl(record)
        append_results_tsv(iteration, aggregate, status, description, current_params)
        history.append(record)

        # Update dashboard every iteration
        write_html_dashboard(history, max(best_score, aggregate))

        # ---- Advance or roll back ----
        if aggregate > best_score:
            # --- Ensemble gate check before accepting new best ---
            if gate_tickers and iteration > 1:
                gate_passed, gate_score = ensemble_gate_check(
                    STRATEGY_FILE, shards_by_ticker, gate_tickers,
                    min_oos_score=-0.30,
                )
                if not gate_passed:
                    vlog("ENSEMBLE GATE REJECTED",
                        f"New score {aggregate:.5f} > best {best_score:.5f} BUT "
                        f"ensemble OOS score {gate_score:.4f} < -0.30. "
                        f"Treating as discard to prevent NVDA overfitting."
                    )
                    status             = "discard"
                    description       += f" [ensemble_rejected gate={gate_score:.3f}]"
                    no_improve        += 1
                    last_failed_source = STRATEGY_FILE.read_text()
                    last_failed_score  = aggregate
                    if BEST_FILE.exists():
                        shutil.copy(BEST_FILE, STRATEGY_FILE)
                    append_results_tsv(iteration, aggregate, status, description, current_params)
                else:
                    best_score         = aggregate
                    no_improve         = 0
                    last_failed_source = None
                    last_failed_score  = None
                    shutil.copy(STRATEGY_FILE, BEST_FILE)
                    vlog("NEW BEST", f"Score improved to {best_score:.6f}. Saved best_strategy.py")
                    write_trigger(best_score, iteration, current_params)
            else:
                best_score         = aggregate
                no_improve         = 0
                last_failed_source = None
                last_failed_score  = None
                shutil.copy(STRATEGY_FILE, BEST_FILE)
                vlog("NEW BEST", f"Score improved to {best_score:.6f}. Saved best_strategy.py")
                write_trigger(best_score, iteration, current_params)

        elif aggregate == best_score:
            no_improve        += 1
            last_failed_source = None
            last_failed_score  = None
            vlog("EXPLORE", f"Score flat at {aggregate:.6f}. no_improve={no_improve}/{patience}")

        else:
            no_improve        += 1
            last_failed_source = STRATEGY_FILE.read_text()
            last_failed_score  = aggregate
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                vlog("ROLLBACK",
                    f"Regression: {aggregate:.6f} < {best_score:.6f}. "
                    f"Rolled back. no_improve={no_improve}/{patience}"
                )
            else:
                vlog("ROLLBACK SKIPPED", "No best_strategy.py exists yet.")

        vlog("AGGREGATE",
            f"Iteration      : {iteration}\n"
            f"Aggregate      : {aggregate:.6f}\n"
            f"Best so far    : {best_score:.6f}\n"
            f"Status         : {status}\n"
            f"no_improve     : {no_improve}/{patience}\n"
            f"stale_count    : {stale_count}/{MAX_STALE}\n"
            f"temperature    : {get_temperature(iteration, no_improve)}"
        )

        if no_improve >= patience and not forever:
            vlog("PLATEAU", f"no_improve={no_improve} >= patience={patience}. Stopping.")
            break

        # ---- Skip LLM if stale ----
        if stale_count >= MAX_STALE:
            vlog("STALE SKIP LLM", "Skipping LLM — metrics flat for 3 consecutive iterations.")
            stale_count = 0  # reset to try again next iter
            continue

        if iteration == 1 and BEST_FILE.exists():
            vlog("ITER1 SKIP", "Iteration 1: copying best_strategy.py directly, skipping LLM.")
            shutil.copy(BEST_FILE, STRATEGY_FILE)
            continue

        # ---- Hash tracking ----
        current_hash = strategy_hash(STRATEGY_FILE)
        hash_repeats = tried_hashes.count(current_hash)
        tried_hashes.append(current_hash)

        if hash_repeats >= 2:
            vlog("HASH REPEAT",
                f"Strategy hash {current_hash} seen {hash_repeats+1} times. "
                f"Forcing diversity prompt."
            )
            # Treat as no_improve to trigger diversity forcing
            no_improve = max(no_improve, 2)

        # ---- Build prompt ----
        temperature     = get_temperature(iteration, no_improve)
        strategy_source = STRATEGY_FILE.read_text()

        messages = build_prompt(
            program_source     = program_source,
            strategy_source    = strategy_source,
            results_by_ticker  = results_by_ticker,
            aggregate_score    = best_score if best_score > -math.inf else aggregate,
            iteration          = iteration,
            history            = history,
            no_improve         = no_improve,
            tried_hashes       = tried_hashes,
            last_failed_source = last_failed_source,
            last_failed_score  = last_failed_score,
        )

        vlog("LLM PROMPT",
            f"Turns       : {len(messages)}\n"
            f"Temperature : {temperature}\n"
            f"no_improve  : {no_improve}\n"
            f"hash_repeats: {hash_repeats}\n"
        )

        # ---- Generate ----
        torch.cuda.empty_cache()
        print(f"\n[llm] Generating next strategy (temperature={temperature}) ...")

        try:
            formatted = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            outputs    = pipe(formatted, temperature=temperature)
            llm_output = outputs[0]["generated_text"]
            vlog("LLM RAW OUTPUT", f"Length : {len(llm_output)} chars\n\n{llm_output[:3000]}")

        except torch.cuda.OutOfMemoryError as exc:
            vlog("OOM ERROR", f"CUDA OOM during generation.\n{exc}")
            torch.cuda.empty_cache()
            continue

        except Exception as exc:
            vlog("GENERATION ERROR", f"{exc}\n\n{traceback.format_exc()}")
            continue

        # ---- Parse ----
        reasoning, new_strategy = parse_llm_response(llm_output)

        if reasoning:
            print(f"\n[llm] Reasoning: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}\n")

        if not new_strategy:
            vlog("PARSE FAILED", "No strategy block found. Re-running current best.")
            continue

        # ---- Auto-repair & sanitise ----
        new_strategy, was_repaired = auto_repair_strategy(new_strategy)
        if was_repaired:
            vlog("AUTO-REPAIR", "Appended missing get_params boilerplate.")

        new_strategy = sanitise_strategy_code(new_strategy)

        # ---- Validate + inner repair loop ----
        MAX_REPAIR_ATTEMPTS = 3
        repair_attempt      = 0
        candidate_code      = new_strategy

        while repair_attempt < MAX_REPAIR_ATTEMPTS:
            repair_attempt += 1

            syntax_ok,   syntax_err   = validate_syntax(candidate_code)
            contract_ok, contract_err = validate_strategy_contract(candidate_code)
            atr_ok,      atr_err      = validate_atr_formula(candidate_code)

            vlog("VALIDATION",
                f"Repair attempt  : {repair_attempt}/{MAX_REPAIR_ATTEMPTS}\n"
                f"Syntax OK       : {syntax_ok}  {syntax_err or ''}\n"
                f"Contract OK     : {contract_ok}  {contract_err or ''}\n"
                f"ATR formula OK  : {atr_ok}  {atr_err or ''}\n"
            )

            # Syntax failure
            if not syntax_ok:
                if repair_attempt >= MAX_REPAIR_ATTEMPTS:
                    candidate_code = None; break
                repair_msgs    = build_repair_prompt(program_source, candidate_code, syntax_err, "syntax", repair_attempt)
                candidate_code = _run_llm_repair(pipe, repair_msgs, temperature)
                if candidate_code is None: break
                candidate_code, _ = auto_repair_strategy(candidate_code)
                candidate_code    = sanitise_strategy_code(candidate_code)
                continue

            # Contract failure
            if not contract_ok:
                if repair_attempt >= MAX_REPAIR_ATTEMPTS:
                    candidate_code = None; break
                repair_msgs    = build_repair_prompt(program_source, candidate_code, contract_err, "contract", repair_attempt)
                candidate_code = _run_llm_repair(pipe, repair_msgs, temperature)
                if candidate_code is None: break
                candidate_code, _ = auto_repair_strategy(candidate_code)
                candidate_code    = sanitise_strategy_code(candidate_code)
                continue

            # ATR formula failure
            if not atr_ok:
                if repair_attempt >= MAX_REPAIR_ATTEMPTS:
                    candidate_code = None; break
                repair_msgs    = build_repair_prompt(program_source, candidate_code, atr_err, "atr_formula", repair_attempt)
                candidate_code = _run_llm_repair(pipe, repair_msgs, temperature)
                if candidate_code is None: break
                candidate_code, _ = auto_repair_strategy(candidate_code)
                candidate_code    = sanitise_strategy_code(candidate_code)
                continue

            # Output validation
            TEMP_FILE = ROOT / "strategy_candidate.py"
            TEMP_FILE.write_text(candidate_code)
            output_ok, output_err = validate_strategy_output(TEMP_FILE, validation_shard)
            TEMP_FILE.unlink(missing_ok=True)

            vlog("VALIDATION OUTPUT",
                f"Repair attempt : {repair_attempt}/{MAX_REPAIR_ATTEMPTS}\n"
                f"Output OK      : {output_ok}\n"
                f"Output error   : {output_err or 'none'}\n"
            )

            if output_ok:
                vlog("VALIDATION PASSED", f"Passed on repair attempt {repair_attempt}")
                break

            if repair_attempt >= MAX_REPAIR_ATTEMPTS:
                candidate_code = None; break

            repair_msgs    = build_repair_prompt(program_source, candidate_code, output_err, "output", repair_attempt)
            candidate_code = _run_llm_repair(pipe, repair_msgs, temperature)
            if candidate_code is None: break
            candidate_code, _ = auto_repair_strategy(candidate_code)
            candidate_code    = sanitise_strategy_code(candidate_code)

        # ---- If all repair attempts exhausted ----
        if candidate_code is None:
            print("[repair] All repair attempts failed. Keeping current best.")
            append_results_tsv(iteration, 0.0, "crash", "repair loop exhausted", {})
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
            continue

        # ---- Write validated code ----
        STRATEGY_FILE.write_text(candidate_code)
        vlog("STRATEGY WRITTEN",
            f"Written to : {STRATEGY_FILE}\n"
            f"Length     : {len(candidate_code)} chars\n"
            f"Hash       : {strategy_hash(STRATEGY_FILE)}\n"
        )
        print("[strategy] strategy.py updated for next iteration.")

    # ---- Final summary ----
    vlog("RUN COMPLETE",
        f"Best aggregate score : {best_score:.5f}\n"
        f"Best strategy        : {BEST_FILE}\n"
        f"Results table        : {RESULTS_TSV}\n"
        f"Dashboard            : {DASHBOARD_HTML}\n"
        f"Full JSONL log       : {LOG_FILE}\n"
        f"Verbose log          : {VERBOSE_LOG}\n"
        f"Total iterations     : {iteration}\n"
        f"Tickers evaluated    : {tickers}"
    )
    print(f"\n{'='*62}")
    print(f"  AutoFin complete.")
    print(f"  Best aggregate score : {best_score:.5f}")
    print(f"  Tickers              : {tickers}")
    print(f"  Dashboard            : {DASHBOARD_HTML}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AutoFin — Autonomous Multi-Ticker Financial Strategy Optimisation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--ticker", default=None,
        help="Single Yahoo Finance ticker (legacy; use --tickers for multi-ticker).",
    )
    p.add_argument(
        "--tickers", default=None,
        help=(
            "Comma-separated tickers, e.g. 'NVDA,^GSPC,MSFT,GLD'\n"
            "First ticker is primary; others are used for ensemble gate.\n"
            f"Default: {','.join(DEFAULT_TICKERS)}"
        ),
    )
    p.add_argument("--start",      default="2018-01-01")
    p.add_argument("--end",        default="2024-12-31")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help=(
            "HuggingFace model ID.\n"
            "  Qwen/Qwen2.5-72B-Instruct  (A100 — best)\n"
            "  Qwen/Qwen2.5-32B-Instruct  (L4  — good)\n"
            "  Qwen/Qwen2.5-7B-Instruct   (T4  — default)\n"
            "  google/gemma-3-27b-it       (alternative)"
        ),
    )
    p.add_argument("--device",     default="cuda")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--forever",    action="store_true")
    p.add_argument("--patience",   type=int, default=5)
    p.add_argument("--start_year", type=int, default=2018)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve tickers: --tickers > --ticker > default
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = DEFAULT_TICKERS

    run(
        tickers    = tickers,
        start      = args.start,
        end        = args.end,
        model_id   = args.model,
        device     = args.device,
        iterations = args.iterations,
        forever    = args.forever,
        patience   = args.patience,
        start_year = args.start_year,
    )