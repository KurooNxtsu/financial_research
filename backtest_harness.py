"""
backtest_harness.py  —  AutoFin Evaluation Engine & LLM Orchestration
=======================================================================
THIS FILE IS READ-ONLY TO THE LLM.

Changes vs v1:
  - Shards are now YEARLY (keyed "YYYY") instead of monthly.
    Each shard gets a 100-day warm-up buffer prepended so that
    Ichimoku / ATR / VWAP are fully initialised on the very first
    bar of the evaluation window. The buffer rows are stripped before
    any metrics are calculated — no look-ahead, no contamination.
  - Rollback logic changed: we only roll back when aggregate < best_score.
    When aggregate == best_score (both 0.0 while the agent is exploring)
    we let the mutated code stand so the LLM can keep searching.
  - The LLM is now shown its failed strategy alongside the current best
    so it learns what NOT to repeat.
  - Full verbose logging to verbose.log for every stage of the pipeline.

Usage:
    python backtest_harness.py --ticker "^GSPC" --device cuda --forever
    python backtest_harness.py --ticker "^NDX"  --device cuda --iterations 20
"""

from __future__ import annotations

import argparse
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
# Paths (all relative to the directory containing this file)
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parent.resolve()
PROGRAM_FILE  = ROOT / "program.md"
STRATEGY_FILE = ROOT / "strategy.py"
BEST_FILE     = ROOT / "best_strategy.py"
LOG_FILE      = ROOT / "autofin_log.jsonl"
RESULTS_TSV   = ROOT / "results.tsv"
TRIGGER_FILE  = ROOT / "push_trigger.json"
VERBOSE_LOG   = ROOT / "verbose.log"

# How many calendar days of history to prepend to each shard so that
# slow indicators (Ichimoku span_b needs 52+26 = 78 bars) are warm.
WARMUP_DAYS = 100

# Boilerplate appended when get_params is missing due to truncation
GET_PARAMS_BOILERPLATE = '''

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
'''

# ---------------------------------------------------------------------------
# Verbose logger
# ---------------------------------------------------------------------------

def vlog(tag: str, content: str) -> None:
    """Write a timestamped verbose entry to verbose.log and print it."""
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
# Data loading
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    vlog("DATA DOWNLOAD", f"Ticker: {ticker}  Range: {start} → {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    vlog("DATA LOADED",
        f"Rows     : {len(df)}\n"
        f"From     : {df.index[0].date()}\n"
        f"To       : {df.index[-1].date()}\n"
        f"Columns  : {list(df.columns)}\n\n"
        f"Head:\n{df.head(3).to_string()}\n\n"
        f"Tail:\n{df.tail(3).to_string()}"
    )
    return df


# ---------------------------------------------------------------------------
# Yearly sharding with warm-up buffer
# ---------------------------------------------------------------------------

def make_shards(df: pd.DataFrame, start_year: int = 2018) -> dict[str, pd.DataFrame]:
    """
    Split DataFrame into YEARLY shards keyed as "YYYY".

    Each shard value is the FULL slice including a WARMUP_DAYS-day history
    buffer prepended before the target year. The evaluation functions must
    call strip_warmup() on the trade/return series before computing metrics
    so that warm-up bars are never scored.

    Shards with fewer than 50 bars in the target year are skipped.
    """
    shards: dict[str, pd.DataFrame] = {}
    years = sorted(set(df.index.year))

    for yr in years:
        if yr < start_year:
            continue

        year_mask = df.index.year == yr
        year_df   = df[year_mask]

        if len(year_df) < 50:
            vlog("SHARD SKIP", f"Year {yr} has only {len(year_df)} bars — skipping.")
            continue

        year_start = year_df.index[0]
        pre_mask   = df.index < year_start
        pre_df     = df[pre_mask].iloc[-WARMUP_DAYS:]

        combined = pd.concat([pre_df, year_df])
        combined.attrs["eval_start"] = year_start
        shards[str(yr)] = combined

    vlog("SHARDS CREATED",
        f"start_year : {start_year}\n"
        f"Shard keys : {list(shards.keys())}\n" +
        "\n".join(
            f"  {k}: {len(v)} total rows  "
            f"(warmup={len(v) - (v.index >= v.attrs['eval_start']).sum()}  "
            f"eval={(v.index >= v.attrs['eval_start']).sum()})"
            for k, v in shards.items()
        )
    )
    return shards


def strip_warmup(series: pd.Series, eval_start: pd.Timestamp) -> pd.Series:
    """Return only the bars from eval_start onward."""
    return series[series.index >= eval_start]

def normalise_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Convert any known signal format into a unified 'signal' column DataFrame."""
    if "signal" in signals.columns:
        return signals

    cols = set(signals.columns)

    if {"long_entry", "long_exit", "short_entry", "short_exit"}.issubset(cols):
        position, unified = 0, []
        for le, lx, se, sx in zip(
            signals["long_entry"], signals["long_exit"],
            signals["short_entry"], signals["short_exit"],
        ):
            if le:        position =  1
            elif se:      position = -1
            elif lx or sx: position = 0
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
# Trade simulator  (fixed — do not modify)
# ---------------------------------------------------------------------------

def simulate_trades(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    # --- Normalise signal column ---
    if "signal" not in signals.columns:
        cols = set(signals.columns)

        # Handle long_entry / long_exit / short_entry / short_exit pattern
        if {"long_entry", "long_exit", "short_entry", "short_exit"}.issubset(cols):
            position = 0
            unified  = []
            for le, lx, se, sx in zip(
                signals["long_entry"],
                signals["long_exit"],
                signals["short_entry"],
                signals["short_exit"],
            ):
                if le:   position =  1
                elif se: position = -1
                elif lx or sx: position = 0
                unified.append(position)
            signals = signals.copy()
            signals["signal"] = unified

        # Handle long_entry / long_exit only (no shorts)
        elif {"long_entry", "long_exit"}.issubset(cols):
            position = 0
            unified  = []
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
# Metrics  (fixed — do not modify)
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
    """
    score = Sharpe × 0.6  +  ProfitFactor × 0.2  −  MaxDrawdown × 0.2
    Fixed metric — do not change.
    """
    return round(sharpe * 0.6 + pf * 0.2 - mdd * 0.2, 6)


def validate_strategy_output(path: Path, df_sample: pd.DataFrame) -> tuple[bool, str]:
    """
    Run generate_signals on a small sample and verify:
    1. Returns a DataFrame with a 'signal' column (after normalisation)
    2. Produces at least 1 non-zero signal (not a do-nothing strategy)
    """
    try:
        strat = load_strategy(path)
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


def evaluate_shard(shard: pd.DataFrame, generate_signals_fn, shard_key: str) -> dict:
    """
    Run strategy on one shard (which includes a warm-up buffer) and return
    metrics computed only over the target evaluation window.
    """
    eval_start = shard.attrs.get("eval_start", shard.index[0])

    try:
        signals = generate_signals_fn(shard)
        signals = normalise_signals(signals)
        trades  = simulate_trades(shard, signals)

        dr_full = trades["daily_return"]
        dr      = strip_warmup(dr_full, eval_start)

        if len(dr) == 0:
            result = _empty_result(shard, "no eval bars after warm-up strip")
            vlog(f"SHARD EVAL [{shard_key}]",
                f"eval_start : {eval_start.date()}\n"
                f"Result     : EMPTY — no bars after warmup strip\n"
                f"Raw result : {json.dumps(result, indent=2)}"
            )
            return result

        sh   = sharpe_ratio(dr)
        pf_  = profit_factor(dr)
        mdd  = max_drawdown(dr)
        sc   = composite_score(sh, pf_, mdd)

        sig_eval = strip_warmup(signals["signal"], eval_start)
        n_trades = int((sig_eval.diff().abs() > 0).sum())
        sig_counts = sig_eval.value_counts().to_dict()

        # Penalise zero-trade shards
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
            f"signal_counts : {sig_counts}\n"
            f"sharpe        : {sh:.4f}\n"
            f"profit_factor : {pf_:.4f}\n"
            f"max_drawdown  : {mdd:.4f}\n"
            f"score         : {sc:.6f}\n"
            f"daily_ret sample (first 5):\n{dr.head().to_string()}"
        )
        return result

    except Exception as exc:
        result = {
            "sharpe": 0.0, "profit_factor": 0.0,
            "max_drawdown": 1.0, "score": -99.0,
            "n_trades": 0, "n_bars": len(shard),
            "error": str(exc),
        }
        vlog(f"SHARD EVAL ERROR [{shard_key}]",
            f"Exception : {exc}\n\n"
            f"{traceback.format_exc()}"
        )
        return result


def _empty_result(shard: pd.DataFrame, reason: str) -> dict:
    return {
        "sharpe": 0.0, "profit_factor": 0.0,
        "max_drawdown": 0.0, "score": 0.0,
        "n_trades": 0, "n_bars": len(shard),
        "error": reason,
    }


# ---------------------------------------------------------------------------
# Strategy hot-loader
# ---------------------------------------------------------------------------

def load_strategy(path: Path):
    spec   = importlib.util.spec_from_file_location("strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Contract validator
# ---------------------------------------------------------------------------

def validate_syntax(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<strategy>", "exec")
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def validate_strategy_contract(code: str) -> tuple[bool, str]:
    """Check that the code defines generate_signals and get_params, and has no shadowing bugs."""
    if "def generate_signals" not in code:
        return False, "missing generate_signals function"
    if "def get_params" not in code:
        return False, "missing get_params function"
    # Catch common name-shadowing that causes UnboundLocalError at runtime.
    # The LLM sometimes writes `rsi = rsi(df, ...)` inside generate_signals,
    # which shadows the module-level function and causes a crash on the second call.
    for name in ["rsi", "atr", "vwap", "ichimoku_cloud"]:
        if f"\n    {name} = {name}(" in code:
            return False, (
                f"name-shadowing detected: local variable '{name}' shadows "
                f"the module-level function '{name}'. Use an alias like '{name}_ = {name}(...)' instead."
            )
    return True, ""


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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=6000,   # enough for full strategy; leaves room in context window
        temperature=0.4,
        top_p=0.8,
        do_sample=True,
        return_full_text=False,
    )

    vlog("LLM LOADED",
        f"Model          : {model_id}\n"
        f"Quantization   : 4-bit NF4 double-quant\n"
        f"max_new_tokens : 6000\n"
        f"temperature    : 0.4\n"
        f"VRAM after load:\n{torch.cuda.memory_summary(abbreviated=True)}"
    )
    return pipe


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    program_source: str,
    strategy_source: str,
    shard_results: dict[str, dict],
    aggregate_score: float,
    iteration: int,
    history: list[dict],
    last_failed_source: Optional[str] = None,
    last_failed_score:  Optional[float] = None,
) -> list[dict]:

    # ---- Per-shard results table ----
    rows = []
    for key, m in shard_results.items():
        err = f"  ERR: {m['error'][:50]}" if m["error"] else ""
        rows.append(
            f"  {key} | Sharpe={m['sharpe']:>7.4f} | PF={m['profit_factor']:>5.2f} "
            f"| MDD={m['max_drawdown']:>5.3f} | Score={m['score']:>8.5f} "
            f"| Trades={m['n_trades']:>4}{err}"
        )
    shard_table = "\n".join(rows)

    # ---- Score history ----
    hist_lines = [
        f"  iter {h['iteration']:>3}: aggregate={h['aggregate_score']:.5f}  "
        f"status={h['status']}"
        for h in history[-5:]
    ] or ["  (no history yet — this is the baseline run)"]
    hist_str = "\n".join(hist_lines)

    # ---- Task line ----
    if iteration == 1:
        task_line = (
            "This is iteration 1 — the BASELINE run. "
            "Output the current strategy.py UNCHANGED.\n\n"
            "YOU MUST RESPOND IN EXACTLY THIS FORMAT — NO EXCEPTIONS:\n"
            "<reasoning>\nBaseline run, no changes.\n</reasoning>\n"
            "<strategy>\n# paste strategy here unchanged\n</strategy>"
        )
    else:
        task_line = (
            f"The CURRENT BEST aggregate score is {aggregate_score:.5f}. "
            "Propose an improved strategy.py. /no_think\n\n"
            "YOU MUST RESPOND IN EXACTLY THIS FORMAT — NO EXCEPTIONS:\n"
            "<reasoning>\nYour analysis here\n</reasoning>\n"
            "<strategy>\n# full python code here\n</strategy>\n\n"
            "Do NOT write anything outside these two tags."
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

        failure_block = (                                          # <-- indented inside the if
            f"\n--- YOUR LAST ATTEMPT (score={last_failed_score:.5f}, NOT an improvement) ---\n"
            f"Study this carefully. Do NOT repeat the same changes. "
            f"Propose something meaningfully different.\n\n"
            f"{failed_display}\n"
            f"--- END OF LAST ATTEMPT ---\n"
        )

    # Only show the LLM the editable parts to save context tokens.
    # The indicator function bodies are frozen — no need to include them in full.
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
        f"--- Per-shard results (current best strategy) ---\n{shard_table}\n\n"
        f"--- Aggregate score: {aggregate_score:.5f} ---\n\n"
        f"--- Score history (last 5) ---\n{hist_str}\n\n"
        f"{failure_block}"
        f"--- Current BEST strategy.py ---\n{strategy_display}\n\n"
        f"{task_line}"
    )

    return [
        {"role": "system", "content": program_source},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> tuple[Optional[str], Optional[str]]:
    r_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    s_match = re.search(r"<strategy>(.*?)</strategy>",   text, re.DOTALL | re.IGNORECASE)

    reasoning = r_match.group(1).strip() if r_match else None
    strategy  = s_match.group(1).strip() if s_match else None

    # Fallback: grab the largest ```python``` block
    if not strategy:
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        if blocks:
            strategy = max(blocks, key=len).strip()
            vlog("PARSE FALLBACK", "No <strategy> tag found — fell back to largest ```python``` block")

    if strategy:
        strategy = re.sub(r"^```[a-zA-Z]*\n?", "", strategy)
        strategy = re.sub(r"\n?```$",           "", strategy)
        strategy = strategy.strip()

    return reasoning, strategy


# ---------------------------------------------------------------------------
# Auto-repair: append missing boilerplate when output is truncated
# ---------------------------------------------------------------------------

def auto_repair_strategy(code: str) -> tuple[str, bool]:
    """
    If the LLM output was truncated and is missing get_params / STRATEGY_NAME,
    append the standard boilerplate so the contract check can pass.
    Returns (repaired_code, was_repaired).
    """
    if "def get_params" not in code:
        repaired = code.rstrip() + GET_PARAMS_BOILERPLATE
        return repaired, True
    return code, False


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
# Main loop
# ---------------------------------------------------------------------------

def run(
    ticker:     str,
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
        f"ticker     : {ticker}\n"
        f"start      : {start}\n"
        f"end        : {end}\n"
        f"model      : {model_id}\n"
        f"device     : {device}\n"
        f"iterations : {iterations}\n"
        f"forever    : {forever}\n"
        f"patience   : {patience}\n"
        f"start_year : {start_year}\n"
        f"program.md : {len(program_source)} chars"
    )

    df     = download_data(ticker, start, end)
    shards = make_shards(df, start_year=start_year)

    pipe = build_llm_pipeline(model_id, device)

    best_score          = -math.inf
    no_improve          = 0
    history: list[dict] = []
    iteration           = 0
    last_failed_source: Optional[str]  = None
    last_failed_score:  Optional[float] = None

    # Stale-detection: skip LLM call if metrics are identical for N iterations
    last_shard_metrics = None
    stale_count        = 0
    MAX_STALE          = 3

    while True:
        iteration += 1
        if not forever and iteration > iterations:
            vlog("LOOP END", "Max iterations reached.")
            break

        print(f"\n{'='*62}")
        print(f"  AutoFin  Iteration {iteration}" + ("" if forever else f"/{iterations}"))
        print(f"{'='*62}")

        # ---- Load strategy ----
        try:
            strat               = load_strategy(STRATEGY_FILE)
            generate_signals_fn = strat.generate_signals
            current_params      = strat.get_params()
            vlog("STRATEGY LOADED",
                f"Iteration : {iteration}\n"
                f"Params    :\n{json.dumps(current_params, indent=2)}\n\n"
                f"Source ({len(STRATEGY_FILE.read_text())} chars):\n{STRATEGY_FILE.read_text()}"
            )
        except Exception as exc:
            vlog("STRATEGY CRASH",
                f"Failed to load strategy.py\n"
                f"Exception : {exc}\n\n"
                f"{traceback.format_exc()}"
            )
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                vlog("ROLLBACK", "Rolled back to best_strategy.py after load crash.")
            append_results_tsv(iteration, 0.0, "crash", f"load error: {exc}", {})
            continue

        # ---- Evaluate on all yearly shards ----
        shard_results: dict[str, dict] = {}
        any_crash = False
        for key, shard_df in shards.items():
            m = evaluate_shard(shard_df, generate_signals_fn, key)
            shard_results[key] = m
            if m["error"] and m["score"] == -99.0:
                any_crash = True
            status_str = (
                f"Sharpe={m['sharpe']:>7.4f}  PF={m['profit_factor']:>5.2f}  "
                f"MDD={m['max_drawdown']:>5.3f}  Score={m['score']:>8.5f}  "
                f"Trades={m['n_trades']:>3}"
            )
            err_str = f"  [ERR: {m['error']}]" if m["error"] else ""
            print(f"  {key}: {status_str}{err_str}")

        valid_scores = [m["score"] for m in shard_results.values() if m["score"] != -99.0]
        aggregate    = float(np.mean(valid_scores)) if valid_scores else -99.0
        print(f"\n  ► Aggregate: {aggregate:.5f}   Best so far: {best_score:.5f}")

        # ---- Stale detection ----
        current_metrics = {k: v["score"] for k, v in shard_results.items()}
        if last_shard_metrics is not None and current_metrics == last_shard_metrics:
            stale_count += 1
            vlog("STALE DETECTION",
                f"Metrics identical for {stale_count}/{MAX_STALE} consecutive iterations.\n"
                f"Scores: {current_metrics}"
            )
            if stale_count >= MAX_STALE:
                vlog("STALE SKIP", f"Skipping LLM call — metrics flat for {MAX_STALE} iterations.")
                last_shard_metrics = current_metrics
        else:
            stale_count = 0
        last_shard_metrics = current_metrics

        # ---- Determine status ----
        if any_crash and not valid_scores:
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

        vlog("AGGREGATE",
            f"Iteration      : {iteration}\n"
            f"Aggregate      : {aggregate:.6f}\n"
            f"Best so far    : {best_score:.6f}\n"
            f"Status         : {status}\n"
            f"Description    : {description}\n"
            f"no_improve     : {no_improve}/{patience}\n"
            f"stale_count    : {stale_count}/{MAX_STALE}\n"
            f"any_crash      : {any_crash}\n"
            f"valid_shards   : {len(valid_scores)}/{len(shard_results)}\n"
            f"All scores     : {json.dumps(current_metrics, indent=2)}"
        )

        # ---- Log ----
        record = {
            "iteration":       iteration,
            "aggregate_score": aggregate,
            "shard_results":   shard_results,
            "params":          current_params,
            "status":          status,
            "timestamp":       datetime.utcnow().isoformat(),
        }
        log_jsonl(record)
        append_results_tsv(iteration, aggregate, status, description, current_params)
        history.append(record)

        # ---- Advance or roll back ----
        if aggregate > best_score:
            best_score         = aggregate
            no_improve         = 0
            last_failed_source = None
            last_failed_score  = None
            shutil.copy(STRATEGY_FILE, BEST_FILE)
            vlog("NEW BEST",
                f"Score improved : {best_score:.6f} (was -inf or lower)\n"
                f"Saved best_strategy.py"
            )
            write_trigger(best_score, iteration, current_params)

        elif aggregate == best_score:
            no_improve += 1
            vlog("EXPLORE",
                f"Score flat at {aggregate:.6f} — keeping mutated code for exploration.\n"
                f"no_improve: {no_improve}/{patience}"
            )
            last_failed_source = None
            last_failed_score  = None

        else:
            no_improve        += 1
            last_failed_source = STRATEGY_FILE.read_text()
            last_failed_score  = aggregate
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                vlog("ROLLBACK",
                    f"Regression: {aggregate:.6f} < {best_score:.6f}\n"
                    f"Rolled back strategy.py to best_strategy.py\n"
                    f"no_improve: {no_improve}/{patience}\n\n"
                    f"--- FAILED STRATEGY ---\n{last_failed_source}"
                )
            else:
                vlog("ROLLBACK SKIPPED", "No best_strategy.py exists yet — keeping current code.")

        if no_improve >= patience and not forever:
            vlog("PLATEAU", f"no_improve={no_improve} >= patience={patience}. Stopping.")
            break

        # Skip LLM call if metrics have been stale too long
        if stale_count >= MAX_STALE:
            vlog("STALE SKIP LLM", "Skipping LLM generation this iteration due to flat metrics.")
            continue
        if iteration == 1 and BEST_FILE.exists():
            vlog("ITER1 SKIP", "Iteration 1: copying best_strategy.py directly, skipping LLM.")
            shutil.copy(BEST_FILE, STRATEGY_FILE)
            continue

        # ---- Build prompt ----
        strategy_source = STRATEGY_FILE.read_text()

        messages = build_prompt(
            program_source     = program_source,
            strategy_source    = strategy_source,
            shard_results      = shard_results,
            aggregate_score    = best_score if best_score > -math.inf else aggregate,
            iteration          = iteration,
            history            = history,
            last_failed_source = last_failed_source,
            last_failed_score  = last_failed_score,
        )

        vlog("LLM PROMPT",
            f"Turns : {len(messages)}\n\n" +
            "\n" + "-"*50 + "\n".join(
                f"[{m['role'].upper()}]\n{m['content']}"
                for m in messages
            )
        )

        # ---- Generate ----
        torch.cuda.empty_cache()
        vlog("VRAM BEFORE GENERATION", torch.cuda.memory_summary(abbreviated=True))
        print("\n[llm] Generating next strategy ...")

        try:
            formatted = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            estimated_tokens = len(formatted) // 4
            vlog("LLM INPUT",
                f"Formatted prompt length : {len(formatted)} chars\n"
                f"Estimated tokens        : ~{estimated_tokens}\n\n"
                f"--- FORMATTED PROMPT (first 3000 chars) ---\n{formatted[:3000]}"
            )

            outputs    = pipe(formatted)
            llm_output = outputs[0]["generated_text"]

            vlog("LLM RAW OUTPUT",
                f"Length : {len(llm_output)} chars\n\n"
                f"--- FULL OUTPUT ---\n{llm_output}"
            )

        except torch.cuda.OutOfMemoryError as exc:
            vlog("OOM ERROR",
                f"CUDA OOM during generation.\n"
                f"Exception : {exc}\n\n"
                f"Memory summary:\n{torch.cuda.memory_summary()}"
            )
            torch.cuda.empty_cache()
            continue

        except Exception as exc:
            vlog("GENERATION ERROR",
                f"Exception : {exc}\n\n"
                f"{traceback.format_exc()}"
            )
            continue

        # ---- Parse ----
        reasoning, new_strategy = parse_llm_response(llm_output)

        vlog("LLM PARSED",
            f"Reasoning found  : {reasoning is not None}\n"
            f"Strategy found   : {new_strategy is not None}\n\n"
            f"--- REASONING ---\n{reasoning or '(none)'}\n\n"
            f"--- EXTRACTED STRATEGY ({len(new_strategy) if new_strategy else 0} chars) ---\n"
            f"{new_strategy or '(none)'}"
        )

        if reasoning:
            print(f"\n[llm] Reasoning: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}\n")

        if not new_strategy:
            vlog("PARSE FAILED", "No strategy block found in LLM output. Re-running current best.")
            print("[parse] No <strategy> block found. Re-running current best next iteration.")
            continue

        # ---- Auto-repair truncated output ----
        new_strategy, was_repaired = auto_repair_strategy(new_strategy)
        if was_repaired:
            vlog("AUTO-REPAIR",
                "Output was missing get_params — appended standard boilerplate.\n"
                "This indicates truncation; consider reducing prompt size further."
            )

        # ---- Validate ----
        syntax_ok, syntax_err     = validate_syntax(new_strategy)
        contract_ok, contract_err = validate_strategy_contract(new_strategy)

        vlog("VALIDATION",
            f"Syntax OK       : {syntax_ok}\n"
            f"Syntax error    : {syntax_err or 'none'}\n"
            f"Contract OK     : {contract_ok}\n"
            f"Contract error  : {contract_err or 'none'}"
        )

        if not syntax_ok:
            print(f"[parse] Syntax error — keeping current best. Error: {syntax_err}")
            append_results_tsv(iteration + 1, 0.0, "crash", f"syntax error: {syntax_err[:80]}", {})
            continue

        if not contract_ok:
            print(f"[parse] Contract error — {contract_err}. Keeping current best.")
            append_results_tsv(iteration + 1, 0.0, "crash", f"contract error: {contract_err}", {})
            continue

        # ---- Validate output ----
        # Write to a temp file so we can load and run it without clobbering strategy.py
        TEMP_FILE = ROOT / "strategy_candidate.py"
        TEMP_FILE.write_text(new_strategy)
        validation_df = list(shards.values())[-1]  # use 2024 shard
        output_ok, output_err = validate_strategy_output(TEMP_FILE, validation_df)

        vlog("VALIDATION OUTPUT",
            f"Output OK    : {output_ok}\n"
            f"Output error : {output_err or 'none'}"
        )

        if not output_ok:
            print(f"[validate] Output check failed — {output_err}. Keeping current best.")
            append_results_tsv(iteration + 1, 0.0, "crash", f"output check: {output_err}", {})
            TEMP_FILE.unlink(missing_ok=True)
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
            continue

        TEMP_FILE.unlink(missing_ok=True)

        # ---- Write ----
        STRATEGY_FILE.write_text(new_strategy)
        vlog("STRATEGY WRITTEN",
            f"Written to : {STRATEGY_FILE}\n"
            f"Length     : {len(new_strategy)} chars\n\n"
            f"--- CONTENT ---\n{new_strategy}"
        )
        print("[strategy] strategy.py updated for next iteration.")

    # ---- Final summary ----
    vlog("RUN COMPLETE",
        f"Best aggregate score : {best_score:.5f}\n"
        f"Best strategy        : {BEST_FILE}\n"
        f"Results table        : {RESULTS_TSV}\n"
        f"Full JSONL log       : {LOG_FILE}\n"
        f"Verbose log          : {VERBOSE_LOG}\n"
        f"Total iterations     : {iteration}"
    )
    print(f"\n{'='*62}")
    print(f"  AutoFin complete.")
    print(f"  Best aggregate score : {best_score:.5f}")
    print(f"  Best strategy        : {BEST_FILE}")
    print(f"  Results table        : {RESULTS_TSV}")
    print(f"  Full JSONL log       : {LOG_FILE}")
    print(f"  Verbose log          : {VERBOSE_LOG}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AutoFin — Autonomous Financial Strategy Optimisation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--ticker", default="^GSPC",
        help="Yahoo Finance ticker.\n  ^GSPC = S&P 500 (default)\n  ^NDX  = NASDAQ-100",
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
    run(
        ticker     = args.ticker,
        start      = args.start,
        end        = args.end,
        model_id   = args.model,
        device     = args.device,
        iterations = args.iterations,
        forever    = args.forever,
        patience   = args.patience,
        start_year = args.start_year,
    )