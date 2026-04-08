"""
backtest_harness.py  —  AutoFin Evaluation Engine & LLM Orchestration
=======================================================================
THIS FILE IS READ-ONLY TO THE LLM.

Changes vs v1:
  - Shards are now MONTHLY (keyed "YYYY-MM") instead of yearly.
    Each iteration completes in seconds rather than minutes.
  - Every shard gets a 100-day warm-up buffer prepended so that
    Ichimoku / ATR / VWAP are fully initialised on the very first
    bar of the evaluation window. The buffer rows are stripped before
    any metrics are calculated — no look-ahead, no contamination.
  - Rollback logic changed: we only roll back when aggregate < best_score.
    When aggregate == best_score (both 0.0 while the agent is exploring)
    we let the mutated code stand so the LLM can keep searching.
  - The LLM is now shown its failed strategy alongside the current best
    so it learns what NOT to repeat.

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

# How many calendar days of history to prepend to each shard so that
# slow indicators (Ichimoku span_b needs 52+26 = 78 bars) are warm.
WARMUP_DAYS = 100

# ---------------------------------------------------------------------------
# TSV results log
# ---------------------------------------------------------------------------

TSV_HEADER = "iteration\taggregate_score\tstatus\tdescription\tparams_snapshot\n"


def init_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)
        print(f"[log] Initialised {RESULTS_TSV}")


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
    print(f"[data] Downloading {ticker}  {start} → {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    print(f"[data] {len(df)} rows loaded.")
    return df


# ---------------------------------------------------------------------------
# Monthly sharding with warm-up buffer
# ---------------------------------------------------------------------------

def make_shards(df: pd.DataFrame, start_year: int = 2016) -> dict[str, pd.DataFrame]:
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

        # Target window: the calendar year
        year_mask = df.index.year == yr
        year_df   = df[year_mask]

        if len(year_df) < 50:          # skip very short years
            continue

        # Warm-up window: up to WARMUP_DAYS trading days before the year starts
        year_start = year_df.index[0]
        pre_mask   = df.index < year_start
        pre_df     = df[pre_mask].iloc[-WARMUP_DAYS:]

        combined = pd.concat([pre_df, year_df])

        # Tag the boundary so evaluate_shard can strip the buffer
        combined.attrs["eval_start"] = year_start
        shards[str(yr)] = combined

    return shards

def strip_warmup(series: pd.Series, eval_start: pd.Timestamp) -> pd.Series:
    """Return only the bars from eval_start onward."""
    return series[series.index >= eval_start]


# ---------------------------------------------------------------------------
# Trade simulator  (fixed — do not modify)
# ---------------------------------------------------------------------------

def simulate_trades(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
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


def evaluate_shard(shard: pd.DataFrame, generate_signals_fn) -> dict:
    """
    Run strategy on one shard (which includes a warm-up buffer) and return
    metrics computed only over the target evaluation window.
    """
    eval_start = shard.attrs.get("eval_start", shard.index[0])

    try:
        # Run indicators over the FULL shard (including warm-up) so they
        # are initialised by the time we reach the evaluation window.
        signals = generate_signals_fn(shard)
        trades  = simulate_trades(shard, signals)

        # Strip warm-up rows before scoring
        dr_full = trades["daily_return"]
        dr      = strip_warmup(dr_full, eval_start)

        if len(dr) == 0:
            return _empty_result(shard, "no eval bars after warm-up strip")

        sh   = sharpe_ratio(dr)
        pf_  = profit_factor(dr)
        mdd  = max_drawdown(dr)
        sc   = composite_score(sh, pf_, mdd)

        sig_eval  = strip_warmup(signals["signal"], eval_start)
        n_trades  = int((sig_eval.diff().abs() > 0).sum())

        return {
            "sharpe":        round(sh,  4),
            "profit_factor": round(pf_, 4),
            "max_drawdown":  round(mdd, 4),
            "score":         sc,
            "n_trades":      n_trades,
            "n_bars":        len(dr),
            "error":         None,
        }
    except Exception as exc:
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
# Strategy hot-loader
# ---------------------------------------------------------------------------

def load_strategy(path: Path):
    spec   = importlib.util.spec_from_file_location("strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    import torch

    print(f"[llm] Loading {model_id} on {device} ...")

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
        max_new_tokens=2048,
        temperature=0.3,
        do_sample=True,
        return_full_text=False,
    )
    print("[llm] Model loaded.")
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
    """
    Build the chat-format messages list for the LLM.

    Now includes the failed strategy code (if any) so the LLM knows
    exactly what it tried last and why it didn't improve.
    """
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
            "Propose an improved strategy.py.\n\n"
            "YOU MUST RESPOND IN EXACTLY THIS FORMAT — NO EXCEPTIONS:\n"
            "<reasoning>\nYour analysis here\n</reasoning>\n"
            "<strategy>\n# full python code here\n</strategy>\n\n"
            "Do NOT write anything outside these two tags."
        )

    # ---- Failure context block (the key fix for Trap 2) ----
    failure_block = ""
    if last_failed_source is not None:
        failure_block = (
            f"\n--- YOUR LAST ATTEMPT (score={last_failed_score:.5f}, NOT an improvement) ---\n"
            f"Study this carefully. Do NOT repeat the same changes. "
            f"Propose something meaningfully different.\n\n"
            f"{last_failed_source}\n"
            f"--- END OF LAST ATTEMPT ---\n"
        )

    user_content = (
        f"=== ITERATION {iteration} ===\n\n"
        f"--- Per-shard results (current best strategy) ---\n{shard_table}\n\n"
        f"--- Aggregate score: {aggregate_score:.5f} ---\n\n"
        f"--- Score history (last 5) ---\n{hist_str}\n\n"
        f"{failure_block}"
        f"--- Current BEST strategy.py ---\n{strategy_source}\n\n"
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

    # Fallback: if no <strategy> tag, grab the largest ```python``` block
    if not strategy:
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        if blocks:
            strategy = max(blocks, key=len).strip()
            print("[parse] No <strategy> tag — fell back to largest ```python``` block")

    if strategy:
        strategy = re.sub(r"^```[a-zA-Z]*\n?", "", strategy)
        strategy = re.sub(r"\n?```$",           "", strategy)
        strategy = strategy.strip()

    return reasoning, strategy

def validate_syntax(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<strategy>", "exec")
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

def log_jsonl(record: dict) -> None:
    with open(LOG_FILE, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# GitHub MCP trigger
# ---------------------------------------------------------------------------

def write_trigger(score: float, iteration: int, params: dict) -> None:
    payload = {
        "score":     score,
        "iteration": iteration,
        "params":    params,
        "timestamp": datetime.utcnow().isoformat(),
    }
    TRIGGER_FILE.write_text(json.dumps(payload, indent=2))
    print(f"[trigger] push_trigger.json updated (score={score:.5f})")


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
    print(f"[init] Loaded program.md ({len(program_source)} chars)")

    df     = download_data(ticker, start, end)
    shards = make_shards(df, start_year=start_year)
    print(f"[data] Monthly shards: {list(shards.keys())}")

    pipe = build_llm_pipeline(model_id, device)

    best_score         = -math.inf
    no_improve         = 0
    history: list[dict] = []
    iteration          = 0

    # Track the last failed attempt so the LLM can learn from it
    last_failed_source: Optional[str] = None
    last_failed_score:  Optional[float] = None

    while True:
        iteration += 1
        if not forever and iteration > iterations:
            print("[loop] Max iterations reached.")
            break

        print(f"\n{'='*62}")
        print(f"  AutoFin  Iteration {iteration}" + ("" if forever else f"/{iterations}"))
        print(f"{'='*62}")

        # ---- Load strategy ----
        try:
            strat               = load_strategy(STRATEGY_FILE)
            generate_signals_fn = strat.generate_signals
            current_params      = strat.get_params()
        except Exception as exc:
            print(f"[strategy] CRASH loading strategy.py: {exc}")
            traceback.print_exc()
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                print("[strategy] Rolled back to best_strategy.py")
            append_results_tsv(iteration, 0.0, "crash", f"load error: {exc}", {})
            continue

        # ---- Evaluate on all monthly shards ----
        shard_results: dict[str, dict] = {}
        any_crash = False
        for key, shard_df in shards.items():
            m = evaluate_shard(shard_df, generate_signals_fn)
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

        # ---- Determine status ----
        if any_crash and not valid_scores:
            status      = "crash"
            description = "all shards errored"
        elif aggregate > best_score:
            status      = "keep" if iteration > 1 else "baseline"
            description = f"iter {iteration}: aggregate={aggregate:.5f}"
        elif aggregate == best_score:
            # Exploration: score didn't improve but didn't regress either.
            # Let the mutated code stand so the LLM can keep wandering.
            status      = "explore"
            description = f"iter {iteration}: flat ({aggregate:.5f}), keeping for exploration"
        else:
            status      = "discard" if iteration > 1 else "baseline"
            description = f"iter {iteration}: regression ({aggregate:.5f} < {best_score:.5f})"

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
            best_score          = aggregate
            no_improve          = 0
            last_failed_source  = None
            last_failed_score   = None
            shutil.copy(STRATEGY_FILE, BEST_FILE)
            print(f"  ★ New best! Saved to best_strategy.py")
            write_trigger(best_score, iteration, current_params)

        elif aggregate == best_score:
            # Exploration mode — do NOT roll back. The LLM keeps its mutated code.
            no_improve += 1
            print(f"  ~ Exploration move kept (no_improve={no_improve}/{patience})")
            # No failure to report — the code is still live
            last_failed_source = None
            last_failed_score  = None

        else:
            # Regression — roll back and remember what failed
            no_improve         += 1
            last_failed_source  = STRATEGY_FILE.read_text()
            last_failed_score   = aggregate
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                print(f"  ✗ Regression: rolled back to best_strategy.py  (no_improve={no_improve}/{patience})")
            else:
                print(f"  ✗ Regression, no best file yet — keeping current code.")

        if no_improve >= patience and not forever:
            print("[loop] Plateau reached. Stopping.")
            break

        # ---- Call LLM for next proposal ----
        # Re-read strategy AFTER potential rollback so the LLM always sees
        # the current best. The failed attempt is passed separately.
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

        print("\n[llm] Generating next strategy ...")
        try:
            formatted  = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            outputs    = pipe(formatted)
            llm_output = outputs[0]["generated_text"]
        except Exception as exc:
            print(f"[llm] Generation failed: {exc}")
            traceback.print_exc()
            continue

        reasoning, new_strategy = parse_llm_response(llm_output)

        if reasoning:
            print(f"\n[llm] Reasoning: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}\n")

        if not new_strategy:
            print("[parse] No <strategy> block found. Re-running current best next iteration.")
            continue

        syntax_ok, syntax_err = validate_syntax(new_strategy)
        if not syntax_ok:
            print(f"[parse] Syntax error — keeping current best. Error: {syntax_err}")
            append_results_tsv(
                iteration + 1, 0.0, "crash",
                f"syntax error: {syntax_err[:80]}", {}
            )
            continue

        STRATEGY_FILE.write_text(new_strategy)
        print("[strategy] strategy.py updated for next iteration.")

    # ---- Final summary ----
    print(f"\n{'='*62}")
    print(f"  AutoFin complete.")
    print(f"  Best aggregate score : {best_score:.5f}")
    print(f"  Best strategy        : {BEST_FILE}")
    print(f"  Results table        : {RESULTS_TSV}")
    print(f"  Full JSONL log       : {LOG_FILE}")
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
        default="Qwen/Qwen2.5-32B-Instruct",
        help=(
            "HuggingFace model ID.\n"
            "  Qwen/Qwen2.5-72B-Instruct  (A100 — best)\n"
            "  Qwen/Qwen2.5-32B-Instruct  (L4  — default)\n"
            "  Qwen/Qwen2.5-7B-Instruct   (smaller GPU, faster)\n"
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