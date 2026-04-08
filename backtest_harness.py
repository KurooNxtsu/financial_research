"""
backtest_harness.py  —  AutoFin Evaluation Engine & LLM Orchestration
=======================================================================
THIS FILE IS READ-ONLY TO THE LLM.

Analogous to prepare.py in AutoResearch:
  - Fixed evaluation metric (composite_score) — do not modify.
  - Data loading, sharding, trade simulation — do not modify.
  - LLM orchestration loop — do not modify.

The LLM only sees program.md and strategy.py. It never sees this file.

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
PROGRAM_FILE  = ROOT / "program.md"          # LLM reads this
STRATEGY_FILE = ROOT / "strategy.py"         # LLM writes this
BEST_FILE     = ROOT / "best_strategy.py"    # snapshot of the best-ever strategy
LOG_FILE      = ROOT / "autofin_log.jsonl"   # machine-readable full log
RESULTS_TSV   = ROOT / "results.tsv"         # human-readable experiment table (untracked)
TRIGGER_FILE  = ROOT / "push_trigger.json"   # watched by external GitHub MCP


# ---------------------------------------------------------------------------
# TSV results log (mirrors AutoResearch's results.tsv)
# ---------------------------------------------------------------------------

TSV_HEADER = "iteration\taggregate_score\tstatus\tdescription\tparams_snapshot\n"


def init_results_tsv() -> None:
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)
        print(f"[log] Initialised {RESULTS_TSV}")


def append_results_tsv(
    iteration: int,
    score: float,
    status: str,       # "keep" | "discard" | "crash" | "baseline"
    description: str,
    params: dict,
) -> None:
    """Append one row to results.tsv (tab-separated, no commas in description)."""
    params_str = json.dumps(params).replace("\t", " ")
    desc       = description.replace("\t", " ").replace("\n", " ")
    row = f"{iteration}\t{score:.6f}\t{status}\t{desc}\t{params_str}\n"
    with open(RESULTS_TSV, "a") as fh:
        fh.write(row)


# ---------------------------------------------------------------------------
# Data loading & sharding
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    print(f"[data] Downloading {ticker}  {start} → {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    print(f"[data] {len(df)} rows loaded.")
    return df


def make_shards(df: pd.DataFrame, start_year: int = 2018) -> dict[str, pd.DataFrame]:
    """
    Split DataFrame into year-wide shards (keyed by 'YYYY').
    Shards with fewer than 20 bars are skipped (incomplete years).
    """
    shards: dict[str, pd.DataFrame] = {}
    for yr in sorted(df.index.year.unique()):
        if yr < start_year:
            continue
        shard = df[df.index.year == yr].copy()
        if len(shard) >= 20:
            shards[str(yr)] = shard
    return shards


# ---------------------------------------------------------------------------
# Trade simulator  (fixed — do not modify)
# ---------------------------------------------------------------------------

def simulate_trades(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a signal series into daily strategy returns.

    Positions are determined at the close of the signal bar and executed on
    the next bar (lag by 1) to avoid look-ahead bias.
    """
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
    score = Sharpe_Ratio × 0.6  +  Profit_Factor × 0.2  −  Max_Drawdown × 0.2
    This is the FIXED metric. Do not change.
    """
    return round(sharpe * 0.6 + pf * 0.2 - mdd * 0.2, 6)


def evaluate_shard(shard: pd.DataFrame, generate_signals_fn) -> dict:
    """Run strategy on one shard and return a metrics dict."""
    try:
        signals  = generate_signals_fn(shard)
        trades   = simulate_trades(shard, signals)
        dr       = trades["daily_return"]
        sh       = sharpe_ratio(dr)
        pf_      = profit_factor(dr)
        mdd      = max_drawdown(dr)
        sc       = composite_score(sh, pf_, mdd)
        n_trades = int((signals["signal"].diff().abs() > 0).sum())
        return {
            "sharpe":        round(sh,   4),
            "profit_factor": round(pf_,  4),
            "max_drawdown":  round(mdd,  4),
            "score":         sc,
            "n_trades":      n_trades,
            "n_bars":        len(shard),
            "error":         None,
        }
    except Exception as exc:
        return {
            "sharpe": 0.0, "profit_factor": 0.0,
            "max_drawdown": 1.0, "score": -99.0,
            "n_trades": 0, "n_bars": len(shard),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Strategy hot-loader
# ---------------------------------------------------------------------------

def load_strategy(path: Path):
    """Dynamically load (or reload) strategy.py. Returns the module."""
    spec   = importlib.util.spec_from_file_location("strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# LLM — HuggingFace open-source (Qwen / Gemma)
# ---------------------------------------------------------------------------

def build_llm_pipeline(model_id: str, device: str):
    """
    Load an instruct model via HuggingFace Transformers.

    For GPTQ models (names contain 'GPTQ' or 'Int4'):
      - The model is ALREADY quantized — never pass BitsAndBytesConfig.
      - Requires: pip install optimum auto-gptq
      - Works on Colab T4 / A100 directly.

    For non-quantized models (e.g. google/gemma-3-27b-it):
      - Loaded in bfloat16 directly — no extra quantization needed on GPU.

    Recommended model IDs for Colab free T4 (16 GB VRAM):
      Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4    ← safest fit on free T4
      Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4   ← fits if VRAM is clean
      Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4   ← Colab Pro / A100 only
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    print(f"[llm] Loading {model_id} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # GPTQ models carry their own quantization config — load as-is.
    # Non-GPTQ models: load in bfloat16 to save VRAM without bitsandbytes.
    is_gptq = any(tag in model_id for tag in ("GPTQ", "gptq", "Int4", "int4", "Int8", "int8"))

    model_kwargs: dict = dict(
        trust_remote_code=True,
        device_map="auto",          # lets HF place layers across GPU/CPU automatically
    )
    if not is_gptq:
        # For plain instruct models load in bfloat16 — no quantization config needed
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.3,
        do_sample=True,
        return_full_text=False,
    )
    print(f"[llm] Model loaded. VRAM used: "
          f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
    return pipe


# ---------------------------------------------------------------------------
# Prompt construction  — injects program.md verbatim, just like AutoResearch
# ---------------------------------------------------------------------------

def build_prompt(
    program_source: str,
    strategy_source: str,
    shard_results: dict[str, dict],
    aggregate_score: float,
    iteration: int,
    history: list[dict],
) -> list[dict]:
    """
    Build the chat-format messages list for the LLM.

    The LLM receives:
      - program.md verbatim as the system prompt  (read-only context)
      - current strategy.py + results as the user message  (writable context)

    This mirrors AutoResearch where the LLM is told to 'read these files for
    full context' and program.md is the live spec.
    """
    # ---- Per-shard results table ----
    rows = []
    for yr, m in shard_results.items():
        err = f"  ERR: {m['error'][:50]}" if m["error"] else ""
        rows.append(
            f"  {yr} | Sharpe={m['sharpe']:>7.4f} | PF={m['profit_factor']:>5.2f} "
            f"| MDD={m['max_drawdown']:>5.3f} | Score={m['score']:>8.5f} "
            f"| Trades={m['n_trades']:>4}{err}"
        )
    shard_table = "\n".join(rows)

    # ---- Last 5 iterations of score history ----
    hist_lines = [
        f"  iter {h['iteration']:>3}: aggregate={h['aggregate_score']:.5f}  "
        f"status={h['status']}"
        for h in history[-5:]
    ] or ["  (no history yet — this is the baseline run)"]
    hist_str = "\n".join(hist_lines)

    # ---- System prompt = program.md verbatim ----
    system_prompt = program_source

    # ---- User message = results + current strategy ----
    if iteration == 1:
        task_line = (
            "This is iteration 1 — the BASELINE run. "
            "Output the current strategy.py UNCHANGED so we can record the baseline score."
        )
    else:
        task_line = (
            f"The previous aggregate score was {aggregate_score:.5f}. "
            "Propose an improved strategy.py. Be hypothesis-driven. "
            "Reference the weakest shards in your reasoning."
        )

    user_content = (
        f"=== ITERATION {iteration} ===\n\n"
        f"--- Per-shard results ---\n{shard_table}\n\n"
        f"--- Aggregate score: {aggregate_score:.5f} ---\n\n"
        f"--- Score history (last 5) ---\n{hist_str}\n\n"
        f"--- Current strategy.py ---\n{strategy_source}\n\n"
        f"{task_line}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract <reasoning> and <strategy> blocks from LLM output."""
    r_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    s_match = re.search(r"<strategy>(.*?)</strategy>",   text, re.DOTALL | re.IGNORECASE)

    reasoning = r_match.group(1).strip() if r_match else None
    strategy  = s_match.group(1).strip() if s_match else None

    if strategy:
        # Strip accidental markdown fences
        strategy = re.sub(r"^```[a-zA-Z]*\n?", "", strategy)
        strategy = re.sub(r"\n?```$",           "", strategy)
        strategy = strategy.strip()

    return reasoning, strategy


def validate_syntax(code: str) -> tuple[bool, str]:
    """Return (ok, error_message)."""
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
# Main loop — mirrors AutoResearch's experiment loop exactly
# ---------------------------------------------------------------------------

def run(
    ticker:     str,
    start:      str,
    end:        str,
    model_id:   str,
    device:     str,
    iterations: int,       # ignored when forever=True
    forever:    bool,
    patience:   int,
    start_year: int,
) -> None:

    # ---- Setup ----
    init_results_tsv()

    program_source = PROGRAM_FILE.read_text()
    print(f"[init] Loaded program.md ({len(program_source)} chars)")

    df     = download_data(ticker, start, end)
    shards = make_shards(df, start_year=start_year)
    print(f"[data] Shards: {list(shards.keys())}")

    pipe = build_llm_pipeline(model_id, device)

    best_score = -math.inf
    no_improve = 0
    history: list[dict] = []
    iteration  = 0

    # ---- Loop forever (like AutoResearch) or up to iterations ----
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
            strat = load_strategy(STRATEGY_FILE)
            generate_signals_fn = strat.generate_signals
            current_params      = strat.get_params()
        except Exception as exc:
            print(f"[strategy] CRASH loading strategy.py: {exc}")
            traceback.print_exc()
            # Roll back to best known good
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                print("[strategy] Rolled back to best_strategy.py")
            append_results_tsv(iteration, 0.0, "crash", f"load error: {exc}", {})
            continue

        # ---- Evaluate on all shards ----
        shard_results: dict[str, dict] = {}
        any_crash = False
        for year, shard_df in shards.items():
            print(f"  Shard {year}: {len(shard_df)} bars", end="  →  ", flush=True)
            m = evaluate_shard(shard_df, generate_signals_fn)
            shard_results[year] = m
            if m["error"]:
                any_crash = True
            print(
                f"Sharpe={m['sharpe']:>7.4f}  PF={m['profit_factor']:>5.2f}  "
                f"MDD={m['max_drawdown']:>5.3f}  Score={m['score']:>8.5f}  "
                f"Trades={m['n_trades']:>4}"
                + (f"  [ERR: {m['error']}]" if m["error"] else "")
            )

        valid_scores = [m["score"] for m in shard_results.values() if not m["error"]]
        aggregate    = float(np.mean(valid_scores)) if valid_scores else -99.0
        print(f"\n  ► Aggregate: {aggregate:.5f}   Best so far: {best_score:.5f}")

        # ---- Determine status: keep / discard / crash ----
        if any_crash and not valid_scores:
            status      = "crash"
            description = "all shards errored"
        elif aggregate > best_score:
            status      = "keep" if iteration > 1 else "baseline"
            description = f"iter {iteration}: aggregate={aggregate:.5f}"
        else:
            status      = "discard" if iteration > 1 else "baseline"
            description = f"iter {iteration}: no improvement ({aggregate:.5f} vs {best_score:.5f})"

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

        # ---- Advance or roll back (AutoResearch pattern) ----
        if aggregate > best_score:
            best_score = aggregate
            no_improve = 0
            shutil.copy(STRATEGY_FILE, BEST_FILE)
            print(f"  ★ New best! Saved to best_strategy.py")
            write_trigger(best_score, iteration, current_params)
        else:
            no_improve += 1
            # ROLL BACK — restore strategy.py to the best known good version
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                print(f"  ✗ Rolled back to best_strategy.py  (no_improve={no_improve}/{patience})")
            if no_improve >= patience and not forever:
                print("[loop] Plateau reached. Stopping.")
                break

        # ---- Call LLM for next proposal ----
        # Re-read strategy source AFTER potential rollback
        strategy_source = STRATEGY_FILE.read_text()

        messages = build_prompt(
            program_source  = program_source,
            strategy_source = strategy_source,
            shard_results   = shard_results,
            aggregate_score = aggregate if aggregate > -99.0 else best_score,
            iteration       = iteration,
            history         = history,
        )

        print("\n[llm] Generating next strategy ...")
        try:
            formatted = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            outputs    = pipe(formatted)
            llm_output = outputs[0]["generated_text"]
        except Exception as exc:
            print(f"[llm] Generation failed: {exc}")
            traceback.print_exc()
            # Don't break the loop — the harness will re-run current best next iter
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
            # Log the crash but don't overwrite strategy.py
            append_results_tsv(
                iteration + 1, 0.0, "crash",
                f"syntax error: {syntax_err[:80]}", {}
            )
            continue

        # ---- Write the new strategy for the next evaluation ----
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
    p.add_argument("--start",      default="2018-01-01",
                   help="Data start date (YYYY-MM-DD)")
    p.add_argument("--end",        default="2024-12-31",
                   help="Data end date  (YYYY-MM-DD)")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        help=(
            "HuggingFace model ID.\n"
            "  Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4  (A100 — best)\n"
            "  Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4  (L4  — default)\n"
            "  google/gemma-3-27b-it                  (alternative)"
        ),
    )
    p.add_argument("--device",     default="cuda",
                   help="cuda | cpu")
    p.add_argument("--iterations", type=int, default=20,
                   help="Max iterations (ignored if --forever)")
    p.add_argument("--forever",    action="store_true",
                   help="Run indefinitely until interrupted (mirrors AutoResearch behaviour)")
    p.add_argument("--patience",   type=int, default=5,
                   help="Stop after N consecutive non-improvements (only without --forever)")
    p.add_argument("--start_year", type=int, default=2018,
                   help="First shard year")
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