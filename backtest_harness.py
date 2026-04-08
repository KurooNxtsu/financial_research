"""
backtest_harness.py  —  AutoFin Evaluation Engine & LLM Orchestration
=======================================================================
THIS FILE IS READ-ONLY TO THE LLM.

Dependencies (all standard, no GPTQ / bitsandbytes needed):
    pip install yfinance pandas numpy transformers accelerate

Recommended models for Colab free T4 (16 GB VRAM):
    Qwen/Qwen2.5-7B-Instruct     ~14 GB bfloat16  ← default, safe on T4
    Qwen/Qwen2.5-3B-Instruct     ~6  GB bfloat16  ← very safe, slightly weaker
    google/gemma-2-2b-it          ~4  GB bfloat16  ← lightest option

Usage:
    python backtest_harness.py --ticker NVDA --device cuda --iterations 10
    python backtest_harness.py --ticker "^GSPC" --device cuda --forever
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
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
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parent.resolve()
PROGRAM_FILE  = ROOT / "program.md"
STRATEGY_FILE = ROOT / "strategy.py"
BEST_FILE     = ROOT / "best_strategy.py"
LOG_FILE      = ROOT / "autofin_log.jsonl"
RESULTS_TSV   = ROOT / "results.tsv"
TRIGGER_FILE  = ROOT / "push_trigger.json"

# ---------------------------------------------------------------------------
# TSV results log
# ---------------------------------------------------------------------------

TSV_HEADER = "iteration\taggregate_score\tstatus\tdescription\tparams_snapshot\n"


def init_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)
        print(f"[log] Initialised {RESULTS_TSV}")
    else:
        print(f"[log] Appending to existing {RESULTS_TSV}")


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
    # Also flush to stdout so we can see it even inside subprocess
    print(f"[tsv] {row.strip()}", flush=True)


# ---------------------------------------------------------------------------
# Data loading & sharding
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"[data] Downloading {ticker}  {start} → {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance ≥0.2 returns MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    print(f"[data] {len(df)} rows loaded.")
    return df


def make_shards(df: pd.DataFrame, start_year: int = 2018) -> dict[str, pd.DataFrame]:
    shards: dict[str, pd.DataFrame] = {}
    for yr in sorted(df.index.year.unique()):
        if yr < start_year:
            continue
        shard = df[df.index.year == yr].copy()
        if len(shard) >= 20:
            shards[str(yr)] = shard
    return shards


# ---------------------------------------------------------------------------
# Trade simulator
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
    if std == 0 or math.isnan(float(std)):
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
    """score = Sharpe×0.6 + ProfitFactor×0.2 − MaxDrawdown×0.2  (fixed metric)"""
    return round(sharpe * 0.6 + pf * 0.2 - mdd * 0.2, 6)


def evaluate_shard(shard: pd.DataFrame, generate_signals_fn) -> dict:
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
            "sharpe":        round(sh,  4),
            "profit_factor": round(pf_, 4),
            "max_drawdown":  round(mdd, 4),
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
    spec   = importlib.util.spec_from_file_location("strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# LLM  — plain HuggingFace Transformers, bfloat16, NO quantization libraries
# ---------------------------------------------------------------------------

def build_llm_pipeline(model_id: str, device: str):
    """
    Load a plain instruct model in bfloat16.

    No GPTQ, no bitsandbytes, no auto-gptq — just transformers + accelerate.
    Works on any Colab GPU tier out of the box.

    VRAM usage at bfloat16:
        Qwen/Qwen2.5-3B-Instruct   ~6  GB   (very safe on free T4)
        Qwen/Qwen2.5-7B-Instruct   ~14 GB   (safe on free T4)
        Qwen/Qwen2.5-14B-Instruct  ~28 GB   (Colab Pro A100 only)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    print(f"[llm] Loading {model_id} in bfloat16 on {device} ...")
    print(f"[llm] Available VRAM: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # half precision — no quantization library needed
        device_map="auto",             # HF places layers on GPU automatically
        trust_remote_code=True,
    )
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

    used_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"[llm] Model loaded. VRAM used: {used_vram:.1f} GB")
    return pipe


# ---------------------------------------------------------------------------
# Prompt construction  — injects program.md verbatim
# ---------------------------------------------------------------------------

def build_prompt(
    program_source: str,
    strategy_source: str,
    shard_results: dict[str, dict],
    aggregate_score: float,
    iteration: int,
    history: list[dict],
) -> list[dict]:
    # Per-shard table
    rows = []
    for yr, m in shard_results.items():
        err = f"  ERR: {m['error'][:50]}" if m["error"] else ""
        rows.append(
            f"  {yr} | Sharpe={m['sharpe']:>7.4f} | PF={m['profit_factor']:>5.2f}"
            f" | MDD={m['max_drawdown']:>5.3f} | Score={m['score']:>8.5f}"
            f" | Trades={m['n_trades']:>4}{err}"
        )
    shard_table = "\n".join(rows)

    # Last 5 iterations of history
    hist_lines = [
        f"  iter {h['iteration']:>3}: aggregate={h['aggregate_score']:.5f}  status={h['status']}"
        for h in history[-5:]
    ] or ["  (no history yet — this is the baseline run)"]
    hist_str = "\n".join(hist_lines)

    # Iteration-specific instruction
    if iteration == 1:
        task_line = (
            "This is iteration 1 — the BASELINE run. "
            "Output the current strategy.py UNCHANGED so we can record the baseline score."
        )
    else:
        task_line = (
            f"The previous aggregate score was {aggregate_score:.5f}. "
            "Propose an improved strategy.py. Be hypothesis-driven and reference "
            "the weakest shards in your reasoning."
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
        {"role": "system", "content": program_source},   # program.md verbatim
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
# Loggers
# ---------------------------------------------------------------------------

def log_jsonl(record: dict) -> None:
    with open(LOG_FILE, "a") as fh:
        fh.write(json.dumps(record) + "\n")


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

    # Load program.md from disk — injected into every LLM prompt verbatim
    if not PROGRAM_FILE.exists():
        raise FileNotFoundError(
            f"program.md not found at {PROGRAM_FILE}. "
            "Make sure all three files are in the same directory."
        )
    program_source = PROGRAM_FILE.read_text()
    print(f"[init] Loaded program.md ({len(program_source)} chars)")

    df     = download_data(ticker, start, end)
    shards = make_shards(df, start_year=start_year)
    print(f"[data] Shards: {list(shards.keys())}")

    # Load model once — stays in VRAM for the entire run
    pipe = build_llm_pipeline(model_id, device)

    best_score = -math.inf
    no_improve = 0
    history: list[dict] = []
    iteration  = 0

    while True:
        iteration += 1
        if not forever and iteration > iterations:
            print("[loop] Max iterations reached.")
            break

        print(f"\n{'='*62}")
        print(f"  AutoFin  Iteration {iteration}" + ("" if forever else f"/{iterations}"))
        print(f"{'='*62}", flush=True)

        # ---- Load & validate strategy ----
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

        # ---- Run backtests on all shards ----
        shard_results: dict[str, dict] = {}
        any_crash = False
        for year, shard_df in shards.items():
            print(f"  Shard {year}: {len(shard_df)} bars", end="  →  ", flush=True)
            m = evaluate_shard(shard_df, generate_signals_fn)
            shard_results[year] = m
            if m["error"]:
                any_crash = True
            print(
                f"Sharpe={m['sharpe']:>7.4f}  PF={m['profit_factor']:>5.2f}"
                f"  MDD={m['max_drawdown']:>5.3f}  Score={m['score']:>8.5f}"
                f"  Trades={m['n_trades']:>4}"
                + (f"  [ERR: {m['error']}]" if m["error"] else ""),
                flush=True,
            )

        valid_scores = [m["score"] for m in shard_results.values() if not m["error"]]
        aggregate    = float(np.mean(valid_scores)) if valid_scores else -99.0
        print(f"\n  ► Aggregate: {aggregate:.5f}   Best so far: {best_score:.5f}", flush=True)

        # ---- Status ----
        if any_crash and not valid_scores:
            status      = "crash"
            description = "all shards errored"
        elif iteration == 1:
            status      = "baseline"
            description = f"baseline: aggregate={aggregate:.5f}"
        elif aggregate > best_score:
            status      = "keep"
            description = f"iter {iteration}: improved to {aggregate:.5f}"
        else:
            status      = "discard"
            description = f"iter {iteration}: no improvement ({aggregate:.5f} <= {best_score:.5f})"

        # ---- Log everything immediately ----
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
            best_score = aggregate
            no_improve = 0
            shutil.copy(STRATEGY_FILE, BEST_FILE)
            print(f"  ★ New best! Saved to best_strategy.py")
            write_trigger(best_score, iteration, current_params)
        else:
            no_improve += 1
            if BEST_FILE.exists():
                shutil.copy(BEST_FILE, STRATEGY_FILE)
                print(f"  ✗ Rolled back to best_strategy.py  "
                      f"(no_improve={no_improve}/{patience})")
            if no_improve >= patience and not forever:
                print("[loop] Plateau reached. Stopping.")
                break

        # ---- Build LLM prompt (re-read after possible rollback) ----
        strategy_source = STRATEGY_FILE.read_text()
        messages = build_prompt(
            program_source  = program_source,
            strategy_source = strategy_source,
            shard_results   = shard_results,
            aggregate_score = aggregate if aggregate > -99.0 else best_score,
            iteration       = iteration,
            history         = history,
        )

        # ---- Generate next strategy ----
        print("\n[llm] Generating next strategy ...", flush=True)
        try:
            formatted  = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            outputs    = pipe(formatted)
            llm_output = outputs[0]["generated_text"]
        except Exception as exc:
            print(f"[llm] Generation failed: {exc}")
            traceback.print_exc()
            continue

        reasoning, new_strategy = parse_llm_response(llm_output)

        if reasoning:
            print(f"\n[llm] Reasoning:\n{reasoning[:400]}"
                  f"{'...' if len(reasoning) > 400 else ''}\n")

        if not new_strategy:
            print("[parse] No <strategy> block found. Keeping current best.")
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
        print("[strategy] strategy.py updated for next iteration.", flush=True)

    # ---- Summary ----
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
        help="Yahoo Finance ticker.\n  ^GSPC = S&P 500\n  ^NDX  = NASDAQ-100\n  NVDA  = single stock",
    )
    p.add_argument("--start",      default="2018-01-01")
    p.add_argument("--end",        default="2024-12-31")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "HuggingFace model ID (plain instruct, no GPTQ needed):\n"
            "  Qwen/Qwen2.5-3B-Instruct   ~6  GB VRAM  (very safe on free T4)\n"
            "  Qwen/Qwen2.5-7B-Instruct   ~14 GB VRAM  (default, fits free T4)\n"
            "  Qwen/Qwen2.5-14B-Instruct  ~28 GB VRAM  (Colab Pro A100 only)\n"
            "  google/gemma-2-2b-it        ~4  GB VRAM  (lightest option)"
        ),
    )
    p.add_argument("--device",     default="cuda")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--forever",    action="store_true",
                   help="Run indefinitely until interrupted")
    p.add_argument("--patience",   type=int, default=5,
                   help="Stop after N consecutive non-improvements (ignored with --forever)")
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