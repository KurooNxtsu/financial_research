# AutoFin — Autonomous Financial Strategy Agent

> A self-improving quantitative trading strategy agent inspired by AI Scientist / AutoResearch, running on open-source models and **free cloud GPUs** that keeps changing as you run/update the code. (Google Colab / Kaggle).

![Aggregate score over 8 iterations, rising from 0.37 to 0.90](results_chart.png)

---

## What is this?

AutoFin is an autonomous loop where an LLM iteratively reads its own backtest results, rewrites a trading strategy, and re-evaluates — with no human in the loop.

It is the AI Scientist idea applied to quantitative finance:

| AI Scientist | AutoFin |
|---|---|
| LLM proposes experiments | LLM proposes strategy logic |
| Runs ML training code | Runs a backtest harness |
| Scores on validation accuracy | Scores on Sharpe + PF − MDD |
| Writes a paper | Saves best_strategy.py |

The key constraint that makes this interesting: **everything runs on a free Colab T4 or Kaggle GPU**. No paid APIs, no proprietary data, no enterprise infrastructure.

---

## Results

After 8 iterations against `^GSPC` and `GLD` (2018–2024):

| Iteration | Score | Status |
|---|---|---|
| 1 | 0.3727 | baseline |
| 2 | 0.3727 | explore (flat) |
| 3 | **0.8998** | keep ✓ |
| 4 | 0.8998 | explore (flat) |
| 5 | 0.4594 | discard ✗ |
| 6 | 0.7574 | discard ✗ |
| 7 | 0.8292 | discard ✗ |
| 8 | **0.8998** | keep ✓ |

The score jump at iteration 3 came from the agent identifying that `^GSPC 2019` was scoring −0.76 because the strategy was shorting the Q4 2018 recovery while Ichimoku was still bearish but price had already recovered above VWAP. It added a VWAP filter on short entries without being told the root cause.

---

## Architecture

Three files. One loop.

```
autofin/
├── program.md           ← agent's constitution (read-only)
├── strategy.py          ← agent's only writable surface
└── backtest_harness.py  ← locked evaluation engine (read-only)
```

### `program.md` — The constitution

Read by the LLM at every iteration. Contains the scoring formula, indicator documentation, known weak shards, structural anti-patterns the agent must not repeat, diversity forcing ideas, and the required response format. Think of it as the system prompt that also encodes institutional memory across iterations.

### `strategy.py` — The writable surface

Four indicator functions with **frozen signatures**: `ichimoku_cloud`, `rsi`, `atr`, `vwap`. The agent can only modify:
- Parameter dicts (`ICHIMOKU_PARAMS`, `RSI_PARAMS`, `ATR_PARAMS`, `VWAP_PARAMS`)
- Signal logic inside `generate_signals()`

It cannot rename functions, change signatures, or add imports.

### `backtest_harness.py` — The locked engine

Never shown to the agent. Handles everything else:
- `yfinance` data download for multiple tickers
- Yearly sharding (2018–2024) with 100-day warm-up buffers
- Composite scoring: `Sharpe×0.6 + PF×0.2 − MDD×0.2`
- Multi-ticker aggregate scoring (mean across all tickers and shards)
- Ensemble gate: rejects strategies that overfit to the primary ticker
- Static validators before backtesting (syntax, ATR formula, no-DataFrame-join)
- Auto-repair loop (up to 3 LLM repair calls on validation failure)
- Rollback to best on regression; diversity forcing when stuck
- Temperature scheduling: 0.55 → 0.20 over iterations
- Strategy hash tracking to detect and break repeated-code loops
- Live HTML dashboard (`results.html`, auto-refreshes every 30s)

### The loop

```
while True:
    1. Load strategy.py
    2. Evaluate across all tickers + shards → aggregate score
    3. If improved + ensemble gate passed → save as new best
    4. If regressed → rollback to best
    5. Build prompt: spec + strategy + heatmap + score history + failure context
    6. LLM proposes new strategy.py
    7. Validate (syntax → contract → ATR formula → no-join → output)
    8. Auto-repair if needed (up to 3 attempts)
    9. Write validated code → next iteration
```

---

## Setup

### Requirements

```
python >= 3.10
torch
transformers
bitsandbytes
accelerate
yfinance
pandas
numpy
```

Install:

```bash
pip install torch transformers bitsandbytes accelerate yfinance pandas numpy
```

> On Colab/Kaggle, `torch` is pre-installed. You only need the remaining packages.

### Files

Clone or copy three files into the same directory:

```bash
git clone https://github.com/yourusername/autofin.git
cd autofin
```

Verify `strategy.py` loads cleanly before starting:

```bash
python strategy.py
# Self-test passed: N long bars, N short bars over 300 days.
```

---

## Usage

### Recommended (multi-ticker)

```bash
python backtest_harness.py \
  --tickers "NVDA,^GSPC,MSFT,GLD" \
  --model "Qwen/Qwen3.5-9B" \
  --device cuda \
  --iterations 30
```

### Minimal (two tickers, good for Colab free tier)

```bash
python backtest_harness.py \
  --tickers "^GSPC,GLD" \
  --model "Qwen/Qwen3.5-9B" \
  --device cuda \
  --iterations 20
```

### Run forever (patience-based stopping)

```bash
python backtest_harness.py \
  --tickers "NVDA,^GSPC,MSFT,GLD" \
  --device cuda \
  --forever \
  --patience 8
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--tickers` | `NVDA,^GSPC,MSFT,GLD` | Comma-separated Yahoo Finance tickers. First ticker is primary. |
| `--ticker` | — | Legacy single-ticker mode. |
| `--start` | `2018-01-01` | Data start date. |
| `--end` | `2024-12-31` | Data end date. |
| `--model` | `Qwen/Qwen3.5-9B` | HuggingFace model ID. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--iterations` | `20` | Max iterations before stopping. |
| `--forever` | `False` | Ignore `--iterations`; stop only on `--patience`. |
| `--patience` | `5` | Stop after N consecutive non-improvements. |
| `--start_year` | `2018` | Earliest year to include in sharding. |

---

## Model selection

| Model | GPU | Notes |
|---|---|---|
| `Qwen/Qwen3.5-9B` | Colab T4 (free) | Default. Works on 15GB VRAM with 4-bit. |
| `Qwen/Qwen2.5-7B-Instruct` | Colab T4 (free) | No thinking mode — more predictable output length. |
| `Qwen/Qwen2.5-32B-Instruct` | Kaggle P100 / L4 | Better reasoning, larger context. |
| `Qwen/Qwen2.5-72B-Instruct` | A100 (Colab Pro) | Best results, requires paid tier. |
| `google/gemma-3-27b-it` | L4 | Good alternative to Qwen. |

All models run 4-bit NF4 double-quantised via `bitsandbytes`.

---

## Output files

| File | Description |
|---|---|
| `best_strategy.py` | Best strategy found so far. Survives restarts. |
| `results.tsv` | Tab-separated log: iteration, score, status, params. |
| `autofin_log.jsonl` | Full per-shard results in JSONL format. |
| `verbose.log` | Complete run log including LLM prompts and outputs. |
| `results.html` | Live dashboard (auto-refreshes every 30s). |
| `push_trigger.json` | Written on every new best — hook for CI/CD or GitHub Actions. |

---

## Scoring

The composite score is fixed and cannot be changed by the agent:

```
score = Sharpe_Ratio × 0.6  +  Profit_Factor × 0.2  −  Max_Drawdown × 0.2
```

The **aggregate score** is the mean of this across all tickers and all yearly shards. This prevents the agent from overfitting to a single asset or a single bull-market year.

### Ensemble gate

Before any new best is accepted, the candidate strategy is evaluated on 2023 and 2024 out-of-sample shards for all non-primary tickers. If the mean OOS score falls below −0.30, the strategy is rejected even if the primary-ticker aggregate improved.

### Yearly shards

Each shard is one calendar year with a 100-day warm-up buffer prepended so indicators (especially Ichimoku, which needs 78 bars to initialise) are fully ready from the first evaluation bar.

---

## Indicators

The agent works with four indicators whose function signatures are frozen:

| Indicator | Key outputs | Default params |
|---|---|---|
| `ichimoku_cloud` | `bullish`, `bearish`, `tenkan_sen`, `kijun_sen`, `cloud_top`, `cloud_bottom` | conversion=9, base=26, lagging=52, displacement=26 |
| `rsi` | `rsi_value`, `buy_signal`, `sell_signal` | period=14, overbought=70, oversold=30 |
| `atr` | `atr_value`, `long_stop`, `short_stop` | period=14, multiplier=2.5 |
| `vwap` | `vwap_value`, `above_vwap`, `below_vwap` | period=14 |

The agent accesses each indicator's columns directly — never by joining DataFrames (a common failure mode the harness validates against).

---

## Robustness features

**Static validators** run before every backtest:
- Syntax check (Python `compile`)
- Contract check (both `generate_signals` and `get_params` present; no name-shadowing)
- ATR formula check (must use `close.shift(1)` for previous close)
- No-DataFrame-join check (catches `atr_value_atr`-style KeyErrors before they reach the LLM)
- Output validation (non-zero signals, not stuck, minimum transitions)

**Auto-repair loop**: on validation failure, a targeted repair prompt is sent to the LLM with the specific error type and anti-pattern guidance. Up to 3 repair attempts before the iteration is abandoned and the best strategy is restored.

**Diversity forcing**: if `no_improve >= 2`, the prompt is augmented with two concrete structural ideas the agent has not tried, preventing pure threshold-tuning loops.

**Temperature scheduling**: generation temperature starts at 0.55 (broad exploration) and decays to 0.20 (refinement) over iterations, with a bump back up to 0.55 when stuck.

**Strategy hash tracking**: detects when the agent is submitting structurally identical code across iterations and injects a stronger diversity prompt.

---

## Resuming a run

The harness is stateless between restarts. `best_strategy.py` is written to disk on every new best. On restart:

```bash
# Just re-run the same command — iteration 1 copies best_strategy.py automatically
python backtest_harness.py --tickers "^GSPC,GLD" --device cuda --iterations 30
```

If `best_strategy.py` exists, iteration 1 skips the LLM entirely and uses it as the starting point.

---

## Colab quickstart

```python
# Install dependencies
!pip install bitsandbytes accelerate yfinance -q

# Clone repo
!git clone https://github.com/yourusername/autofin.git
%cd autofin

# Verify strategy loads
!python strategy.py

# Run
!python backtest_harness.py \
    --tickers "^GSPC,GLD" \
    --model "Qwen/Qwen3.5-9B" \
    --device cuda \
    --iterations 20
```

Monitor progress by opening `results.html` or tailing `verbose.log`:

```python
!tail -f verbose.log
```

---

## Limitations

- **Backtesting ≠ live trading.** This system optimises historical performance. No transaction costs, slippage, or market impact are modelled.
- **Indicator set is fixed.** The agent cannot add new data sources or indicators beyond the four provided.
- **No imports.** The agent can only use `numpy`, `pandas`, and `math` inside `strategy.py`.
- **Context window.** At 9B parameters with 4-bit quantisation, the model has a limited context window. Long prompt histories may cause truncation. The harness mitigates this by trimming strategy source and failure context.
- **Not investment advice.** This is a research project. Do not use it to make real financial decisions.

---

## Contributing

Pull requests welcome. The most useful contributions are:

- Additional indicator functions (with frozen signatures matching the existing pattern)
- Alternative scoring metrics as optional harness flags
- Colab/Kaggle notebook wrappers
- Integration tests for the validator pipeline

Please do not modify `backtest_harness.py` in ways that break the three-file contract — the agent relies on the harness being a stable environment.

---

## Acknowledgements

- [AI Scientist](https://github.com/SakanaAI/AI-Scientist) by Sakana AI — the self-improving research loop this project is modelled on
- [Qwen](https://huggingface.co/Qwen) by Alibaba — the open-source models powering the agent
- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for 4-bit quantisation

---

## License

MIT
