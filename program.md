# AutoFin — Autonomous Financial Strategy Agent

This is an autonomous financial strategy research system. The LLM reads this file
as its live instruction manual at the start of every iteration.

---

## Setup

To set up a new AutoFin run:

1. Agree on a run tag based on today's date (e.g. `apr8`).
2. Check that `strategy.py` exists and is syntactically valid — run it directly
   to confirm it imports cleanly: `python strategy.py`
3. Verify market data can be fetched (the harness handles this automatically via
   `yfinance`).
4. Initialise `results.tsv` with just the header row — the baseline will be
   recorded after the very first run.
5. Confirm setup and go.

---

## Three-File Architecture

```
autofin/
├── program.md          ← THIS FILE — read by the LLM at every iteration (read-only)
├── strategy.py         ← LLM's ONLY writable surface
└── backtest_harness.py ← fixed evaluation engine (read-only to the LLM)
```

| File | Owner | Purpose |
|---|---|---|
| `program.md` | Human / read-only | Live spec the LLM reads at runtime |
| `strategy.py` | **LLM** | Indicator parameters + signal logic |
| `backtest_harness.py` | Human / read-only | Data, sharding, metrics, loop orchestration |

The LLM receives the **contents of both `program.md` and `strategy.py`** in its
context every iteration. It proposes a new `strategy.py`. The harness executes it,
scores it, and feeds results back. The LLM **never sees `backtest_harness.py`**.

---

## What You CAN Do

- Modify `strategy.py` — this is the **only** file you edit.
- Change any parameter values inside the `ICHIMOKU_PARAMS`, `RSI_PARAMS`,
  `ATR_PARAMS`, and `VWAP_PARAMS` dicts.
- Change the entry/exit logic inside `generate_signals()`.
- Adjust thresholds, add derived intermediate signals, tune position sizing.

## What You CANNOT Do

- Modify `program.md` or `backtest_harness.py`. They are read-only.
- Add new import statements — only `numpy`, `pandas`, and `math` are available.
- Change any function **name** or **signature** in `strategy.py`.
- Access external data, APIs, or the filesystem.

---

## Indicators

Your strategy has exactly four indicator functions. Their **signatures are
frozen** — you control only the parameter values in the `*_PARAMS` dicts and the
logic inside `generate_signals()`.

### 1. `ichimoku_cloud` — Trend Filter

```python
ichimoku_cloud(df, conversion_period, base_period, lagging_span2_period, displacement)
```

Returns a DataFrame with columns:
`tenkan_sen`, `kijun_sen`, `senkou_span_a`, `senkou_span_b`,
`cloud_top`, `cloud_bottom`, `bullish` (bool), `bearish` (bool).

- **Long bias**: `bullish == True` (price above cloud)
- **Short bias**: `bearish == True` (price below cloud)
- Defaults: `conversion_period=9, base_period=26, lagging_span2_period=52, displacement=26`

> **Warm-up note**: the full Ichimoku cloud (senkou_span_b) requires
> `lagging_span2_period + displacement` bars to initialise — 78 bars at defaults.
> The harness prepends a 100-day warm-up buffer to every monthly shard so your
> indicators are fully ready from the first bar of the evaluation window.
> You do not need to handle this yourself.

### 2. `rsi` — Momentum Filter

```python
rsi(df, period, overbought, oversold)
```

Returns columns: `rsi_value`, `buy_signal` (RSI crossed up through `oversold`),
`sell_signal` (RSI crossed down through `overbought`).

- Defaults: `period=14, overbought=70, oversold=30`

### 3. `atr` — Volatility / Stop-Loss

```python
atr(df, period, multiplier)
```

Returns columns: `atr_value`, `long_stop` (close − multiplier × ATR),
`short_stop` (close + multiplier × ATR).

- Defaults: `period=14, multiplier=2.0`

### 4. `vwap` — Volume Filter

```python
vwap(df, period)
```

Rolling VWAP over `period` trading days.
Returns columns: `vwap_value`, `above_vwap` (bool), `below_vwap` (bool).

- Default: `period=14`

---

## Default Entry / Exit Rules

These rules live inside `generate_signals()` and **you may change them**:

```
LONG  entry : ichimoku bullish  AND  rsi buy_signal   AND  price > vwap
LONG  exit  : rsi sell_signal   OR   price drops below atr long_stop
SHORT entry : ichimoku bearish  AND  rsi sell_signal  AND  price < vwap
SHORT exit  : rsi buy_signal    OR   price rises above atr short_stop
```

The default entry condition is intentionally strict (a "perfect storm" of three
simultaneous signals). If you are seeing zero trades on most shards, **loosen at
least one condition** — for example, drop the VWAP filter, or replace the RSI
crossover signal with a zone check (`rsi_value < oversold` rather than a single
crossover bar).

---

## Evaluation Metric (fixed — lives in backtest_harness.py, do not try to change it)

```
score = Sharpe_Ratio × 0.6  +  Profit_Factor × 0.2  −  Max_Drawdown × 0.2
```

- **Higher is better.**
- `Max_Drawdown` is a positive decimal (0.25 = 25% drawdown).
- `Profit_Factor` is clipped to `[0, 10]` to avoid infinite values.
- `Sharpe_Ratio` is annualised (252 trading days, risk-free rate = 0).

The **aggregate score** is the mean across all valid shards. Do NOT sacrifice
out-of-sample performance (2023, 2024 shards) for in-sample gains — watch both.

---

## Data Shards

Shards are **monthly** windows (keyed `"YYYY-MM"`), covering every calendar month
from `start_year` through the end of the data. Each shard has a 100-day warm-up
buffer prepended by the harness — you never see cold indicators.

Example shard keys: `2018-01`, `2018-02`, ..., `2024-12`

| Period | Character |
|---|---|
| 2018 | In-sample — late bull / Q4 crash |
| 2019 | In-sample — recovery bull run |
| 2020 | In-sample — COVID crash + V-recovery |
| 2021 | In-sample — strong bull market |
| 2022 | In-sample — inflation bear market |
| 2023 | **Out-of-sample** |
| 2024 | **Out-of-sample (most recent)** |

Because shards are monthly (~21 trading bars each), a strategy that generates
zero trades in a month scores 0.0 for that shard. Watch the `Trades` column in
the results table — if most shards show 0, your entry conditions are too strict.

---

## Simplicity Criterion

All else being equal, **simpler is better**.

- A ~0 improvement that deletes or simplifies code → **keep it** (simplification win).
- A large improvement with clean logic → **keep it**.
- A small improvement (+0.001 score) that adds 20 lines of hacky code → probably
  not worth it. Weigh the complexity cost honestly.

---

## Rollback and Exploration Rules

The harness uses a three-way outcome system each iteration:

| Outcome | Condition | Action |
|---|---|---|
| **keep** | `aggregate > best_score` | Save as new best, update trigger |
| **explore** | `aggregate == best_score` | Keep the mutated code live — let the LLM keep searching |
| **discard** | `aggregate < best_score` | Roll back to `best_strategy.py`; pass the failed code to the LLM |

When a strategy is **discarded**, the harness sends you your failed code in the
next prompt under the heading `YOUR LAST ATTEMPT`. Study it carefully — do NOT
repeat the same change. Propose something meaningfully different.

When in **explore** mode (flat score), your code is kept live. This lets you
accumulate incremental improvements even when starting from 0.0.

---

## The Experiment Loop

**LOOP FOREVER:**

1. Read `program.md` (this file) and `strategy.py`.
2. Examine the per-shard results table and aggregate score from the harness.
3. Identify the weakest shards. Think about why: is the strategy over-trading in
   volatile regimes? Are stops too tight in the COVID crash months? Are signals
   too rare in 2022?
4. If you see a `YOUR LAST ATTEMPT` block, understand why it failed before
   proposing a new approach.
5. Propose a new `strategy.py` with a hypothesis-driven change.
6. Respond in the required format (below).
7. The harness will:
   - Execute the new `strategy.py`
   - Score it on all monthly shards
   - **Keep** if aggregate improved, **Explore** if flat, **Discard** if worse
   - Log to `results.tsv` with status `keep`, `explore`, `discard`, or `crash`
   - Feed you the results for the next iteration

**The first iteration** always runs the **unmodified baseline** to establish a
reference. Do not change anything on iteration 1 — just output the current
`strategy.py` unchanged.

**NEVER STOP**: Do not ask whether to continue. Do not pause for confirmation.
Run until manually interrupted. If you exhaust obvious ideas, try: relaxing the
entry conditions one at a time, replacing RSI crossover with RSI zone checks,
using the Ichimoku kijun/tenkan cross as a secondary signal, adjusting the ATR
multiplier for different market regimes, or asymmetric long/short thresholds.

---

## Crash / Failure Handling

- **Syntax error**: The harness rejects your code and re-runs the previous best.
  Fix the syntax in your next proposal.
- **Zero trades on most shards**: Your entry condition is too strict for a
  monthly window (~21 bars). Drop or relax at least one filter.
- **Profit_Factor = 10.0** (capped): The strategy may have no losing trades in
  that shard — check if ATR stop is unrealistically wide.
- **Score = -99.0**: A runtime exception occurred. The harness logs the error; fix
  the logic in your next proposal.

---

## Response Format

Respond with **exactly two XML sections** and nothing else:

```
<reasoning>
Concise explanation (≤ 150 words) of what you changed and why, referencing:
  - The previous aggregate score
  - Which monthly shards were weakest and why you think so
  - What specifically you changed and the hypothesis
  - (If applicable) Why your last attempt failed and what you are doing differently
</reasoning>

<strategy>
[complete, valid Python — the FULL new contents of strategy.py]
</strategy>
```

**Hard rules for the `<strategy>` block:**
- No markdown fences (no triple backticks anywhere inside the block).
- Must be the **complete file**, not a diff or partial snippet.
- All four indicator functions must be present with their original implementations.
- `generate_signals(df: pd.DataFrame) -> pd.DataFrame` signature must not change.
- `get_params() -> dict` must return a flat dict of all current parameter values.
- Do not add new imports.

---

## Parameter Search Space (reference bounds)

| Parameter | Sensible Min | Sensible Max | Note |
|---|---|---|---|
| `ichimoku conversion_period` | 5 | 15 | Fast line |
| `ichimoku base_period` | 20 | 40 | Must be > conversion_period |
| `ichimoku lagging_span2_period` | 40 | 80 | Must be > base_period |
| `ichimoku displacement` | 20 | 35 | Cloud shift |
| `rsi period` | 7 | 21 | |
| `rsi overbought` | 60 | 80 | Must be > oversold |
| `rsi oversold` | 20 | 40 | Must be < overbought |
| `atr period` | 7 | 21 | |
| `atr multiplier` | 1.0 | 4.0 | Wider = fewer stops hit |
| `vwap period` | 5 | 30 | Shorter = more reactive |

---

## GitHub Integration (external MCP — no logic here)

When a new best score is found, the harness writes `push_trigger.json`:
```json
{ "score": 1.2345, "iteration": 7, "params": { ... }, "timestamp": "..." }
```
An external MCP server (with a fine-grained GitHub token) watches this file and
handles the commit and push. Neither `strategy.py` nor `backtest_harness.py`
contain any git logic.