# AutoFin — Autonomous Financial Strategy Agent

This is an autonomous financial strategy research system. The LLM reads this file
as its live instruction manual at the start of every iteration.

---

## Setup

To set up a new AutoFin run:

1. Agree on a run tag based on today's date (e.g. `apr12`).
2. Check that `strategy.py` exists and is syntactically valid:
   `python strategy.py`
3. Initialise `results.tsv` with just the header row — the baseline will be
   recorded after the very first run.
4. Start the harness. Recommended Colab command:
   ```
   python backtest_harness.py \
     --tickers "NVDA,^GSPC,MSFT,GLD" \
     --model "Qwen/Qwen3.5-9B" \
     --device cuda \
     --iterations 30
   ```
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
- Add intermediate derived signals (using only numpy/pandas/math — no new imports).

## What You CANNOT Do

- Modify `program.md` or `backtest_harness.py`. They are read-only.
- Add new import statements — only `numpy`, `pandas`, and `math` are available.
- Change any function **name** or **signature** in `strategy.py`.
- Access external data, APIs, or the filesystem.

---

## CRITICAL CODE RULES (read before writing any strategy.py)

### Rule 1 — Never join indicator DataFrames

Call each indicator separately and access its columns directly. **Never** use
`.join()`, `rsuffix=`, or `lsuffix=` inside `generate_signals()`.

**CORRECT:**
```python
ichi  = ichimoku_cloud(df, **ICHIMOKU_PARAMS)
rsi_  = rsi(df,           **RSI_PARAMS)
atr_  = atr(df,           **ATR_PARAMS)
vwap_ = vwap(df,          **VWAP_PARAMS)

# Access columns directly:
bool(ichi["bullish"].iloc[i])
rsi_["rsi_value"].iloc[i]
atr_["atr_value"].iloc[i]
atr_["long_stop"].iloc[i]
vwap_["above_vwap"].iloc[i]
```

**WRONG (causes 'atr_value_atr' KeyError crashes):**
```python
ichi = ichi.join(rsi_, rsuffix='_rsi')   # DO NOT DO THIS
ichi = ichi.join(atr_, rsuffix='_atr')   # DO NOT DO THIS
```

### Rule 2 — generate_signals must return signal + stop columns

```python
return pd.DataFrame({"signal": signal, "stop": stop}, index=df.index)
```
Do not return columns named `entry`, `exit`, `long_entry`, etc. unless you
also include a `signal` column.

### Rule 3 — Use zone RSI checks, not strict crossovers

`rsi_value < oversold + 15` (i.e. RSI < 45) generates entries in trending markets.
`rsi_value < oversold` (RSI < 30) is too strict — you will get zero trades.

### Rule 4 — ATR formula must use previous close

```python
prev_close = df["Close"].shift(1)
(df["High"] - prev_close).abs()   # CORRECT
(df["High"] - df["Close"]).abs()  # WRONG — harness will reject this
```

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
- You can also use `tenkan_sen > kijun_sen` as a secondary momentum signal.

> **Warm-up note**: the full Ichimoku cloud requires `lagging_span2_period + displacement`
> bars to initialise — 78 bars at defaults. The harness prepends a 100-day warm-up
> buffer so your indicators are fully ready from the first eval bar.

### 2. `rsi` — Momentum Filter

```python
rsi(df, period, overbought, oversold)
```

Returns columns: `rsi_value`, `buy_signal` (RSI crossed UP through `oversold`),
`sell_signal` (RSI crossed DOWN through `overbought`).

- Defaults: `period=14, overbought=70, oversold=30`
- Zone checks (e.g. `rsi_value < oversold + 15`) generate more signals than
  strict crossovers on trending stocks.

### 3. `atr` — Volatility / Stop-Loss

```python
atr(df, period, multiplier)
```

Returns columns: `atr_value`, `long_stop` (close − multiplier × ATR),
`short_stop` (close + multiplier × ATR).

- Defaults: `period=14, multiplier=2.5`
- **CRITICAL**: The ATR true range MUST use `close.shift(1)` for previous close.
  Never use `(high - close)` or `(low - close)` directly. The harness validates this.
- You can build **adaptive stops** using the ATR value and a rolling mean:
  ```python
  atr_ma = atr_["atr_value"].rolling(20).mean()
  high_vol = atr_["atr_value"] > atr_ma * 1.5
  adaptive_mult = np.where(high_vol, ATR_PARAMS["multiplier"] * 0.7, ATR_PARAMS["multiplier"])
  ```

### 4. `vwap` — Volume Filter

```python
vwap(df, period)
```

Rolling VWAP over `period` trading days.
Returns columns: `vwap_value`, `above_vwap` (bool), `below_vwap` (bool).

- Default: `period=14`

---

## Default Entry / Exit Rules (v2 baseline)

These rules live inside `generate_signals()` and **you may change them**:

```
LONG  entry : ichimoku bullish
              AND tenkan_sen > kijun_sen   (momentum confirmation)
              AND rsi_value < oversold+15  (mild pullback in uptrend)
LONG  exit  : rsi sell_signal   OR  low < adaptive_stop

SHORT entry : ichimoku bearish
              AND tenkan_sen < kijun_sen   (momentum confirmation)
              AND rsi_value > overbought-15
              AND below_vwap == True       (VWAP confirms real downtrend, not a bounce)
SHORT exit  : rsi buy_signal    OR  high > adaptive_stop
```

The Tenkan/Kijun filter prevents entering during dead-cat bounces in sustained
downtrends (this was the main cause of the 2022 losses in v1). If you are still
getting zero trades, remove ONE condition at a time and test.

The VWAP filter on short entries is critical for avoiding false shorts during
early-recovery rallies (e.g. the 2019 ^GSPC recovery from Q4 2018 crash, which
caused a −0.758 score when the strategy shorted the early recovery because
Ichimoku was still bearish but price/VWAP had already turned).

---

## Evaluation Metric (fixed — do not try to change it)

```
score = Sharpe_Ratio × 0.6  +  Profit_Factor × 0.2  −  Max_Drawdown × 0.2
```

The **aggregate score** is the **mean across ALL tickers AND all yearly shards**.
This prevents overfitting to a single momentum stock (e.g. NVDA in a bull year).
Do NOT sacrifice out-of-sample performance (2023, 2024) or cross-asset
generalisation for in-sample NVDA gains.

### Ensemble Gate

Before any new best is accepted, the candidate is tested on out-of-sample shards
for non-primary tickers (e.g. ^GSPC, MSFT, GLD). If the mean OOS score across
those tickers is below -0.30, the strategy is **rejected** even if the primary
ticker aggregate improved. This catches NVDA-overfit strategies.

---

## Data Shards

Shards are **yearly** windows (keyed `"YYYY"`) from `start_year` to end of data.
Each shard has a 100-day warm-up buffer prepended.

| Period | Character |
|---|---|
| 2018 | In-sample — late bull / Q4 crash |
| 2019 | In-sample — recovery bull run |
| 2020 | In-sample — COVID crash + V-recovery |
| 2021 | In-sample — strong bull market |
| 2022 | In-sample — inflation bear market |
| 2023 | **Out-of-sample** |
| 2024 | **Out-of-sample (most recent)** |

---

## Known Weak Shards — Address These First

Based on observed results, the following shards are consistently negative and
should be the primary focus of improvement attempts:

| Shard | Score | Root Cause | Hypothesis |
|---|---|---|---|
| `^GSPC 2019` | −0.758 | Strategy shorts Q4 2018 recovery | Ichimoku still bearish while price already recovered above VWAP. Add `below_vwap` to short entry. |
| `^GSPC 2023` | −0.817 | False signals in choppy sideways market | ATR multiplier too tight; stops getting hit. Widen multiplier or add volatility gate. |
| `^GSPC 2020` | −0.360 | COVID crash V-recovery whipsaws | Short entries during recovery leg. VWAP filter should help. |
| `GLD 2020` | −0.780 | Only 2 trades, both losers | Signals too rare for GLD's volatility profile. Loosen RSI threshold. |
| `GLD 2021` | −0.853 | 4 trades, Sharpe −1.43 | GLD sideways chop with trend-following strategy. Volatility gate may skip this. |
| `GLD 2024` | −0.559 | Recent poor fit | GLD had a strong bull run in 2024; check if strategy is going short into it. |

---

## Diversity System

If `no_improve >= 2`, the harness injects a **DIVERSITY REQUIREMENT** block into
your prompt with two concrete structural ideas. You MUST try something
structurally different — do not just re-tune a threshold value.

Structural changes that have NOT been tried yet (attempt these before parameter tuning):
- Use Tenkan/Kijun cross as the **primary** entry trigger (not cloud position)
- Add a trailing stop that tightens after a trade has been open > 15 bars
- Use RSI divergence: enter long when price lower-low but RSI higher-low
- Volatility gate: skip entries entirely when ATR > 2x its 20-day average
- Asymmetric RSI: long entry RSI < 35, short entry RSI > 68
- Exit via cloud re-entry (price moves back inside cloud = trend weakening)

---

## Simplicity Criterion

All else being equal, **simpler is better**.

- A ~0 improvement that simplifies code → **keep it**.
- A large improvement with clean logic → **keep it**.
- A +0.001 improvement with 30 lines of hacky code → probably not worth it.

---

## Rollback and Exploration Rules

| Outcome | Condition | Action |
|---|---|---|
| **keep** | `aggregate > best_score` AND ensemble gate passed | Save as new best |
| **explore** | `aggregate == best_score` | Keep code live for further search |
| **discard** | `aggregate < best_score` OR ensemble gate failed | Roll back |

When discarded, the harness sends your failed code in the next prompt under
`YOUR LAST ATTEMPT`. Study it — do NOT repeat the same change.

---

## The Experiment Loop

**LOOP FOREVER:**

1. Read `program.md` (this file) and `strategy.py`.
2. Examine the per-ticker per-shard heatmap and aggregate score.
3. Identify the weakest tickers/shards. Think about why:
   - Is the strategy long-only in bear markets (2022)?
   - Are stops getting hit too fast in high-volatility periods (2020)?
   - Does it work on NVDA but fail on ^GSPC (overfitting)?
4. If you see a `YOUR LAST ATTEMPT` block or a DIVERSITY REQUIREMENT, respond
   to those constraints first before proposing.
5. Propose a new `strategy.py` with a hypothesis-driven change.
6. Respond in the required format.

**NEVER STOP**: Do not ask whether to continue. Do not pause for confirmation.

---

## Crash / Failure Handling

- **Syntax error**: Fix in next proposal.
- **ATR formula error**: True range MUST use `close.shift(1)`. Fix immediately.
- **structural_error (DF join)**: Do NOT use `.join()` on indicator DataFrames.
  Access each indicator's columns directly. See CRITICAL CODE RULES above.
- **Zero trades**: Loosen at least one entry condition.
- **Profit_Factor = 10.0** (capped): ATR stop may be unrealistically wide.
- **Ensemble gate rejected**: Strategy fits primary ticker but not other assets. Add regime
  filtering or loosen conditions so it generalises.
- **Score = -99.0**: Runtime exception. Fix the logic.

---

## Response Format

Respond with **exactly two XML sections** and nothing else:

```
<reasoning>
Concise explanation (150 words max) referencing:
  - The previous aggregate score (across all tickers)
  - Which tickers/shards were weakest and why
  - What specifically you changed and the hypothesis
  - (If applicable) Why your last attempt failed and what you're doing differently
  - (If DIVERSITY REQUIREMENT shown) Which structural idea you're implementing
</reasoning>

<strategy>
[complete, valid Python — the FULL new contents of strategy.py]
</strategy>
```

**Hard rules for the `<strategy>` block:**
- No markdown fences (no triple backticks).
- Must be the **complete file**, not a diff or partial snippet.
- All four indicator functions must be present with their ORIGINAL implementations.
- `generate_signals(df: pd.DataFrame) -> pd.DataFrame` signature must not change.
- `get_params() -> dict` must return a flat dict of ALL current parameter values.
- Do not add new imports.
- **CRITICAL — no name-shadowing**: Use `ichi = ichimoku_cloud(...)`,
  `rsi_ = rsi(...)`, `atr_ = atr(...)`, `vwap_ = vwap(...)`.
- **CRITICAL — no DataFrame joins**: Never use `.join()`, `rsuffix=`, or `lsuffix=`
  inside `generate_signals()`. Access each indicator's columns directly.
- **CRITICAL — ATR formula**: Use `prev_close = df["Close"].shift(1)` and
  `(df["High"] - prev_close).abs()`. NEVER `(high - close)` or `(low - close)`.
- **CRITICAL — signals must fire**: Use `rsi_value < oversold + 15` (RSI < 45)
  for long entry, not strict `rsi_value < oversold` (RSI < 30).
- **CRITICAL — return format**: Return `pd.DataFrame({"signal": signal, "stop": stop})`.
  Do not rename these columns.

---

## Parameter Search Space

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
An external MCP server watches this file and handles the commit and push.