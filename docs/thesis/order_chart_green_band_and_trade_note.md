# Plan — ema_21 green band + below-plot trade note (with all SL/TP options)

- Repo: `/scratch/work/nguyenl37/Aalto_MS_Thesis`
- Date: 2026-10-03
- Status: **PROPOSED** — no code changed yet
- Scope: **order-chart rendering only**. No reward, no prior, no env training
  behavior, no launch-script change.
- Python: `.venv-x86/bin/python`
- Does not disturb the running Oct 3 training pair (jobs `20712216` gpuarm1 /
  `20712217` gpuarm2). This is offline post-processing code.

---

## 1. Why this plan exists

### 1.1 The missing green TP band

Observed: per-trade order plots no longer show the green TP background band;
only the red SL band is visible.

Root cause (verified, **not** a plotting regression):

- `order_levels()` in `quant_rl/eval/order_chart.py` builds the bands at lines
  246–249:

  ```python
  red = _clip_band(metrics.entry_price, metrics.sl_price, y0, y1) if show_sl_tp else None
  green = _clip_band(metrics.entry_price, metrics.tp_price, y0, y1) if show_sl_tp else None
  ```

- `_clip_band` (lines 297–307) returns `None` immediately when `other is None`.
- For an **ema_21 exit** there is no fixed take-profit, so `tp_price` is `NaN`
  (`quant_rl/eval/trade_metrics.py:160-170` only sets it when
  `open_row["tp_price"]` is non-null or a `take_profit_per_trade_usd` fallback is
  configured — neither applies here). Therefore `green is None` and **no green
  band is drawn**.

Empirical confirmation on the Sep 30 idea 1 run
(`outputs/20260930_153627_935800_rl_train_seed50_tcn/training/trades.csv`):

| metric | value |
|---|---|
| opens with `exit_mode=ema_21` and `tp_ref=ema_21` | **948 / 1076 (88%)** |
| opens with `exit_mode=structural` and a real TP | 128 |
| opens with empty `tp_price` | 948 |
| opens with a `tp_price` | 128 |
| `exit_u >= 0.5` (the ema_21 branch) | 948 |
| mean `exit_u` | **0.745** |

So the policy overwhelmingly picks the EMA-21 exit, those trades have no target
price, and the green band legitimately disappears. The red SL band always
renders because `sl_price` is always populated.

**Confounder to document:** the green/red rectangles visible on these charts are
mostly **FVG/IFVG zones**, not the SL/TP bands. They are drawn by
`_draw_zones()` (`quant_rl/eval/po3_plots.py:125`), called from
`quant_rl/eval/plots.py:1105`, with colors from `_zone_color()` — bullish FVG
green, bearish red, at alpha 0.28–0.40. Anyone reviewing these plots must not
confuse a zone with the TP band.

### 1.2 The legend / info box

The current trade info lives in a **corner box on the price panel**
(`plots.py:1157-1194`) and, in HTML, inside the plot title `<sub>` block
(`plots_interactive.py:869-881`). At the bottom-right of a dense chart it covers
candles and is easy to miss, and it does not explain *why* a stop or target was
chosen. Raw refs like `M5_last_swing_low` or `ema_21` also require knowing the
strategy's internals.

---

## 2. Goals

1. **Green block for ema_21 exits.** When a trade exits on the EMA-21 rule (no
   fixed TP), still draw a green band so the chart always shows a reward /
   realized zone.
2. **Below-plot note** spanning the figure width, below both subplots, with:
   - `Direction · Open → Close · Duration · PnL · Volume`
   - **Entry reason** — why the trade was taken at all (see 3.4)
   - **SL reason** and **TP reason** in plain language
   - **all SL and TP options** (the ladder the agent chose from), with the
     chosen one marked `★`
   - the existing PnL logged-vs-calc reconciliation, preserved
3. A reader who knows nothing about the RL agent can still understand each
   chart.
---

## 3. Note design

### 3.1 Layout — PNG

The corner box is removed. The note is drawn under the MACD/RSI subplot:

```
+------------------------------------------------------------------------------+
|  [ price candles + EMA50 + VWAP + swing levels + manipulation/FVG zones ]      |
|  Entry ^   #### SL band (red)   @@@@ TP or ema-exit band (green)   Exit v     |
+------------------------------------------------------------------------------+
|  [ MACD / Signal ]                                                            |
+------------------------------------------------------------------------------+
|  [ Histogram + RSI ]                                                         |
+------------------------------------------------------------------------------+
|  Long - Open 2025-10-27 17:44 -> Close 17:48 (4m00s) - PnL -21.53 - Vol 1.49  |  <- new
|  Entry: swept Asia session low, manipulation done - manipulation "done"       |
|  SL 25683.39 swing low (M5)  |  Exit 25729.85 EMA-21 exit (no fixed target)     |
|  PnL logged -21.53 / calc -21.53 / closed: normal                             |
|  SL options (6) * swing low (M5) 25683.39 . swing low (H1) 25718.9 . ...      |
|  TP options (0)  none - exit on EMA-21                                        |
+------------------------------------------------------------------------------+
```

Structural-TP variant:

```
|  Short - Open 2025-10-28 19:02 -> Close 19:27 (25m01s) - PnL -230.08 - Vol 11.98
|  Entry: swept London session low, distribution phase 1 - manipulation "done"
|  SL 25951.66 swing high (M15)  |  TP 25876.83 swing low (H1)  (1.14R)
|  Stop = swing high on M15. Target = next H1 swing low at 2.66 planned RR.
|  SL options (1) * swing high (M15) 25951.66
|  TP options (9) * swing low (H1) 25876.83 . Asia session low 25840.1 . ...
```

### 3.2 Layout — HTML

Identical text, rendered as a Plotly annotation **below** the two-row grid
(instead of inside the title `<sub>`), so PNG and HTML agree.

### 3.3 Plain-language ref mapping

New helper `pretty_ref(name)` in a shared module. Raw ref -> readable text:

| raw ref | plain text |
|---|---|
| `M5_last_swing_low` / `M15_...` / `H1_...` | `swing low (M5)` / `swing high (H1)` |
| `last_swing_low` / `last_swing_high` | `swing low (M1)` / `swing high (M1)` |
| `asian_low` / `asian_high` | `Asia session low` / `Asia session high` |
| `london_low` / `london_high` | `London session low` / `London session high` |
| `prev_day_high` / `prev_day_low` | `previous day high` / `previous day low` |
| `ctx_prev_week_high` / `_low` | `previous week high` / `previous week low` |
| `sweep_high_level` / `sweep_low_level` | `swept liquidity high` / `swept liquidity low` |
| `H1_fvg_bull_high` | `1H fair-value gap (bullish)` |
| `M15_ifvg_bear_low` | `15m inverse gap (bearish)` |
| `smt_swing` | `SMT swing` |
| `ema_21` | `EMA-21 exit` |
| joined names, e.g. `asian_low\|H1_last_swing_low` | join the parts with `+` |

Reason sentences:

- ema exit: `Stop = <sl reason>. Exit = closed on the EMA-21 cross (no fixed target).`
- structural: `Stop = <sl reason>. Target = <tp reason>` (+ planned RR when present).

### 3.4 Entry reason — why the trade was taken

This is the missing third piece. The SL/TP reasons explain *where the levels came
from*; the entry reason explains *why a trade existed at all*, and which side was
picked. It is assembled from columns already present on the `open` row — no new
logging required for the first version.

Source columns (`trades.csv`, `type == "open"`):

| column | meaning | example values (Sep 30 idea1) |
|---|---|---|
| `manipulation` | PO3 manipulation state at entry | `done`, `none`, `against` |
| `level_type` | liquidity level the entry matched | `london_low`, `asian_low`, `london_high` |
| `distribution_phase` | PO3 distribution leg active | `1.0` |
| `sweep_delay_s` | seconds since the liquidity sweep | `2340.0`, empty |
| `manipulation_end` | manipulation-end bar flag | `1.0`, `0.0` |
| `strategy` | strategy name | `po3_ifvg`, `distribution` |
| `ifvg_zone_low` / `ifvg_zone_high` | IFVG zone the retest happened in | `21196.85` / `21198.05` |

Proposed composition (first line of the reason, before SL/TP):

```
Entry: <level pretty> swept, manipulation <state>            (idea1, po3_ifvg)
Entry: <level pretty> swept, distribution phase 1            (idea2, distribution)
Entry: <no level> - strategy offered this side              (fallback)
```

with the manipulation/distribution clause swapped on `strategy`, and the
`distribution_phase` clause shown only when `distribution_phase == 1`.

Mapping for `manipulation`:

| value | plain text |
|---|---|
| `done` | `manipulation "done" (phase completed)` |
| `none` | `no manipulation yet` |
| `against` | `manipulation "against" (entry not confirmed by the side)` |

Mapping for `level_type` reuses `pretty_ref()`:

| value | plain text |
|---|---|
| `london_low` | `swept London session low` |
| `asian_low` | `swept Asia session low` |
| `london_high` | `swept London session high` |
| empty | `strategy offered this side (no swept level)` |

Notes:

- **Gate is diagnostic only.** Both `config/2_po3_ifvg.yaml:42` and
  `config/3_distribution.yaml:35` set `strategy.entry.enforce_gate: false`,
  and both strategies' `validate_entry` return `True` immediately when the gate
  is off (`po3_ifvg.py:79-81`, `distribution.py:98-100`). So the entry reason
  describes the **setup the strategy offered**, not a hard filter that blocked
  other trades. The note must not claim "the gate required X".
- **The side is the agent's, not the strategy's.** With
  `agent_direction_control=true`, `_apply_direction_control` lets the agent's
  signed conviction override the strategy's `ctx` whenever
  `|conviction| > direction_override_threshold`. So the note should say the side
  was "agent-selected", and the SL/TP menus then show what the *strategy* would
  have offered for that side. (On the new Oct 3 run `agent_direction_control` is
  `false`, so the side is `ctx` — the note reads "strategy side". Either way the
  text stays accurate because it is derived, not hard-coded.)
- If the entry column set is ever found insufficient, the same Option-B pattern
  (log a small `entry_reason` dict on the open row) can be added later without
  touching the note format.

---

## 4. The ema_21 green band

In `order_levels()`, when `ema_exit` is true and there is no `tp_price`, build the
green band from **entry -> exit_price** instead of leaving `green = None`:

```python
red = _clip_band(metrics.entry_price, metrics.sl_price, y0, y1) if show_sl_tp else None
green = _clip_band(metrics.entry_price, metrics.tp_price, y0, y1) if show_sl_tp else None
if show_sl_tp and green is None and ema_exit and metrics.exit_price is not None:
    green = _clip_band(metrics.entry_price, metrics.exit_price, y0, y1)
```

- `exit_price` is always present on `TradeChartMetrics` (`trade_metrics.py:44`),
  so this always yields a band for an ema exit.
- Clipped by the same `_clip_band`, so it stays inside `ylim`.
- On a **losing** ema exit the band sits on the loss side of entry. Per the
  explicit request it stays **green**; the note states the PnL so there is no
  ambiguity. (Easy one-line switch to red if that is ever preferred.)
- The existing `exit ema_21` note is kept, so the reason stays visible.
- Structural-TP plots are unchanged: `green` still comes from `tp_price`.

---

## 5. Getting "all options" — Option B (recommended)

The trade row carries the **chosen** option and the **menu size**
(`sl_ref`, `tp_index`, `n_sl`, `n_tp`, `tp_index`, `exit_mode`), but **not** the
individual ladder prices — confirmed: `trades.csv` has no ladder/candidate columns.

Two ways to surface the full menu:

- **Option A — regenerate in the chart.** Rebuild the ladder at `t_open` via
  `_stop_menu()` (`quant_rl/backtest/entry_levels.py:350`) and `_target_menu()`
  (line 391), mirroring `place_strategy_order()` (line 140). Faithful, but it
  pushes the strategy object and feature row through
  `order_window.prepare_order_chart`, `plots.py`, and `plots_interactive.py`.
- **Option B — log the ladder at open (recommended).** Record `sl_menu` and
  `tp_menu` (list of `{name, price}`) on the open row when the order is placed,
  next to the existing decision payload in
  `trading_env._decision_fields()` (~lines 1500–1544) / the open `record`
  dict (~line 1402). The chart then just prints what the agent actually saw —
  deterministic, honest even if the strategy changes later, and it keeps
  strategy plumbing out of the chart path.

Chosen: **Option B**. Trade-off: it widens the trade-record schema by two list
columns; `trades.csv` grows modestly.

---

## 6. Files to change

| file | change |
|---|---|
| `quant_rl/envs/trading_env.py` | (Option B) write `sl_menu` / `tp_menu` on the open record |
| `quant_rl/eval/order_chart.py` | ema_21 green band (entry->exit); accept the menus from the row |
| `quant_rl/eval/trade_note.py` *(new)* | `pretty_ref()`, `entry_reason()`, `build_trade_note()` shared by both renderers |
| `quant_rl/eval/plots.py` | drop corner box, add `fig.text` note below subplots; widen `tight_layout(rect=...)` |
| `quant_rl/eval/plots_interactive.py` | drop title `<sub>`, add bottom annotation with the same text |

No change is needed to the entry columns: `manipulation`, `level_type`,
`distribution_phase`, `sweep_delay_s`, `strategy`, and the IFVG zone bounds are
already written on every `open` row.

---

## 7. Tests to add

`tests/test_eval/`:

- `test_order_chart.py` — ema_21 exit with `tp_price=None` =>
  `levels.green is not None`, equals the entry->exit band, and is inside `ylim`.
- `test_trade_note.py` *(new)*:
  - `pretty_ref("M5_last_swing_low") == "swing low (M5)"`,
    `pretty_ref("ema_21") == "EMA-21 exit"`.
  - note contains direction / open / close / duration / PnL / volume.
  - note contains an **entry reason** line.
  - `entry_reason()` with `level_type="london_low"`, `manipulation="done"` on
    `po3_ifvg` mentions the London low and the manipulation state; with
    `distribution_phase=1` on `distribution` mentions the distribution phase.
  - `entry_reason()` with all entry columns empty degrades to
    "strategy offered this side", never raises and never returns empty.
  - ema note contains "no fixed target"; structural note contains the RR.
  - with `sl_menu` / `tp_menu` present, the note lists them and marks the
    chosen one with `*`.
  - menu capping: chosen + nearest 4, never exceeding the documented cap.

Regression safety: existing tests only exercise the `tp_price` path, so the ema
band is additive. The corner-box removal must be checked against any assertions
on that box — none found in `tests/test_eval/test_order_chart.py`, but this is
re-verified before the change lands.

---

## 8. Guardrails

- No change to reward, prior (`mode_prior_coef`), `ent_coef`, `log_std_min`,
  environment training behavior, or the launch script.
- Keep the PnL logged-vs-calc reconciliation visible.
- PNG and HTML must render the **same** note text (one shared helper).
- Green band appears only when there is no fixed TP.
- No commit, no push.
- Do not disturb jobs `20712216` / `20712217`.

---

## 9. Open questions and proposed defaults

| # | Question | Proposed default |
|---|---|---|
| 1 | All options via regenerate-in-chart (A) or log-at-open (B)? | **B** |
| 2 | Losing ema_21 exit — green or red block? | **Stay green**, PnL shown in note |
| 3 | Full ladder or capped? | **Cap at chosen `*` + nearest 4 per side** |
| 4 | Keep the corner box as well? | **Replace** it with the below-plot note |

---

## 10. Rollback

- `order_chart.py` / `plots.py` / `plots_interactive.py` are tracked and clean
  before this change, so rollback is a `git checkout --` of those paths.
- `trade_note.py` is new and can simply be deleted.
- The `sl_menu` / `tp_menu` columns are additive; older `trades.csv` files render
  fine because the note falls back to chosen-only when the menus are absent.