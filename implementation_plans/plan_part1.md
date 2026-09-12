# Swing Detection & Data Pipeline Fix — Implementation Plan

## Overview

Fixes for five core issues in the trading system:

1. **Data Loading**: Pre-NY session data (Asian + Europe) must be available as reference
2. **Buffer Context Window**: Agent must use full buffer context including pre-NY bars
3. **EOD Close**: Agent-managed with risk guardrails + configurable overnight block mode
4. **Plotting**: Must load full data range (opening to closing + buffer), not just trading session
5. **Swing High/Low**: Must use confirmed, volatility-adaptive swing detector

---

## Issue 5: Swing High/Low — Confirmed Volatility-Adaptive Detector

### Current Problem

 uses rolling max/min with shift:



**Issues:**
- Uses wicks (high/low) instead of confirmed pivot close
- No ATR filtering — many tiny swings in choppy markets
- No market structure classification (HH/HL/LH/LL)
- No continuous features (distance_to_last_high, bars_since_last_swing, etc.)
- Same parameters for all timeframes

### Solution: Multi-Layer Detection Pipeline



### New Functions in 

#### 1.  — Confirmed Fractal Pivots

Uses fractal detection with left/right confirmation. Pivot at candle  becomes
known at candle . No look-ahead bias.

- Uses  price by default (not wick high/low)
- : candle i higher than 2 bars before AND after
- Returns: , , , 

#### 2.  — ATR-Filtered ZigZag Swings

Combines fractal confirmation with ATR minimum movement. ZigZag-style: new swing
high only accepted when price has reversed by .

- Prevents tiny swings in choppy markets
- Returns: , , , 

#### 3.  — HH/HL/LH/LL Classification

- **HH** (Higher High): current swing high > previous
- **LH** (Lower High): current swing high < previous
- **HL** (Higher Low): current swing low > previous
- **LL** (Lower Low): current swing low < previous

#### 4.  — Continuous Features for RL

- : (last_swing_high - close) / atr
- : (close - last_swing_low) / atr
- : distance to nearest swing / atr
- : 1 for high, -1 for low
- : size of last swing in ATR units
- , , 
- : HH, HL, LH, LL, or empty

#### 5.  — Timeframe-Adaptive Parameters


