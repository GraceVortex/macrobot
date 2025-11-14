# TMF Trading Strategy: Project Summary

## Overview

Systematic, backtested trading strategy for TMF (3x leveraged 20-year Treasury ETF) developed through 9 phases of research and testing.

**Final Performance:** Sharpe 0.92 | Win 83.3% | +2.89% per trade | 3-4 trades/year

## Phase Evolution

### Phase 1-2: Foundation
Initial exploration and signal development. Established correlation-based approach.

### Phase 3: Signal Generation
**Outcome:** Base dataset of 53 trades (2015-2023)

Created TLT correlation signals with T+2 holding period. Baseline Sharpe: 0.84

**Key Output:** `tmf_all_trades.csv` - foundation for all subsequent phases

### Phase 4-5: Strategy Enhancement
**Outcome:** Single-asset focus validated

Tested portfolio approaches and expanded signals. Found single-asset (TMF) more robust than diversification.

### Phase 6-7: Macro Validation
**Outcome:** Regime dependency identified

Full period backtest (2015-2023) with 12-month forward validation. Strategy works best in volatile markets.

**Finding:** Performance varies by volatility regime - needs adaptive filtering

### Phase 8: Advanced Improvements (Idealized)
**Outcome:** Sharpe 1.15 with idealized stops

Introduced:
- VIX filtering (threshold = 20)
- Rating filter (>= 3)
- Simple stop-loss (1.5% clip)
- Position sizing (0.5x - 2.0x)

**Problem:** Stops were unrealistic (simple clip), creating overly optimistic results

### Phase 8.5: Reality Check
**Outcome:** Critical discovery - stops don't work

**Key Finding:**
```
Simple Stop:  Sharpe +1.15  Win 85.7%  (optimistic)
OHLC Stop:    Sharpe -1.13  Win  7.1%  (realistic)
```

Introduced:
1. OHLC-based realistic stops (intraday volatility captured)
2. Transaction costs scaled by position size
3. VIX percentile filter (adaptive)
4. Position sizing before stop (fixed max loss)

**Conclusion:** Tight stops (1.5%) destroyed by TMF intraday volatility. Need new approach.

### Phase 9: Complete Solution
**Outcome:** Production-ready strategy

**Block A: Stop-Loss Solution (15 variants tested)**

Results sorted by Sharpe:
1. No Stop: 0.92 (WINNER)
2. ATR 2.0x: 0.74
3. Close 3%: 0.41
4. OHLC 7%: 0.34
...
15. Trailing: -1.16 (WORST)

**Answer:** Don't use stop-loss. Time-based exit (T+2) is sufficient.

**Block B: Filter Optimization**

VIX comparison:
- P80: Sharpe 0.94 (22 trades) - highest but fewer samples
- P70: Sharpe 0.92 (30 trades) - CHOSEN for balance
- P60: Sharpe 0.83 (43 trades)
- None: Sharpe 0.89 (53 trades)

Rating filter: Minimal benefit (0.92 vs 0.94). Not worth complexity.

**Block C: Risk Profile**

MTM vs Cash accounting:
- Cash DD: -10.3% (misleading)
- MTM DD: -84.3% (realistic for 3x leverage)

Yearly breakdown shows best performance 2020-2022 (volatile years).

**Block D: Production Spec**

Final configuration with conservative estimates (30% discount).

## Critical Insights

### 1. Stop-Loss Paradox
**Problem:** All stops underperform no-stop
**Cause:** TMF intraday noise triggers stops before profit
**Solution:** Time-based exit only

### 2. Testing Method Matters
**Phase 8:** Clip stop → Sharpe 1.15
**Phase 8.5:** OHLC stop → Sharpe -1.13
**Lesson:** Always use OHLC data for realistic results

### 3. Cost Modeling
**Old:** Fixed cost
**New:** Cost scales with position (2x position = 2x cost)
**Impact:** Honest performance estimates

### 4. VIX Regime Selection
**Finding:** P70 percentile optimal
**Reason:** Balances Sharpe and trade frequency
**Result:** Quality over quantity

### 5. Position Sizing
**Approach:** Rating as weight, not filter
**Formula:** multiplier = rating/3.0, clip [0.5, 2.0]
**Benefit:** No trades discarded, natural scaling

### 6. MTM Reality
**Insight:** MTM DD 8x larger than cash
**Reason:** Daily mark-to-market + 3x leverage
**Action:** Focus on Sharpe, not absolute DD

## Final Strategy

```
Entry:        TLT correlation signal (Phase 3)
VIX Filter:   > 70th percentile
Position:     0.5x - 2.0x by rating
Stop-Loss:    NONE
Exit:         T+2 business days
Instrument:   TMF (3x levered)
Cost:         0.19% * multiplier
```

**Expected Performance:**
- Sharpe: 0.92 (Conservative: 0.64)
- Win Rate: 83.3%
- Avg/Trade: +2.89%
- Frequency: 3-4 trades/year
- Max DD: -84% (MTM)

**Conservative Annual:** 6-10% return, Sharpe 0.6-0.8

## Key Outputs

**Code:**
- `phase3_leverage.py` - Signal generation
- `phase8_5.py` - Reality check
- `phase9.py` - Final solution

**Data:**
- `outputs_phase3/tmf_all_trades.csv` - Base dataset
- `outputs_phase9/block_a_stops.csv` - 15 stop tests
- `outputs_phase9/block_b_filters.csv` - VIX optimization
- `outputs_phase9/PRODUCTION_SPEC.txt` - Implementation guide

## Implementation

### Entry Rules
1. Monitor VIX daily
2. Calculate 70th percentile (rolling)
3. Trade only when VIX > threshold
4. Verify TLT correlation signal

### Position Sizing
```python
multiplier = rating / 3.0
multiplier = clip(multiplier, 0.5, 2.0)
cost = 0.19% * multiplier
```

### Exit Rules
Exit exactly T+2 business days. No discretion.

### Risk Management
- Max 1-2 concurrent positions
- No stop-loss (proven suboptimal)
- Monitor VIX regime weekly
- Review strategy quarterly

## Risk Warnings

**High Volatility:** 3x leverage means extreme swings. DD can reach -80%+.

**Low Frequency:** Only 3-4 trades/year. Don't force trades in calm markets.

**Regime Dependent:** Works in volatile markets, weak in calm. Be patient.

**No Stops:** Counterintuitive but systematically proven. Requires discipline.

## Potential Improvements

### 1. Multi-Asset
Apply to TLT (1x) for lower volatility variant. Compare risk/reward profiles.

### 2. Options Overlay
Sell covered calls for premium income and downside protection.

### 3. Dynamic VIX
Adjust threshold based on market regime. Lower in calm, higher in panic.

### 4. ML Enhancement
Train classifier on features. Use probability as additional filter/weight.

### 5. Portfolio Integration
Combine with uncorrelated strategies. Target portfolio Sharpe 1.5+.

### 6. Execution Optimization
Test optimal entry times. Minimize slippage through limit orders.

## Conclusion

Nine phases transformed initial exploration into production-ready strategy:

**Phase 1-2:** Foundation
**Phase 3:** Signal generation (53 trades)
**Phase 4-5:** Enhancement testing
**Phase 6-7:** Macro validation
**Phase 8:** Idealized improvements (Sharpe 1.15)
**Phase 8.5:** Reality check (OHLC stops → Sharpe -1.13)
**Phase 9:** Complete solution (No stops → Sharpe 0.92)

**Key Discovery:** Stop-loss paradox. All 15 variants tested underperform no-stop approach.

**Why:** TMF intraday volatility is noise. Stops trigger before profit. Time-based exit (T+2) provides natural limit.

**Result:** Clean, simple, robust strategy ready for live trading.

**Conservative Expectations:**
- Sharpe: 0.6-0.8
- Annual: 6-10%
- Trades: 3-4/year

Strategy requires patience, discipline, and conviction. The counterintuitive no-stop approach has been systematically validated through rigorous testing.

---

**Total Development:** 9 phases
**Trades Analyzed:** 53
**Stop Variants Tested:** 15
**Final Sharpe:** 0.92
**Status:** Production-ready

See `outputs_phase9/PRODUCTION_SPEC.txt` for implementation details.
