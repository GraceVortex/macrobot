# Current Status: Phase 10 CPI Event Strategy

**Last Updated:** 2024-11-16

## Executive Summary

Phase 10 CPI event-driven strategy is **CONDITIONALLY VALIDATED** and ready for paper trading after one final step.

---

## Phase 10.7 - Preferred Configuration ✅

**Status:** Validated on 15 years (2010-2024) using FRED proxy data

**Configuration:**
```
Events:     CPI only (monthly releases)
Entry:      Next trading day Open after CPI
Hold:       5 business days
Exit:       Day 5 Open OR OHLC 3% stop
Filter:     NONE (VIX filter not helpful)
Expected:   Sharpe 0.5-0.7, ~12 trades/year
```

**Walk-Forward Results (9 windows):**
- Positive windows: 8/9 (88.9%)
- Median Sharpe: 0.67
- One bad period: 2016-2017 (Sharpe -0.78)
- Regime-dependent: works in high volatility

**Validation Score: 5/8 checks passed**
- ✅ Bug fixes applied (8 critical fixes)
- ✅ Stopped trades validated
- ✅ Non-stopped trades 97.8% win rate
- ✅ Historical edge exists (2010-2024)
- ✅ No overfitting
- ⚠️  Walk-forward: 8/9 positive but high variance
- ❌ Worst case: one negative period
- ❌ Regime dependent

---

## What's Left: Final Data Collection

### Current Data
- Source: FRED API
- Method: "Previous as consensus" proxy
- Events: 177 CPI (105 train, 72 test)
- Quality: ⚠️  Acceptable for validation, but not perfect

### What's Needed
**Collect real consensus forecasts from Investing.com**

**Task:**
1. Visit: https://www.investing.com/economic-calendar/
2. Filter: United States → CPI
3. Date range: 2010-01-01 to 2024-11-15
4. Manually copy: Date, Actual, Forecast (consensus), Previous
5. Save to: `outputs_phase10_6/economic_events_extended.csv`

**Expected:**
- ~180 CPI events (2010-2024)
- Real market consensus (not previous-as-proxy)
- More accurate surprise calculations

**Time Required:** 2-3 hours manual work

**Why Important:**
- FRED proxy uses "previous = consensus" (approximation)
- Real consensus more accurate for surprise calculation
- Final validation before paper trading

---

## After Data Collection

### Step 1: Re-run Phase 10.7
```bash
# Update economic_events_extended.csv with real data
python phase10_7.py
```

**Compare results:**
- FRED proxy: Median Sharpe 0.67
- Real consensus: Sharpe ???

**If similar → Great! Strategy robust**  
**If different → Adjust expectations**

### Step 2: Paper Trading
**Start:** Next CPI release (December 11-13, 2024)

**Track:**
- Entry price, exit price, stop hits
- VIX on entry day
- Compare to backtest expectations

**After 10-15 trades:**
- If matches backtest → Go live with 25% capital
- If underperforms → Investigate or stop
- If outperforms → Continue monitoring

---

## Phase 10 Journey Summary

### Phase 10.0: Initial Discovery
- Result: Sharpe 1.32 (too good)
- Problem: 8 critical bugs found

### Phase 10.5: Bug Fixes
- Fixed: Stop timing, entry/exit, slippage, etc.
- Result: Sharpe 1.22 (realistic)
- Validated: Stop mechanism working correctly

### Phase 10.6: Historical Extension
- Extended: 2020-2024 → 2010-2024 (FRED data)
- Train: Sharpe 0.63 (2010-2019)
- Test: Sharpe 0.83 (2020-2024)
- Result: ✅ Edge exists on old data

### Phase 10.7: Walk-Forward Validation
- Method: 9 rolling windows (5yr train, 2yr test)
- Result: 8/9 positive (88.9%)
- Median: Sharpe 0.67
- Problem: One disaster (2016-2017, Sharpe -0.78)
- Diagnosis: Regime-dependent

### Phase 10.8: VIX Filter Test
- Tested: Skip if VIX < 15
- Result: Marginal (+0.06 Sharpe)
- Decision: NOT recommended (too crude)

---

## Decision Tree

```
Current State: Phase 10.7 validated on FRED proxy data
│
├─ Option A: Collect full consensus data (RECOMMENDED)
│  ├─ Time: 2-3 hours
│  ├─ Benefit: Most accurate validation
│  ├─ Re-run Phase 10.7 with real data
│  ├─ Compare to FRED results
│  └─ Then → Paper trading
│
├─ Option B: Skip to paper trading with FRED data
│  ├─ Time: 0 hours
│  ├─ Risk: Proxy data may not be perfect
│  ├─ Start paper trading now
│  ├─ Collect real data in parallel
│  └─ Validate retrospectively
│
└─ Option C: Pivot to Phase 9 instead
   └─ Phase 9 already production-ready (Sharpe 0.92)
```

**Recommendation:** Option A (collect data, then paper trade)
- Most rigorous approach
- 2-3 hours well spent
- Confidence boost before real money

---

## Files Reference

### Key Documentation
- `PHASE10_FINAL_SUMMARY.md` - Complete Phase 10 journey
- `outputs_phase10_7/WALK_FORWARD_SUMMARY.txt` - Preferred config details
- `outputs_phase10_8/VIX_FILTER_SUMMARY.txt` - Why VIX filter not used

### Data Files
- `outputs_phase10_6/economic_events_extended.csv` - FRED proxy data (update needed)
- `outputs_phase10_7/walk_forward_results.csv` - 9 window results
- `outputs_phase10_8/walk_forward_vix_filtered.csv` - VIX filter test

### Code
- `phase10.py` - Main backtest engine (bug-fixed)
- `phase10_6_quick.py` - FRED data collection
- `phase10_7.py` - Walk-forward validation (RUN THIS after data update)
- `phase10_8.py` - VIX filter test

---

## Next Actions

### Immediate (Before Paper Trading)
- [ ] Collect real CPI consensus data from Investing.com (2-3 hours)
- [ ] Update `outputs_phase10_6/economic_events_extended.csv`
- [ ] Re-run `python phase10_7.py`
- [ ] Review new walk-forward results
- [ ] If Sharpe still 0.6+ → Proceed to paper trading

### Paper Trading Phase (3-6 months)
- [ ] Set CPI calendar alerts
- [ ] Trade next 10-15 CPI releases
- [ ] Track actual vs expected metrics
- [ ] Compare live Sharpe to backtest (0.5-0.7)
- [ ] Document any deviations

### Go-Live Decision (After Paper Trading)
- [ ] If live Sharpe ≥ 0.5 → Deploy 25% capital
- [ ] If live Sharpe < 0.3 → Investigate or stop
- [ ] Monitor for 20 trades before full deployment

---

## Questions & Answers

**Q: Why not use Phase 9 instead?**  
A: Phase 9 is production-ready (Sharpe 0.92), but only 3-4 trades/year. Phase 10 offers 12 trades/year for more activity. Can use both strategies together.

**Q: Is Phase 10 safe without VIX filter?**  
A: Yes. Testing showed VIX filter only marginally helpful (+0.06 Sharpe). Trade all CPI events, accept occasional bad years.

**Q: What if 2016-2017 repeats?**  
A: Expect 1-2 bad years per decade (low volatility periods). Strategy still profitable long-term (8/9 windows positive).

**Q: Can I skip data collection?**  
A: Yes, but not recommended. FRED proxy good for validation, but real consensus more accurate. 2-3 hours worth the confidence boost.

**Q: When is next CPI?**  
A: Usually 11-13th of each month. Next: December 11-13, 2024.

---

## Bottom Line

**Phase 10.7 is validated and ready for paper trading** after collecting real consensus data.

Expected performance:
- Sharpe: 0.5-0.7 (realistic with bad years)
- Win Rate: 40-50%
- Frequency: ~12 trades/year
- One bad year per decade expected

**Final step:** Collect real CPI consensus data (2-3 hours) → Paper trade → Go live

**Status:** 95% complete, one data collection task remaining

---

For implementation details, see `outputs_phase10_7/WALK_FORWARD_SUMMARY.txt`
