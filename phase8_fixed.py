"""
PHASE 8: ADVANCED STRATEGY IMPROVEMENTS (FIXED)
================================================
Professional-grade enhancements - all bugs fixed

IMPROVEMENTS:
1. Real VIX filtering
2. Signal strength filter
3. Realistic stop-loss
4. Position sizing
5. Combined strategies

All pandas ambiguity errors resolved.
"""

import pandas as pd
import numpy as np
import warnings
import os
from pandas.tseries.offsets import BDay

warnings.filterwarnings('ignore')

TRANSACTION_COSTS = 0.0019

print("=" * 70)
print("PHASE 8: ADVANCED IMPROVEMENTS (FIXED)")
print("=" * 70)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

def load_vix_simple():
    """Load VIX or use proxy"""
    print("Loading VIX data...")
    try:
        import yfinance as yf
        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        vix = vix[['Close']].rename(columns={'Close': 'vix'})
        print(f"  VIX loaded: {len(vix)} days\n")
        return vix
    except:
        print("  VIX unavailable, using year proxy\n")
        return None

def load_trades():
    """Load trades"""
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades['exit_date'] = trades['date'] + BDay(2)
    trades['year'] = trades['date'].dt.year
    return trades.sort_values('date').reset_index(drop=True)

vix_data = load_vix_simple()
all_trades = load_trades()

print(f"Loaded: {len(all_trades)} trades ({all_trades['date'].min().date()} to {all_trades['date'].max().date()})")
print()

# ============================================================================
# FILTERS
# ============================================================================

def filter_by_vix(trades, vix_data, threshold=20):
    """Filter by VIX threshold"""
    if vix_data is None:
        # Fallback
        result = trades[trades['year'].isin([2020, 2021])].copy()
        print(f"  VIX filter (proxy): {len(trades)} -> {len(result)} trades")
        return result
    
    keep_indices = []
    for idx in range(len(trades)):
        trade = trades.iloc[idx]
        trade_date = trade['date']
        
        # Get VIX value
        vix_slice = vix_data[vix_data.index <= trade_date]
        if len(vix_slice) == 0:
            continue
        
        vix_val = float(vix_slice.iloc[-1]['vix'])
        if vix_val >= threshold:
            keep_indices.append(idx)
    
    result = trades.iloc[keep_indices].copy() if keep_indices else pd.DataFrame()
    print(f"  VIX > {threshold}: {len(trades)} -> {len(result)} trades")
    return result

def filter_by_rating(trades, min_rating=3):
    """Filter by signal strength"""
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades[trades['rating'].abs() >= min_rating].copy()
    print(f"  Rating >= {min_rating}: {len(trades)} -> {len(result)} trades")
    return result

def apply_simple_stop(trades, stop_pct=1.5):
    """Simple stop-loss"""
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades.copy()
    result['pnl'] = result['pnl'].clip(lower=-stop_pct)
    return result

def apply_position_sizing(trades):
    """Scale by signal strength"""
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades.copy()
    
    # Size: rating 5 = 2x, rating 3 = 1x, rating 1 = 0.5x
    result['multiplier'] = result['rating'].abs() / 3.0
    result['multiplier'] = result['multiplier'].clip(lower=0.5, upper=2.0)
    result['pnl'] = result['pnl'] * result['multiplier']
    
    return result

# ============================================================================
# METRICS
# ============================================================================

def build_daily_curve(trades):
    """Build equity curve"""
    if len(trades) == 0:
        return pd.DataFrame(columns=['date', 'pnl_net', 'cum_net', 'drawdown'])
    
    start = trades['date'].min()
    end = trades['exit_date'].max()
    dates = pd.date_range(start=start, end=end, freq='B')
    
    daily = pd.DataFrame({'date': dates, 'pnl_net': 0.0})
    daily.set_index('date', inplace=True)
    
    for idx in range(len(trades)):
        trade = trades.iloc[idx]
        exit_date = trade['exit_date']
        pnl_net = trade['pnl'] - (TRANSACTION_COSTS * 100)
        
        if exit_date in daily.index:
            daily.loc[exit_date, 'pnl_net'] += pnl_net
    
    daily['cum_net'] = daily['pnl_net'].cumsum()
    daily['running_max'] = daily['cum_net'].expanding().max()
    daily['drawdown'] = daily['cum_net'] - daily['running_max']
    
    return daily.reset_index()

def calc_metrics(daily, trades, label):
    """Calculate metrics"""
    if len(trades) == 0 or len(daily) == 0:
        return None
    
    trades_net = trades.copy()
    trades_net['pnl_net'] = trades_net['pnl'] - (TRANSACTION_COSTS * 100)
    
    pnl = daily['pnl_net']
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0
    
    total_ret = float(daily['cum_net'].iloc[-1])
    max_dd = float(daily['drawdown'].min())
    win_rate = float((trades_net['pnl_net'] > 0).mean() * 100)
    avg_ret = float(trades_net['pnl_net'].mean())
    
    print(f"{label}:")
    print(f"  Trades: {len(trades)}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Avg: {avg_ret:+.2f}%")
    print(f"  Win: {win_rate:.1f}%")
    print(f"  Total: {total_ret:+.1f}%")
    print(f"  DD: {max_dd:.1f}%")
    print()
    
    return {
        'label': label,
        'trades': len(trades),
        'sharpe': sharpe,
        'avg_ret': avg_ret,
        'win_rate': win_rate,
        'total_ret': total_ret,
        'max_dd': max_dd
    }

# ============================================================================
# TESTS
# ============================================================================

results = []

# Baseline
print("=" * 70)
print("BASELINE")
print("=" * 70)
print()

daily_base = build_daily_curve(all_trades)
base = calc_metrics(daily_base, all_trades, "Baseline")
if base:
    results.append(base)

# Test 1: VIX
print("=" * 70)
print("TEST 1: VIX Filter")
print("=" * 70)
print()

vix_trades = filter_by_vix(all_trades, vix_data, threshold=20)
if len(vix_trades) > 0:
    daily_vix = build_daily_curve(vix_trades)
    vix_m = calc_metrics(daily_vix, vix_trades, "VIX > 20")
    if vix_m:
        results.append(vix_m)
else:
    print("  No trades\n")

# Test 2: Strong signals
print("=" * 70)
print("TEST 2: Strong Signals")
print("=" * 70)
print()

strong_trades = filter_by_rating(all_trades, min_rating=3)
if len(strong_trades) > 0:
    daily_strong = build_daily_curve(strong_trades)
    strong_m = calc_metrics(daily_strong, strong_trades, "Rating >= 3")
    if strong_m:
        results.append(strong_m)
else:
    print("  No trades\n")

# Test 3: Simple stop
print("=" * 70)
print("TEST 3: Stop Loss")
print("=" * 70)
print()

stopped = apply_simple_stop(all_trades, stop_pct=1.5)
if len(stopped) > 0:
    daily_stop = build_daily_curve(stopped)
    stop_m = calc_metrics(daily_stop, stopped, "Stop 1.5%")
    if stop_m:
        results.append(stop_m)

# Test 4: Position sizing
print("=" * 70)
print("TEST 4: Position Sizing")
print("=" * 70)
print()

sized = apply_position_sizing(all_trades)
if len(sized) > 0:
    daily_sized = build_daily_curve(sized)
    sized_m = calc_metrics(daily_sized, sized, "Pos Sizing")
    if sized_m:
        results.append(sized_m)

# Test 5: VIX + Strong
print("=" * 70)
print("TEST 5: VIX + Strong Signals")
print("=" * 70)
print()

combined = filter_by_vix(all_trades, vix_data, threshold=20)
if len(combined) > 0:
    combined = filter_by_rating(combined, min_rating=3)

if len(combined) > 0:
    daily_comb = build_daily_curve(combined)
    comb_m = calc_metrics(daily_comb, combined, "VIX + Strong")
    if comb_m:
        results.append(comb_m)
else:
    print("  No trades\n")

# Test 6: Ultimate (VIX + Strong + Stop + Size)
print("=" * 70)
print("TEST 6: ULTIMATE")
print("=" * 70)
print()

ultimate = filter_by_vix(all_trades, vix_data, threshold=20)
print(f"  Step 1: {len(ultimate)} trades")

if len(ultimate) > 0:
    ultimate = filter_by_rating(ultimate, min_rating=3)
    print(f"  Step 2: {len(ultimate)} trades")

if len(ultimate) > 0:
    ultimate = apply_simple_stop(ultimate, stop_pct=1.5)
    print(f"  Step 3: {len(ultimate)} trades")

if len(ultimate) > 0:
    ultimate = apply_position_sizing(ultimate)
    print(f"  Step 4: {len(ultimate)} trades")
    print()

if len(ultimate) > 0:
    daily_ult = build_daily_curve(ultimate)
    ult_m = calc_metrics(daily_ult, ultimate, "ULTIMATE")
    if ult_m:
        results.append(ult_m)
else:
    print("  No trades\n")

# ============================================================================
# RESULTS
# ============================================================================

print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print()

comp = pd.DataFrame(results)
print(comp.to_string(index=False))
print()

best_idx = comp['sharpe'].idxmax()
best = comp.iloc[best_idx]

print("=" * 70)
print("BEST STRATEGY")
print("=" * 70)
print()
print(f"Strategy: {best['label']}")
print(f"Sharpe: {best['sharpe']:.2f}")
print(f"Trades: {int(best['trades'])}")
print(f"Win Rate: {best['win_rate']:.1f}%")
print()

# Save
os.makedirs('outputs_phase8', exist_ok=True)
comp.to_csv('outputs_phase8/results.csv', index=False)

with open('outputs_phase8/SUMMARY.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE 8 SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Baseline: Sharpe {base['sharpe']:.2f}\n")
    f.write(f"Best: {best['label']} - Sharpe {best['sharpe']:.2f}\n\n")
    f.write(comp.to_string(index=False))

print("Files saved to outputs_phase8/")
print("Phase 8 complete!")
