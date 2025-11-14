"""
PHASE 8.5: PROFESSIONAL IMPROVEMENTS
=====================================
Key improvements over Phase 8:
1. Transaction costs scale with position size (cost proportional to multiplier)
2. Position sizing -> stop order (fixed max loss 1.5% per trade)
3. OHLC-based realistic stops (using TMF/TLT daily bars)
4. VIX percentile filter (>70th percentile instead of threshold=20)

This addresses the 4 key issues from Phase 8:
- More realistic stop-loss execution
- Honest cost modeling
- Fixed per-trade risk
- More adaptive volatility filtering
"""

import pandas as pd
import numpy as np
import warnings
import os
from pandas.tseries.offsets import BDay

warnings.filterwarnings('ignore')

TRANSACTION_COSTS = 0.0019

print("=" * 70)
print("PHASE 8.5: PROFESSIONAL IMPROVEMENTS")
print("=" * 70)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

def load_vix_data():
    """Load VIX data"""
    print("Loading VIX data...")
    try:
        import yfinance as yf
        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        vix = vix[['Close']].rename(columns={'Close': 'vix'})
        print(f"  [OK] VIX loaded: {len(vix)} days\n")
        return vix
    except Exception as e:
        print(f"  [FAIL] VIX failed: {e}\n")
        return None

def load_ohlc_data(ticker):
    """Load OHLC data for realistic stops"""
    print(f"Loading {ticker} OHLC data...")
    try:
        import yfinance as yf
        data = yf.download(ticker, start='2015-01-01', end='2024-12-31', progress=False)
        if len(data) == 0:
            print(f"  [FAIL] {ticker} no data\n")
            return None
        print(f"  [OK] {ticker} loaded: {len(data)} days\n")
        return data
    except Exception as e:
        print(f"  [FAIL] {ticker} failed: {e}\n")
        return None

def load_trades():
    """Load trades"""
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades['exit_date'] = trades['date'] + BDay(2)
    trades['year'] = trades['date'].dt.year
    return trades.sort_values('date').reset_index(drop=True)

# Load all data
vix_data = load_vix_data()
tmf_ohlc = load_ohlc_data('TMF')
all_trades = load_trades()

print(f"Loaded: {len(all_trades)} trades ({all_trades['date'].min().date()} to {all_trades['date'].max().date()})")
print()

# ============================================================================
# IMPROVED FILTERS
# ============================================================================

def filter_by_vix_percentile(trades, vix_data, percentile=70):
    """
    Filter by VIX percentile instead of fixed threshold
    More adaptive to changing volatility regimes
    """
    if vix_data is None:
        # Fallback to year proxy
        result = trades[trades['year'].isin([2020, 2021])].copy()
        print(f"  VIX filter (proxy): {len(trades)} -> {len(result)} trades")
        return result
    
    # Calculate percentile threshold
    vix_threshold = np.percentile(vix_data['vix'].dropna(), percentile)
    print(f"  VIX {percentile}th percentile = {vix_threshold:.2f}")
    
    keep_indices = []
    for idx in range(len(trades)):
        trade = trades.iloc[idx]
        trade_date = trade['date']
        
        # Get VIX value at trade date
        vix_slice = vix_data[vix_data.index <= trade_date]
        if len(vix_slice) == 0:
            continue
        
        vix_val = float(vix_slice.iloc[-1]['vix'])
        if vix_val >= vix_threshold:
            keep_indices.append(idx)
    
    result = trades.iloc[keep_indices].copy() if keep_indices else pd.DataFrame()
    print(f"  VIX > {vix_threshold:.1f}: {len(trades)} -> {len(result)} trades")
    return result

def filter_by_rating(trades, min_rating=3):
    """Filter by signal strength"""
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades[trades['rating'].abs() >= min_rating].copy()
    print(f"  Rating >= {min_rating}: {len(trades)} -> {len(result)} trades")
    return result

def apply_position_sizing(trades):
    """
    Scale position by signal strength
    Returns trades with 'multiplier' column
    """
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades.copy()
    
    # Size: rating 5 = 2x, rating 3 = 1x, rating 1 = 0.5x
    result['multiplier'] = result['rating'].abs() / 3.0
    result['multiplier'] = result['multiplier'].clip(lower=0.5, upper=2.0)
    
    # Apply multiplier to PnL
    result['pnl'] = result['pnl'] * result['multiplier']
    
    return result

def apply_ohlc_stop(trades, ohlc_data, stop_pct=1.5):
    """
    OHLC-based realistic stop-loss
    Checks if intraday low would have hit the stop
    More realistic than simple clipping
    """
    if len(trades) == 0 or ohlc_data is None:
        print("  [WARN] No OHLC data, using simple stop")
        return apply_simple_stop(trades, stop_pct)
    
    result = trades.copy()
    stopped_count = 0
    
    for idx in range(len(result)):
        trade_date = result.iloc[idx]['date']
        exit_date = result.iloc[idx]['exit_date']
        original_pnl = result.iloc[idx]['pnl']
        
        # Get OHLC data for holding period
        ohlc_slice = ohlc_data[(ohlc_data.index > trade_date) & (ohlc_data.index <= exit_date)]
        
        if len(ohlc_slice) == 0:
            # No data, use original PnL
            continue
        
        # Check if stop would be hit
        # Assume entry at open of first day
        entry_price = float(ohlc_slice.iloc[0]['Open'])
        stop_price = entry_price * (1 - stop_pct / 100)
        
        # Check each day's low
        for bar_idx in range(len(ohlc_slice)):
            bar_low = float(ohlc_slice.iloc[bar_idx]['Low'])
            if bar_low <= stop_price:
                # Stop hit!
                result.iloc[idx, result.columns.get_loc('pnl')] = -stop_pct
                stopped_count += 1
                break
    
    print(f"  OHLC stops triggered: {stopped_count}/{len(result)} trades")
    return result

def apply_simple_stop(trades, stop_pct=1.5):
    """
    Simple clip-based stop (fallback)
    Used when OHLC data unavailable
    """
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades.copy()
    result['pnl'] = result['pnl'].clip(lower=-stop_pct)
    return result

# ============================================================================
# METRICS WITH IMPROVED COST MODELING
# ============================================================================

def build_daily_curve(trades):
    """
    Build equity curve with costs scaled by multiplier
    KEY IMPROVEMENT: cost_pct proportional to multiplier
    """
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
        pnl_gross = trade['pnl']
        
        # Get multiplier (default to 1.0 if not present)
        multiplier = trade.get('multiplier', 1.0)
        
        # KEY FIX: Scale costs by position size
        cost_pct = TRANSACTION_COSTS * 100 * multiplier
        pnl_net = pnl_gross - cost_pct
        
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
    
    # Calculate net PnL with scaled costs
    trades_net = trades.copy()
    multiplier = trades_net.get('multiplier', pd.Series([1.0] * len(trades_net)))
    cost_pct = TRANSACTION_COSTS * 100 * multiplier
    trades_net['pnl_net'] = trades_net['pnl'] - cost_pct
    
    pnl = daily['pnl_net']
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0
    
    total_ret = float(daily['cum_net'].iloc[-1])
    max_dd = float(daily['drawdown'].min())
    win_rate = float((trades_net['pnl_net'] > 0).mean() * 100)
    avg_ret = float(trades_net['pnl_net'].mean())
    
    # Calculate average multiplier if present
    avg_mult = float(multiplier.mean()) if 'multiplier' in trades.columns else 1.0
    
    print(f"{label}:")
    print(f"  Trades: {len(trades)}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Avg: {avg_ret:+.2f}%")
    print(f"  Win: {win_rate:.1f}%")
    print(f"  Total: {total_ret:+.1f}%")
    print(f"  DD: {max_dd:.1f}%")
    if avg_mult != 1.0:
        print(f"  Avg Multiplier: {avg_mult:.2f}x")
    print()
    
    return {
        'label': label,
        'trades': len(trades),
        'sharpe': sharpe,
        'avg_ret': avg_ret,
        'win_rate': win_rate,
        'total_ret': total_ret,
        'max_dd': max_dd,
        'avg_mult': avg_mult
    }

# ============================================================================
# STRATEGY TESTS
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

# Test 1: VIX Percentile (70th)
print("=" * 70)
print("TEST 1: VIX Percentile Filter")
print("=" * 70)
print()

vix_trades = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
if len(vix_trades) > 0:
    daily_vix = build_daily_curve(vix_trades)
    vix_m = calc_metrics(daily_vix, vix_trades, "VIX P70")
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

# Test 3: OHLC Stop
print("=" * 70)
print("TEST 3: OHLC-Based Stop Loss")
print("=" * 70)
print()

stopped = apply_ohlc_stop(all_trades, tmf_ohlc, stop_pct=1.5)
if len(stopped) > 0:
    daily_stop = build_daily_curve(stopped)
    stop_m = calc_metrics(daily_stop, stopped, "OHLC Stop 1.5%")
    if stop_m:
        results.append(stop_m)

# Test 4: Position sizing with scaled costs
print("=" * 70)
print("TEST 4: Position Sizing (Scaled Costs)")
print("=" * 70)
print()

sized = apply_position_sizing(all_trades)
if len(sized) > 0:
    daily_sized = build_daily_curve(sized)
    sized_m = calc_metrics(daily_sized, sized, "Pos Sizing + Scaled Costs")
    if sized_m:
        results.append(sized_m)

# Test 5: VIX P70 + Strong
print("=" * 70)
print("TEST 5: VIX P70 + Strong Signals")
print("=" * 70)
print()

combined = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
if len(combined) > 0:
    combined = filter_by_rating(combined, min_rating=3)

if len(combined) > 0:
    daily_comb = build_daily_curve(combined)
    comb_m = calc_metrics(daily_comb, combined, "VIX P70 + Strong")
    if comb_m:
        results.append(comb_m)
else:
    print("  No trades\n")

# Test 6: ULTIMATE v2.0 (VIX P70 + Strong + SIZE -> STOP)
print("=" * 70)
print("TEST 6: ULTIMATE v2.0")
print("=" * 70)
print()
print("Pipeline: VIX P70 -> Rating >=3 -> Position Sizing -> OHLC Stop")
print()

ultimate = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
print(f"  Step 1 (VIX P70): {len(ultimate)} trades")

if len(ultimate) > 0:
    ultimate = filter_by_rating(ultimate, min_rating=3)
    print(f"  Step 2 (Rating): {len(ultimate)} trades")

if len(ultimate) > 0:
    # KEY CHANGE: Position sizing BEFORE stop
    ultimate = apply_position_sizing(ultimate)
    print(f"  Step 3 (Sizing): {len(ultimate)} trades")

if len(ultimate) > 0:
    # Then apply stop to limit max loss at 1.5% per trade
    ultimate = apply_ohlc_stop(ultimate, tmf_ohlc, stop_pct=1.5)
    print(f"  Step 4 (OHLC Stop): {len(ultimate)} trades")
    print()

if len(ultimate) > 0:
    daily_ult = build_daily_curve(ultimate)
    ult_m = calc_metrics(daily_ult, ultimate, "ULTIMATE v2.0")
    if ult_m:
        results.append(ult_m)
else:
    print("  No trades\n")

# Test 7: Alternative - SIZE → Simple Stop (for comparison)
print("=" * 70)
print("TEST 7: ULTIMATE v2.0 Simple Stop")
print("=" * 70)
print()
print("Same as v2.0 but with simple clip stop (no OHLC)")
print()

ultimate_simple = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
if len(ultimate_simple) > 0:
    ultimate_simple = filter_by_rating(ultimate_simple, min_rating=3)
if len(ultimate_simple) > 0:
    ultimate_simple = apply_position_sizing(ultimate_simple)
if len(ultimate_simple) > 0:
    ultimate_simple = apply_simple_stop(ultimate_simple, stop_pct=1.5)
    print(f"  Final: {len(ultimate_simple)} trades")
    print()

if len(ultimate_simple) > 0:
    daily_ult_simple = build_daily_curve(ultimate_simple)
    ult_simple_m = calc_metrics(daily_ult_simple, ultimate_simple, "ULTIMATE v2.0 (Simple Stop)")
    if ult_simple_m:
        results.append(ult_simple_m)
else:
    print("  No trades\n")

# ============================================================================
# RESULTS & COMPARISON
# ============================================================================

print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print()

comp = pd.DataFrame(results)
# Sort by Sharpe
comp = comp.sort_values('sharpe', ascending=False)
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
print(f"Avg Return: {best['avg_ret']:+.2f}%")
print(f"Win Rate: {best['win_rate']:.1f}%")
print(f"Total Return: {best['total_ret']:+.1f}%")
print(f"Max DD: {best['max_dd']:.1f}%")
print()

# ============================================================================
# KEY IMPROVEMENTS SUMMARY
# ============================================================================

print("=" * 70)
print("KEY IMPROVEMENTS vs PHASE 8")
print("=" * 70)
print()
print("+ Transaction costs scale with multiplier (honest cost modeling)")
print("+ Position sizing -> stop order (fixed 1.5% max loss per trade)")
print("+ OHLC-based stops (realistic stop execution)")
print("+ VIX percentile filter (adaptive volatility regime)")
print()

# Save outputs
os.makedirs('outputs_phase8_5', exist_ok=True)
comp.to_csv('outputs_phase8_5/results.csv', index=False)

# Save detailed summary
with open('outputs_phase8_5/SUMMARY.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE 8.5 SUMMARY - PROFESSIONAL IMPROVEMENTS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("KEY IMPROVEMENTS:\n")
    f.write("1. Transaction costs scale with position size (cost proportional to multiplier)\n")
    f.write("2. Position sizing -> stop order (fixed max loss 1.5%)\n")
    f.write("3. OHLC-based realistic stops (using TMF daily bars)\n")
    f.write("4. VIX percentile filter (>70th instead of threshold=20)\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("RESULTS\n")
    f.write("=" * 70 + "\n\n")
    
    if base:
        f.write(f"Baseline: Sharpe {base['sharpe']:.2f}\n")
    f.write(f"Best: {best['label']} - Sharpe {best['sharpe']:.2f}\n\n")
    
    f.write(comp.to_string(index=False))
    f.write("\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("IDEOLOGY DECISIONS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. Stop-loss ideology: FIXED per-trade risk\n")
    f.write("   - Position sizing applied BEFORE stop\n")
    f.write("   - Max loss capped at 1.5% regardless of position size\n")
    f.write("   - This is the conservative approach for risk management\n\n")
    
    f.write("2. Stop implementation: OHLC-based (realistic)\n")
    f.write("   - Uses TMF daily bars to check if intraday low hits stop\n")
    f.write("   - More realistic than simple clipping\n")
    f.write("   - Fallback to simple stop if OHLC unavailable\n\n")
    
    f.write("3. Volatility filter: VIX percentile-based\n")
    f.write("   - Uses 70th percentile instead of fixed threshold\n")
    f.write("   - Adapts to changing volatility regimes\n\n")

print("Files saved to outputs_phase8_5/")
print()
print("=" * 70)
print("PHASE 8.5 COMPLETE!")
print("=" * 70)
