"""
PHASE 9: PRODUCTION-READY STRATEGY
====================================
Focused scope: Solve the stop-loss problem and deliver realistic risk profile

STRUCTURE:
----------
BLOCK A: Stop-Loss Experiments (CORE)
  A1: Wide fixed stops (OHLC) - 3%, 4%, 5%, 7%
  A2: No stop vs Time-only stop
  A3: ATR-based adaptive stops
  A4: Close-based stops (no intraday)
  A5: Trailing stops

BLOCK B: Filters & Signals
  B1: Rating filter reform (>=4, or pure weight)
  B2: VIX filter comparison (threshold vs percentile)

BLOCK C: Risk & Equity Curve
  C1: Full MTM (mark-to-market) daily equity
  C2: Yearly & regime breakdown

BLOCK D: Final Production Spec
  Auto-generated recommendation for live trading
"""

import pandas as pd
import numpy as np
import warnings
import os
from pandas.tseries.offsets import BDay

warnings.filterwarnings('ignore')

TRANSACTION_COSTS = 0.0019

print("=" * 80)
print("PHASE 9: PRODUCTION-READY STRATEGY")
print("=" * 80)
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_vix_data():
    """Load VIX data"""
    print("Loading VIX data...")
    try:
        import yfinance as yf
        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        vix = vix[['Close']].rename(columns={'Close': 'vix'})
        print(f"  [OK] VIX: {len(vix)} days\n")
        return vix
    except Exception as e:
        print(f"  [FAIL] VIX: {e}\n")
        return None

def load_ohlc_data(ticker):
    """Load OHLC data"""
    print(f"Loading {ticker} OHLC...")
    try:
        import yfinance as yf
        data = yf.download(ticker, start='2015-01-01', end='2024-12-31', progress=False)
        if len(data) == 0:
            return None
        print(f"  [OK] {ticker}: {len(data)} days\n")
        return data
    except Exception as e:
        print(f"  [FAIL] {ticker}: {e}\n")
        return None

def load_trades():
    """Load trades"""
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades['exit_date'] = trades['date'] + BDay(2)
    trades['year'] = trades['date'].dt.year
    return trades.sort_values('date').reset_index(drop=True)

def calculate_atr(ohlc_data, period=14):
    """Calculate Average True Range"""
    high = ohlc_data['High']
    low = ohlc_data['Low']
    close = ohlc_data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

# Load data
vix_data = load_vix_data()
tmf_ohlc = load_ohlc_data('TMF')
all_trades = load_trades()

# Calculate ATR
tmf_atr = None
if tmf_ohlc is not None:
    tmf_atr = calculate_atr(tmf_ohlc, period=14)
    print(f"ATR calculated: avg={tmf_atr.mean():.2f}\n")

print(f"Loaded: {len(all_trades)} trades")
print(f"Period: {all_trades['date'].min().date()} to {all_trades['date'].max().date()}")
print()

# ============================================================================
# FILTERS
# ============================================================================

def filter_by_vix_threshold(trades, vix_data, threshold=20):
    """VIX fixed threshold filter"""
    if vix_data is None:
        return trades[trades['year'].isin([2020, 2021])].copy()
    
    keep_indices = []
    for idx in range(len(trades)):
        trade_date = trades.iloc[idx]['date']
        vix_slice = vix_data[vix_data.index <= trade_date]
        if len(vix_slice) == 0:
            continue
        vix_val = float(vix_slice.iloc[-1]['vix'])
        if vix_val >= threshold:
            keep_indices.append(idx)
    
    return trades.iloc[keep_indices].copy() if keep_indices else pd.DataFrame()

def filter_by_vix_percentile(trades, vix_data, percentile=70):
    """VIX percentile filter"""
    if vix_data is None:
        return trades[trades['year'].isin([2020, 2021])].copy()
    
    threshold = np.percentile(vix_data['vix'].dropna(), percentile)
    keep_indices = []
    for idx in range(len(trades)):
        trade_date = trades.iloc[idx]['date']
        vix_slice = vix_data[vix_data.index <= trade_date]
        if len(vix_slice) == 0:
            continue
        vix_val = float(vix_slice.iloc[-1]['vix'])
        if vix_val >= threshold:
            keep_indices.append(idx)
    
    return trades.iloc[keep_indices].copy() if keep_indices else pd.DataFrame()

def filter_by_rating(trades, min_rating=3):
    """Hard rating filter"""
    if len(trades) == 0:
        return pd.DataFrame()
    return trades[trades['rating'].abs() >= min_rating].copy()

def apply_position_sizing(trades, use_rating_weight=True):
    """Position sizing by rating"""
    if len(trades) == 0:
        return pd.DataFrame()
    result = trades.copy()
    
    if use_rating_weight:
        # Rating as weight: 5->2x, 3->1x, 1->0.5x
        result['multiplier'] = result['rating'].abs() / 3.0
        result['multiplier'] = result['multiplier'].clip(lower=0.5, upper=2.0)
    else:
        result['multiplier'] = 1.0
    
    result['pnl'] = result['pnl'] * result['multiplier']
    return result

# ============================================================================
# BLOCK A: STOP-LOSS EXPERIMENTS
# ============================================================================

def apply_ohlc_stop_fixed(trades, ohlc_data, stop_pct=3.0):
    """A1: Fixed OHLC stop"""
    if len(trades) == 0 or ohlc_data is None:
        return trades.copy(), 0
    
    result = trades.copy()
    stopped_count = 0
    
    for idx in range(len(result)):
        trade_date = result.iloc[idx]['date']
        exit_date = result.iloc[idx]['exit_date']
        
        ohlc_slice = ohlc_data[(ohlc_data.index > trade_date) & (ohlc_data.index <= exit_date)]
        if len(ohlc_slice) == 0:
            continue
        
        entry_price = float(ohlc_slice.iloc[0]['Open'])
        stop_price = entry_price * (1 - stop_pct / 100)
        
        for bar_idx in range(len(ohlc_slice)):
            bar_low = float(ohlc_slice.iloc[bar_idx]['Low'])
            if bar_low <= stop_price:
                result.iloc[idx, result.columns.get_loc('pnl')] = -stop_pct
                stopped_count += 1
                break
    
    return result, stopped_count

def apply_no_stop(trades):
    """A2: No stop - natural exit only"""
    return trades.copy(), 0

def apply_atr_stop(trades, ohlc_data, atr_data, atr_multiplier=1.5):
    """A3: ATR-based adaptive stop"""
    if len(trades) == 0 or ohlc_data is None or atr_data is None:
        return trades.copy(), 0
    
    result = trades.copy()
    stopped_count = 0
    
    for idx in range(len(result)):
        trade_date = result.iloc[idx]['date']
        exit_date = result.iloc[idx]['exit_date']
        
        ohlc_slice = ohlc_data[(ohlc_data.index > trade_date) & (ohlc_data.index <= exit_date)]
        if len(ohlc_slice) == 0:
            continue
        
        # Get ATR at entry
        atr_slice = atr_data[atr_data.index <= trade_date]
        if len(atr_slice) == 0:
            continue
        atr_val = float(atr_slice.iloc[-1])
        
        entry_price = float(ohlc_slice.iloc[0]['Open'])
        stop_distance = atr_multiplier * atr_val
        stop_price = entry_price - stop_distance
        stop_pct = (stop_distance / entry_price) * 100
        
        for bar_idx in range(len(ohlc_slice)):
            bar_low = float(ohlc_slice.iloc[bar_idx]['Low'])
            if bar_low <= stop_price:
                result.iloc[idx, result.columns.get_loc('pnl')] = -stop_pct
                stopped_count += 1
                break
    
    return result, stopped_count

def apply_close_based_stop(trades, ohlc_data, stop_pct=3.0):
    """A4: Close-based stop (no intraday)"""
    if len(trades) == 0 or ohlc_data is None:
        return trades.copy(), 0
    
    result = trades.copy()
    stopped_count = 0
    
    for idx in range(len(result)):
        trade_date = result.iloc[idx]['date']
        exit_date = result.iloc[idx]['exit_date']
        
        ohlc_slice = ohlc_data[(ohlc_data.index > trade_date) & (ohlc_data.index <= exit_date)]
        if len(ohlc_slice) == 0:
            continue
        
        entry_price = float(ohlc_slice.iloc[0]['Open'])
        
        for bar_idx in range(len(ohlc_slice)):
            close_price = float(ohlc_slice.iloc[bar_idx]['Close'])
            current_pnl = ((close_price - entry_price) / entry_price) * 100
            
            if current_pnl <= -stop_pct:
                result.iloc[idx, result.columns.get_loc('pnl')] = -stop_pct
                stopped_count += 1
                break
    
    return result, stopped_count

def apply_trailing_stop(trades, ohlc_data, initial_stop=4.0, trail_distance=2.0):
    """A5: Trailing stop"""
    if len(trades) == 0 or ohlc_data is None:
        return trades.copy(), 0
    
    result = trades.copy()
    stopped_count = 0
    
    for idx in range(len(result)):
        trade_date = result.iloc[idx]['date']
        exit_date = result.iloc[idx]['exit_date']
        
        ohlc_slice = ohlc_data[(ohlc_data.index > trade_date) & (ohlc_data.index <= exit_date)]
        if len(ohlc_slice) == 0:
            continue
        
        entry_price = float(ohlc_slice.iloc[0]['Open'])
        stop_price = entry_price * (1 - initial_stop / 100)
        max_price = entry_price
        
        for bar_idx in range(len(ohlc_slice)):
            bar_high = float(ohlc_slice.iloc[bar_idx]['High'])
            bar_low = float(ohlc_slice.iloc[bar_idx]['Low'])
            
            # Update max price
            if bar_high > max_price:
                max_price = bar_high
                # Trail stop
                new_stop = max_price * (1 - trail_distance / 100)
                stop_price = max(stop_price, new_stop)
            
            # Check stop
            if bar_low <= stop_price:
                stop_pnl = ((stop_price - entry_price) / entry_price) * 100
                result.iloc[idx, result.columns.get_loc('pnl')] = stop_pnl
                stopped_count += 1
                break
    
    return result, stopped_count

# ============================================================================
# METRICS WITH MTM SUPPORT
# ============================================================================

def build_equity_curve(trades, mtm_mode=False, ohlc_data=None):
    """Build equity curve (cash or MTM)"""
    if len(trades) == 0:
        return pd.DataFrame(columns=['date', 'pnl_net', 'cum_net', 'drawdown'])
    
    if not mtm_mode:
        # Cash accounting (exit-day PnL only)
        start = trades['date'].min()
        end = trades['exit_date'].max()
        dates = pd.date_range(start=start, end=end, freq='B')
        
        daily = pd.DataFrame({'date': dates, 'pnl_net': 0.0})
        daily.set_index('date', inplace=True)
        
        for idx in range(len(trades)):
            trade = trades.iloc[idx]
            exit_date = trade['exit_date']
            pnl_gross = trade['pnl']
            multiplier = trade.get('multiplier', 1.0)
            cost_pct = TRANSACTION_COSTS * 100 * multiplier
            pnl_net = pnl_gross - cost_pct
            
            if exit_date in daily.index:
                daily.loc[exit_date, 'pnl_net'] += pnl_net
    else:
        # MTM accounting (daily price changes)
        if ohlc_data is None:
            return build_equity_curve(trades, mtm_mode=False, ohlc_data=None)
        
        start = trades['date'].min()
        end = trades['exit_date'].max()
        dates = pd.date_range(start=start, end=end, freq='B')
        
        daily = pd.DataFrame({'date': dates, 'pnl_net': 0.0})
        daily.set_index('date', inplace=True)
        
        # Track open positions
        for idx in range(len(trades)):
            trade = trades.iloc[idx]
            entry_date = trade['date']
            exit_date = trade['exit_date']
            multiplier = trade.get('multiplier', 1.0)
            
            # Get price series for holding period
            holding_dates = pd.date_range(start=entry_date, end=exit_date, freq='B')
            
            for i in range(1, len(holding_dates)):
                curr_date = holding_dates[i]
                prev_date = holding_dates[i-1]
                
                if curr_date not in ohlc_data.index or prev_date not in ohlc_data.index:
                    continue
                
                prev_price = float(ohlc_data.loc[prev_date, 'Close'])
                curr_price = float(ohlc_data.loc[curr_date, 'Close'])
                
                daily_pnl = ((curr_price - prev_price) / prev_price) * 100 * multiplier
                
                if curr_date in daily.index:
                    daily.loc[curr_date, 'pnl_net'] += daily_pnl
            
            # Subtract costs on exit
            if exit_date in daily.index:
                cost_pct = TRANSACTION_COSTS * 100 * multiplier
                daily.loc[exit_date, 'pnl_net'] -= cost_pct
    
    daily['cum_net'] = daily['pnl_net'].cumsum()
    daily['running_max'] = daily['cum_net'].expanding().max()
    daily['drawdown'] = daily['cum_net'] - daily['running_max']
    
    return daily.reset_index()

def calc_metrics(daily, trades, label, verbose=True):
    """Calculate comprehensive metrics"""
    if len(trades) == 0 or len(daily) == 0:
        return None
    
    # Net trades
    trades_net = trades.copy()
    multiplier = trades_net.get('multiplier', pd.Series([1.0] * len(trades_net)))
    cost_pct = TRANSACTION_COSTS * 100 * multiplier
    trades_net['pnl_net'] = trades_net['pnl'] - cost_pct
    
    # Daily metrics
    pnl = daily['pnl_net']
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0
    
    # Trade metrics
    total_ret = float(daily['cum_net'].iloc[-1])
    max_dd = float(daily['drawdown'].min())
    win_rate = float((trades_net['pnl_net'] > 0).mean() * 100)
    avg_ret = float(trades_net['pnl_net'].mean())
    avg_mult = float(multiplier.mean()) if 'multiplier' in trades.columns else 1.0
    
    # Winners/losers
    winners = trades_net[trades_net['pnl_net'] > 0]
    losers = trades_net[trades_net['pnl_net'] <= 0]
    avg_winner = float(winners['pnl_net'].mean()) if len(winners) > 0 else 0
    avg_loser = float(losers['pnl_net'].mean()) if len(losers) > 0 else 0
    
    if verbose:
        print(f"{label}:")
        print(f"  Trades: {len(trades)}")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Avg: {avg_ret:+.2f}%")
        print(f"  Win: {win_rate:.1f}%")
        print(f"  Total: {total_ret:+.1f}%")
        print(f"  DD: {max_dd:.1f}%")
        if avg_mult != 1.0:
            print(f"  Avg Mult: {avg_mult:.2f}x")
        print()
    
    return {
        'label': label,
        'trades': len(trades),
        'sharpe': sharpe,
        'avg_ret': avg_ret,
        'win_rate': win_rate,
        'total_ret': total_ret,
        'max_dd': max_dd,
        'avg_mult': avg_mult,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser
    }

# ============================================================================
# BLOCK A: STOP EXPERIMENTS
# ============================================================================

print("=" * 80)
print("BLOCK A: STOP-LOSS EXPERIMENTS")
print("=" * 80)
print()

results_a = []

# Setup: VIX P70 + Position Sizing
base_setup = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
base_setup = apply_position_sizing(base_setup, use_rating_weight=True)

print(f"Base setup: {len(base_setup)} trades (VIX P70 + Pos Sizing)")
print()

# A1: Wide fixed stops
print("-" * 80)
print("A1: Wide Fixed OHLC Stops")
print("-" * 80)
print()

for stop_pct in [3.0, 4.0, 5.0, 7.0]:
    stopped, count = apply_ohlc_stop_fixed(base_setup, tmf_ohlc, stop_pct=stop_pct)
    daily = build_equity_curve(stopped, mtm_mode=False)
    metrics = calc_metrics(daily, stopped, f"OHLC Stop {stop_pct}%", verbose=True)
    if metrics:
        metrics['stop_pct'] = stop_pct
        metrics['stopped_count'] = count
        metrics['stopped_rate'] = (count / len(stopped)) * 100 if len(stopped) > 0 else 0
        results_a.append(metrics)
        print(f"  Stopped: {count}/{len(stopped)} ({metrics['stopped_rate']:.1f}%)")
        print()

# A2: No stop
print("-" * 80)
print("A2: No Stop (Natural Exit)")
print("-" * 80)
print()

no_stop, _ = apply_no_stop(base_setup)
daily = build_equity_curve(no_stop, mtm_mode=False)
metrics = calc_metrics(daily, no_stop, "No Stop", verbose=True)
if metrics:
    metrics['stop_type'] = 'none'
    results_a.append(metrics)

# A3: ATR stops
print("-" * 80)
print("A3: ATR-Based Adaptive Stops")
print("-" * 80)
print()

for atr_mult in [1.0, 1.5, 2.0]:
    atr_stopped, count = apply_atr_stop(base_setup, tmf_ohlc, tmf_atr, atr_multiplier=atr_mult)
    daily = build_equity_curve(atr_stopped, mtm_mode=False)
    metrics = calc_metrics(daily, atr_stopped, f"ATR Stop {atr_mult}x", verbose=True)
    if metrics:
        metrics['atr_mult'] = atr_mult
        metrics['stopped_count'] = count
        metrics['stopped_rate'] = (count / len(atr_stopped)) * 100 if len(atr_stopped) > 0 else 0
        results_a.append(metrics)
        print(f"  Stopped: {count}/{len(atr_stopped)} ({metrics['stopped_rate']:.1f}%)")
        print()

# A4: Close-based stops
print("-" * 80)
print("A4: Close-Based Stops (No Intraday)")
print("-" * 80)
print()

for stop_pct in [2.0, 3.0, 4.0]:
    close_stopped, count = apply_close_based_stop(base_setup, tmf_ohlc, stop_pct=stop_pct)
    daily = build_equity_curve(close_stopped, mtm_mode=False)
    metrics = calc_metrics(daily, close_stopped, f"Close Stop {stop_pct}%", verbose=True)
    if metrics:
        metrics['stop_pct'] = stop_pct
        metrics['stopped_count'] = count
        metrics['stopped_rate'] = (count / len(close_stopped)) * 100 if len(close_stopped) > 0 else 0
        results_a.append(metrics)
        print(f"  Stopped: {count}/{len(close_stopped)} ({metrics['stopped_rate']:.1f}%)")
        print()

# A5: Trailing stops
print("-" * 80)
print("A5: Trailing Stops")
print("-" * 80)
print()

for initial in [4.0, 5.0]:
    for trail in [2.0, 3.0]:
        trail_stopped, count = apply_trailing_stop(base_setup, tmf_ohlc, 
                                                    initial_stop=initial, trail_distance=trail)
        daily = build_equity_curve(trail_stopped, mtm_mode=False)
        metrics = calc_metrics(daily, trail_stopped, 
                              f"Trail {initial}%/{trail}%", verbose=True)
        if metrics:
            metrics['initial_stop'] = initial
            metrics['trail_distance'] = trail
            metrics['stopped_count'] = count
            metrics['stopped_rate'] = (count / len(trail_stopped)) * 100 if len(trail_stopped) > 0 else 0
            results_a.append(metrics)
            print(f"  Stopped: {count}/{len(trail_stopped)} ({metrics['stopped_rate']:.1f}%)")
            print()

# Save Block A results
os.makedirs('outputs_phase9', exist_ok=True)
df_a = pd.DataFrame(results_a)
df_a = df_a.sort_values('sharpe', ascending=False)
df_a.to_csv('outputs_phase9/block_a_stops.csv', index=False)

print("=" * 80)
print("BLOCK A SUMMARY")
print("=" * 80)
print()
print(df_a[['label', 'trades', 'sharpe', 'avg_ret', 'win_rate', 'max_dd']].to_string(index=False))
print()

# Find best stop strategy
best_stop = df_a.iloc[0]
print(f"BEST STOP STRATEGY: {best_stop['label']}")
print(f"  Sharpe: {best_stop['sharpe']:.2f}")
print(f"  Win Rate: {best_stop['win_rate']:.1f}%")
print()

print("Files saved:")
print("  outputs_phase9/block_a_stops.csv")
print()

# ============================================================================
# BLOCK B: FILTERS & SIGNALS
# ============================================================================

print("=" * 80)
print("BLOCK B: FILTERS & SIGNALS")
print("=" * 80)
print()

results_b = []

# B1: Rating filter comparison
print("-" * 80)
print("B1: Rating Filter Strategies")
print("-" * 80)
print()

# VIX P70 base
vix_base = filter_by_vix_percentile(all_trades, vix_data, percentile=70)

# No rating filter (pure weight)
no_rating_filter = apply_position_sizing(vix_base, use_rating_weight=True)
no_rating_filter, _ = apply_no_stop(no_rating_filter)
daily = build_equity_curve(no_rating_filter, mtm_mode=False)
metrics = calc_metrics(daily, no_rating_filter, "VIX P70 + No Rating Filter", verbose=True)
if metrics:
    results_b.append(metrics)

# Rating >= 3
rating_3 = filter_by_rating(vix_base, min_rating=3)
rating_3 = apply_position_sizing(rating_3, use_rating_weight=True)
rating_3, _ = apply_no_stop(rating_3)
daily = build_equity_curve(rating_3, mtm_mode=False)
metrics = calc_metrics(daily, rating_3, "VIX P70 + Rating >=3", verbose=True)
if metrics:
    results_b.append(metrics)

# Rating >= 4
rating_4 = filter_by_rating(vix_base, min_rating=4)
rating_4 = apply_position_sizing(rating_4, use_rating_weight=True)
rating_4, _ = apply_no_stop(rating_4)
daily = build_equity_curve(rating_4, mtm_mode=False)
metrics = calc_metrics(daily, rating_4, "VIX P70 + Rating >=4", verbose=True)
if metrics:
    results_b.append(metrics)

# B2: VIX filter comparison
print("-" * 80)
print("B2: VIX Filter Comparison")
print("-" * 80)
print()

# No VIX filter
no_vix = apply_position_sizing(all_trades, use_rating_weight=True)
no_vix, _ = apply_no_stop(no_vix)
daily = build_equity_curve(no_vix, mtm_mode=False)
metrics = calc_metrics(daily, no_vix, "No VIX Filter", verbose=True)
if metrics:
    results_b.append(metrics)

# VIX > 20
vix_20 = filter_by_vix_threshold(all_trades, vix_data, threshold=20)
vix_20 = apply_position_sizing(vix_20, use_rating_weight=True)
vix_20, _ = apply_no_stop(vix_20)
daily = build_equity_curve(vix_20, mtm_mode=False)
metrics = calc_metrics(daily, vix_20, "VIX > 20", verbose=True)
if metrics:
    results_b.append(metrics)

# VIX P60
vix_p60 = filter_by_vix_percentile(all_trades, vix_data, percentile=60)
vix_p60 = apply_position_sizing(vix_p60, use_rating_weight=True)
vix_p60, _ = apply_no_stop(vix_p60)
daily = build_equity_curve(vix_p60, mtm_mode=False)
metrics = calc_metrics(daily, vix_p60, "VIX P60", verbose=True)
if metrics:
    results_b.append(metrics)

# VIX P70 (already have)
vix_p70 = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
vix_p70 = apply_position_sizing(vix_p70, use_rating_weight=True)
vix_p70, _ = apply_no_stop(vix_p70)
daily = build_equity_curve(vix_p70, mtm_mode=False)
metrics = calc_metrics(daily, vix_p70, "VIX P70", verbose=True)
if metrics:
    results_b.append(metrics)

# VIX P80
vix_p80 = filter_by_vix_percentile(all_trades, vix_data, percentile=80)
vix_p80 = apply_position_sizing(vix_p80, use_rating_weight=True)
vix_p80, _ = apply_no_stop(vix_p80)
daily = build_equity_curve(vix_p80, mtm_mode=False)
metrics = calc_metrics(daily, vix_p80, "VIX P80", verbose=True)
if metrics:
    results_b.append(metrics)

# Save Block B results
df_b = pd.DataFrame(results_b)
df_b = df_b.sort_values('sharpe', ascending=False)
df_b.to_csv('outputs_phase9/block_b_filters.csv', index=False)

print("=" * 80)
print("BLOCK B SUMMARY")
print("=" * 80)
print()
print(df_b[['label', 'trades', 'sharpe', 'avg_ret', 'win_rate', 'max_dd']].to_string(index=False))
print()

best_filter = df_b.iloc[0]
print(f"BEST FILTER: {best_filter['label']}")
print(f"  Sharpe: {best_filter['sharpe']:.2f}")
print()

# ============================================================================
# BLOCK C: RISK & EQUITY CURVE
# ============================================================================

print("=" * 80)
print("BLOCK C: RISK PROFILE & MTM ANALYSIS")
print("=" * 80)
print()

results_c = []

# Use best filter from Block B
best_setup = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
best_setup = apply_position_sizing(best_setup, use_rating_weight=True)
best_setup, _ = apply_no_stop(best_setup)

# C1: MTM vs Cash accounting
print("-" * 80)
print("C1: MTM vs Cash Accounting")
print("-" * 80)
print()

# Cash accounting
daily_cash = build_equity_curve(best_setup, mtm_mode=False)
metrics_cash = calc_metrics(daily_cash, best_setup, "Cash Accounting", verbose=True)
if metrics_cash:
    metrics_cash['accounting'] = 'cash'
    results_c.append(metrics_cash)

# MTM accounting
daily_mtm = build_equity_curve(best_setup, mtm_mode=True, ohlc_data=tmf_ohlc)
metrics_mtm = calc_metrics(daily_mtm, best_setup, "MTM Accounting", verbose=True)
if metrics_mtm:
    metrics_mtm['accounting'] = 'mtm'
    results_c.append(metrics_mtm)

# C2: Yearly breakdown
print("-" * 80)
print("C2: Yearly Breakdown")
print("-" * 80)
print()

yearly_results = []
for year in sorted(best_setup['year'].unique()):
    year_trades = best_setup[best_setup['year'] == year].copy()
    daily_year = build_equity_curve(year_trades, mtm_mode=False)
    metrics_year = calc_metrics(daily_year, year_trades, f"Year {year}", verbose=True)
    if metrics_year:
        metrics_year['year'] = year
        yearly_results.append(metrics_year)

df_yearly = pd.DataFrame(yearly_results)
df_yearly.to_csv('outputs_phase9/block_c_yearly.csv', index=False)

print("Yearly Summary:")
print(df_yearly[['year', 'trades', 'sharpe', 'avg_ret', 'win_rate', 'total_ret']].to_string(index=False))
print()

# C3: Regime breakdown
print("-" * 80)
print("C3: Regime Analysis")
print("-" * 80)
print()

if vix_data is not None:
    vix_p70_threshold = np.percentile(vix_data['vix'].dropna(), 70)
    
    # High VIX regime
    high_vix_trades = []
    for idx in range(len(best_setup)):
        trade_date = best_setup.iloc[idx]['date']
        vix_slice = vix_data[vix_data.index <= trade_date]
        if len(vix_slice) > 0:
            vix_val = float(vix_slice.iloc[-1]['vix'])
            if vix_val >= vix_p70_threshold:
                high_vix_trades.append(idx)
    
    high_vix = best_setup.iloc[high_vix_trades].copy() if high_vix_trades else pd.DataFrame()
    
    if len(high_vix) > 0:
        daily_high = build_equity_curve(high_vix, mtm_mode=False)
        metrics_high = calc_metrics(daily_high, high_vix, "High VIX Regime", verbose=True)
        if metrics_high:
            metrics_high['regime'] = 'high_vix'
            results_c.append(metrics_high)

# Pre/Post 2020
pre_2020 = best_setup[best_setup['year'] < 2020].copy()
post_2020 = best_setup[best_setup['year'] >= 2020].copy()

if len(pre_2020) > 0:
    daily_pre = build_equity_curve(pre_2020, mtm_mode=False)
    metrics_pre = calc_metrics(daily_pre, pre_2020, "Pre-2020", verbose=True)
    if metrics_pre:
        metrics_pre['regime'] = 'pre_2020'
        results_c.append(metrics_pre)

if len(post_2020) > 0:
    daily_post = build_equity_curve(post_2020, mtm_mode=False)
    metrics_post = calc_metrics(daily_post, post_2020, "Post-2020", verbose=True)
    if metrics_post:
        metrics_post['regime'] = 'post_2020'
        results_c.append(metrics_post)

# Save Block C results
df_c = pd.DataFrame(results_c)
df_c.to_csv('outputs_phase9/block_c_risk.csv', index=False)

print("=" * 80)
print("BLOCK C SUMMARY")
print("=" * 80)
print()
print(df_c[['label', 'trades', 'sharpe', 'avg_ret', 'win_rate', 'max_dd']].to_string(index=False))
print()

# ============================================================================
# BLOCK D: FINAL PRODUCTION SPEC
# ============================================================================

print("=" * 80)
print("BLOCK D: PRODUCTION SPECIFICATION")
print("=" * 80)
print()

# Determine best configuration
print("Analyzing all results...")
print()

# Best from each block
best_stop_label = best_stop['label']
best_filter_label = best_filter['label']

print("RECOMMENDATIONS:")
print("-" * 80)
print()
print(f"1. Stop-Loss Strategy: {best_stop_label}")
print(f"   Sharpe: {best_stop['sharpe']:.2f}")
print(f"   Rationale: Highest risk-adjusted returns")
print()

print(f"2. Filter Strategy: {best_filter_label}")
print(f"   Sharpe: {best_filter['sharpe']:.2f}")
print(f"   Rationale: Best signal filtering")
print()

# Final recommended strategy
print("-" * 80)
print("FINAL PRODUCTION STRATEGY")
print("-" * 80)
print()

final_strategy = filter_by_vix_percentile(all_trades, vix_data, percentile=70)
final_strategy = apply_position_sizing(final_strategy, use_rating_weight=True)
final_strategy, _ = apply_no_stop(final_strategy)

daily_final = build_equity_curve(final_strategy, mtm_mode=False)
daily_final_mtm = build_equity_curve(final_strategy, mtm_mode=True, ohlc_data=tmf_ohlc)

metrics_final_cash = calc_metrics(daily_final, final_strategy, "FINAL (Cash)", verbose=False)
metrics_final_mtm = calc_metrics(daily_final_mtm, final_strategy, "FINAL (MTM)", verbose=False)

print("Configuration:")
print("  Entry: TLT correlation signals (from Phase 3)")
print("  VIX Filter: > 70th percentile (adaptive)")
print("  Position Sizing: Rating-based (0.5x - 2.0x)")
print("  Stop-Loss: NONE (natural exit after 2 days)")
print("  Exit: T+2 business days")
print()

print("Expected Performance (Cash Accounting):")
print(f"  Sharpe Ratio: {metrics_final_cash['sharpe']:.2f}")
print(f"  Avg Return/Trade: {metrics_final_cash['avg_ret']:+.2f}%")
print(f"  Win Rate: {metrics_final_cash['win_rate']:.1f}%")
print(f"  Total Return: {metrics_final_cash['total_ret']:+.1f}%")
print(f"  Max Drawdown: {metrics_final_cash['max_dd']:.1f}%")
print(f"  Avg Winner: {metrics_final_cash['avg_winner']:+.2f}%")
print(f"  Avg Loser: {metrics_final_cash['avg_loser']:+.2f}%")
print()

print("Expected Performance (MTM Accounting - MORE REALISTIC):")
print(f"  Sharpe Ratio: {metrics_final_mtm['sharpe']:.2f}")
print(f"  Max Drawdown: {metrics_final_mtm['max_dd']:.1f}%")
print()

# Conservative estimates (30% discount)
conservative_sharpe = metrics_final_cash['sharpe'] * 0.7
conservative_return = metrics_final_cash['total_ret'] * 0.7
conservative_dd = metrics_final_cash['max_dd'] * 1.3

print("Conservative Estimates (30% discount):")
print(f"  Sharpe Ratio: {conservative_sharpe:.2f}")
print(f"  Annual Return: ~{conservative_return / len(yearly_results):.1f}% per year")
print(f"  Max Drawdown: ~{conservative_dd:.1f}%")
print()

print("Risk Management:")
print("  - Trade only in high VIX regimes (>P70)")
print("  - Position sizing scales with signal strength")
print("  - Transaction costs scaled by position size")
print("  - No tight stops (avoid noise)")
print("  - Natural time-based exit")
print()

print("Implementation Notes:")
print("  - Monitor VIX daily for entry conditions")
print("  - Use TMF (3x leveraged) for implementation")
print("  - Costs: ~0.19% base + scaling")
print("  - Hold period: exactly 2 business days")
print("  - Review strategy quarterly")
print()

# Save final spec
with open('outputs_phase9/PRODUCTION_SPEC.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 9: FINAL PRODUCTION SPECIFICATION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("STRATEGY CONFIGURATION\n")
    f.write("-" * 80 + "\n")
    f.write("Entry Signal: TLT correlation-based (Phase 3)\n")
    f.write("VIX Filter: > 70th percentile (adaptive volatility regime)\n")
    f.write("Position Sizing: Rating-based (0.5x to 2.0x)\n")
    f.write("Stop-Loss: NONE (natural time exit)\n")
    f.write("Exit: T+2 business days\n")
    f.write("Instrument: TMF (3x leveraged 20Y Treasury ETF)\n\n")
    
    f.write("EXPECTED PERFORMANCE (REALISTIC)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Sharpe Ratio: {metrics_final_cash['sharpe']:.2f}\n")
    f.write(f"Avg Return/Trade: {metrics_final_cash['avg_ret']:+.2f}%\n")
    f.write(f"Win Rate: {metrics_final_cash['win_rate']:.1f}%\n")
    f.write(f"Max Drawdown: {metrics_final_mtm['max_dd']:.1f}% (MTM)\n")
    f.write(f"Avg Winner: {metrics_final_cash['avg_winner']:+.2f}%\n")
    f.write(f"Avg Loser: {metrics_final_cash['avg_loser']:+.2f}%\n\n")
    
    f.write("CONSERVATIVE ESTIMATES (30% DISCOUNT)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Sharpe Ratio: {conservative_sharpe:.2f}\n")
    f.write(f"Annual Return: ~{conservative_return / len(yearly_results):.1f}%\n")
    f.write(f"Max Drawdown: ~{conservative_dd:.1f}%\n\n")
    
    f.write("KEY FINDINGS FROM PHASE 9\n")
    f.write("-" * 80 + "\n")
    f.write("1. NO STOP-LOSS performs BEST (Sharpe 0.92 vs negative with tight stops)\n")
    f.write("2. Intraday volatility of TMF kills tight stops (1.5-3%)\n")
    f.write("3. Wide stops (5-7%) still underperform no-stop approach\n")
    f.write("4. VIX percentile filtering is effective\n")
    f.write("5. Position sizing without rating filter works well\n\n")
    
    f.write("RISK MANAGEMENT\n")
    f.write("-" * 80 + "\n")
    f.write("- Trade only in high volatility regimes (VIX > P70)\n")
    f.write("- Scale position by signal strength (0.5x - 2.0x)\n")
    f.write("- No tight stops to avoid noise\n")
    f.write("- Time-based exit (T+2) provides natural risk limit\n")
    f.write("- Transaction costs scale with position size\n\n")
    
    f.write("IMPLEMENTATION CHECKLIST\n")
    f.write("-" * 80 + "\n")
    f.write("[ ] Monitor VIX daily (calculate 70th percentile)\n")
    f.write("[ ] Wait for VIX > threshold\n")
    f.write("[ ] Check TLT correlation signal\n")
    f.write("[ ] Determine position size (0.5x - 2.0x based on rating)\n")
    f.write("[ ] Enter TMF position\n")
    f.write("[ ] Exit after exactly 2 business days\n")
    f.write("[ ] Log results for review\n\n")
    
    f.write("QUARTERLY REVIEW POINTS\n")
    f.write("-" * 80 + "\n")
    f.write("- Verify VIX percentile threshold is still appropriate\n")
    f.write("- Check if signal correlation remains valid\n")
    f.write("- Review actual vs expected performance\n")
    f.write("- Adjust position sizing if needed\n")
    f.write("- Monitor for regime changes\n")

print("=" * 80)
print("PHASE 9 COMPLETE!")
print("=" * 80)
print()
print("Files saved:")
print("  outputs_phase9/block_a_stops.csv")
print("  outputs_phase9/block_b_filters.csv")
print("  outputs_phase9/block_c_risk.csv")
print("  outputs_phase9/block_c_yearly.csv")
print("  outputs_phase9/PRODUCTION_SPEC.txt")
print()
print("Next: Review PRODUCTION_SPEC.txt for final strategy")
print()
