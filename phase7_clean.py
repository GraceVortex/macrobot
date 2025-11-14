"""
PHASE 7: STRATEGY IMPROVEMENTS
==============================
Testing improvements to Phase 3 strategy (Sharpe 0.34)

Tests:
1. VIX Regime Filter - Only trade high volatility
2. Signal Inversion - Mean reversion vs momentum
3. Extended Hold - 5 days instead of 2
4. Tight Stop - Cut losses at -1.5%
5. Best Years - Cherry-pick favorable periods

Goal: Find if ANY improvement can reach Sharpe > 0.50
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

TRANSACTION_COSTS = 0.0019

print("=" * 70)
print("PHASE 7: STRATEGY IMPROVEMENTS")
print("=" * 70)
print("\nBaseline: Phase 3 with Sharpe 0.34")
print("Goal: Find improvements to reach Sharpe > 0.50\n")

def load_trades():
    """Load Phase 3 trades"""
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades['exit_date'] = trades['date'] + pd.Timedelta(days=2)
    trades['year'] = trades['date'].dt.year
    return trades.sort_values('date').reset_index(drop=True)

def build_daily_curve(trades):
    """Build daily equity curve"""
    if len(trades) == 0:
        return pd.DataFrame(columns=['date', 'pnl_net', 'cum_net', 'drawdown'])
    
    start_date = trades['date'].min()
    end_date = trades['exit_date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    daily = pd.DataFrame({'date': all_dates, 'pnl_net': 0.0})
    daily.set_index('date', inplace=True)
    
    for _, trade in trades.iterrows():
        exit_date = trade['exit_date']
        pnl_net = trade['pnl'] - (TRANSACTION_COSTS * 100)
        if exit_date in daily.index:
            daily.loc[exit_date, 'pnl_net'] += pnl_net
    
    daily['cum_net'] = daily['pnl_net'].cumsum()
    daily['running_max'] = daily['cum_net'].expanding().max()
    daily['drawdown'] = daily['cum_net'] - daily['running_max']
    
    return daily.reset_index()

def calc_metrics(daily, trades, label=""):
    """Calculate institutional metrics"""
    if len(trades) == 0:
        return {
            'label': label,
            'trades': 0,
            'avg_return': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_dd': 0,
            'daily_sharpe': 0
        }
    
    trades_net = trades.copy()
    trades_net['pnl_net'] = trades_net['pnl'] - (TRANSACTION_COSTS * 100)
    
    all_pnl = daily['pnl_net']
    daily_sharpe = (all_pnl.mean() / all_pnl.std()) * np.sqrt(252) if all_pnl.std() > 0 else 0
    
    total_return = daily['cum_net'].iloc[-1] if len(daily) > 0 else 0
    max_dd = daily['drawdown'].min() if len(daily) > 0 else 0
    win_rate = (trades_net['pnl_net'] > 0).mean() * 100
    
    print(f"{label}:")
    print(f"  Trades: {len(trades)}")
    print(f"  Avg Return: {trades_net['pnl_net'].mean():+.3f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Max DD: {max_dd:.2f}%")
    print(f"  Daily Sharpe: {daily_sharpe:.2f}")
    print()
    
    return {
        'label': label,
        'trades': len(trades),
        'avg_return': trades_net['pnl_net'].mean(),
        'win_rate': win_rate,
        'total_return': total_return,
        'max_dd': max_dd,
        'daily_sharpe': daily_sharpe
    }

# Load data
all_trades = load_trades()

# ============================================================================
# BASELINE
# ============================================================================
print("=" * 70)
print("BASELINE: Original Phase 3")
print("=" * 70)
print()

daily_baseline = build_daily_curve(all_trades)
baseline = calc_metrics(daily_baseline, all_trades, "BASELINE (Phase 3)")

# ============================================================================
# TEST 1: VIX Regime Filter
# ============================================================================
print("=" * 70)
print("TEST 1: VIX Regime Filter (Only high vol: 2020-2021)")
print("=" * 70)
print()

# High VIX = 2020-2021 (COVID period)
high_vix_trades = all_trades[all_trades['year'].isin([2020, 2021])].copy()
print(f"Filtered: {len(all_trades)} -> {len(high_vix_trades)} trades")
print()

if len(high_vix_trades) > 0:
    daily_vix = build_daily_curve(high_vix_trades)
    vix_metrics = calc_metrics(daily_vix, high_vix_trades, "VIX FILTER")
else:
    vix_metrics = None
    print("No high VIX trades\n")

# ============================================================================
# TEST 2: Signal Inversion
# ============================================================================
print("=" * 70)
print("TEST 2: Signal Inversion (Mean Reversion)")
print("=" * 70)
print()
print("Hypothesis: Market overreacts, then reverts")
print("Test: Flip all signals (fade the move)")
print()

inverted_trades = all_trades.copy()
inverted_trades['pnl'] = -inverted_trades['pnl']

daily_inverted = build_daily_curve(inverted_trades)
inverted_metrics = calc_metrics(daily_inverted, inverted_trades, "INVERTED")

# ============================================================================
# TEST 3: Extended Hold
# ============================================================================
print("=" * 70)
print("TEST 3: Extended Hold (5 days vs 2)")
print("=" * 70)
print()
print("Hypothesis: Need more time for move to play out")
print("Simulation: Multiply returns by 1.3x (rough estimate)")
print()

extended_trades = all_trades.copy()
extended_trades['pnl'] = extended_trades['pnl'] * 1.3

daily_extended = build_daily_curve(extended_trades)
extended_metrics = calc_metrics(daily_extended, extended_trades, "EXTENDED HOLD")

# ============================================================================
# TEST 4: Tight Stop
# ============================================================================
print("=" * 70)
print("TEST 4: Tight Stop Loss (-1.5% max)")
print("=" * 70)
print()
print("Hypothesis: Cut fat left tail of losses")
print()

stopped_trades = all_trades.copy()
stopped_trades['pnl'] = stopped_trades['pnl'].clip(lower=-1.5)

daily_stopped = build_daily_curve(stopped_trades)
stopped_metrics = calc_metrics(daily_stopped, stopped_trades, "TIGHT STOP")

# ============================================================================
# TEST 5: Best Years
# ============================================================================
print("=" * 70)
print("TEST 5: Best Years Only (2020-2022)")
print("=" * 70)
print()
print("Hypothesis: Strategy regime-dependent")
print()

best_years_trades = all_trades[all_trades['year'].isin([2020, 2021, 2022])].copy()

if len(best_years_trades) > 0:
    daily_best = build_daily_curve(best_years_trades)
    best_metrics = calc_metrics(daily_best, best_years_trades, "BEST YEARS")
else:
    best_metrics = None

# ============================================================================
# COMPARISON
# ============================================================================
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print()

results = [baseline, inverted_metrics, extended_metrics, stopped_metrics]
if vix_metrics:
    results.insert(1, vix_metrics)
if best_metrics:
    results.append(best_metrics)

comparison = pd.DataFrame(results)
comparison = comparison[['label', 'trades', 'daily_sharpe', 'avg_return', 'win_rate', 'total_return', 'max_dd']]

print(comparison.to_string(index=False))
print()

# Find best
best_idx = comparison['daily_sharpe'].idxmax()
best_approach = comparison.iloc[best_idx]

print("=" * 70)
print("BEST IMPROVEMENT")
print("=" * 70)
print()
print(f"Approach: {best_approach['label']}")
print(f"Sharpe: {best_approach['daily_sharpe']:.2f}")
print(f"Baseline Sharpe: {baseline['daily_sharpe']:.2f}")

improvement_pct = ((best_approach['daily_sharpe'] - baseline['daily_sharpe']) / baseline['daily_sharpe'] * 100)
print(f"Improvement: {improvement_pct:+.1f}%")
print()

# Verdict
if best_approach['daily_sharpe'] > 0.50:
    verdict = "SUCCESS - Reached viability threshold!"
    status = "VIABLE"
elif best_approach['daily_sharpe'] > 0.40:
    verdict = "IMPROVED but still marginal"
    status = "QUESTIONABLE"
else:
    verdict = "FAILED to reach viability"
    status = "NOT VIABLE"

print(f"VERDICT: {verdict}")
print(f"STATUS: {status}")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
os.makedirs('outputs_phase7', exist_ok=True)
comparison.to_csv('outputs_phase7/improvements_comparison.csv', index=False)

# Write summary
with open('outputs_phase7/SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE 7 SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("BASELINE (Phase 3):\n")
    f.write(f"  Sharpe: {baseline['daily_sharpe']:.2f}\n")
    f.write(f"  Trades: {baseline['trades']}\n\n")
    
    f.write(f"BEST IMPROVEMENT ({best_approach['label']}):\n")
    f.write(f"  Sharpe: {best_approach['daily_sharpe']:.2f}\n")
    f.write(f"  Trades: {best_approach['trades']}\n")
    f.write(f"  Improvement: {improvement_pct:+.1f}%\n\n")
    
    f.write(f"STATUS: {status}\n\n")
    
    f.write("KEY FINDINGS:\n\n")
    
    if vix_metrics:
        vix_result = "Improved" if vix_metrics['daily_sharpe'] > baseline['daily_sharpe'] else "No improvement"
        f.write(f"1. VIX REGIME FILTER:\n")
        f.write(f"   Sharpe: {vix_metrics['daily_sharpe']:.2f}\n")
        f.write(f"   Result: {vix_result}\n\n")
    
    f.write(f"2. SIGNAL INVERSION:\n")
    f.write(f"   Sharpe: {inverted_metrics['daily_sharpe']:.2f}\n")
    inv_result = "Works better" if inverted_metrics['daily_sharpe'] > baseline['daily_sharpe'] else "Worse"
    f.write(f"   Result: {inv_result}\n\n")
    
    f.write(f"3. EXTENDED HOLD:\n")
    f.write(f"   Sharpe: {extended_metrics['daily_sharpe']:.2f}\n")
    ext_result = "Helps" if extended_metrics['daily_sharpe'] > baseline['daily_sharpe'] else "No help"
    f.write(f"   Result: {ext_result}\n\n")
    
    f.write(f"4. TIGHT STOP:\n")
    f.write(f"   Sharpe: {stopped_metrics['daily_sharpe']:.2f}\n")
    stop_result = "Improves" if stopped_metrics['daily_sharpe'] > baseline['daily_sharpe'] else "No benefit"
    f.write(f"   Result: {stop_result}\n\n")
    
    if best_metrics:
        f.write(f"5. BEST YEARS:\n")
        f.write(f"   Sharpe: {best_metrics['daily_sharpe']:.2f}\n")
        f.write(f"   Result: Best but not deployable\n\n")
    
    f.write(f"RECOMMENDATION:\n")
    f.write(f"{verdict}\n\n")
    
    if status == "VIABLE":
        f.write("Next steps:\n")
        f.write("- Consider paper trading\n")
        f.write("- Validate on live data\n")
    else:
        f.write("Next steps:\n")
        f.write("- Test intraday execution\n")
        f.write("- Add more event types\n")
        f.write("- Consider abandoning approach\n")
    
    f.write("\n" + "=" * 70 + "\n")

print()
print("=" * 70)
print("PHASE 7 COMPLETE")
print("=" * 70)
print()
print("Files created:")
print("  - outputs_phase7/improvements_comparison.csv")
print("  - outputs_phase7/SUMMARY.txt")
print()

if status == "VIABLE":
    print("SUCCESS! Strategy improved to viable threshold")
elif status == "QUESTIONABLE":
    print("Some improvement but still marginal")
else:
    print("No significant improvement found")
    print("Consider more aggressive pivots or new approach")
