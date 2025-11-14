"""
CRITICAL TEST: Phase 6.7 on FULL 2015-2024 PERIOD
=================================================
Testing if Phase 6.7 methodology works on ALL years
or just got lucky with 2020-2024
"""

import pandas as pd
import numpy as np
import os

TRANSACTION_COSTS = 0.0019

print("""
================================================================================
       CRITICAL TEST: Phase 6.7 on FULL PERIOD (2015-2024)
================================================================================

Testing Phase 6.7 institutional Sharpe on ALL Phase 3 trades.
This will show if 2020-2024 was cherry-picked or strategy is robust.
""")

def load_all_trades():
    """Load ALL Phase 3 trades"""
    print("Loading Phase 3 trades...")
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades.sort_values('date').reset_index(drop=True)
    trades['exit_date'] = trades['date'] + pd.Timedelta(days=2)
    
    print(f"  Period: {trades['date'].min().date()} to {trades['date'].max().date()}")
    print(f"  Total: {len(trades)} trades\n")
    return trades

def build_daily_curve(trades, label=""):
    """Build daily equity curve"""
    start_date = trades['date'].min()
    end_date = trades['exit_date'].max()
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    daily = pd.DataFrame({
        'date': all_dates,
        'pnl_gross': 0.0,
        'pnl_net': 0.0
    })
    daily.set_index('date', inplace=True)
    
    for _, trade in trades.iterrows():
        exit_date = trade['exit_date']
        pnl_gross = trade['pnl']
        pnl_net = pnl_gross - (TRANSACTION_COSTS * 100)
        
        if exit_date in daily.index:
            daily.loc[exit_date, 'pnl_gross'] += pnl_gross
            daily.loc[exit_date, 'pnl_net'] += pnl_net
    
    daily['cum_net'] = daily['pnl_net'].cumsum()
    daily['running_max'] = daily['cum_net'].expanding().max()
    daily['drawdown'] = daily['cum_net'] - daily['running_max']
    
    return daily.reset_index()

def calc_daily_sharpe(daily, trades, label=""):
    """Calculate institutional metrics"""
    trades_net = trades.copy()
    trades_net['pnl_net'] = trades_net['pnl'] - (TRANSACTION_COSTS * 100)
    
    # Daily Sharpe
    all_pnl = daily['pnl_net']
    if all_pnl.std() > 0:
        daily_sharpe = (all_pnl.mean() / all_pnl.std()) * np.sqrt(252)
    else:
        daily_sharpe = 0
    
    # Per-trade Sharpe (old method)
    if trades_net['pnl_net'].std() > 0:
        old_sharpe = (trades_net['pnl_net'].mean() / trades_net['pnl_net'].std()) * np.sqrt(252/2)
    else:
        old_sharpe = 0
    
    total_return = daily['cum_net'].iloc[-1]
    max_dd = daily['drawdown'].min()
    
    print(f"{label}:")
    print(f"  Trades: {len(trades)}")
    print(f"  Avg Return (net): {trades_net['pnl_net'].mean():+.3f}%")
    print(f"  Win Rate: {(trades_net['pnl_net'] > 0).mean() * 100:.1f}%")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Max DD: {max_dd:.2f}%")
    print(f"  Daily Sharpe: {daily_sharpe:.2f} ← INSTITUTIONAL")
    print(f"  Old Sharpe (Per-Trade): {old_sharpe:.2f}")
    print()
    
    return {
        'trades': len(trades),
        'daily_sharpe': daily_sharpe,
        'old_sharpe': old_sharpe,
        'total_return': total_return,
        'max_dd': max_dd,
        'win_rate': (trades_net['pnl_net'] > 0).mean() * 100
    }

# Load all trades
all_trades = load_all_trades()

# Test 1: FULL PERIOD
print(f"{'='*70}")
print("TEST 1: FULL PERIOD (2015-2024)")
print(f"{'='*70}\n")

daily_full = build_daily_curve(all_trades)
metrics_full = calc_daily_sharpe(daily_full, all_trades, "FULL PERIOD (2015-2024)")

# Test 2: 2020-2024 ONLY (Phase 6.7 original)
print(f"{'='*70}")
print("TEST 2: 2020-2024 ONLY (Phase 6.7 Original)")
print(f"{'='*70}\n")

trades_2020 = all_trades[all_trades['date'] >= '2020-01-01']
daily_2020 = build_daily_curve(trades_2020)
metrics_2020 = calc_daily_sharpe(daily_2020, trades_2020, "2020-2024 ONLY")

# Test 3: 2015-2019 (Excluded period)
print(f"{'='*70}")
print("TEST 3: 2015-2019 (Excluded from Phase 6.7)")
print(f"{'='*70}\n")

trades_early = all_trades[all_trades['date'] < '2020-01-01']
if len(trades_early) > 0:
    daily_early = build_daily_curve(trades_early)
    metrics_early = calc_daily_sharpe(daily_early, trades_early, "2015-2019 EXCLUDED")

# Comparison
print(f"{'='*70}")
print("CRITICAL COMPARISON")
print(f"{'='*70}\n")

comparison = pd.DataFrame([
    {
        'Period': 'FULL (2015-2024)',
        'Trades': metrics_full['trades'],
        'Daily Sharpe': f"{metrics_full['daily_sharpe']:.2f}",
        'Win Rate': f"{metrics_full['win_rate']:.1f}%",
        'Total Return': f"{metrics_full['total_return']:+.2f}%"
    },
    {
        'Period': '2020-2024 (Phase 6.7)',
        'Trades': metrics_2020['trades'],
        'Daily Sharpe': f"{metrics_2020['daily_sharpe']:.2f}",
        'Win Rate': f"{metrics_2020['win_rate']:.1f}%",
        'Total Return': f"{metrics_2020['total_return']:+.2f}%"
    },
    {
        'Period': '2015-2019 (Excluded)',
        'Trades': metrics_early['trades'],
        'Daily Sharpe': f"{metrics_early['daily_sharpe']:.2f}",
        'Win Rate': f"{metrics_early['win_rate']:.1f}%",
        'Total Return': f"{metrics_early['total_return']:+.2f}%"
    }
])

print(comparison.to_string(index=False))
print()

# Verdict
print(f"{'='*70}")
print("VERDICT")
print(f"{'='*70}\n")

full_sharpe = metrics_full['daily_sharpe']
phase67_sharpe = metrics_2020['daily_sharpe']
improvement = ((phase67_sharpe - full_sharpe) / full_sharpe * 100) if full_sharpe > 0 else 0

print(f"Full Period Sharpe:  {full_sharpe:.2f}")
print(f"Phase 6.7 Sharpe:    {phase67_sharpe:.2f}")
print(f"Difference:          {phase67_sharpe - full_sharpe:+.2f} ({improvement:+.1f}%)\n")

if full_sharpe > 0.65:
    verdict = "EXCELLENT - Strategy works across all periods!"
    status = "ROBUST"
elif full_sharpe > 0.50:
    verdict = "GOOD - Strategy is solid but period-dependent"
    status = "ACCEPTABLE"
elif full_sharpe > 0.35:
    verdict = "MARGINAL - Strategy shows weakness in some periods"
    status = "QUESTIONABLE"
else:
    verdict = "WEAK - Phase 6.7 likely cherry-picked strong period"
    status = "NOT ROBUST"

print(f"VERDICT: {verdict}")
print(f"STATUS: {status}\n")

if improvement > 30:
    print("WARNING: Phase 6.7 shows 30%+ improvement over full period")
    print("This suggests 2020-2024 was an EASY MODE for the strategy")
    print("Expected live performance closer to FULL PERIOD Sharpe\n")

# Save results
os.makedirs('outputs_phase6_7_full', exist_ok=True)
comparison.to_csv('outputs_phase6_7_full/period_comparison.csv', index=False)

summary = f"""
================================================================================
                    FULL PERIOD TEST SUMMARY
================================================================================

FULL PERIOD (2015-2024):
Daily Sharpe: {metrics_full['daily_sharpe']:.2f}
Total Return: {metrics_full['total_return']:+.2f}%
Max DD: {metrics_full['max_dd']:.2f}%

PHASE 6.7 (2020-2024):
Daily Sharpe: {metrics_2020['daily_sharpe']:.2f}
Total Return: {metrics_2020['total_return']:+.2f}%
Max DD: {metrics_2020['max_dd']:.2f}%

VERDICT: {status}
{verdict}

RECOMMENDATION:
Use FULL PERIOD Sharpe ({metrics_full['daily_sharpe']:.2f}) for honest expectations,
not Phase 6.7 Sharpe ({phase67_sharpe:.2f}) which may be period-specific.

Conservative Live Expectation: {metrics_full['daily_sharpe'] * 0.70:.2f} (30% degradation)
================================================================================
"""

with open('outputs_phase6_7_full/SUMMARY.txt', 'w') as f:
    f.write(summary)

print(summary)
print(f"Files saved to outputs_phase6_7_full/")
