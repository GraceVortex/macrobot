"""
PHASE 10.5: DEEP AUDIT & VALIDATION
====================================
Analyze Phase 10 results after bug fixes:
1. Audit stopped vs non-stopped trades
2. CPI-only deep dive
3. Walk-forward validation
4. Conservative configuration testing
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = 'outputs_phase10_5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 10.5: DEEP AUDIT & VALIDATION")
print("=" * 80)
print()

# ============================================================================
# PART 1: AUDIT STOPPED TRADES
# ============================================================================

print("=" * 80)
print("PART 1: STOPPED TRADES AUDIT")
print("=" * 80)
print()

# Load all trades from Phase 10
all_trades = pd.read_csv('outputs_phase10/all_trades_block_b.csv')

print(f"Total trades loaded: {len(all_trades)}")
print(f"Configurations: {all_trades['config'].unique().tolist()}")
print()

# Focus on OHLC 2% stop (the best performer)
ohlc_2 = all_trades[all_trades['config'] == 'OHLC stop 2%'].copy()

print("OHLC 2% STOP ANALYSIS:")
print("-" * 80)
print(f"Total trades: {len(ohlc_2)}")

stopped = ohlc_2[ohlc_2['stopped'] == True]
non_stopped = ohlc_2[ohlc_2['stopped'] == False]

print(f"Stopped trades: {len(stopped)} ({len(stopped)/len(ohlc_2)*100:.1f}%)")
print(f"Non-stopped trades: {len(non_stopped)} ({len(non_stopped)/len(ohlc_2)*100:.1f}%)")
print()

print("STOPPED TRADES:")
print(f"  Count: {len(stopped)}")
print(f"  Avg P&L: {stopped['pnl_net'].mean():+.2f}%")
print(f"  Median P&L: {stopped['pnl_net'].median():+.2f}%")
print(f"  Std Dev: {stopped['pnl_net'].std():.2f}%")
print(f"  Min: {stopped['pnl_net'].min():+.2f}%")
print(f"  Max: {stopped['pnl_net'].max():+.2f}%")
print(f"  Win Rate: {(stopped['pnl_net'] > 0).sum() / len(stopped) * 100:.1f}%")
print()

print("NON-STOPPED TRADES:")
print(f"  Count: {len(non_stopped)}")
print(f"  Avg P&L: {non_stopped['pnl_net'].mean():+.2f}%")
print(f"  Median P&L: {non_stopped['pnl_net'].median():+.2f}%")
print(f"  Std Dev: {non_stopped['pnl_net'].std():.2f}%")
print(f"  Min: {non_stopped['pnl_net'].min():+.2f}%")
print(f"  Max: {non_stopped['pnl_net'].max():+.2f}%")
print(f"  Win Rate: {(non_stopped['pnl_net'] > 0).sum() / len(non_stopped) * 100:.1f}%")
print()

# Critical check
print("CRITICAL CHECK:")
if stopped['pnl_net'].mean() > 0:
    print("  ⚠️  WARNING: Stopped trades avg P&L is POSITIVE!")
    print("      This is unusual - stops should cut losses")
    print("      Possible issues:")
    print("      - Stopped trades hitting profit targets by chance")
    print("      - Stop logic still has subtle bugs")
    print("      - Small sample size creating noise")
else:
    print("  ✅ Stopped trades avg P&L is negative (expected)")
    expected_stop_loss = -2.0 - 0.3  # -2% stop - 0.3% costs
    if stopped['pnl_net'].mean() < expected_stop_loss * 0.8:
        print(f"  ⚠️  WARNING: Stopped avg ({stopped['pnl_net'].mean():.2f}%) worse than expected ({expected_stop_loss:.2f}%)")
    else:
        print(f"  ✅ Stopped avg ({stopped['pnl_net'].mean():.2f}%) close to expected ({expected_stop_loss:.2f}%)")

print()

# Save detailed breakdown
stopped_analysis = pd.DataFrame({
    'metric': ['count', 'avg_pnl', 'median_pnl', 'std', 'min', 'max', 'win_rate'],
    'stopped': [
        len(stopped),
        stopped['pnl_net'].mean(),
        stopped['pnl_net'].median(),
        stopped['pnl_net'].std(),
        stopped['pnl_net'].min(),
        stopped['pnl_net'].max(),
        (stopped['pnl_net'] > 0).sum() / len(stopped) * 100
    ],
    'non_stopped': [
        len(non_stopped),
        non_stopped['pnl_net'].mean(),
        non_stopped['pnl_net'].median(),
        non_stopped['pnl_net'].std(),
        non_stopped['pnl_net'].min(),
        non_stopped['pnl_net'].max(),
        (non_stopped['pnl_net'] > 0).sum() / len(non_stopped) * 100
    ]
})
stopped_analysis.to_csv(f'{OUTPUT_DIR}/stopped_trades_analysis.csv', index=False)

# ============================================================================
# PART 2: CPI-ONLY DEEP DIVE
# ============================================================================

print("=" * 80)
print("PART 2: CPI-ONLY DEEP DIVE")
print("=" * 80)
print()

# Need to re-run strategy with CPI only
# Load merged data
merged_df = pd.read_csv('outputs_phase10/merged_data.csv')
merged_df['entry_date'] = pd.to_datetime(merged_df['entry_date'])

cpi_only = merged_df[merged_df['event_type'] == 'CPI'].copy()

print(f"CPI events: {len(cpi_only)}")
print(f"Date range: {cpi_only['entry_date'].min()} to {cpi_only['entry_date'].max()}")
print()

# Load TMF OHLC
import yfinance as yf
print("Loading TMF OHLC...")
tmf = yf.download('TMF', start='2018-01-01', end='2024-11-15', progress=False)
if isinstance(tmf.columns, pd.MultiIndex):
    tmf.columns = tmf.columns.get_level_values(0)
tmf = tmf[['Open', 'High', 'Low', 'Close']].copy()
print(f"  Loaded {len(tmf)} days")
print()

# Import backtest function from phase10
import sys
sys.path.insert(0, '.')
from phase10 import backtest_strategy, calculate_metrics

print("Testing CPI-only with different configurations:")
print("-" * 80)

cpi_results = []

# Config 1: CPI + 5-day + no stop
print("Config 1: CPI only, 5-day hold, no stop")
trades = backtest_strategy(cpi_only, tmf, holding_days=5, stop_type='none')
metrics = calculate_metrics(trades, "CPI 5d no-stop")
if metrics:
    cpi_results.append(metrics)
    print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Total: {metrics['total_return']:+.1f}%")

# Config 2: CPI + 5-day + OHLC 2%
print("Config 2: CPI only, 5-day hold, OHLC 2% stop")
trades = backtest_strategy(cpi_only, tmf, holding_days=5, stop_type='ohlc', stop_pct=2)
metrics = calculate_metrics(trades, "CPI 5d OHLC2%")
if metrics:
    cpi_results.append(metrics)
    print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Total: {metrics['total_return']:+.1f}%")

# Config 3: CPI + 5-day + OHLC 3% (CONSERVATIVE)
print("Config 3: CPI only, 5-day hold, OHLC 3% stop (CONSERVATIVE)")
trades = backtest_strategy(cpi_only, tmf, holding_days=5, stop_type='ohlc', stop_pct=3)
metrics = calculate_metrics(trades, "CPI 5d OHLC3%")
if metrics:
    cpi_results.append(metrics)
    print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Total: {metrics['total_return']:+.1f}%")

# Config 4: CPI + 2-day + OHLC 2% (original)
print("Config 4: CPI only, 2-day hold, OHLC 2% stop")
trades = backtest_strategy(cpi_only, tmf, holding_days=2, stop_type='ohlc', stop_pct=2)
metrics = calculate_metrics(trades, "CPI 2d OHLC2%")
if metrics:
    cpi_results.append(metrics)
    print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Total: {metrics['total_return']:+.1f}%")

print()

# Save CPI results
pd.DataFrame(cpi_results).to_csv(f'{OUTPUT_DIR}/cpi_only_configs.csv', index=False)

# ============================================================================
# PART 3: WALK-FORWARD VALIDATION
# ============================================================================

print("=" * 80)
print("PART 3: WALK-FORWARD VALIDATION")
print("=" * 80)
print()

# Use full dataset (CPI + NFP)
merged_df_full = merged_df.copy()
merged_df_full['year'] = pd.to_datetime(merged_df_full['entry_date']).dt.year

# Define walk-forward windows
windows = [
    ('2020-2021', '2022'),
    ('2020-2022', '2023'),
    ('2021-2022', '2023'),
    ('2020-2023', '2024'),
    ('2021-2023', '2024'),
]

wf_results = []

print("Walk-Forward Validation (Train → Test):")
print("-" * 80)

for train_years_str, test_year_str in windows:
    train_years = [int(y) for y in train_years_str.split('-')]
    test_year = int(test_year_str)
    
    train_data = merged_df_full[merged_df_full['year'].isin(range(train_years[0], train_years[-1]+1))]
    test_data = merged_df_full[merged_df_full['year'] == test_year]
    
    if len(train_data) < 10 or len(test_data) < 5:
        print(f"{train_years_str} → {test_year_str}: SKIP (insufficient data)")
        continue
    
    # Test with OHLC 2% stop, 2-day hold
    train_trades = backtest_strategy(train_data, tmf, holding_days=2, stop_type='ohlc', stop_pct=2)
    test_trades = backtest_strategy(test_data, tmf, holding_days=2, stop_type='ohlc', stop_pct=2)
    
    train_metrics = calculate_metrics(train_trades, f"Train {train_years_str}")
    test_metrics = calculate_metrics(test_trades, f"Test {test_year_str}")
    
    if train_metrics and test_metrics:
        wf_results.append({
            'train_period': train_years_str,
            'test_period': test_year_str,
            'train_trades': train_metrics['trades'],
            'train_sharpe': train_metrics['sharpe'],
            'train_win_rate': train_metrics['win_rate'],
            'test_trades': test_metrics['trades'],
            'test_sharpe': test_metrics['sharpe'],
            'test_win_rate': test_metrics['win_rate'],
            'sharpe_delta': test_metrics['sharpe'] - train_metrics['sharpe']
        })
        
        print(f"{train_years_str} → {test_year_str}:")
        print(f"  Train: Sharpe {train_metrics['sharpe']:+.2f}, Win {train_metrics['win_rate']:.1f}%, Trades {train_metrics['trades']}")
        print(f"  Test:  Sharpe {test_metrics['sharpe']:+.2f}, Win {test_metrics['win_rate']:.1f}%, Trades {test_metrics['trades']}")
        print(f"  Delta: {test_metrics['sharpe'] - train_metrics['sharpe']:+.2f}")
        print()

wf_df = pd.DataFrame(wf_results)
if len(wf_df) > 0:
    wf_df.to_csv(f'{OUTPUT_DIR}/walk_forward_validation.csv', index=False)
    
    avg_delta = wf_df['sharpe_delta'].mean()
    print("WALK-FORWARD SUMMARY:")
    print(f"  Windows tested: {len(wf_df)}")
    print(f"  Avg Sharpe delta (test - train): {avg_delta:+.2f}")
    
    if avg_delta > 0.3:
        print("  ⚠️  WARNING: Test consistently outperforms train - suspicious!")
    elif avg_delta < -0.3:
        print("  ⚠️  WARNING: Test consistently underperforms - overfitting!")
    else:
        print("  ✅ Test vs Train delta reasonable")
    print()

# ============================================================================
# PART 4: CONSERVATIVE CONFIGURATION
# ============================================================================

print("=" * 80)
print("PART 4: CONSERVATIVE CONFIGURATION TESTING")
print("=" * 80)
print()

print("Testing RECOMMENDED conservative config:")
print("  Events: CPI only")
print("  Hold: 5 days")
print("  Stop: OHLC 3%")
print("-" * 80)

# Full period test
trades_conservative = backtest_strategy(cpi_only, tmf, holding_days=5, stop_type='ohlc', stop_pct=3)
metrics_conservative = calculate_metrics(trades_conservative, "Conservative Config")

if metrics_conservative:
    print("FULL PERIOD (2020-2024):")
    print(f"  Trades: {metrics_conservative['trades']}")
    print(f"  Sharpe: {metrics_conservative['sharpe']:.2f}")
    print(f"  Win Rate: {metrics_conservative['win_rate']:.1f}%")
    print(f"  Avg Return: {metrics_conservative['avg_return']:+.2f}%")
    print(f"  Total Return: {metrics_conservative['total_return']:+.1f}%")
    print(f"  Max DD: {metrics_conservative['max_dd']:.1f}%")
    print(f"  Avg Winner: {metrics_conservative['avg_winner']:+.2f}%")
    print(f"  Avg Loser: {metrics_conservative['avg_loser']:+.2f}%")
    print()
    
    # Apply 30% haircut
    conservative_sharpe = metrics_conservative['sharpe'] * 0.7
    conservative_annual = metrics_conservative['total_return'] / 5 * 0.7  # 5 years
    
    print("CONSERVATIVE ESTIMATES (30% haircut):")
    print(f"  Expected Sharpe: {conservative_sharpe:.2f}")
    print(f"  Expected Annual Return: {conservative_annual:.1f}%")
    print(f"  Trade Frequency: ~12 trades/year (monthly CPI)")
    print()

# Train/Test on conservative config
cpi_only['year'] = pd.to_datetime(cpi_only['entry_date']).dt.year
cpi_train = cpi_only[cpi_only['year'] <= 2022]
cpi_test = cpi_only[cpi_only['year'] >= 2023]

print("TRAIN/TEST SPLIT:")
print(f"Train: 2020-2022 ({len(cpi_train)} events)")
trades_train_cons = backtest_strategy(cpi_train, tmf, holding_days=5, stop_type='ohlc', stop_pct=3)
metrics_train_cons = calculate_metrics(trades_train_cons, "Train Conservative")
if metrics_train_cons:
    print(f"  Sharpe: {metrics_train_cons['sharpe']:.2f}, Win: {metrics_train_cons['win_rate']:.1f}%")

print(f"Test: 2023-2024 ({len(cpi_test)} events)")
trades_test_cons = backtest_strategy(cpi_test, tmf, holding_days=5, stop_type='ohlc', stop_pct=3)
metrics_test_cons = calculate_metrics(trades_test_cons, "Test Conservative")
if metrics_test_cons:
    print(f"  Sharpe: {metrics_test_cons['sharpe']:.2f}, Win: {metrics_test_cons['win_rate']:.1f}%")
    
    if metrics_train_cons:
        delta = metrics_test_cons['sharpe'] - metrics_train_cons['sharpe']
        print(f"  Delta: {delta:+.2f}")
        
        if delta > 0.5:
            print("  ⚠️  Test outperforms train significantly")
        elif delta < -0.5:
            print("  ⚠️  Test underperforms train significantly")
        else:
            print("  ✅ Train/test results consistent")

print()

# Save conservative results
conservative_results = {
    'config': 'CPI only, 5-day, OHLC 3%',
    'full_sharpe': metrics_conservative['sharpe'] if metrics_conservative else None,
    'train_sharpe': metrics_train_cons['sharpe'] if metrics_train_cons else None,
    'test_sharpe': metrics_test_cons['sharpe'] if metrics_test_cons else None,
    'conservative_sharpe_30pct': conservative_sharpe if metrics_conservative else None,
    'trades': metrics_conservative['trades'] if metrics_conservative else None
}

pd.DataFrame([conservative_results]).to_csv(f'{OUTPUT_DIR}/conservative_config.csv', index=False)

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 10.5 COMPLETE")
print("=" * 80)
print()
print("Files saved:")
print(f"  {OUTPUT_DIR}/stopped_trades_analysis.csv")
print(f"  {OUTPUT_DIR}/cpi_only_configs.csv")
print(f"  {OUTPUT_DIR}/walk_forward_validation.csv")
print(f"  {OUTPUT_DIR}/conservative_config.csv")
print()
print("Key findings:")
print("1. Stopped trades analysis - check if avg P&L makes sense")
print("2. CPI-only deep dive - best configurations identified")
print("3. Walk-forward validation - overfitting check")
print("4. Conservative config - realistic expectations")
print()
