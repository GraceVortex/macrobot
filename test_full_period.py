"""Quick test: Phase 6.7 on FULL period"""
import pandas as pd
import numpy as np

# Load Phase 3 trades
trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
trades['date'] = pd.to_datetime(trades['date'])

print(f"Phase 3 trades period: {trades['date'].min().date()} to {trades['date'].max().date()}")
print(f"Total trades: {len(trades)}")
print()

# Group by year
trades['year'] = trades['date'].dt.year
yearly = trades.groupby('year').agg({
    'pnl': ['count', 'mean', 'sum'],
    'returns': 'mean'
}).round(2)

print("Year-by-year breakdown:")
print(yearly)
print()

# Calculate metrics by period
TRANSACTION_COSTS = 0.0019

def calc_metrics(df, label):
    df = df.copy()
    df['pnl_net'] = df['pnl'] - (TRANSACTION_COSTS * 100)
    
    print(f"\n{label}:")
    print(f"  Trades: {len(df)}")
    print(f"  Avg P&L (net): {df['pnl_net'].mean():+.3f}%")
    print(f"  Win rate: {(df['pnl_net'] > 0).mean() * 100:.1f}%")
    print(f"  Total: {df['pnl_net'].sum():+.2f}%")

# Full period
calc_metrics(trades, "FULL PERIOD (2015-2024)")

# 2020-2024 only (Phase 6.7)
trades_2020 = trades[trades['year'] >= 2020]
calc_metrics(trades_2020, "2020-2024 ONLY (Phase 6.7)")

# 2015-2019 (excluded)
trades_early = trades[trades['year'] < 2020]
calc_metrics(trades_early, "2015-2019 (Excluded from 6.7)")
