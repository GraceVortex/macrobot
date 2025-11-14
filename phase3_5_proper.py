"""
PHASE 3.5: PHASE 3 WITH PROPER SHARPE CALCULATION
=================================================
Phase 3 strategy with institutional-grade metrics

Improvements:
1. Daily equity curve (not per-trade)
2. Institutional Sharpe: daily returns * sqrt(252)
3. Transaction costs: 0.19% per trade
4. Professional reporting

This uses Phase 3 trades but calculates metrics properly.
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Transaction costs per trade
TRANSACTION_COSTS = 0.0019  # 0.19%

print("""
================================================================================
         PHASE 3.5: PHASE 3 WITH INSTITUTIONAL SHARPE
================================================================================

Taking Phase 3 trades and calculating metrics properly:
• Daily equity curve (not per-trade)
• Institutional Sharpe = mean(daily) / std(daily) * sqrt(252)
• Transaction costs included (0.19%)

This gives HONEST metrics comparable to industry standards.
""")

def load_phase3_trades():
    """Load Phase 3 TMF trades"""
    print("Loading Phase 3 TMF trades...")
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades.sort_values('date').reset_index(drop=True)
    
    # Add exit date (2 days later)
    trades['exit_date'] = trades['date'] + pd.Timedelta(days=2)
    
    print(f"  Loaded: {len(trades)} trades")
    print(f"  Period: {trades['date'].min().date()} to {trades['exit_date'].max().date()}\n")
    return trades

def build_daily_equity_curve(trades):
    """
    Build daily equity curve - INSTITUTIONAL METHOD
    
    Returns daily time-series with:
    - Daily P&L (gross and net)
    - Cumulative equity
    - Drawdown
    """
    
    print("Building daily equity curve...")
    
    # Get date range
    start_date = trades['date'].min()
    end_date = trades['exit_date'].max()
    
    # Create daily date range (business days only)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initialize daily returns
    daily_returns = pd.DataFrame({
        'date': all_dates,
        'pnl_gross': 0.0,
        'pnl_net': 0.0,
        'in_position': False,
        'trade_count': 0
    })
    daily_returns.set_index('date', inplace=True)
    
    # For each trade, attribute P&L to exit date
    for _, trade in trades.iterrows():
        exit_date = trade['exit_date']
        pnl_gross = trade['pnl']
        
        # Net P&L (subtract transaction costs)
        pnl_net = pnl_gross - (TRANSACTION_COSTS * 100)
        
        if exit_date in daily_returns.index:
            daily_returns.loc[exit_date, 'pnl_gross'] += pnl_gross
            daily_returns.loc[exit_date, 'pnl_net'] += pnl_net
            daily_returns.loc[exit_date, 'trade_count'] += 1
        
        # Mark position days
        entry_date = trade['date']
        if entry_date in daily_returns.index:
            daily_returns.loc[entry_date:exit_date, 'in_position'] = True
    
    # Cumulative equity
    daily_returns['cum_pnl_gross'] = daily_returns['pnl_gross'].cumsum()
    daily_returns['cum_pnl_net'] = daily_returns['pnl_net'].cumsum()
    
    # Drawdown (on net equity)
    daily_returns['running_max'] = daily_returns['cum_pnl_net'].expanding().max()
    daily_returns['drawdown'] = daily_returns['cum_pnl_net'] - daily_returns['running_max']
    
    print(f"  Generated {len(daily_returns)} trading days\n")
    
    return daily_returns.reset_index()

def calculate_institutional_metrics(daily_returns, trades):
    """
    Calculate institutional-grade metrics
    
    KEY: Daily Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
    This is the STANDARD industry calculation
    """
    
    print(f"{'='*70}")
    print("INSTITUTIONAL METRICS")
    print(f"{'='*70}\n")
    
    # Per-trade metrics
    trades_with_costs = trades.copy()
    trades_with_costs['pnl_net'] = trades_with_costs['pnl'] - (TRANSACTION_COSTS * 100)
    
    num_trades = len(trades_with_costs)
    avg_return_gross = trades_with_costs['pnl'].mean()
    avg_return_net = trades_with_costs['pnl_net'].mean()
    win_rate = (trades_with_costs['pnl_net'] > 0).mean() * 100
    
    # Daily metrics
    all_daily_pnl = daily_returns['pnl_net']
    
    # Daily Sharpe (INSTITUTIONAL)
    if all_daily_pnl.std() > 0:
        daily_sharpe = (all_daily_pnl.mean() / all_daily_pnl.std()) * np.sqrt(252)
    else:
        daily_sharpe = 0
    
    # Total return and drawdown
    total_return_gross = daily_returns['cum_pnl_gross'].iloc[-1]
    total_return_net = daily_returns['cum_pnl_net'].iloc[-1]
    max_dd = daily_returns['drawdown'].min()
    
    # OLD METHOD (per-trade Sharpe - INFLATED)
    active_returns = trades_with_costs['pnl_net']
    if active_returns.std() > 0:
        old_sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(252/2)
    else:
        old_sharpe = 0
    
    print("RESULTS:")
    print(f"  Trades: {num_trades}")
    print(f"  Avg Return (Gross): {avg_return_gross:+.3f}%")
    print(f"  Avg Return (Net):   {avg_return_net:+.3f}%")
    print(f"  Total Return (Net): {total_return_net:+.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print()
    
    print("SHARPE COMPARISON:")
    print(f"  Old Method (Per-Trade): {old_sharpe:.2f} ← INFLATED")
    print(f"  New Method (Daily):     {daily_sharpe:.2f} ← INSTITUTIONAL ✓")
    print()
    
    print("WHY THE DIFFERENCE?")
    print("  • Old: assumes ~126 independent bets/year")
    print("  • New: accounts for actual calendar time")
    print("  • Daily Sharpe is the TRUE institutional measure")
    print()
    
    return {
        'trades': num_trades,
        'avg_return_gross': avg_return_gross,
        'avg_return_net': avg_return_net,
        'total_return_gross': total_return_gross,
        'total_return_net': total_return_net,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'daily_sharpe': daily_sharpe,
        'old_sharpe': old_sharpe,
        'trading_days': len(daily_returns)
    }

def main():
    print("\nStarting Phase 3.5 (Proper Sharpe Calculation)...\n")
    
    # Load Phase 3 trades
    trades = load_phase3_trades()
    
    # Build daily equity curve
    daily_returns = build_daily_equity_curve(trades)
    
    # Calculate institutional metrics
    metrics = calculate_institutional_metrics(daily_returns, trades)
    
    # Create output directory
    output_dir = 'outputs_phase3_5'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    daily_returns.to_csv(f'{output_dir}/daily_equity_curve.csv', index=False)
    
    # Save trades with net P&L
    trades_net = trades.copy()
    trades_net['pnl_net'] = trades_net['pnl'] - (TRANSACTION_COSTS * 100)
    trades_net.to_csv(f'{output_dir}/tmf_trades_with_costs.csv', index=False)
    
    # Summary
    summary = pd.DataFrame([{
        'metric': 'Trades',
        'value': metrics['trades']
    }, {
        'metric': 'Avg Return (Net) %',
        'value': f"{metrics['avg_return_net']:.3f}"
    }, {
        'metric': 'Total Return (Net) %',
        'value': f"{metrics['total_return_net']:.2f}"
    }, {
        'metric': 'Win Rate %',
        'value': f"{metrics['win_rate']:.1f}"
    }, {
        'metric': 'Max Drawdown %',
        'value': f"{metrics['max_dd']:.2f}"
    }, {
        'metric': 'Daily Sharpe (Institutional)',
        'value': f"{metrics['daily_sharpe']:.2f}"
    }, {
        'metric': 'Old Sharpe (Per-Trade)',
        'value': f"{metrics['old_sharpe']:.2f}"
    }])
    summary.to_csv(f'{output_dir}/summary.csv', index=False)
    
    # Final summary
    print(f"{'='*70}")
    print("CONSERVATIVE LIVE EXPECTATIONS")
    print(f"{'='*70}\n")
    
    live_sharpe = metrics['daily_sharpe'] * 0.70  # 30% degradation
    print(f"Backtest Daily Sharpe: {metrics['daily_sharpe']:.2f}")
    print(f"Expected Live Sharpe:  {live_sharpe:.2f} (30% degradation)")
    print(f"Realistic Range:       {live_sharpe*0.85:.2f} - {live_sharpe*1.15:.2f}")
    print(f"Annual Return Target:  4-6%")
    print()
    
    print("COMPARISON:")
    print(f"  S&P 500 Sharpe:     ~0.4-0.5")
    print(f"  Our Strategy:       {metrics['daily_sharpe']:.2f} (backtest)")
    print(f"  Expected Live:      {live_sharpe:.2f}")
    print(f"  Outperformance:     {live_sharpe/0.45:.1f}x vs S&P")
    print()
    
    print("FILES CREATED:")
    print(f"  • {output_dir}/summary.csv")
    print(f"  • {output_dir}/daily_equity_curve.csv")
    print(f"  • {output_dir}/tmf_trades_with_costs.csv")
    
    print(f"\n{'='*70}")
    print("PHASE 3.5 COMPLETE")
    print(f"{'='*70}\n")
    
    print("✓ Institutional Sharpe calculated properly")
    print("✓ Transaction costs included")
    print("✓ Ready for honest presentation")

if __name__ == "__main__":
    main()
