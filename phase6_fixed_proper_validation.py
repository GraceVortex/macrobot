"""
PHASE 6 FIXED: PROPER OUT-OF-SAMPLE VALIDATION
===============================================
CRITICAL FIX: Train/test split to avoid p-hacking

Previous Phase 6 errors:
1. Tested multiple stops on same data (p-hacking)
2. Selected best performer (overfitting)
3. No out-of-sample validation

Proper methodology:
1. Split: Train (2014-2021) vs Test (2022-2024)
2. Optimize ONLY on train data
3. Test ONCE on holdout data
4. Report ONLY test performance
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

print("""
================================================================================
              PHASE 6 FIXED: PROPER OUT-OF-SAMPLE VALIDATION
================================================================================

CRITICAL METHODOLOGY FIXES:
1. Train/test split (2014-2021 train, 2022-2024 test)
2. Optimize stops ONLY on train data
3. Validate ONCE on test data
4. Report honest out-of-sample Sharpe

This prevents p-hacking and overfitting!
""")

def load_phase3_trades():
    """Load Phase 3 trades"""
    print("Loading Phase 3 trades...")
    trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades.sort_values('date').reset_index(drop=True)
    print(f"  Total: {len(trades)} trades\n")
    return trades

def split_train_test(trades, split_date='2022-01-01'):
    """Split into train and test sets"""
    split = pd.to_datetime(split_date)
    
    train = trades[trades['date'] < split].copy()
    test = trades[trades['date'] >= split].copy()
    
    print(f"Data Split:")
    print(f"  Train: {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"  Train trades: {len(train)}")
    print(f"  Test: {test['date'].min().date()} to {test['date'].max().date()}")
    print(f"  Test trades: {len(test)}")
    print()
    
    return train, test

def calculate_metrics(returns_series, label=""):
    """Calculate performance metrics"""
    if len(returns_series) == 0:
        return None
    
    avg_return = returns_series.mean()
    
    if returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252/2)
    else:
        sharpe = 0
    
    win_rate = (returns_series > 0).mean() * 100
    
    cumulative = returns_series.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    if label:
        print(f"{label}:")
        print(f"  Trades: {len(returns_series)}")
        print(f"  Avg Return: {avg_return:+.3f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Max DD: {max_dd:.2f}%\n")
    
    return {
        'trades': len(returns_series),
        'avg_return': avg_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd': max_dd
    }

def apply_stop_loss(trades, stop_pct):
    """Apply stop loss (simplified, no intraday data)"""
    trades_copy = trades.copy()
    trades_copy['stopped_pnl'] = trades_copy['pnl'].clip(lower=-stop_pct)
    stopped_count = (trades_copy['pnl'] < -stop_pct).sum()
    return trades_copy, stopped_count

def optimize_on_train(train_data):
    """Optimize stop loss on train data ONLY"""
    
    print(f"{'='*70}")
    print("STEP 1: OPTIMIZATION ON TRAIN DATA ONLY")
    print(f"{'='*70}\n")
    
    stop_levels = [None, 2.0, 2.5, 3.0, 4.0, 5.0]
    train_results = []
    
    for stop in stop_levels:
        if stop is None:
            label = 'No Stop'
            stopped = train_data.copy()
            stopped['final_pnl'] = stopped['pnl']
            stopped_count = 0
        else:
            label = f'{stop}% Stop'
            stopped, stopped_count = apply_stop_loss(train_data, stop)
            stopped['final_pnl'] = stopped['stopped_pnl']
        
        metrics = calculate_metrics(stopped['final_pnl'], label)
        metrics['stop_level'] = stop
        metrics['stopped_trades'] = stopped_count
        train_results.append(metrics)
    
    results_df = pd.DataFrame(train_results)
    
    # Select best by Sharpe on TRAIN
    best_idx = results_df['sharpe'].idxmax()
    best_stop = results_df.loc[best_idx, 'stop_level']
    
    print(f"{'='*70}")
    print(f"BEST ON TRAIN DATA: {best_stop}% stop (Sharpe {results_df.loc[best_idx, 'sharpe']:.2f})")
    print(f"{'='*70}\n")
    
    return best_stop, results_df

def validate_on_test(test_data, best_stop):
    """Validate on test data with fixed parameters"""
    
    print(f"{'='*70}")
    print("STEP 2: VALIDATION ON TEST DATA (OUT-OF-SAMPLE)")
    print(f"{'='*70}\n")
    
    if best_stop is None:
        print("Applying: No Stop")
        test_result = test_data.copy()
        test_result['final_pnl'] = test_result['pnl']
        stopped_count = 0
    else:
        print(f"Applying: {best_stop}% Stop (from train optimization)")
        test_result, stopped_count = apply_stop_loss(test_data, best_stop)
        test_result['final_pnl'] = test_result['stopped_pnl']
    
    print()
    metrics = calculate_metrics(test_result['final_pnl'], "TEST SET RESULTS")
    metrics['stopped_trades'] = stopped_count
    
    return metrics, test_result

def monte_carlo_stop_adjustment(trades, stop_pct, n_sims=1000):
    """
    Monte Carlo simulation to estimate intraday stop hits
    
    Problem: We only have close-to-close returns
    Reality: Stops can be hit intraday even if close is profitable
    
    Solution: Simulate intraday paths and estimate stop hit probability
    """
    
    print(f"{'='*70}")
    print("STEP 3: MONTE CARLO INTRADAY STOP ADJUSTMENT")
    print(f"{'='*70}\n")
    
    adjusted_trades = []
    total_adjustments = 0
    
    for _, trade in trades.iterrows():
        final_pnl = trade['final_pnl']
        
        # Only adjust trades that finished positive but might have hit stop
        if final_pnl > 0:
            # Estimate intraday volatility (2x daily is reasonable for TMF)
            intraday_vol = abs(final_pnl) * 2.0
            
            # Simulate paths
            stopped_count = 0
            for _ in range(n_sims):
                # 8 intraday periods (hourly buckets)
                path = np.random.normal(0, intraday_vol/4, 8)
                cumulative = np.cumsum(path)
                
                # Check if stop hit
                if (cumulative <= -stop_pct).any():
                    stopped_count += 1
            
            stop_probability = stopped_count / n_sims
            
            # If >30% chance stop was hit, adjust
            if stop_probability > 0.3:
                adjusted_pnl = -stop_pct
                total_adjustments += 1
            else:
                adjusted_pnl = final_pnl
        else:
            # Negative trades - stop definitely hit if below stop level
            adjusted_pnl = max(final_pnl, -stop_pct)
        
        trade_copy = trade.copy()
        trade_copy['adjusted_pnl'] = adjusted_pnl
        adjusted_trades.append(trade_copy)
    
    adjusted_df = pd.DataFrame(adjusted_trades)
    
    print(f"Intraday adjustments made: {total_adjustments}/{len(trades)}")
    print(f"Adjustment rate: {total_adjustments/len(trades)*100:.1f}%\n")
    
    metrics = calculate_metrics(adjusted_df['adjusted_pnl'], "AFTER INTRADAY ADJUSTMENT")
    
    return metrics, adjusted_df

def main():
    print("\nStarting Phase 6 Fixed (Proper Validation)...\n")
    
    # Load data
    trades = load_phase3_trades()
    
    # Split train/test
    train, test = split_train_test(trades, '2022-01-01')
    
    # STEP 1: Optimize on train
    best_stop, train_results = optimize_on_train(train)
    
    # STEP 2: Validate on test
    test_metrics, test_results = validate_on_test(test, best_stop)
    
    # STEP 3: Monte Carlo adjustment for intraday stops
    if best_stop is not None:
        adjusted_metrics, adjusted_results = monte_carlo_stop_adjustment(
            test_results, best_stop, n_sims=1000
        )
    else:
        adjusted_metrics = test_metrics
        adjusted_results = test_results
    
    # Save results
    os.makedirs('outputs_phase6_fixed', exist_ok=True)
    train_results.to_csv('outputs_phase6_fixed/train_optimization.csv', index=False)
    
    # Test results summary
    summary = pd.DataFrame([{
        'dataset': 'Train',
        'trades': len(train),
        'best_stop': best_stop,
        'sharpe': train_results.loc[train_results['stop_level'] == best_stop, 'sharpe'].values[0]
    }, {
        'dataset': 'Test (Out-of-Sample)',
        'trades': test_metrics['trades'],
        'best_stop': best_stop,
        'sharpe': test_metrics['sharpe']
    }, {
        'dataset': 'Test (Adjusted for Intraday)',
        'trades': adjusted_metrics['trades'],
        'best_stop': best_stop,
        'sharpe': adjusted_metrics['sharpe']
    }])
    
    summary.to_csv('outputs_phase6_fixed/summary.csv', index=False)
    adjusted_results.to_csv('outputs_phase6_fixed/test_results_adjusted.csv', index=False)
    
    # Final report
    print(f"{'='*70}")
    print("FINAL HONEST RESULTS")
    print(f"{'='*70}\n")
    
    print("TRAIN (2014-2021):")
    print(f"  Best stop: {best_stop}%")
    print(f"  Train Sharpe: {train_results.loc[train_results['stop_level'] == best_stop, 'sharpe'].values[0]:.2f}")
    print()
    
    print("TEST (2022-2024, Out-of-Sample):")
    print(f"  Test Sharpe: {test_metrics['sharpe']:.2f}")
    print()
    
    print("TEST (Adjusted for Intraday Stops):")
    print(f"  Adjusted Sharpe: {adjusted_metrics['sharpe']:.2f}")
    print(f"  Avg Return: {adjusted_metrics['avg_return']:+.3f}%")
    print(f"  Win Rate: {adjusted_metrics['win_rate']:.1f}%")
    print()
    
    print("HONEST EXPECTATION FOR LIVE TRADING:")
    print(f"  Conservative Sharpe: {adjusted_metrics['sharpe'] * 0.7:.2f} (30% degradation)")
    print(f"  Realistic Range: {adjusted_metrics['sharpe'] * 0.6:.2f} - {adjusted_metrics['sharpe'] * 0.8:.2f}")
    
    print(f"\n{'='*70}")
    print("PHASE 6 FIXED COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
