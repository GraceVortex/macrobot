"""
PHASE 10.7: WALK-FORWARD VALIDATION
====================================
Multiple rolling train/test windows to validate robustness
Tests if strategy is consistently profitable across different periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

OUTPUT_DIR = 'outputs_phase10_7'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 10.7: WALK-FORWARD VALIDATION")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("STEP 1: Loading data...")
print("-" * 80)

# Load extended events from Phase 10.6
events_file = 'outputs_phase10_6/economic_events_extended.csv'
if not os.path.exists(events_file):
    print(f"ERROR: {events_file} not found!")
    print("Run phase10_6_quick.py first to generate extended data")
    exit(1)

events_df = pd.read_csv(events_file)
events_df['Date'] = pd.to_datetime(events_df['Date'])

print(f"Events loaded: {len(events_df)}")
print(f"Date range: {events_df['Date'].min().date()} to {events_df['Date'].max().date()}")
print()

# Load TMF
import yfinance as yf
print("Loading TMF OHLC...")
tmf = yf.download('TMF', start='2010-01-01', end='2024-11-15', progress=False)
if isinstance(tmf.columns, pd.MultiIndex):
    tmf.columns = tmf.columns.get_level_values(0)
tmf = tmf[['Open', 'High', 'Low', 'Close']].copy()
print(f"TMF: {len(tmf)} days")
print()

# Import backtest functions
import sys
sys.path.insert(0, '.')
try:
    from phase10 import backtest_strategy, calculate_metrics, merge_events_with_prices
except ImportError as e:
    print(f"ERROR: Cannot import from phase10.py: {e}")
    exit(1)

# ============================================================================
# STEP 2: WALK-FORWARD VALIDATION FUNCTION
# ============================================================================

def walk_forward_validation(events_df, tmf_ohlc, 
                            train_years=5, 
                            test_years=2,
                            step_years=1):
    """
    Walk-forward validation with rolling windows
    
    Args:
        events_df: Economic events dataframe
        tmf_ohlc: TMF OHLC dataframe
        train_years: Training window size (5 years)
        test_years: Test window size (2 years)
        step_years: Step size (1 year = rolling by 1)
    
    Returns:
        DataFrame with results for each window
    """
    
    results = []
    
    # Get date range
    min_year = events_df['Date'].dt.year.min()
    max_year = events_df['Date'].dt.year.max()
    
    print(f"Date range: {min_year} to {max_year}")
    print(f"Train window: {train_years} years")
    print(f"Test window: {test_years} years")
    print(f"Step: {step_years} year(s)")
    print()
    
    # Create windows
    for train_start_year in range(min_year, max_year - train_years - test_years + 2, step_years):
        train_end_year = train_start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1
        
        if test_end_year > max_year:
            break
        
        # Filter data by year
        train_data = events_df[
            (events_df['Date'].dt.year >= train_start_year) &
            (events_df['Date'].dt.year <= train_end_year)
        ].copy()
        
        test_data = events_df[
            (events_df['Date'].dt.year >= test_start_year) &
            (events_df['Date'].dt.year <= test_end_year)
        ].copy()
        
        if len(train_data) < 20 or len(test_data) < 10:
            print(f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}: SKIP (insufficient data)")
            continue
        
        # Merge with prices
        train_merged = merge_events_with_prices(train_data, tmf_ohlc)
        test_merged = merge_events_with_prices(test_data, tmf_ohlc)
        
        # CPI only
        train_cpi = train_merged[train_merged['event_type'] == 'CPI'].copy()
        test_cpi = test_merged[test_merged['event_type'] == 'CPI'].copy()
        
        if len(train_cpi) < 10 or len(test_cpi) < 5:
            print(f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}: SKIP (insufficient CPI events)")
            continue
        
        # Backtest on train
        trades_train = backtest_strategy(train_cpi, tmf_ohlc,
                                        holding_days=5,
                                        stop_type='ohlc',
                                        stop_pct=3)
        metrics_train = calculate_metrics(trades_train, f"Train {train_start_year}-{train_end_year}")
        
        # Backtest on test
        trades_test = backtest_strategy(test_cpi, tmf_ohlc,
                                       holding_days=5,
                                       stop_type='ohlc',
                                       stop_pct=3)
        metrics_test = calculate_metrics(trades_test, f"Test {test_start_year}-{test_end_year}")
        
        if not metrics_train or not metrics_test:
            print(f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}: SKIP (metrics failed)")
            continue
        
        # Store results
        window_result = {
            'window': f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}",
            'train_years': f"{train_start_year}-{train_end_year}",
            'test_years': f"{test_start_year}-{test_end_year}",
            'train_events': len(train_cpi),
            'test_events': len(test_cpi),
            'train_sharpe': metrics_train['sharpe'],
            'test_sharpe': metrics_test['sharpe'],
            'train_win': metrics_train['win_rate'],
            'test_win': metrics_test['win_rate'],
            'train_total': metrics_train['total_return'],
            'test_total': metrics_test['total_return'],
            'sharpe_delta': metrics_test['sharpe'] - metrics_train['sharpe']
        }
        
        results.append(window_result)
        
        print(f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}:")
        print(f"  Train: {len(train_cpi):3d} events, Sharpe {metrics_train['sharpe']:+.2f}, Win {metrics_train['win_rate']:.1f}%")
        print(f"  Test:  {len(test_cpi):3d} events, Sharpe {metrics_test['sharpe']:+.2f}, Win {metrics_test['win_rate']:.1f}%")
        print(f"  Delta: {metrics_test['sharpe'] - metrics_train['sharpe']:+.2f}")
        print()
    
    return pd.DataFrame(results)

# ============================================================================
# STEP 3: RUN WALK-FORWARD VALIDATION
# ============================================================================

print("=" * 80)
print("STEP 2: RUNNING WALK-FORWARD VALIDATION")
print("=" * 80)
print()

results_df = walk_forward_validation(events_df, tmf, 
                                     train_years=5, 
                                     test_years=2, 
                                     step_years=1)

# Save results
results_df.to_csv(f'{OUTPUT_DIR}/walk_forward_results.csv', index=False)
print(f"Saved results to {OUTPUT_DIR}/walk_forward_results.csv")
print()

# ============================================================================
# STEP 4: ANALYSIS
# ============================================================================

print("=" * 80)
print("STEP 3: ANALYSIS")
print("=" * 80)
print()

# 1. CONSISTENCY CHECK
print("1. CONSISTENCY CHECK")
print("-" * 80)

avg_test_sharpe = results_df['test_sharpe'].mean()
std_test_sharpe = results_df['test_sharpe'].std()
min_test_sharpe = results_df['test_sharpe'].min()
max_test_sharpe = results_df['test_sharpe'].max()

print(f"Test Sharpe statistics:")
print(f"  Mean:   {avg_test_sharpe:+.2f}")
print(f"  Median: {results_df['test_sharpe'].median():+.2f}")
print(f"  Std:    {std_test_sharpe:.2f}")
print(f"  Min:    {min_test_sharpe:+.2f}")
print(f"  Max:    {max_test_sharpe:+.2f}")
print()

if std_test_sharpe < 0.3:
    print("✅ CONSISTENT: Low variance across periods")
elif std_test_sharpe < 0.5:
    print("⚠️  MODERATELY CONSISTENT: Some variance")
else:
    print("❌ HIGH VARIANCE: Regime dependent")
print()

# 2. DEGRADATION DETECTION
print("2. DEGRADATION DETECTION")
print("-" * 80)

x = np.arange(len(results_df))
y = results_df['test_sharpe'].values

if len(x) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    print(f"Trend analysis:")
    print(f"  Slope: {slope:+.3f} (Sharpe change per window)")
    print(f"  R²: {r_value**2:.3f}")
    print(f"  P-value: {p_value:.3f}")
    print()
    
    if slope < -0.05:
        print("❌ DEGRADING: Strategy performance declining over time")
    elif slope > 0.05:
        print("✅ IMPROVING: Strategy performance increasing over time")
    else:
        print("✅ STABLE: No significant trend over time")
else:
    print("Not enough windows for trend analysis")
    slope = 0
    
print()

# 3. WORST-CASE ANALYSIS
print("3. WORST-CASE ANALYSIS")
print("-" * 80)

worst_idx = results_df['test_sharpe'].idxmin()
worst_window = results_df.loc[worst_idx]

print(f"Worst test period:")
print(f"  Window: {worst_window['window']}")
print(f"  Test Sharpe: {worst_window['test_sharpe']:+.2f}")
print(f"  Test Win Rate: {worst_window['test_win']:.1f}%")
print(f"  Test Return: {worst_window['test_total']:+.1f}%")
print()

if worst_window['test_sharpe'] > 0.2:
    print("✅ ROBUST: Even worst period is positive (Sharpe > 0.2)")
elif worst_window['test_sharpe'] > 0:
    print("⚠️  MARGINAL: Worst period is slightly positive")
else:
    print("❌ RISKY: Some periods have negative Sharpe")
print()

# Best case
best_idx = results_df['test_sharpe'].idxmax()
best_window = results_df.loc[best_idx]

print(f"Best test period:")
print(f"  Window: {best_window['window']}")
print(f"  Test Sharpe: {best_window['test_sharpe']:+.2f}")
print(f"  Test Win Rate: {best_window['test_win']:.1f}%")
print(f"  Test Return: {best_window['test_total']:+.1f}%")
print()

# 4. OVERFITTING CHECK
print("4. OVERFITTING CHECK")
print("-" * 80)

avg_delta = results_df['sharpe_delta'].mean()
positive_deltas = (results_df['sharpe_delta'] > 0).sum()
total_windows = len(results_df)

print(f"Train vs Test comparison:")
print(f"  Avg Sharpe Delta (test - train): {avg_delta:+.2f}")
print(f"  Windows where test > train: {positive_deltas}/{total_windows} ({positive_deltas/total_windows*100:.1f}%)")
print()

if avg_delta < -0.2:
    print("❌ OVERFITTING DETECTED: Test consistently worse than train")
elif avg_delta > 0.3:
    print("⚠️  SUSPICIOUS: Test >> Train (possible lookahead bias or lucky periods)")
else:
    print("✅ NO OVERFITTING: Train/test results balanced")
print()

if positive_deltas / total_windows > 0.7:
    print("⚠️  Too many test > train periods - may indicate lucky recent years")
elif positive_deltas / total_windows < 0.3:
    print("⚠️  Too many train > test periods - possible overfitting")
else:
    print("✅ Balanced mix of test > train and train > test")
print()

# 5. POSITIVE PERIODS CHECK
print("5. POSITIVE PERIODS CHECK")
print("-" * 80)

positive_test = (results_df['test_sharpe'] > 0).sum()
print(f"Test periods with positive Sharpe: {positive_test}/{total_windows} ({positive_test/total_windows*100:.1f}%)")

if positive_test == total_windows:
    print("✅ EXCELLENT: All test periods positive!")
elif positive_test / total_windows >= 0.7:
    print("✅ GOOD: Most test periods positive")
elif positive_test / total_windows >= 0.5:
    print("⚠️  MIXED: About half periods positive")
else:
    print("❌ POOR: Most periods negative")
print()

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

print("=" * 80)
print("STEP 4: GENERATING VISUALIZATIONS")
print("=" * 80)
print()

try:
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Walk-Forward Validation Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Test Sharpe across windows
    ax1 = axes[0, 0]
    ax1.plot(range(len(results_df)), results_df['test_sharpe'], marker='o', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax1.axhline(y=avg_test_sharpe, color='g', linestyle='--', alpha=0.7, label=f'Mean: {avg_test_sharpe:.2f}')
    
    # Add trend line
    if len(x) > 1:
        trend_line = slope * x + intercept
        ax1.plot(x, trend_line, 'r--', alpha=0.5, label=f'Trend: {slope:+.3f}')
    
    ax1.set_title('Test Sharpe Across Windows')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('Test Sharpe')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train vs Test Sharpe
    ax2 = axes[0, 1]
    ax2.scatter(results_df['train_sharpe'], results_df['test_sharpe'], s=100, alpha=0.6)
    
    # Add diagonal line (perfect correlation)
    min_val = min(results_df['train_sharpe'].min(), results_df['test_sharpe'].min())
    max_val = max(results_df['train_sharpe'].max(), results_df['test_sharpe'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
    
    ax2.set_title('Train vs Test Sharpe')
    ax2.set_xlabel('Train Sharpe')
    ax2.set_ylabel('Test Sharpe')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sharpe Delta distribution
    ax3 = axes[1, 0]
    ax3.bar(range(len(results_df)), results_df['sharpe_delta'], alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax3.axhline(y=avg_delta, color='g', linestyle='--', alpha=0.7, label=f'Mean: {avg_delta:+.2f}')
    ax3.set_title('Sharpe Delta (Test - Train)')
    ax3.set_xlabel('Window Index')
    ax3.set_ylabel('Sharpe Delta')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Win Rate comparison
    ax4 = axes[1, 1]
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax4.bar(x_pos - width/2, results_df['train_win'], width, label='Train', alpha=0.7)
    ax4.bar(x_pos + width/2, results_df['test_win'], width, label='Test', alpha=0.7)
    ax4.set_title('Win Rate: Train vs Test')
    ax4.set_xlabel('Window Index')
    ax4.set_ylabel('Win Rate (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/walk_forward_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {OUTPUT_DIR}/walk_forward_analysis.png")
    print()
    
except Exception as e:
    print(f"Error creating visualization: {e}")
    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 10.7 COMPLETE - FINAL VERDICT")
print("=" * 80)
print()

# Calculate overall score
score = 0
max_score = 5

print("SCORING:")
print("-" * 80)

# 1. Consistency
if std_test_sharpe < 0.3:
    print("✅ Consistency: PASS (+1)")
    score += 1
else:
    print("❌ Consistency: FAIL (0)")

# 2. No degradation
if abs(slope) < 0.05:
    print("✅ Stability: PASS (+1)")
    score += 1
else:
    print("❌ Stability: FAIL (0)")

# 3. Worst case positive
if worst_window['test_sharpe'] > 0:
    print("✅ Worst-case: PASS (+1)")
    score += 1
else:
    print("❌ Worst-case: FAIL (0)")

# 4. No overfitting
if abs(avg_delta) < 0.3:
    print("✅ No overfitting: PASS (+1)")
    score += 1
else:
    print("❌ No overfitting: FAIL (0)")

# 5. Most periods positive
if positive_test / total_windows >= 0.7:
    print("✅ Positive periods: PASS (+1)")
    score += 1
else:
    print("❌ Positive periods: FAIL (0)")

print()
print(f"FINAL SCORE: {score}/{max_score}")
print()

# Final recommendation
if score >= 4:
    print("🎉 EXCELLENT: Strategy is ROBUST and VALIDATED")
    print("   → Proceed to paper trading with HIGH confidence")
    print("   → Expected to work consistently across market regimes")
elif score >= 3:
    print("✅ GOOD: Strategy is VALIDATED with some caveats")
    print("   → Proceed to paper trading with MEDIUM confidence")
    print("   → Monitor closely for regime changes")
elif score >= 2:
    print("⚠️  MARGINAL: Strategy shows WEAK validation")
    print("   → Paper trade with LOW confidence")
    print("   → Be ready to stop if live results diverge")
else:
    print("❌ POOR: Strategy FAILED validation")
    print("   → Do NOT paper trade")
    print("   → Consider pivoting to different approach")

print()
print("Files saved:")
print(f"  {OUTPUT_DIR}/walk_forward_results.csv")
print(f"  {OUTPUT_DIR}/walk_forward_analysis.png")
print()
