"""
PHASE 10.8: VIX REGIME FILTER
==============================
Add VIX filter to skip trades in calm markets (VIX < 15)
Compare filtered vs unfiltered performance

Hypothesis: Strategy works better in high-volatility regimes
Expected: Avoid the 2016-2017 disaster period
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

OUTPUT_DIR = 'outputs_phase10_8'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 10.8: VIX REGIME FILTER")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("STEP 1: Loading data...")
print("-" * 80)

# Load events from Phase 10.6
events_file = 'outputs_phase10_6/economic_events_extended.csv'
if not os.path.exists(events_file):
    print(f"ERROR: {events_file} not found!")
    exit(1)

events_df = pd.read_csv(events_file)
events_df['Date'] = pd.to_datetime(events_df['Date'])

print(f"Events: {len(events_df)}")
print(f"Date range: {events_df['Date'].min().date()} to {events_df['Date'].max().date()}")
print()

# Load TMF
print("Loading TMF OHLC...")
tmf = yf.download('TMF', start='2010-01-01', end='2024-11-15', progress=False)
if isinstance(tmf.columns, pd.MultiIndex):
    tmf.columns = tmf.columns.get_level_values(0)
tmf = tmf[['Open', 'High', 'Low', 'Close']].copy()
print(f"TMF: {len(tmf)} days")
print()

# Load VIX
print("Loading VIX data...")
vix = yf.download('^VIX', start='2010-01-01', end='2024-11-15', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix = vix[['Close']].copy()
vix.columns = ['VIX']
print(f"VIX: {len(vix)} days")
print(f"VIX range: {vix['VIX'].min():.1f} to {vix['VIX'].max():.1f}")
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
# STEP 2: VIX-FILTERED BACKTEST FUNCTION
# ============================================================================

def backtest_with_vix_filter(events_df, tmf_ohlc, vix_df, 
                             vix_threshold=15,
                             holding_days=5, 
                             stop_type='ohlc', 
                             stop_pct=3):
    """
    Backtest with VIX regime filter
    
    Args:
        events_df: Economic events
        tmf_ohlc: TMF OHLC data
        vix_df: VIX data
        vix_threshold: Skip if VIX < threshold (default 15)
        holding_days: Days to hold
        stop_type: Stop type
        stop_pct: Stop percentage
    
    Returns:
        (trades_df, skipped_count, avg_vix_traded, avg_vix_skipped)
    """
    
    # Merge events with prices first
    merged_df = merge_events_with_prices(events_df, tmf_ohlc)
    
    # Filter CPI only
    cpi_df = merged_df[merged_df['event_type'] == 'CPI'].copy()
    
    # Add VIX values on entry dates
    cpi_df['vix_on_entry'] = cpi_df['entry_date'].apply(
        lambda date: vix_df.loc[vix_df.index >= date, 'VIX'].iloc[0] 
        if len(vix_df.loc[vix_df.index >= date]) > 0 else np.nan
    )
    
    # Split into traded vs skipped
    traded = cpi_df[cpi_df['vix_on_entry'] >= vix_threshold].copy()
    skipped = cpi_df[cpi_df['vix_on_entry'] < vix_threshold].copy()
    
    skipped_count = len(skipped)
    avg_vix_traded = traded['vix_on_entry'].mean() if len(traded) > 0 else 0
    avg_vix_skipped = skipped['vix_on_entry'].mean() if len(skipped) > 0 else 0
    
    # Backtest only traded
    if len(traded) == 0:
        return pd.DataFrame(), skipped_count, avg_vix_traded, avg_vix_skipped
    
    trades = backtest_strategy(traded, tmf_ohlc, 
                               holding_days=holding_days,
                               stop_type=stop_type, 
                               stop_pct=stop_pct)
    
    return trades, skipped_count, avg_vix_traded, avg_vix_skipped

# ============================================================================
# STEP 3: WALK-FORWARD WITH VIX FILTER
# ============================================================================

print("=" * 80)
print("STEP 2: WALK-FORWARD VALIDATION (WITH VIX FILTER)")
print("=" * 80)
print()

def walk_forward_with_vix(events_df, tmf_ohlc, vix_df,
                         train_years=5, 
                         test_years=2,
                         step_years=1,
                         vix_threshold=15):
    """Walk-forward validation with VIX filter"""
    
    results = []
    
    min_year = events_df['Date'].dt.year.min()
    max_year = events_df['Date'].dt.year.max()
    
    print(f"VIX threshold: {vix_threshold}")
    print(f"Trades will be skipped if VIX < {vix_threshold}")
    print()
    
    for train_start_year in range(min_year, max_year - train_years - test_years + 2, step_years):
        train_end_year = train_start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1
        
        if test_end_year > max_year:
            break
        
        # Filter data
        train_data = events_df[
            (events_df['Date'].dt.year >= train_start_year) &
            (events_df['Date'].dt.year <= train_end_year)
        ].copy()
        
        test_data = events_df[
            (events_df['Date'].dt.year >= test_start_year) &
            (events_df['Date'].dt.year <= test_end_year)
        ].copy()
        
        if len(train_data) < 20 or len(test_data) < 10:
            continue
        
        # Backtest TRAIN (no VIX filter on train - we want to see full performance)
        train_merged = merge_events_with_prices(train_data, tmf_ohlc)
        train_cpi = train_merged[train_merged['event_type'] == 'CPI'].copy()
        
        if len(train_cpi) < 10:
            continue
        
        trades_train = backtest_strategy(train_cpi, tmf_ohlc,
                                        holding_days=5,
                                        stop_type='ohlc',
                                        stop_pct=3)
        metrics_train = calculate_metrics(trades_train, f"Train {train_start_year}-{train_end_year}")
        
        # Backtest TEST (WITH VIX filter)
        trades_test, skipped, avg_vix_traded, avg_vix_skipped = backtest_with_vix_filter(
            test_data, tmf_ohlc, vix_df,
            vix_threshold=vix_threshold,
            holding_days=5,
            stop_type='ohlc',
            stop_pct=3
        )
        
        if len(trades_test) == 0:
            print(f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}: All trades skipped by VIX filter")
            continue
        
        metrics_test = calculate_metrics(trades_test, f"Test {test_start_year}-{test_end_year}")
        
        if not metrics_train or not metrics_test:
            continue
        
        # Also run unfiltered test for comparison
        test_merged = merge_events_with_prices(test_data, tmf_ohlc)
        test_cpi = test_merged[test_merged['event_type'] == 'CPI'].copy()
        
        trades_test_unfiltered = backtest_strategy(test_cpi, tmf_ohlc,
                                                   holding_days=5,
                                                   stop_type='ohlc',
                                                   stop_pct=3)
        metrics_test_unfiltered = calculate_metrics(trades_test_unfiltered, f"Test Unfiltered {test_start_year}-{test_end_year}")
        
        # Store results
        window_result = {
            'window': f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}",
            'train_sharpe': metrics_train['sharpe'],
            'test_sharpe_filtered': metrics_test['sharpe'],
            'test_sharpe_unfiltered': metrics_test_unfiltered['sharpe'] if metrics_test_unfiltered else 0,
            'test_events_total': len(test_cpi),
            'test_events_traded': len(trades_test),
            'test_events_skipped': skipped,
            'avg_vix_traded': avg_vix_traded,
            'avg_vix_skipped': avg_vix_skipped,
            'improvement': metrics_test['sharpe'] - metrics_test_unfiltered['sharpe'] if metrics_test_unfiltered else 0
        }
        
        results.append(window_result)
        
        print(f"{train_start_year}-{train_end_year} → {test_start_year}-{test_end_year}:")
        print(f"  Train:               Sharpe {metrics_train['sharpe']:+.2f}")
        print(f"  Test (unfiltered):   Sharpe {metrics_test_unfiltered['sharpe']:+.2f} ({len(test_cpi)} events)")
        print(f"  Test (VIX filtered): Sharpe {metrics_test['sharpe']:+.2f} ({len(trades_test)} traded, {skipped} skipped)")
        print(f"  Improvement:         {window_result['improvement']:+.2f}")
        print(f"  VIX traded avg:      {avg_vix_traded:.1f}")
        print(f"  VIX skipped avg:     {avg_vix_skipped:.1f}")
        print()
    
    return pd.DataFrame(results)

# Run walk-forward with VIX filter
results_vix = walk_forward_with_vix(events_df, tmf, vix,
                                    train_years=5,
                                    test_years=2,
                                    step_years=1,
                                    vix_threshold=15)

# Save results
results_vix.to_csv(f'{OUTPUT_DIR}/walk_forward_vix_filtered.csv', index=False)
print(f"Saved to {OUTPUT_DIR}/walk_forward_vix_filtered.csv")
print()

# ============================================================================
# STEP 4: ANALYSIS
# ============================================================================

print("=" * 80)
print("STEP 3: ANALYSIS - VIX FILTER IMPACT")
print("=" * 80)
print()

# Load Phase 10.7 unfiltered results for comparison
phase10_7_results = pd.read_csv('outputs_phase10_7/walk_forward_results.csv')

print("1. FILTER EFFECTIVENESS")
print("-" * 80)

avg_improvement = results_vix['improvement'].mean()
positive_improvements = (results_vix['improvement'] > 0).sum()
total_windows = len(results_vix)

print(f"Avg Sharpe improvement: {avg_improvement:+.2f}")
print(f"Windows improved: {positive_improvements}/{total_windows} ({positive_improvements/total_windows*100:.1f}%)")
print()

if avg_improvement > 0.1:
    print("✅ VIX FILTER HELPS: Consistent improvement across windows")
elif avg_improvement > 0:
    print("⚠️  VIX FILTER MARGINAL: Small positive effect")
else:
    print("❌ VIX FILTER HURTS: Making performance worse")
print()

# 2. Trade frequency impact
print("2. TRADE FREQUENCY IMPACT")
print("-" * 80)

total_events = results_vix['test_events_total'].sum()
total_traded = results_vix['test_events_traded'].sum()
total_skipped = results_vix['test_events_skipped'].sum()

print(f"Total test events: {total_events}")
print(f"Traded: {total_traded} ({total_traded/total_events*100:.1f}%)")
print(f"Skipped: {total_skipped} ({total_skipped/total_events*100:.1f}%)")
print()

avg_vix_traded = results_vix['avg_vix_traded'].mean()
avg_vix_skipped = results_vix['avg_vix_skipped'].mean()

print(f"Avg VIX when traded: {avg_vix_traded:.1f}")
print(f"Avg VIX when skipped: {avg_vix_skipped:.1f}")
print()

# 3. Performance comparison
print("3. PERFORMANCE COMPARISON")
print("-" * 80)

filtered_sharpe = results_vix['test_sharpe_filtered'].mean()
unfiltered_sharpe = results_vix['test_sharpe_unfiltered'].mean()

filtered_positive = (results_vix['test_sharpe_filtered'] > 0).sum()
unfiltered_positive = (results_vix['test_sharpe_unfiltered'] > 0).sum()

print(f"Filtered (VIX >= 15):")
print(f"  Avg Sharpe: {filtered_sharpe:.2f}")
print(f"  Positive windows: {filtered_positive}/{total_windows}")
print()

print(f"Unfiltered (all trades):")
print(f"  Avg Sharpe: {unfiltered_sharpe:.2f}")
print(f"  Positive windows: {unfiltered_positive}/{total_windows}")
print()

print(f"Improvement: {filtered_sharpe - unfiltered_sharpe:+.2f} Sharpe points")
print()

# 4. Worst case comparison
print("4. WORST CASE COMPARISON")
print("-" * 80)

worst_filtered = results_vix.loc[results_vix['test_sharpe_filtered'].idxmin()]
worst_unfiltered = results_vix.loc[results_vix['test_sharpe_unfiltered'].idxmin()]

print(f"Worst window (filtered): {worst_filtered['window']}")
print(f"  Sharpe: {worst_filtered['test_sharpe_filtered']:+.2f}")
print(f"  Events traded: {worst_filtered['test_events_traded']}")
print()

print(f"Worst window (unfiltered): {worst_unfiltered['window']}")
print(f"  Sharpe: {worst_unfiltered['test_sharpe_unfiltered']:+.2f}")
print()

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

print("=" * 80)
print("STEP 4: GENERATING VISUALIZATIONS")
print("=" * 80)
print()

try:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VIX Filter Impact Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Filtered vs Unfiltered Sharpe
    ax1 = axes[0, 0]
    x = np.arange(len(results_vix))
    width = 0.35
    ax1.bar(x - width/2, results_vix['test_sharpe_unfiltered'], width, 
            label='Unfiltered', alpha=0.7, color='blue')
    ax1.bar(x + width/2, results_vix['test_sharpe_filtered'], width, 
            label='VIX Filtered', alpha=0.7, color='green')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Test Sharpe: Filtered vs Unfiltered')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement per window
    ax2 = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in results_vix['improvement']]
    ax2.bar(range(len(results_vix)), results_vix['improvement'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=avg_improvement, color='blue', linestyle='--', 
                label=f'Mean: {avg_improvement:+.2f}')
    ax2.set_title('Sharpe Improvement from VIX Filter')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Sharpe Improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trade frequency
    ax3 = axes[1, 0]
    traded_pct = results_vix['test_events_traded'] / results_vix['test_events_total'] * 100
    skipped_pct = results_vix['test_events_skipped'] / results_vix['test_events_total'] * 100
    
    ax3.bar(range(len(results_vix)), traded_pct, label='Traded', alpha=0.7, color='green')
    ax3.bar(range(len(results_vix)), skipped_pct, bottom=traded_pct, 
            label='Skipped', alpha=0.7, color='red')
    ax3.set_title('Trade Frequency (% of events)')
    ax3.set_xlabel('Window Index')
    ax3.set_ylabel('Percentage (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: VIX levels
    ax4 = axes[1, 1]
    ax4.scatter(results_vix['avg_vix_traded'], results_vix['test_sharpe_filtered'], 
                s=100, alpha=0.6, label='Traded')
    ax4.axvline(x=15, color='r', linestyle='--', label='VIX Threshold')
    ax4.set_title('VIX Level vs Performance')
    ax4.set_xlabel('Avg VIX (Traded)')
    ax4.set_ylabel('Test Sharpe (Filtered)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/vix_filter_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {OUTPUT_DIR}/vix_filter_analysis.png")
    print()
    
except Exception as e:
    print(f"Error creating visualization: {e}")
    print()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("=" * 80)
print("PHASE 10.8 COMPLETE - VIX FILTER VERDICT")
print("=" * 80)
print()

score = 0
max_score = 3

print("SCORING:")
print("-" * 80)

# 1. Improves average Sharpe
if avg_improvement > 0.1:
    print("✅ Average improvement: PASS (+1)")
    score += 1
else:
    print("❌ Average improvement: FAIL (0)")

# 2. Improves most windows
if positive_improvements / total_windows > 0.6:
    print("✅ Consistency: PASS (+1)")
    score += 1
else:
    print("❌ Consistency: FAIL (0)")

# 3. Improves worst case
if worst_filtered['test_sharpe_filtered'] > worst_unfiltered['test_sharpe_unfiltered']:
    print("✅ Worst-case: PASS (+1)")
    score += 1
else:
    print("❌ Worst-case: FAIL (0)")

print()
print(f"FINAL SCORE: {score}/{max_score}")
print()

if score >= 2:
    print("✅ VIX FILTER RECOMMENDED")
    print(f"   → Use VIX >= 15 threshold")
    print(f"   → Expected Sharpe improvement: +{avg_improvement:.2f}")
    print(f"   → Trade frequency: {total_traded/total_events*100:.1f}% of events")
elif score == 1:
    print("⚠️  VIX FILTER MARGINAL")
    print("   → Small benefit, optional to use")
else:
    print("❌ VIX FILTER NOT RECOMMENDED")
    print("   → Trade all events without filter")

print()
print("Files saved:")
print(f"  {OUTPUT_DIR}/walk_forward_vix_filtered.csv")
print(f"  {OUTPUT_DIR}/vix_filter_analysis.png")
print()
