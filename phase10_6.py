"""
PHASE 10.6: EXTENDED HISTORICAL VALIDATION (2010-2024)
=======================================================
Extend data from 2020-2024 (5 years) to 2010-2024 (15 years)
for proper statistical validation

Goals:
1. Collect CPI/NFP data 2010-2024 (~360 events vs 117 current)
2. Test strategy across different macro regimes
3. Honest train/test split (2010-2019 train, 2020-2024 test)
4. Check if recent years were just lucky
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os

OUTPUT_DIR = 'outputs_phase10_6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 10.6: EXTENDED HISTORICAL VALIDATION (2010-2024)")
print("=" * 80)
print()

# ============================================================================
# STEP 1: DOWNLOAD TMF OHLC (2010-2024)
# ============================================================================

print("=" * 80)
print("STEP 1: DOWNLOAD TMF OHLC DATA (2010-2024)")
print("=" * 80)
print()

# TMF inception: 2009-04-16
START_DATE = '2010-01-01'
END_DATE = '2024-11-15'

print(f"Downloading TMF from {START_DATE} to {END_DATE}...")
tmf = yf.download('TMF', start=START_DATE, end=END_DATE, progress=False)

if isinstance(tmf.columns, pd.MultiIndex):
    tmf.columns = tmf.columns.get_level_values(0)

tmf = tmf[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

print(f"  Downloaded: {len(tmf)} trading days")
print(f"  Date range: {tmf.index[0].date()} to {tmf.index[-1].date()}")
print(f"  Missing days: {tmf.isnull().sum().sum()}")

tmf.to_csv(f'{OUTPUT_DIR}/tmf_ohlc_extended.csv')
print(f"  Saved to {OUTPUT_DIR}/tmf_ohlc_extended.csv")
print()

# ============================================================================
# STEP 2: LOAD/CREATE EXTENDED ECONOMIC EVENTS
# ============================================================================

print("=" * 80)
print("STEP 2: ECONOMIC EVENTS DATA (2010-2024)")
print("=" * 80)
print()

events_file = f'{OUTPUT_DIR}/economic_events_extended.csv'

if not os.path.exists(events_file):
    print("Creating extended events template...")
    print()
    print("ACTION REQUIRED:")
    print("=" * 80)
    print("1. Go to: https://www.investing.com/economic-calendar/")
    print("2. Filter: United States")
    print("3. Select events: CPI, Core CPI, NFP")
    print("4. Date range: 2010-01-01 to 2024-11-15")
    print("5. Export data and fill into template")
    print()
    print("Template columns: Date, Event_Type, Actual, Consensus, Previous")
    print()
    print("For now, creating template with recent data only...")
    print()
    
    # Load current Phase 10 data as starting point
    current_events = pd.read_csv('outputs_phase10/economic_events.csv')
    current_events.to_csv(events_file, index=False)
    
    print(f"  Template created with {len(current_events)} events")
    print(f"  Saved to {events_file}")
    print()
    print("  NOTE: This is PARTIAL data (2020-2024 only)")
    print("  For full validation, extend back to 2010!")
    print()
    print("  Quick way to extend:")
    print("  - Manually add ~120 more CPI events (2010-2019)")
    print("  - Takes ~2-3 hours but worth it for validation")
    print()
    
    events = current_events
    print("  CONTINUING WITH PARTIAL DATA FOR NOW...")
    print("  Results will be same as Phase 10 until data extended")
    print()
else:
    print(f"Loading extended events from {events_file}...")
    events = pd.read_csv(events_file)
    events['Date'] = pd.to_datetime(events['Date'])
    
    print(f"  Loaded: {len(events)} events")
    print(f"  Event types: {events['Event_Type'].unique().tolist()}")
    print(f"  Date range: {events['Date'].min().date()} to {events['Date'].max().date()}")
    
    # Check coverage by year
    events['year'] = events['Date'].dt.year
    yearly_counts = events.groupby('year').size()
    print()
    print("  Events per year:")
    for year, count in yearly_counts.items():
        print(f"    {year}: {count} events")
    print()

# ============================================================================
# STEP 3: REGIME ANALYSIS
# ============================================================================

print("=" * 80)
print("STEP 3: MACRO REGIME BREAKDOWN")
print("=" * 80)
print()

# Define macro regimes
regimes = {
    'QE Era (2010-2015)': (2010, 2015),
    'Normalization (2016-2019)': (2016, 2019),
    'COVID Era (2020-2021)': (2020, 2021),
    'Inflation Fight (2022-2024)': (2022, 2024)
}

print("Checking event distribution by regime:")
print("-" * 80)

events['year'] = pd.to_datetime(events['Date']).dt.year

for regime_name, (start_year, end_year) in regimes.items():
    regime_events = events[(events['year'] >= start_year) & (events['year'] <= end_year)]
    cpi_count = len(regime_events[regime_events['Event_Type'] == 'CPI'])
    nfp_count = len(regime_events[regime_events['Event_Type'] == 'NFP'])
    
    print(f"{regime_name}:")
    print(f"  Years: {start_year}-{end_year} ({end_year - start_year + 1} years)")
    print(f"  CPI events: {cpi_count}")
    print(f"  NFP events: {nfp_count}")
    print(f"  Total: {len(regime_events)}")
    print()

# ============================================================================
# STEP 4: QUICK SANITY CHECK (if data extended)
# ============================================================================

print("=" * 80)
print("STEP 4: QUICK SANITY CHECK ON OLD DATA")
print("=" * 80)
print()

# Check if we have pre-2020 data
pre_2020_events = events[events['year'] < 2020]

if len(pre_2020_events) > 10:
    print(f"Found {len(pre_2020_events)} events before 2020")
    print("Running quick test on 2010-2019 data...")
    print()
    
    # Import backtest functions
    import sys
    sys.path.insert(0, '.')
    from phase10 import backtest_strategy, calculate_metrics, merge_events_with_prices
    
    # Merge with prices
    merged_old = merge_events_with_prices(pre_2020_events, tmf)
    
    if len(merged_old) > 10:
        # Filter CPI only
        cpi_old = merged_old[merged_old['event_type'] == 'CPI']
        
        if len(cpi_old) > 5:
            print(f"CPI events 2010-2019: {len(cpi_old)}")
            
            # Test with conservative config
            trades_old = backtest_strategy(cpi_old, tmf, holding_days=5, 
                                          stop_type='ohlc', stop_pct=3)
            metrics_old = calculate_metrics(trades_old, "CPI 2010-2019")
            
            if metrics_old:
                print(f"  Sharpe: {metrics_old['sharpe']:.2f}")
                print(f"  Win Rate: {metrics_old['win_rate']:.1f}%")
                print(f"  Avg Return: {metrics_old['avg_return']:+.2f}%")
                print(f"  Total: {metrics_old['total_return']:+.1f}%")
                print()
                
                if metrics_old['sharpe'] > 0.3:
                    print("  ✅ Strategy shows edge on old data!")
                elif metrics_old['sharpe'] > 0:
                    print("  ⚠️  Weak edge on old data (Sharpe < 0.3)")
                else:
                    print("  ❌ No edge on old data (Sharpe < 0)")
                print()
        else:
            print("  Not enough CPI events in old data")
            print()
    else:
        print("  Not enough merged events for testing")
        print()
else:
    print("No pre-2020 data found.")
    print("Current dataset only covers 2020-2024.")
    print()
    print("TO EXTEND DATA:")
    print("1. Manually collect CPI/NFP from Investing.com")
    print("2. Add to economic_events_extended.csv")
    print("3. Re-run this script")
    print()

# ============================================================================
# STEP 5: COMPREHENSIVE TRAIN/TEST (if full data)
# ============================================================================

print("=" * 80)
print("STEP 5: COMPREHENSIVE TRAIN/TEST SPLIT")
print("=" * 80)
print()

if events['year'].min() <= 2012:  # Check if we have meaningful old data
    print("Running comprehensive train/test validation...")
    print()
    
    from phase10 import backtest_strategy, calculate_metrics, merge_events_with_prices
    
    # Merge all events
    merged_all = merge_events_with_prices(events, tmf)
    cpi_all = merged_all[merged_all['event_type'] == 'CPI']
    
    # Main split: 2010-2019 vs 2020-2024
    cpi_all['year'] = pd.to_datetime(cpi_all['entry_date']).dt.year
    
    train_data = cpi_all[cpi_all['year'] < 2020]
    test_data = cpi_all[cpi_all['year'] >= 2020]
    
    print(f"Train period: 2010-2019 ({len(train_data)} CPI events)")
    print(f"Test period:  2020-2024 ({len(test_data)} CPI events)")
    print()
    
    # Test conservative config on both
    print("Testing CPI only, 5-day hold, OHLC 3% stop:")
    print("-" * 80)
    
    if len(train_data) > 10:
        trades_train = backtest_strategy(train_data, tmf, holding_days=5, 
                                         stop_type='ohlc', stop_pct=3)
        metrics_train = calculate_metrics(trades_train, "Train 2010-2019")
        
        if metrics_train:
            print(f"TRAIN: Sharpe {metrics_train['sharpe']:.2f}, Win {metrics_train['win_rate']:.1f}%, Total {metrics_train['total_return']:+.1f}%")
    
    if len(test_data) > 10:
        trades_test = backtest_strategy(test_data, tmf, holding_days=5, 
                                        stop_type='ohlc', stop_pct=3)
        metrics_test = calculate_metrics(trades_test, "Test 2020-2024")
        
        if metrics_test:
            print(f"TEST:  Sharpe {metrics_test['sharpe']:.2f}, Win {metrics_test['win_rate']:.1f}%, Total {metrics_test['total_return']:+.1f}%")
    
    if 'metrics_train' in locals() and 'metrics_test' in locals() and metrics_train and metrics_test:
        delta = metrics_test['sharpe'] - metrics_train['sharpe']
        print(f"DELTA: {delta:+.2f}")
        print()
        
        if delta > 0.5:
            print("⚠️  WARNING: Test significantly outperforms train!")
            print("   Possible explanations:")
            print("   - Recent years (2020-2024) are lucky period")
            print("   - Strategy works better in high-volatility regimes")
            print("   - Small sample variance")
        elif delta < -0.5:
            print("⚠️  WARNING: Train significantly outperforms test!")
            print("   Possible overfitting to old data")
        else:
            print("✅ Train/test results consistent - strategy is robust")
        print()
        
        # Save results
        comparison = pd.DataFrame({
            'period': ['Train 2010-2019', 'Test 2020-2024'],
            'events': [len(train_data), len(test_data)],
            'sharpe': [metrics_train['sharpe'], metrics_test['sharpe']],
            'win_rate': [metrics_train['win_rate'], metrics_test['win_rate']],
            'total_return': [metrics_train['total_return'], metrics_test['total_return']]
        })
        comparison.to_csv(f'{OUTPUT_DIR}/train_test_comparison.csv', index=False)
else:
    print("Insufficient historical data for comprehensive validation")
    print(f"Current data: {events['year'].min()}-{events['year'].max()}")
    print()

# ============================================================================
# STEP 6: REGIME-SPECIFIC PERFORMANCE
# ============================================================================

print("=" * 80)
print("STEP 6: PERFORMANCE BY MACRO REGIME")
print("=" * 80)
print()

if 'merged_all' in locals() and 'cpi_all' in locals():
    print("Testing CPI strategy across different regimes:")
    print("-" * 80)
    
    from phase10 import backtest_strategy, calculate_metrics
    
    regime_results = []
    
    for regime_name, (start_year, end_year) in regimes.items():
        regime_data = cpi_all[(cpi_all['year'] >= start_year) & (cpi_all['year'] <= end_year)]
        
        if len(regime_data) < 3:
            print(f"{regime_name}: SKIP (only {len(regime_data)} events)")
            continue
        
        trades_regime = backtest_strategy(regime_data, tmf, holding_days=5, 
                                         stop_type='ohlc', stop_pct=3)
        metrics_regime = calculate_metrics(trades_regime, regime_name)
        
        if metrics_regime:
            print(f"{regime_name}:")
            print(f"  Events: {len(regime_data)}")
            print(f"  Sharpe: {metrics_regime['sharpe']:+.2f}")
            print(f"  Win Rate: {metrics_regime['win_rate']:.1f}%")
            print(f"  Total: {metrics_regime['total_return']:+.1f}%")
            print()
            
            regime_results.append({
                'regime': regime_name,
                'years': f"{start_year}-{end_year}",
                'events': len(regime_data),
                'sharpe': metrics_regime['sharpe'],
                'win_rate': metrics_regime['win_rate'],
                'total_return': metrics_regime['total_return']
            })
    
    if len(regime_results) > 0:
        pd.DataFrame(regime_results).to_csv(f'{OUTPUT_DIR}/regime_performance.csv', index=False)
        
        print("REGIME SUMMARY:")
        print("-" * 80)
        best_regime = max(regime_results, key=lambda x: x['sharpe'])
        worst_regime = min(regime_results, key=lambda x: x['sharpe'])
        
        print(f"Best regime:  {best_regime['regime']} (Sharpe {best_regime['sharpe']:+.2f})")
        print(f"Worst regime: {worst_regime['regime']} (Sharpe {worst_regime['sharpe']:+.2f})")
        print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 10.6 COMPLETE")
print("=" * 80)
print()

print("Data Coverage:")
print(f"  TMF OHLC: {tmf.index[0].date()} to {tmf.index[-1].date()} ({len(tmf)} days)")
if len(events) > 0:
    events_temp = events.copy()
    events_temp['Date'] = pd.to_datetime(events_temp['Date'])
    print(f"  Events: {events_temp['Date'].min().date()} to {events_temp['Date'].max().date()} ({len(events)} events)")
else:
    print(f"  Events: N/A")
print()

print("Files saved:")
print(f"  {OUTPUT_DIR}/tmf_ohlc_extended.csv")
print(f"  {OUTPUT_DIR}/economic_events_extended.csv")
if os.path.exists(f'{OUTPUT_DIR}/train_test_comparison.csv'):
    print(f"  {OUTPUT_DIR}/train_test_comparison.csv")
if os.path.exists(f'{OUTPUT_DIR}/regime_performance.csv'):
    print(f"  {OUTPUT_DIR}/regime_performance.csv")
print()

if events['year'].min() > 2015:
    print("⚠️  DATA EXTENSION NEEDED:")
    print("=" * 80)
    print()
    print("Current data only covers recent years.")
    print("For full validation, extend to 2010!")
    print()
    print("QUICK GUIDE:")
    print("1. Visit: https://www.investing.com/economic-calendar/")
    print("2. Filter: United States, CPI")
    print("3. Set dates: 2010-01-01 to 2019-12-31")
    print("4. Copy data to economic_events_extended.csv")
    print("5. Re-run this script")
    print()
    print("Expected additions:")
    print("  - CPI: ~120 events (2010-2019)")
    print("  - Takes ~2-3 hours manual work")
    print("  - Worth it for statistical validation!")
    print()
else:
    print("✅ Full historical data validated!")
    print()
    print("Next steps:")
    print("1. Review regime_performance.csv")
    print("2. Check train_test_comparison.csv")
    print("3. If strategy robust → paper trade")
    print("4. If recent years lucky → be cautious")
    print()
