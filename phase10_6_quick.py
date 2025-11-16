"""
PHASE 10.6 QUICK: Quick & Dirty Historical Test using FRED
===========================================================
Use FRED API with "previous as consensus" proxy
Fast validation (30 min) to check if strategy works pre-2020

NOTE: This is NOT perfect (consensus = previous)
But gives quick indication before spending 3 hours on manual collection
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
import os

OUTPUT_DIR = 'outputs_phase10_6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 10.6 QUICK: FRED API TEST (2010-2024)")
print("=" * 80)
print()

# ============================================================================
# STEP 1: GET FRED DATA
# ============================================================================

print("STEP 1: Fetching data from FRED...")
print("-" * 80)

# Try pandas_datareader first (works without API key!)
USE_FRED_DATA = False
events_df = None

try:
    print("Attempting to fetch FRED data via pandas_datareader (no API key needed)...")
    from pandas_datareader import data as pdr
    
    # Fetch CPI
    print("  Fetching CPI (CPIAUCSL)...")
    cpi_raw = pdr.get_data_fred('CPIAUCSL', start='2010-01-01', end='2024-11-15')
    print(f"    ✅ CPI fetched: {len(cpi_raw)} observations")
    
    # Fetch NFP  
    print("  Fetching NFP (PAYEMS)...")
    nfp_raw = pdr.get_data_fred('PAYEMS', start='2010-01-01', end='2024-11-15')
    print(f"    ✅ NFP fetched: {len(nfp_raw)} observations")
    
    # Process CPI - convert to YoY % change
    cpi_df = pd.DataFrame({
        'Date': cpi_raw.index,
        'Actual': cpi_raw.values.flatten() if hasattr(cpi_raw.values, 'flatten') else cpi_raw.values
    })
    cpi_df['Actual_YoY'] = cpi_df['Actual'].pct_change(periods=12) * 100
    cpi_df['Consensus'] = cpi_df['Actual_YoY'].shift(1)  # Previous as consensus proxy
    cpi_df['Previous'] = cpi_df['Actual_YoY'].shift(1)
    cpi_df = cpi_df.iloc[13:].copy()  # Drop first 13 rows
    cpi_df['Actual'] = cpi_df['Actual_YoY']
    cpi_df['Event_Type'] = 'CPI'
    cpi_df = cpi_df[['Date', 'Event_Type', 'Actual', 'Consensus', 'Previous']].copy()
    
    # Process NFP - calculate monthly change
    nfp_df = pd.DataFrame({
        'Date': nfp_raw.index,
        'Level': nfp_raw.values.flatten() if hasattr(nfp_raw.values, 'flatten') else nfp_raw.values
    })
    nfp_df['Actual'] = nfp_df['Level'].diff()
    nfp_df['Consensus'] = nfp_df['Actual'].shift(1)  # Previous change as consensus proxy
    nfp_df['Previous'] = nfp_df['Actual'].shift(2)
    nfp_df = nfp_df.iloc[2:].copy()  # Drop first 2 rows
    nfp_df['Event_Type'] = 'NFP'
    nfp_df = nfp_df[['Date', 'Event_Type', 'Actual', 'Consensus', 'Previous']].copy()
    
    # Combine
    events_df = pd.concat([cpi_df, nfp_df], ignore_index=True)
    events_df = events_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n✅ Total events created: {len(events_df)}")
    print(f"   Date range: {events_df['Date'].min().date()} to {events_df['Date'].max().date()}")
    
    events_df.to_csv(f'{OUTPUT_DIR}/economic_events_extended.csv', index=False)
    print(f"   Saved to {OUTPUT_DIR}/economic_events_extended.csv")
    
    USE_FRED_DATA = True

except Exception as e:
    print(f"\n❌ pandas_datareader failed: {e}")
    print("   Trying fredapi with API key...")
    
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    if not FRED_API_KEY:
        FRED_API_KEY = 'ef636b6d99542d08f7d0ab6152932290'  # Hardcoded as fallback
    
    if FRED_API_KEY:
        fred = Fred(api_key=FRED_API_KEY)
    
    try:
        print("Fetching CPI (CPIAUCSL)...")
        # Get CPI - Consumer Price Index for All Urban Consumers
        cpi_series = fred.get_series('CPIAUCSL', 
                                     observation_start='2010-01-01', 
                                     observation_end='2024-11-15')
        
        # Convert to dataframe
        cpi_df = pd.DataFrame({
            'Date': cpi_series.index,
            'Actual': cpi_series.values
        })
        
        # Calculate year-over-year % change (this is what CPI reports)
        cpi_df['Actual_YoY'] = cpi_df['Actual'].pct_change(periods=12) * 100
        
        # Use previous month's YoY as "consensus" (proxy)
        cpi_df['Consensus'] = cpi_df['Actual_YoY'].shift(1)
        cpi_df['Previous'] = cpi_df['Actual_YoY'].shift(1)
        
        # Drop first 13 rows (need 12 for YoY + 1 for previous)
        cpi_df = cpi_df.iloc[13:].copy()
        
        cpi_df['Actual'] = cpi_df['Actual_YoY']
        cpi_df['Event_Type'] = 'CPI'
        cpi_df = cpi_df[['Date', 'Event_Type', 'Actual', 'Consensus', 'Previous']]
        
        print(f"  CPI: {len(cpi_df)} monthly observations")
        
        print("\nFetching NFP (PAYEMS)...")
        # Get NFP - Total Nonfarm Payrolls
        nfp_series = fred.get_series('PAYEMS', 
                                     observation_start='2010-01-01', 
                                     observation_end='2024-11-15')
        
        # Convert to dataframe
        nfp_df = pd.DataFrame({
            'Date': nfp_series.index,
            'Level': nfp_series.values
        })
        
        # Calculate monthly change (in thousands)
        nfp_df['Actual'] = nfp_df['Level'].diff()
        
        # Use previous month's change as "consensus" (proxy)
        nfp_df['Consensus'] = nfp_df['Actual'].shift(1)
        nfp_df['Previous'] = nfp_df['Actual'].shift(2)
        
        # Drop first 2 rows
        nfp_df = nfp_df.iloc[2:].copy()
        
        nfp_df['Event_Type'] = 'NFP'
        nfp_df = nfp_df[['Date', 'Event_Type', 'Actual', 'Consensus', 'Previous']]
        
        print(f"  NFP: {len(nfp_df)} monthly observations")
        
        # Combine
        events_df = pd.concat([cpi_df, nfp_df], ignore_index=True)
        events_df = events_df.sort_values('Date').reset_index(drop=True)
        
        print(f"\nTotal events: {len(events_df)}")
        print(f"Date range: {events_df['Date'].min().date()} to {events_df['Date'].max().date()}")
        
        # Save
        events_df.to_csv(f'{OUTPUT_DIR}/economic_events_extended.csv', index=False)
        print(f"Saved to {OUTPUT_DIR}/economic_events_extended.csv")
        
    except Exception as e:
        print(f"ERROR fetching FRED data: {e}")
        print("Falling back to existing data...")
        
        existing = pd.read_csv('outputs_phase10/economic_events.csv')
        events_df = existing
        events_df['Date'] = pd.to_datetime(events_df['Date'])

print()

# ============================================================================
# STEP 2: LOAD TMF DATA
# ============================================================================

print("STEP 2: Loading TMF OHLC...")
print("-" * 80)

tmf = yf.download('TMF', start='2010-01-01', end='2024-11-15', progress=False)
if isinstance(tmf.columns, pd.MultiIndex):
    tmf.columns = tmf.columns.get_level_values(0)
tmf = tmf[['Open', 'High', 'Low', 'Close']].copy()

print(f"TMF: {len(tmf)} days ({tmf.index[0].date()} to {tmf.index[-1].date()})")
print()

# ============================================================================
# STEP 3: TEST STRATEGY ON 2010-2019
# ============================================================================

print("=" * 80)
print("STEP 3: TESTING ON 2010-2019 (PRE-COVID)")
print("=" * 80)
print()

# Import backtest with error handling
import sys
sys.path.insert(0, '.')

try:
    from phase10 import backtest_strategy, calculate_metrics, merge_events_with_prices
except ImportError as e:
    print(f"\n❌ ERROR: Cannot import from phase10.py")
    print(f"   Error: {e}")
    print("\n   Make sure:")
    print("   1. phase10.py exists in current directory")
    print("   2. phase10.py has these functions:")
    print("      - backtest_strategy()")
    print("      - calculate_metrics()")
    print("      - merge_events_with_prices()")
    sys.exit(1)

# Filter to 2010-2019 with safe date handling
if events_df is not None and len(events_df) > 0:
    # Ensure Date is datetime type
    if 'Date' in events_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(events_df['Date']):
            events_df['Date'] = pd.to_datetime(events_df['Date'])
        events_df['year'] = events_df['Date'].dt.year
    else:
        print("ERROR: No 'Date' column in events_df!")
        print(f"Available columns: {events_df.columns.tolist()}")
        sys.exit(1)
    
    old_data = events_df[events_df['year'] < 2020].copy()
else:
    print("ERROR: No events data available!")
    sys.exit(1)

if len(old_data) == 0:
    print("WARNING: No pre-2020 data found!")
    print("Using fallback to existing Phase 10 data...")
    old_data = events_df.copy()

if len(old_data) < 10:
    print("⚠️  Not enough pre-2020 data")
    print("FRED API key may be missing or FRED data incomplete")
    print()
else:
    print(f"Pre-2020 events: {len(old_data)}")
    
    # Merge with prices
    merged_old = merge_events_with_prices(old_data, tmf)
    
    print(f"Merged with prices: {len(merged_old)}")
    print()
    
    # Test CPI only
    cpi_old = merged_old[merged_old['event_type'] == 'CPI']
    
    print(f"CPI events 2010-2019: {len(cpi_old)}")
    print()
    
    if len(cpi_old) > 5:
        print("Testing configurations on 2010-2019:")
        print("-" * 80)
        
        results = []
        
        # Config 1: No stop
        print("1. CPI, 5-day, no stop")
        trades = backtest_strategy(cpi_old, tmf, holding_days=5, stop_type='none')
        metrics = calculate_metrics(trades, "CPI 5d no-stop (2010-2019)")
        if metrics:
            results.append(metrics)
            print(f"   Sharpe: {metrics['sharpe']:.2f}, Win: {metrics['win_rate']:.1f}%, Total: {metrics['total_return']:+.1f}%")
        
        # Config 2: OHLC 2%
        print("2. CPI, 5-day, OHLC 2%")
        trades = backtest_strategy(cpi_old, tmf, holding_days=5, stop_type='ohlc', stop_pct=2)
        metrics = calculate_metrics(trades, "CPI 5d OHLC2% (2010-2019)")
        if metrics:
            results.append(metrics)
            print(f"   Sharpe: {metrics['sharpe']:.2f}, Win: {metrics['win_rate']:.1f}%, Total: {metrics['total_return']:+.1f}%")
        
        # Config 3: OHLC 3% (conservative)
        print("3. CPI, 5-day, OHLC 3% (CONSERVATIVE)")
        trades = backtest_strategy(cpi_old, tmf, holding_days=5, stop_type='ohlc', stop_pct=3)
        metrics = calculate_metrics(trades, "CPI 5d OHLC3% (2010-2019)")
        if metrics:
            results.append(metrics)
            print(f"   Sharpe: {metrics['sharpe']:.2f}, Win: {metrics['win_rate']:.1f}%, Total: {metrics['total_return']:+.1f}%")
        
        print()
        
        # Save results
        pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}/quick_test_2010_2019.csv', index=False)

# ============================================================================
# STEP 4: FULL TRAIN/TEST (2010-2019 vs 2020-2024)
# ============================================================================

print("=" * 80)
print("STEP 4: TRAIN/TEST SPLIT (2010-2019 vs 2020-2024)")
print("=" * 80)
print()

if events_df is not None and len(old_data) > 10:
    # Get 2020-2024 data
    new_data = events_df[events_df['year'] >= 2020].copy()
    
    print(f"Train: 2010-2019 ({len(old_data)} events)")
    print(f"Test:  2020-2024 ({len(new_data)} events)")
    print()
    
    # Merge
    merged_old = merge_events_with_prices(old_data, tmf)
    merged_new = merge_events_with_prices(new_data, tmf)
    
    # CPI only
    cpi_train = merged_old[merged_old['event_type'] == 'CPI']
    cpi_test = merged_new[merged_new['event_type'] == 'CPI']
    
    print(f"CPI Train: {len(cpi_train)} events")
    print(f"CPI Test:  {len(cpi_test)} events")
    print()
    
    if len(cpi_train) > 5 and len(cpi_test) > 5:
        print("Conservative config (CPI 5d OHLC3%):")
        print("-" * 80)
        
        # Train
        trades_train = backtest_strategy(cpi_train, tmf, holding_days=5, 
                                         stop_type='ohlc', stop_pct=3)
        metrics_train = calculate_metrics(trades_train, "Train 2010-2019")
        
        # Test
        trades_test = backtest_strategy(cpi_test, tmf, holding_days=5, 
                                        stop_type='ohlc', stop_pct=3)
        metrics_test = calculate_metrics(trades_test, "Test 2020-2024")
        
        if metrics_train and metrics_test:
            print(f"TRAIN: Sharpe {metrics_train['sharpe']:.2f}, Win {metrics_train['win_rate']:.1f}%, Total {metrics_train['total_return']:+.1f}%")
            print(f"TEST:  Sharpe {metrics_test['sharpe']:.2f}, Win {metrics_test['win_rate']:.1f}%, Total {metrics_test['total_return']:+.1f}%")
            
            delta = metrics_test['sharpe'] - metrics_train['sharpe']
            print(f"DELTA: {delta:+.2f}")
            print()
            
            # Interpretation
            print("INTERPRETATION:")
            print("-" * 80)
            
            if metrics_train['sharpe'] < 0.2:
                print("❌ TRAIN SHARPE < 0.2: Strategy doesn't work on old data")
                print("   Recent years (2020-2024) were LUCKY period")
                print("   BE CAUTIOUS with live trading!")
            elif metrics_train['sharpe'] < 0.5:
                print("⚠️  TRAIN SHARPE 0.2-0.5: Weak edge on old data")
                print("   Strategy marginal, may be regime-dependent")
            else:
                print("✅ TRAIN SHARPE > 0.5: Strategy works on old data!")
                print("   Edge appears robust across time periods")
            
            print()
            
            if abs(delta) > 0.5:
                if delta > 0:
                    print("⚠️  Test >> Train: 2020-2024 significantly better")
                    print("   Possible reasons:")
                    print("   - Higher volatility in recent years")
                    print("   - Lucky test period")
                    print("   - Strategy works better in inflation regime")
                else:
                    print("⚠️  Train >> Test: 2010-2019 significantly better")
                    print("   Possible overfitting to old data")
            else:
                print("✅ Train/Test consistent: Strategy appears robust")
            
            print()
            
            # Save comparison
            comparison = pd.DataFrame({
                'period': ['Train 2010-2019', 'Test 2020-2024'],
                'events': [len(cpi_train), len(cpi_test)],
                'sharpe': [metrics_train['sharpe'], metrics_test['sharpe']],
                'win_rate': [metrics_train['win_rate'], metrics_test['win_rate']],
                'total_return': [metrics_train['total_return'], metrics_test['total_return']]
            })
            comparison.to_csv(f'{OUTPUT_DIR}/train_test_comparison_quick.csv', index=False)
            
            print("Saved to train_test_comparison_quick.csv")
            print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 10.6 QUICK TEST COMPLETE")
print("=" * 80)
print()

print("⚠️  IMPORTANT NOTE:")
print("-" * 80)
print("This test uses FRED data with 'previous as consensus' proxy")
print("NOT the same as real market consensus forecasts!")
print()
print("Real surprises may be different:")
print("- FRED proxy: surprise = actual - previous")
print("- Real world: surprise = actual - consensus")
print()
print("Use this test to get ROUGH IDEA if strategy works pre-2020")
print("For ACCURATE results, collect real consensus data from Investing.com")
print()

if len(old_data) > 10 and 'metrics_train' in locals() and metrics_train:
    if metrics_train['sharpe'] > 0.5:
        print("✅ QUICK TEST RESULT: Promising on old data!")
        print("   → Worth collecting real consensus data for validation")
        print("   → Proceed with paper trading cautiously")
    elif metrics_train['sharpe'] > 0.2:
        print("⚠️  QUICK TEST RESULT: Weak on old data")
        print("   → May be regime-dependent")
        print("   → Paper trade but be cautious")
    else:
        print("❌ QUICK TEST RESULT: Doesn't work on old data")
        print("   → 2020-2024 was likely lucky period")
        print("   → Consider pivoting or only trading high-vol regimes")
else:
    print("⚠️  Insufficient data for pre-2020 test")
    print("   → Get FRED API key and re-run")
    print("   → Or collect manual data from Investing.com")

print()
