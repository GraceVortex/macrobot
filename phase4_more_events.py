"""
PHASE 4: EXPANDING EVENT COVERAGE
==================================
Adding more macro events to increase trade frequency and statistical power

Current: NFP + CPI = ~20 events/year
Phase 4 adds:
  - GDP (Gross Domestic Product): 4 releases/year
  - Retail Sales: 12 releases/year
  - Initial Jobless Claims: 52 releases/year (weekly)
  - PMI (Manufacturing): 12 releases/year

Target: 100+ events/year, 1000+ over 10 years
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration
FRED_API_KEY = "ef636b6d99542d08f7d0ab6152932290"
START_DATE = "2014-01-01"
END_DATE = "2024-11-14"

INSTRUMENT = 'TMF'  # Best from Phase 3
HOLDING_PERIOD = 2

# Phase 4 parameters - same as Phase 3
VIX_THRESHOLD = 15
MIN_RATING_THRESHOLD = 2.0
TREND_MA_SHORT = 20
TREND_MA_LONG = 50
ML_CONFIDENCE_THRESHOLD = 0.60

XGBOOST_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42
}

print("""
================================================================================
                    PHASE 4: EXPANDED EVENT COVERAGE
================================================================================

Expanding beyond NFP and CPI to include:
  - GDP (Quarterly): 4 releases/year
  - Retail Sales (Monthly): 12 releases/year
  - Initial Claims (Weekly): 52 releases/year
  - PMI Manufacturing (Monthly): 12 releases/year

Current baseline: ~20 events/year (NFP + CPI)
Phase 4 target: 80-100 events/year

Goal: More trading opportunities + better statistical validation
""")

# Download functions (reuse from phase3)
def download_prices(ticker):
    """Download historical prices"""
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data.index = pd.to_datetime(data.index).date
    return data


def download_vix():
    """Download VIX"""
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    return vix


def download_all_macro_data():
    """Download all macro events including new Phase 4 events"""
    print("Downloading macro data from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    
    macro_data = {}
    
    # Existing events (Phase 1-3)
    print("  NFP (Non-Farm Payrolls)...")
    nfp = fred.get_series('PAYEMS', observation_start=START_DATE, observation_end=END_DATE)
    nfp_df = pd.DataFrame({'value': nfp})
    nfp_df['change'] = nfp_df['value'].diff() * 1000
    nfp_df = nfp_df.dropna()
    macro_data['nfp'] = nfp_df
    print(f"    {len(nfp_df)} releases")
    
    print("  CPI (Consumer Price Index)...")
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    cpi_df = pd.DataFrame({'value': cpi})
    cpi_df['yoy'] = cpi_df['value'].pct_change(12) * 100
    cpi_df = cpi_df.dropna()
    macro_data['cpi'] = cpi_df
    print(f"    {len(cpi_df)} releases")
    
    # NEW Phase 4 events
    print("  GDP (Gross Domestic Product)...")
    gdp = fred.get_series('GDP', observation_start=START_DATE, observation_end=END_DATE)
    gdp_df = pd.DataFrame({'value': gdp})
    gdp_df['qoq'] = gdp_df['value'].pct_change() * 100  # Quarter-over-quarter
    gdp_df = gdp_df.dropna()
    macro_data['gdp'] = gdp_df
    print(f"    {len(gdp_df)} releases")
    
    print("  Retail Sales...")
    retail = fred.get_series('RSXFS', observation_start=START_DATE, observation_end=END_DATE)
    retail_df = pd.DataFrame({'value': retail})
    retail_df['mom'] = retail_df['value'].pct_change() * 100  # Month-over-month
    retail_df = retail_df.dropna()
    macro_data['retail'] = retail_df
    print(f"    {len(retail_df)} releases")
    
    print("  Initial Jobless Claims...")
    claims = fred.get_series('ICSA', observation_start=START_DATE, observation_end=END_DATE)
    claims_df = pd.DataFrame({'value': claims})
    claims_df['change'] = claims_df['value'].diff()
    claims_df = claims_df.dropna()
    macro_data['claims'] = claims_df
    print(f"    {len(claims_df)} releases")
    
    print("  PMI Manufacturing...")
    pmi = fred.get_series('MANEMP', observation_start=START_DATE, observation_end=END_DATE)
    pmi_df = pd.DataFrame({'value': pmi})
    pmi_df['change'] = pmi_df['value'].diff()
    pmi_df = pmi_df.dropna()
    macro_data['pmi'] = pmi_df
    print(f"    {len(pmi_df)} releases")
    
    # Fed Funds
    fed_funds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    macro_data['fed_funds'] = fed_funds
    
    total_events = sum([len(macro_data[k]) for k in ['nfp', 'cpi', 'gdp', 'retail', 'claims', 'pmi']])
    print(f"\nTotal events: {total_events}")
    
    return macro_data


def calculate_rating(actual, consensus, event_type):
    """Calculate event rating based on surprise"""
    if pd.isna(actual) or pd.isna(consensus) or consensus == 0:
        return 0
    
    surprise = actual - consensus
    surprise_pct = (surprise / abs(consensus)) * 100
    
    # Different thresholds for different events
    if event_type in ['nfp', 'claims']:
        # Absolute numbers - use same scale as NFP
        thresholds = [5, 15, 30, 50]
    elif event_type in ['cpi', 'gdp', 'retail', 'pmi']:
        # Percentages - use CPI scale
        thresholds = [2, 5, 10, 20]
    else:
        thresholds = [5, 15, 30, 50]
    
    if abs(surprise_pct) < thresholds[0]:
        rating = 0
    elif abs(surprise_pct) < thresholds[1]:
        rating = np.sign(surprise_pct) * 2
    elif abs(surprise_pct) < thresholds[2]:
        rating = np.sign(surprise_pct) * 3
    elif abs(surprise_pct) < thresholds[3]:
        rating = np.sign(surprise_pct) * 4
    else:
        rating = np.sign(surprise_pct) * 5
    
    return rating


# Feature engineering (reuse from phase3)
def calculate_technical_indicators(prices, date):
    """Calculate technical indicators"""
    if hasattr(date, 'date'):
        date = date.date()
    
    prices_to_date = prices[prices.index < date]  # No look-ahead bias
    
    if len(prices_to_date) < 50:
        return {}
    
    # RSI
    delta = prices_to_date['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Momentum
    momentum_5d = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-6] - 1) * 100 if len(prices_to_date) >= 6 else 0
    momentum_20d = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-21] - 1) * 100 if len(prices_to_date) >= 21 else 0
    
    # Volatility
    returns = prices_to_date['close'].pct_change()
    volatility_20 = returns.rolling(20).std().iloc[-1] * 100 if len(returns) >= 20 else 0
    
    return {
        'rsi_14': rsi.iloc[-1] if len(rsi) > 0 else 50,
        'volatility_20': volatility_20,
        'momentum_5d': momentum_5d,
        'momentum_20d': momentum_20d
    }


def get_vix_features(vix, date):
    """Calculate VIX features"""
    if hasattr(date, 'date'):
        date = date.date()
    
    vix_to_date = vix[vix.index <= date].copy()
    
    if len(vix_to_date) < 20:
        return {}
    
    current_vix = vix_to_date['vix'].iloc[-1]
    vix_ma_20 = vix_to_date['vix'].rolling(20).mean().iloc[-1]
    vix_change_5d = vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-6] if len(vix_to_date) >= 6 else 0
    vix_change_20d = vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-21] if len(vix_to_date) >= 21 else 0
    
    return {
        'vix': current_vix,
        'vix_ma_20': vix_ma_20,
        'vix_change_5d': vix_change_5d,
        'vix_change_20d': vix_change_20d,
        'vix_above_20': 1 if current_vix > 20 else 0
    }


def get_fed_funds_features(fed_funds, date):
    """Get Fed policy features"""
    if hasattr(date, 'date'):
        date = date.date()
    
    fed_dates = [d for d in fed_funds.index if pd.Timestamp(d).date() <= date]
    
    if len(fed_dates) == 0:
        return {}
    
    current_rate = fed_funds.loc[fed_dates[-1]]
    three_months_ago = [d for d in fed_dates if pd.Timestamp(d).date() <= date - timedelta(days=90)]
    rate_change_3m = current_rate - fed_funds.loc[three_months_ago[-1]] if three_months_ago else 0
    
    return {
        'fed_funds_rate': current_rate,
        'fed_rate_change_3m': rate_change_3m
    }


def calculate_trend(prices, date):
    """Calculate trend"""
    if hasattr(date, 'date'):
        date = date.date()
    
    prices_to_date = prices[prices.index < date]  # No look-ahead bias
    
    if len(prices_to_date) < TREND_MA_LONG:
        return 'neutral'
    
    ma_short = prices_to_date['close'].iloc[-TREND_MA_SHORT:].mean()
    ma_long = prices_to_date['close'].iloc[-TREND_MA_LONG:].mean()
    
    if ma_short > ma_long:
        return 'uptrend'
    elif ma_short < ma_long:
        return 'downtrend'
    else:
        return 'neutral'


def calculate_returns(prices, event_date, holding_period=2):
    """Calculate forward returns"""
    if hasattr(event_date, 'date'):
        event_date = event_date.date()
    
    if event_date not in prices.index:
        future_dates = [d for d in prices.index if d > event_date]
        if not future_dates:
            return None
        event_date = future_dates[0]
    
    base_price = prices.loc[event_date, 'close']
    future_dates = [d for d in prices.index if d > event_date]
    
    if holding_period - 1 < len(future_dates):
        future_price = prices.loc[future_dates[holding_period - 1], 'close']
        return (future_price - base_price) / base_price * 100
    
    return None


def process_event(date, data, event_type, prices, vix, fed_funds):
    """Process single event and create features"""
    
    # Get consensus and actual
    if event_type == 'nfp':
        idx = data.index.get_loc(date)
        if idx < 6:
            return None
        consensus = data.iloc[idx-6:idx]['change'].mean()
        actual = data.loc[date, 'change']
    elif event_type == 'cpi':
        idx = data.index.get_loc(date)
        if idx < 6:
            return None
        consensus = data.iloc[idx-6:idx]['yoy'].mean()
        actual = data.loc[date, 'yoy']
    elif event_type == 'gdp':
        idx = data.index.get_loc(date)
        if idx < 4:
            return None
        consensus = data.iloc[idx-4:idx]['qoq'].mean()
        actual = data.loc[date, 'qoq']
    elif event_type == 'retail':
        idx = data.index.get_loc(date)
        if idx < 6:
            return None
        consensus = data.iloc[idx-6:idx]['mom'].mean()
        actual = data.loc[date, 'mom']
    elif event_type == 'claims':
        idx = data.index.get_loc(date)
        if idx < 4:
            return None
        consensus = data.iloc[idx-4:idx]['change'].mean()
        actual = data.loc[date, 'change']
    elif event_type == 'pmi':
        idx = data.index.get_loc(date)
        if idx < 6:
            return None
        consensus = data.iloc[idx-6:idx]['change'].mean()
        actual = data.loc[date, 'change']
    else:
        return None
    
    # Calculate rating
    rating = calculate_rating(actual, consensus, event_type)
    
    if abs(rating) < MIN_RATING_THRESHOLD:
        return None
    
    # VIX filter
    vix_value = vix.loc[vix.index <= (date.date() if hasattr(date, 'date') else date)]['vix'].iloc[-1] if len(vix[vix.index <= (date.date() if hasattr(date, 'date') else date)]) > 0 else None
    
    if vix_value is None or vix_value < VIX_THRESHOLD:
        return None
    
    # Trend filter
    trend = calculate_trend(prices, date)
    if rating > 0 and trend == 'uptrend':
        return None
    if rating < 0 and trend == 'downtrend':
        return None
    
    # ML features
    tech_features = calculate_technical_indicators(prices, date)
    if not tech_features:
        return None
    
    vix_features = get_vix_features(vix, date)
    if not vix_features:
        return None
    
    fed_features = get_fed_funds_features(fed_funds, date)
    if not fed_features:
        return None
    
    # Returns
    ret = calculate_returns(prices, date, HOLDING_PERIOD)
    if ret is None:
        return None
    
    surprise = actual - consensus
    surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0
    
    return {
        'date': date,
        'type': event_type.upper(),
        'rating': rating,
        'surprise_pct': surprise_pct,
        **tech_features,
        **vix_features,
        **fed_features,
        'returns': ret,
        'target': 1 if ret > 0 else 0
    }


def create_full_dataset(prices, vix, macro_data):
    """Create dataset with all events"""
    
    print("\nCreating dataset with all macro events...")
    
    data = []
    fed_funds = macro_data['fed_funds']
    
    event_counts = {}
    
    # Process each event type
    for event_type in ['nfp', 'cpi', 'gdp', 'retail', 'claims', 'pmi']:
        print(f"\nProcessing {event_type.upper()}...")
        event_data = macro_data[event_type]
        count = 0
        
        for date, row in event_data.iterrows():
            result = process_event(date, event_data, event_type, prices, vix, fed_funds)
            if result is not None:
                data.append(result)
                count += 1
        
        event_counts[event_type] = count
        print(f"  Added {count} {event_type.upper()} events")
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"TOTAL DATASET: {len(df)} high-quality events")
    print(f"{'='*60}")
    for event_type, count in event_counts.items():
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {event_type.upper():10s}: {count:3d} ({pct:5.1f}%)")
    
    return df


def train_and_evaluate(df):
    """Train ML model with walk-forward validation"""
    
    print(f"\n{'='*60}")
    print("TRAINING AND EVALUATION")
    print(f"{'='*60}")
    
    feature_cols = [
        'rating', 'surprise_pct',
        'vix', 'vix_ma_20', 'vix_change_5d', 'vix_change_20d', 'vix_above_20',
        'rsi_14', 'volatility_20', 'momentum_5d', 'momentum_20d',
        'fed_funds_rate', 'fed_rate_change_3m'
    ]
    
    # Event type dummies
    df['type_nfp'] = (df['type'] == 'NFP').astype(int)
    df['type_cpi'] = (df['type'] == 'CPI').astype(int)
    df['type_gdp'] = (df['type'] == 'GDP').astype(int)
    df['type_retail'] = (df['type'] == 'RETAIL').astype(int)
    df['type_claims'] = (df['type'] == 'CLAIMS').astype(int)
    
    feature_cols.extend(['type_nfp', 'type_cpi', 'type_gdp', 'type_retail', 'type_claims'])
    
    # Walk-forward validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    all_trades = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]['target']
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx]['target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        test_df = df.iloc[test_idx].copy()
        test_df['prediction'] = y_pred
        test_df['confidence'] = np.maximum(y_pred_proba, 1 - y_pred_proba)
        
        # High confidence only
        high_conf = test_df[test_df['confidence'] >= ML_CONFIDENCE_THRESHOLD].copy()
        
        if len(high_conf) > 0:
            high_conf['pnl'] = 0.0
            high_conf.loc[high_conf['prediction'] == 1, 'pnl'] = high_conf.loc[high_conf['prediction'] == 1, 'returns']
            high_conf.loc[high_conf['prediction'] == 0, 'pnl'] = -high_conf.loc[high_conf['prediction'] == 0, 'returns']
            
            accuracy = accuracy_score(high_conf['target'], high_conf['prediction'])
            avg_pnl = high_conf['pnl'].mean()
            total_pnl = high_conf['pnl'].sum()
            
            high_conf['fold'] = fold
            all_trades.append(high_conf[['date', 'type', 'rating', 'prediction', 'confidence', 'returns', 'pnl', 'fold']])
            
            print(f"\nFold {fold}: {len(high_conf)} trades, Acc: {accuracy*100:.1f}%, Avg P&L: {avg_pnl:+.3f}%")
        else:
            accuracy, avg_pnl, total_pnl = 0, 0, 0
            print(f"\nFold {fold}: No high-confidence trades")
        
        results.append({
            'fold': fold,
            'trades': len(high_conf) if len(high_conf) > 0 else 0,
            'accuracy': accuracy,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl
        })
    
    results_df = pd.DataFrame(results)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Total trades: {results_df['trades'].sum()}")
    print(f"Avg Accuracy: {results_df['accuracy'].mean()*100:.1f}%")
    print(f"Avg P&L: {results_df['avg_pnl'].mean():+.4f}%")
    print(f"Total P&L: {results_df['total_pnl'].sum():+.2f}%")
    
    return results_df, trades_df


def main():
    print("\nStarting Phase 4...")
    
    # Download data
    print("\nDownloading price data...")
    prices = download_prices(INSTRUMENT)
    vix = download_vix()
    
    print(f"Downloaded {len(prices)} days of {INSTRUMENT}")
    print(f"Downloaded {len(vix)} days of VIX")
    
    # Download all macro data
    macro_data = download_all_macro_data()
    
    # Create dataset
    df = create_full_dataset(prices, vix, macro_data)
    
    # Save dataset
    os.makedirs('outputs_phase4', exist_ok=True)
    df.to_csv('outputs_phase4/phase4_dataset.csv', index=False)
    print(f"\nSaved dataset: outputs_phase4/phase4_dataset.csv")
    
    # Train and evaluate
    if len(df) > 0:
        results, trades = train_and_evaluate(df)
        
        # Save results
        results.to_csv('outputs_phase4/phase4_results.csv', index=False)
        if len(trades) > 0:
            trades.to_csv('outputs_phase4/phase4_all_trades.csv', index=False)
        
        print(f"\nSaved results to outputs_phase4/")
    
    print(f"\n{'='*60}")
    print("PHASE 4 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
