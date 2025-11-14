"""
PHASE 3: LEVERAGED INSTRUMENTS
===============================
Testing leveraged ETFs to amplify returns while keeping same signal quality

Strategy:
- Use Phase 2.5 filters and ML model (proven to work)
- Test leveraged instruments:
  * TMF: 3x leveraged long Treasury (3x TLT)
  * TBT: 2x inverse Treasury (for short positions)
  
Target: 3x returns with same accuracy = +0.75% avg P&L per trade
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
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

# Test multiple instruments
INSTRUMENTS = {
    'IEF': '7-10Y Treasury (baseline)',
    'TMF': '3x Leveraged 20Y+ Treasury',
    'TBT': '2x Inverse 20Y+ Treasury'
}

# Phase 2.5 parameters
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
                    PHASE 3: LEVERAGED INSTRUMENTS
================================================================================

Testing leveraged ETFs to amplify returns:
  - IEF: Baseline (1x)
  - TMF: 3x leveraged long (Direxion Daily 20Y+ Treasury Bull 3X)
  - TBT: 2x inverse (ProShares UltraShort 20Y+ Treasury)

Goal: Increase avg P&L from +0.25% to +0.60-0.75% per trade
Risk: Higher volatility and tracking error
""")

# Data loading functions (reuse from phase2_5)
def download_prices(ticker):
    """Download historical prices"""
    print(f"  Downloading {ticker}...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    
    if data.empty:
        print(f"  Warning: No data for {ticker}")
        return None
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data.index = pd.to_datetime(data.index).date
    print(f"  Downloaded {len(data)} days")
    return data


def download_vix():
    """Download VIX"""
    print("  Downloading VIX...")
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    return vix


def download_macro_data():
    """Download macro events"""
    print("  Downloading macro data...")
    fred = Fred(api_key=FRED_API_KEY)
    
    nfp = fred.get_series('PAYEMS', observation_start=START_DATE, observation_end=END_DATE)
    nfp_df = pd.DataFrame({'value': nfp})
    nfp_df['change'] = nfp_df['value'].diff() * 1000
    nfp_df = nfp_df.dropna()
    
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    cpi_df = pd.DataFrame({'value': cpi})
    cpi_df['yoy'] = cpi_df['value'].pct_change(12) * 100
    cpi_df = cpi_df.dropna()
    
    fed_funds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    
    return {'nfp': nfp_df, 'cpi': cpi_df, 'fed_funds': fed_funds}


# Feature engineering (from phase2_5)
def calculate_technical_indicators(prices, date):
    """Calculate technical indicators"""
    if hasattr(date, 'date'):
        date = date.date()
    
    prices_to_date = prices[prices.index <= date].copy()
    
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
    
    # Use only data BEFORE event date to avoid look-ahead bias
    prices_to_date = prices[prices.index < date]
    
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


def calculate_rating_and_returns(row, data, event_type, prices):
    """Calculate rating and returns"""
    if event_type == 'NFP':
        actual = row['change']
        idx = data.index.get_loc(row.name)
        if idx < 6:
            return None
        consensus = data.iloc[idx-6:idx]['change'].mean()
    else:
        actual = row['yoy']
        idx = data.index.get_loc(row.name)
        if idx < 6:
            return None
        consensus = data.iloc[idx-6:idx]['yoy'].mean()
    
    surprise = actual - consensus
    surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0
    
    # Rating
    thresholds = [5, 15, 30, 50] if event_type == 'NFP' else [2, 5, 10, 20]
    
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
    
    # Returns
    event_date = row.name if not hasattr(row.name, 'date') else row.name.date()
    
    if event_date not in prices.index:
        future_dates = [d for d in prices.index if d > event_date]
        if not future_dates:
            return None
        event_date = future_dates[0]
    
    base_price = prices.loc[event_date, 'close']
    future_dates = [d for d in prices.index if d > event_date]
    
    period = 2
    if period - 1 < len(future_dates):
        future_price = prices.loc[future_dates[period - 1], 'close']
        ret = (future_price - base_price) / base_price * 100
    else:
        return None
    
    return rating, surprise_pct, ret


def create_dataset_for_instrument(prices, vix, macro_data, fed_funds, instrument_name):
    """Create dataset with Phase 2.5 filters"""
    
    print(f"\nCreating dataset for {instrument_name}...")
    
    data = []
    
    # Process NFP
    nfp_data = macro_data['nfp']
    
    for date, row in nfp_data.iterrows():
        result = calculate_rating_and_returns(row, nfp_data, 'NFP', prices)
        if result is None:
            continue
        
        rating, surprise_pct, ret = result
        
        # Phase 1 filters
        vix_value = vix.loc[vix.index <= date.date() if hasattr(date, 'date') else date]['vix'].iloc[-1] if len(vix[vix.index <= (date.date() if hasattr(date, 'date') else date)]) > 0 else None
        
        if vix_value is None or vix_value < VIX_THRESHOLD:
            continue
        
        if abs(rating) < MIN_RATING_THRESHOLD:
            continue
        
        trend = calculate_trend(prices, date)
        if rating > 0 and trend == 'uptrend':
            continue
        if rating < 0 and trend == 'downtrend':
            continue
        
        # ML features
        ml_features = calculate_technical_indicators(prices, date)
        if not ml_features:
            continue
        
        vix_features = get_vix_features(vix, date)
        if not vix_features:
            continue
        
        fed_features = get_fed_funds_features(fed_funds, date)
        if not fed_features:
            continue
        
        features = {
            'date': date,
            'type': 'NFP',
            'rating': rating,
            'surprise_pct': surprise_pct,
            **ml_features,
            **vix_features,
            **fed_features,
            'returns': ret,
            'target': 1 if ret > 0 else 0
        }
        
        data.append(features)
    
    # Process CPI
    cpi_data = macro_data['cpi']
    
    for date, row in cpi_data.iterrows():
        result = calculate_rating_and_returns(row, cpi_data, 'CPI', prices)
        if result is None:
            continue
        
        rating, surprise_pct, ret = result
        
        vix_value = vix.loc[vix.index <= date.date() if hasattr(date, 'date') else date]['vix'].iloc[-1] if len(vix[vix.index <= (date.date() if hasattr(date, 'date') else date)]) > 0 else None
        
        if vix_value is None or vix_value < VIX_THRESHOLD:
            continue
        
        if abs(rating) < MIN_RATING_THRESHOLD:
            continue
        
        trend = calculate_trend(prices, date)
        if rating > 0 and trend == 'uptrend':
            continue
        if rating < 0 and trend == 'downtrend':
            continue
        
        ml_features = calculate_technical_indicators(prices, date)
        if not ml_features:
            continue
        
        vix_features = get_vix_features(vix, date)
        if not vix_features:
            continue
        
        fed_features = get_fed_funds_features(fed_funds, date)
        if not fed_features:
            continue
        
        features = {
            'date': date,
            'type': 'CPI',
            'rating': rating,
            'surprise_pct': surprise_pct,
            **ml_features,
            **vix_features,
            **fed_features,
            'returns': ret,
            'target': 1 if ret > 0 else 0
        }
        
        data.append(features)
    
    df = pd.DataFrame(data)
    
    print(f"  Final dataset: {len(df)} events")
    
    return df


def train_and_test(df, instrument_name):
    """Train ML model and test with confidence threshold"""
    
    if len(df) < 20:
        print(f"  Not enough data for {instrument_name}")
        return None, None
    
    feature_cols = [
        'rating', 'surprise_pct',
        'vix', 'vix_ma_20', 'vix_change_5d', 'vix_change_20d', 'vix_above_20',
        'rsi_14', 'volatility_20', 'momentum_5d', 'momentum_20d',
        'fed_funds_rate', 'fed_rate_change_3m'
    ]
    
    df['type_nfp'] = (df['type'] == 'NFP').astype(int)
    feature_cols.append('type_nfp')
    
    # Walk-forward validation
    n_splits = min(5, len(df) // 15)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    all_trades = []  # Collect all trades for detailed analysis
    
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
            # Calculate P&L correctly for inverse ETFs
            if instrument_name == 'TBT':
                # TBT is inverse - flip the logic
                high_conf['pnl'] = 0.0
                high_conf.loc[high_conf['prediction'] == 1, 'pnl'] = -high_conf.loc[high_conf['prediction'] == 1, 'returns']
                high_conf.loc[high_conf['prediction'] == 0, 'pnl'] = high_conf.loc[high_conf['prediction'] == 0, 'returns']
            else:
                # Normal long-only ETFs (IEF, TMF)
                high_conf['pnl'] = 0.0
                high_conf.loc[high_conf['prediction'] == 1, 'pnl'] = high_conf.loc[high_conf['prediction'] == 1, 'returns']
                high_conf.loc[high_conf['prediction'] == 0, 'pnl'] = -high_conf.loc[high_conf['prediction'] == 0, 'returns']
            
            accuracy = accuracy_score(high_conf['target'], high_conf['prediction'])
            avg_pnl = high_conf['pnl'].mean()
            total_pnl = high_conf['pnl'].sum()
            
            # Save trades for detailed analysis
            high_conf['fold'] = fold
            all_trades.append(high_conf[['date', 'type', 'rating', 'prediction', 'confidence', 'returns', 'pnl', 'fold']])
        else:
            accuracy, avg_pnl, total_pnl = 0, 0, 0
        
        results.append({
            'fold': fold,
            'high_conf_trades': len(high_conf) if len(high_conf) > 0 else 0,
            'accuracy': accuracy,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{instrument_name} Results:")
    print(f"  Avg Accuracy: {results_df['accuracy'].mean()*100:.1f}%")
    print(f"  Avg P&L: {results_df['avg_pnl'].mean():+.4f}%")
    print(f"  Total P&L: {results_df['total_pnl'].sum():+.2f}%")
    
    # Combine all trades
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    
    return results_df, trades_df


def main():
    print("\nDownloading data...")
    
    # Common data
    vix = download_vix()
    macro_data = download_macro_data()
    fed_funds = macro_data['fed_funds']
    
    # Test each instrument
    all_results = {}
    
    for ticker, description in INSTRUMENTS.items():
        print(f"\n{'='*70}")
        print(f"Testing {ticker}: {description}")
        print(f"{'='*70}")
        
        prices = download_prices(ticker)
        
        if prices is None:
            print(f"  Skipping {ticker} - no data available")
            continue
        
        df = create_dataset_for_instrument(prices, vix, macro_data, fed_funds, ticker)
        
        if len(df) > 0:
            results, trades = train_and_test(df, ticker)
            if results is not None:
                all_results[ticker] = {
                    'results': results,
                    'avg_pnl': results['avg_pnl'].mean(),
                    'total_pnl': results['total_pnl'].sum(),
                    'accuracy': results['accuracy'].mean()
                }
                
                # Save detailed trades
                if len(trades) > 0:
                    trades.to_csv(f'outputs_phase3/{ticker.lower()}_all_trades.csv', index=False)
                    print(f"  Saved detailed trades: {ticker.lower()}_all_trades.csv")
    
    # Final comparison
    print(f"\n{'='*70}")
    print("PHASE 3 COMPARISON")
    print(f"{'='*70}\n")
    
    comparison = []
    for ticker, data in all_results.items():
        comparison.append({
            'Instrument': ticker,
            'Avg_PnL': data['avg_pnl'],
            'Total_PnL': data['total_pnl'],
            'Accuracy': data['accuracy'] * 100
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # Save results
    os.makedirs('outputs_phase3', exist_ok=True)
    comp_df.to_csv('outputs_phase3/leverage_comparison.csv', index=False)
    
    print(f"\n{'='*70}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*70}")
    print("\nFiles created in outputs_phase3/:")
    print("  1. leverage_comparison.csv")
    print()


if __name__ == "__main__":
    main()
