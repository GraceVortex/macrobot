"""
PHASE 4.5: SELECTIVE EVENT EXPANSION
====================================
Based on Phase 4 learnings: Remove Claims, keep quality events

Events included:
  - NFP (Non-Farm Payrolls): Monthly, high impact
  - CPI (Consumer Price Index): Monthly, high impact  
  - GDP (Gross Domestic Product): Quarterly, high impact
  - Retail Sales: Monthly, moderate impact
  - PMI Manufacturing: Monthly, moderate impact

Events excluded:
  - Initial Jobless Claims: Weekly, too noisy

Target: 120-130 trades with Phase 3 quality
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

FRED_API_KEY = "ef636b6d99542d08f7d0ab6152932290"
START_DATE = "2014-01-01"
END_DATE = "2024-11-14"

INSTRUMENT = 'TMF'
HOLDING_PERIOD = 2

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
                   PHASE 4.5: SELECTIVE EVENT EXPANSION
================================================================================

Quality-focused event selection based on Phase 4 analysis:

INCLUDED (Quality events):
  - NFP, CPI, GDP, Retail Sales, PMI

EXCLUDED (Noisy events):
  - Initial Jobless Claims (weekly noise)

Target: 120+ trades, +0.9% avg P&L, 60%+ accuracy
""")

# Reuse all helper functions from phase4
def download_prices(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data.index = pd.to_datetime(data.index).date
    return data

def download_vix():
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    return vix

def download_macro_data():
    print("Downloading macro data from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    
    macro_data = {}
    
    print("  NFP...")
    nfp = fred.get_series('PAYEMS', observation_start=START_DATE, observation_end=END_DATE)
    nfp_df = pd.DataFrame({'value': nfp})
    nfp_df['change'] = nfp_df['value'].diff() * 1000
    nfp_df = nfp_df.dropna()
    macro_data['nfp'] = nfp_df
    print(f"    {len(nfp_df)} releases")
    
    print("  CPI...")
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    cpi_df = pd.DataFrame({'value': cpi})
    cpi_df['yoy'] = cpi_df['value'].pct_change(12) * 100
    cpi_df = cpi_df.dropna()
    macro_data['cpi'] = cpi_df
    print(f"    {len(cpi_df)} releases")
    
    print("  GDP...")
    gdp = fred.get_series('GDP', observation_start=START_DATE, observation_end=END_DATE)
    gdp_df = pd.DataFrame({'value': gdp})
    gdp_df['qoq'] = gdp_df['value'].pct_change() * 100
    gdp_df = gdp_df.dropna()
    macro_data['gdp'] = gdp_df
    print(f"    {len(gdp_df)} releases")
    
    print("  Retail Sales...")
    retail = fred.get_series('RSXFS', observation_start=START_DATE, observation_end=END_DATE)
    retail_df = pd.DataFrame({'value': retail})
    retail_df['mom'] = retail_df['value'].pct_change() * 100
    retail_df = retail_df.dropna()
    macro_data['retail'] = retail_df
    print(f"    {len(retail_df)} releases")
    
    print("  PMI Manufacturing...")
    pmi = fred.get_series('MANEMP', observation_start=START_DATE, observation_end=END_DATE)
    pmi_df = pd.DataFrame({'value': pmi})
    pmi_df['change'] = pmi_df['value'].diff()
    pmi_df = pmi_df.dropna()
    macro_data['pmi'] = pmi_df
    print(f"    {len(pmi_df)} releases")
    
    fed_funds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    macro_data['fed_funds'] = fed_funds
    
    total = len(nfp_df) + len(cpi_df) + len(gdp_df) + len(retail_df) + len(pmi_df)
    print(f"\nTotal raw events: {total}")
    
    return macro_data

def calculate_rating(actual, consensus, event_type):
    if pd.isna(actual) or pd.isna(consensus) or consensus == 0:
        return 0
    
    surprise_pct = ((actual - consensus) / abs(consensus)) * 100
    
    thresholds = [2, 5, 10, 20] if event_type != 'nfp' else [5, 15, 30, 50]
    
    if abs(surprise_pct) < thresholds[0]:
        return 0
    elif abs(surprise_pct) < thresholds[1]:
        return np.sign(surprise_pct) * 2
    elif abs(surprise_pct) < thresholds[2]:
        return np.sign(surprise_pct) * 3
    elif abs(surprise_pct) < thresholds[3]:
        return np.sign(surprise_pct) * 4
    else:
        return np.sign(surprise_pct) * 5

def calculate_technical_indicators(prices, date):
    if hasattr(date, 'date'):
        date = date.date()
    prices_to_date = prices[prices.index < date]
    if len(prices_to_date) < 50:
        return {}
    delta = prices_to_date['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    momentum_5d = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-6] - 1) * 100 if len(prices_to_date) >= 6 else 0
    momentum_20d = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-21] - 1) * 100 if len(prices_to_date) >= 21 else 0
    returns = prices_to_date['close'].pct_change()
    volatility_20 = returns.rolling(20).std().iloc[-1] * 100 if len(returns) >= 20 else 0
    return {'rsi_14': rsi.iloc[-1] if len(rsi) > 0 else 50, 'volatility_20': volatility_20, 'momentum_5d': momentum_5d, 'momentum_20d': momentum_20d}

def get_vix_features(vix, date):
    if hasattr(date, 'date'):
        date = date.date()
    vix_to_date = vix[vix.index <= date].copy()
    if len(vix_to_date) < 20:
        return {}
    current_vix = vix_to_date['vix'].iloc[-1]
    vix_ma_20 = vix_to_date['vix'].rolling(20).mean().iloc[-1]
    vix_change_5d = vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-6] if len(vix_to_date) >= 6 else 0
    vix_change_20d = vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-21] if len(vix_to_date) >= 21 else 0
    return {'vix': current_vix, 'vix_ma_20': vix_ma_20, 'vix_change_5d': vix_change_5d, 'vix_change_20d': vix_change_20d, 'vix_above_20': 1 if current_vix > 20 else 0}

def get_fed_funds_features(fed_funds, date):
    if hasattr(date, 'date'):
        date = date.date()
    fed_dates = [d for d in fed_funds.index if pd.Timestamp(d).date() <= date]
    if len(fed_dates) == 0:
        return {}
    current_rate = fed_funds.loc[fed_dates[-1]]
    three_months_ago = [d for d in fed_dates if pd.Timestamp(d).date() <= date - timedelta(days=90)]
    rate_change_3m = current_rate - fed_funds.loc[three_months_ago[-1]] if three_months_ago else 0
    return {'fed_funds_rate': current_rate, 'fed_rate_change_3m': rate_change_3m}

def calculate_trend(prices, date):
    if hasattr(date, 'date'):
        date = date.date()
    prices_to_date = prices[prices.index < date]
    if len(prices_to_date) < TREND_MA_LONG:
        return 'neutral'
    ma_short = prices_to_date['close'].iloc[-TREND_MA_SHORT:].mean()
    ma_long = prices_to_date['close'].iloc[-TREND_MA_LONG:].mean()
    if ma_short > ma_long:
        return 'uptrend'
    elif ma_short < ma_long:
        return 'downtrend'
    return 'neutral'

def calculate_returns(prices, event_date, holding_period=2):
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
    metrics = {'nfp': ('change', 6), 'cpi': ('yoy', 6), 'gdp': ('qoq', 4), 'retail': ('mom', 6), 'pmi': ('change', 6)}
    
    if event_type not in metrics:
        return None
    
    col, window = metrics[event_type]
    idx = data.index.get_loc(date)
    
    if idx < window:
        return None
    
    consensus = data.iloc[idx-window:idx][col].mean()
    actual = data.loc[date, col]
    rating = calculate_rating(actual, consensus, event_type)
    
    if abs(rating) < MIN_RATING_THRESHOLD:
        return None
    
    vix_value = vix.loc[vix.index <= (date.date() if hasattr(date, 'date') else date)]['vix'].iloc[-1] if len(vix[vix.index <= (date.date() if hasattr(date, 'date') else date)]) > 0 else None
    if vix_value is None or vix_value < VIX_THRESHOLD:
        return None
    
    trend = calculate_trend(prices, date)
    if (rating > 0 and trend == 'uptrend') or (rating < 0 and trend == 'downtrend'):
        return None
    
    tech_features = calculate_technical_indicators(prices, date)
    if not tech_features:
        return None
    
    vix_features = get_vix_features(vix, date)
    if not vix_features:
        return None
    
    fed_features = get_fed_funds_features(fed_funds, date)
    if not fed_features:
        return None
    
    ret = calculate_returns(prices, date, HOLDING_PERIOD)
    if ret is None:
        return None
    
    surprise_pct = ((actual - consensus) / abs(consensus)) * 100 if consensus != 0 else 0
    
    return {
        'date': date, 'type': event_type.upper(), 'rating': rating, 'surprise_pct': surprise_pct,
        **tech_features, **vix_features, **fed_features, 'returns': ret, 'target': 1 if ret > 0 else 0
    }

def create_dataset(prices, vix, macro_data):
    print("\nCreating dataset...")
    data = []
    fed_funds = macro_data['fed_funds']
    event_counts = {}
    
    for event_type in ['nfp', 'cpi', 'gdp', 'retail', 'pmi']:
        print(f"  Processing {event_type.upper()}...")
        count = 0
        for date, row in macro_data[event_type].iterrows():
            result = process_event(date, macro_data[event_type], event_type, prices, vix, fed_funds)
            if result:
                data.append(result)
                count += 1
        event_counts[event_type] = count
        print(f"    {count} events")
    
    df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
    
    print(f"\nTotal: {len(df)} events")
    for k, v in event_counts.items():
        print(f"  {k.upper()}: {v}")
    
    return df

def train_and_evaluate(df):
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    feature_cols = ['rating', 'surprise_pct', 'vix', 'vix_ma_20', 'vix_change_5d', 'vix_change_20d', 'vix_above_20',
                    'rsi_14', 'volatility_20', 'momentum_5d', 'momentum_20d', 'fed_funds_rate', 'fed_rate_change_3m']
    
    df['type_nfp'] = (df['type'] == 'NFP').astype(int)
    df['type_cpi'] = (df['type'] == 'CPI').astype(int)
    df['type_gdp'] = (df['type'] == 'GDP').astype(int)
    df['type_retail'] = (df['type'] == 'RETAIL').astype(int)
    feature_cols.extend(['type_nfp', 'type_cpi', 'type_gdp', 'type_retail'])
    
    tscv = TimeSeriesSplit(n_splits=5)
    results, all_trades = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx]['target']
        X_test, y_test = df.iloc[test_idx][feature_cols], df.iloc[test_idx]['target']
        
        scaler = StandardScaler()
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(scaler.fit_transform(X_train), y_train, verbose=False)
        
        y_pred = model.predict(scaler.transform(X_test))
        y_pred_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
        
        test_df = df.iloc[test_idx].copy()
        test_df['prediction'] = y_pred
        test_df['confidence'] = np.maximum(y_pred_proba, 1 - y_pred_proba)
        
        high_conf = test_df[test_df['confidence'] >= ML_CONFIDENCE_THRESHOLD].copy()
        
        if len(high_conf) > 0:
            high_conf['pnl'] = 0.0
            high_conf.loc[high_conf['prediction'] == 1, 'pnl'] = high_conf.loc[high_conf['prediction'] == 1, 'returns']
            high_conf.loc[high_conf['prediction'] == 0, 'pnl'] = -high_conf.loc[high_conf['prediction'] == 0, 'returns']
            
            acc = accuracy_score(high_conf['target'], high_conf['prediction'])
            avg_pnl = high_conf['pnl'].mean()
            
            high_conf['fold'] = fold
            all_trades.append(high_conf[['date', 'type', 'rating', 'prediction', 'confidence', 'returns', 'pnl', 'fold']])
            
            print(f"Fold {fold}: {len(high_conf)} trades, Acc: {acc*100:.1f}%, P&L: {avg_pnl:+.3f}%")
        
        results.append({'fold': fold, 'trades': len(high_conf) if len(high_conf) > 0 else 0,
                       'accuracy': acc if len(high_conf) > 0 else 0, 'avg_pnl': avg_pnl if len(high_conf) > 0 else 0,
                       'total_pnl': high_conf['pnl'].sum() if len(high_conf) > 0 else 0})
    
    results_df = pd.DataFrame(results)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results_df['trades'].sum()} trades")
    print(f"Accuracy: {results_df['accuracy'].mean()*100:.1f}%")
    print(f"Avg P&L: {results_df['avg_pnl'].mean():+.4f}%")
    print(f"Total P&L: {results_df['total_pnl'].sum():+.2f}%")
    
    return results_df, trades_df

def main():
    print("\nStarting Phase 4.5...")
    prices = download_prices(INSTRUMENT)
    vix = download_vix()
    macro_data = download_macro_data()
    
    df = create_dataset(prices, vix, macro_data)
    
    os.makedirs('outputs_phase4_5', exist_ok=True)
    df.to_csv('outputs_phase4_5/dataset.csv', index=False)
    
    if len(df) > 0:
        results, trades = train_and_evaluate(df)
        results.to_csv('outputs_phase4_5/results.csv', index=False)
        if len(trades) > 0:
            trades.to_csv('outputs_phase4_5/all_trades.csv', index=False)
    
    print(f"\n{'='*60}")
    print("PHASE 4.5 COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
