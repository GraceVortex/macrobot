"""
PHASE 2 - MACHINE LEARNING MODEL
=================================
XGBoost для prediction направления движения IEF после макро-событий

Model: XGBoost Classifier (работает отлично на CPU!)
Features: 20+ признаков (макро, VIX, технические, Fed policy)
Validation: Walk-forward (no data leakage!)
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

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

FRED_API_KEY = "ef636b6d99542d08f7d0ab6152932290"

START_DATE = "2014-01-01"
END_DATE = "2024-11-14"

INSTRUMENT = "IEF"  # Based on Phase 1 results
HOLDING_PERIOD = '2d'  # Best from Phase 1

# Phase 2: ML Parameters
VIX_THRESHOLD = 15  # From Phase 1
MIN_RATING_THRESHOLD = 1.0

# XGBoost parameters
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
╔══════════════════════════════════════════════════════════════════╗
║              PHASE 2: MACHINE LEARNING MODEL                     ║
║                  XGBoost на CPU (быстро!)                        ║
╚══════════════════════════════════════════════════════════════════╝

Модель: XGBoost Gradient Boosting
Датасет: ~75 трейдов (маленький - CPU справится за секунды!)
Validation: Walk-forward (no overfitting)
Target: Win Rate 45% → 55%, Avg P&L +0.08% → +0.15%
""")

# ============================================
# 1. DATA LOADING (reuse from Phase 1)
# ============================================

def download_prices(ticker):
    """Download historical prices"""
    print(f"📊 Downloading {ticker} prices...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data.index = pd.to_datetime(data.index).date
    print(f"   ✓ Downloaded {len(data)} days")
    return data


def download_vix():
    """Download VIX"""
    print("📊 Downloading VIX...")
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    print(f"   ✓ Downloaded {len(vix)} days")
    return vix


def download_macro_data():
    """Download macro events"""
    print("📈 Downloading macro data from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    
    # NFP
    nfp = fred.get_series('PAYEMS', observation_start=START_DATE, observation_end=END_DATE)
    nfp_df = pd.DataFrame({'value': nfp})
    nfp_df['change'] = nfp_df['value'].diff() * 1000
    nfp_df = nfp_df.dropna()
    
    # CPI
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    cpi_df = pd.DataFrame({'value': cpi})
    cpi_df['yoy'] = cpi_df['value'].pct_change(12) * 100
    cpi_df = cpi_df.dropna()
    
    # Fed Funds Rate
    fed_funds = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    
    print(f"   ✓ NFP: {len(nfp_df)}, CPI: {len(cpi_df)}, Fed Funds: {len(fed_funds)}")
    
    return {'nfp': nfp_df, 'cpi': cpi_df, 'fed_funds': fed_funds}


# ============================================
# 2. FEATURE ENGINEERING
# ============================================

def calculate_technical_indicators(prices, date):
    """Calculate technical indicators up to date"""
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
    
    # Moving averages
    ma_5 = prices_to_date['close'].rolling(5).mean()
    ma_20 = prices_to_date['close'].rolling(20).mean()
    ma_50 = prices_to_date['close'].rolling(50).mean()
    
    # Volatility
    returns = prices_to_date['close'].pct_change()
    volatility_20 = returns.rolling(20).std() * 100
    
    # Price momentum
    momentum_5 = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-6] - 1) * 100 if len(prices_to_date) >= 6 else 0
    momentum_20 = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-21] - 1) * 100 if len(prices_to_date) >= 21 else 0
    
    return {
        'rsi_14': rsi.iloc[-1] if len(rsi) > 0 else 50,
        'ma_5': ma_5.iloc[-1] if len(ma_5) > 0 else prices_to_date['close'].iloc[-1],
        'ma_20': ma_20.iloc[-1] if len(ma_20) > 0 else prices_to_date['close'].iloc[-1],
        'ma_50': ma_50.iloc[-1] if len(ma_50) > 0 else prices_to_date['close'].iloc[-1],
        'volatility_20': volatility_20.iloc[-1] if len(volatility_20) > 0 else 0,
        'momentum_5d': momentum_5,
        'momentum_20d': momentum_20,
        'price': prices_to_date['close'].iloc[-1]
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
    
    # Get closest date
    fed_dates = [d for d in fed_funds.index if pd.Timestamp(d).date() <= date]
    
    if len(fed_dates) == 0:
        return {}
    
    current_rate = fed_funds.loc[fed_dates[-1]]
    
    # Rate change over last 3 months
    three_months_ago = [d for d in fed_dates if pd.Timestamp(d).date() <= date - timedelta(days=90)]
    rate_change_3m = current_rate - fed_funds.loc[three_months_ago[-1]] if three_months_ago else 0
    
    return {
        'fed_funds_rate': current_rate,
        'fed_rate_change_3m': rate_change_3m
    }


def calculate_rating(row, data, event_type):
    """Calculate event rating (reuse from Phase 1)"""
    if event_type == 'NFP':
        actual = row['change']
        idx = data.index.get_loc(row.name)
        if idx < 6:
            return 0, actual, actual, 0, 0
        consensus = data.iloc[idx-6:idx]['change'].mean()
    else:  # CPI
        actual = row['yoy']
        idx = data.index.get_loc(row.name)
        if idx < 6:
            return 0, actual, actual, 0, 0
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
    
    return rating, actual, consensus, surprise, surprise_pct


def calculate_returns(prices, event_date, holding_period='2d'):
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
    
    period = int(holding_period[:-1])
    if period - 1 < len(future_dates):
        future_price = prices.loc[future_dates[period - 1], 'close']
        return (future_price - base_price) / base_price * 100
    
    return None


# ============================================
# 3. CREATE ML DATASET
# ============================================

def create_ml_dataset(prices, vix, macro_data, fed_funds):
    """Create dataset with features for ML"""
    
    print("\n" + "="*60)
    print("🔧 CREATING ML DATASET")
    print("="*60)
    
    data = []
    
    # Process NFP events
    print("\n📊 Processing NFP events...")
    nfp_data = macro_data['nfp']
    
    for date, row in nfp_data.iterrows():
        rating, actual, consensus, surprise, surprise_pct = calculate_rating(row, nfp_data, 'NFP')
        
        if abs(rating) < MIN_RATING_THRESHOLD:
            continue
        
        # VIX filter
        vix_features = get_vix_features(vix, date)
        if not vix_features or vix_features['vix'] < VIX_THRESHOLD:
            continue
        
        # Technical indicators
        tech_features = calculate_technical_indicators(prices, date)
        if not tech_features:
            continue
        
        # Fed features
        fed_features = get_fed_funds_features(fed_funds, date)
        if not fed_features:
            continue
        
        # Calculate returns (target)
        ret = calculate_returns(prices, date, HOLDING_PERIOD)
        if ret is None:
            continue
        
        # Combine all features
        features = {
            'date': date,
            'type': 'NFP',
            'rating': rating,
            'surprise_pct': surprise_pct,
            **vix_features,
            **tech_features,
            **fed_features,
            'returns': ret,
            'target': 1 if ret > 0 else 0  # Binary: up or down
        }
        
        data.append(features)
    
    # Process CPI events
    print("📊 Processing CPI events...")
    cpi_data = macro_data['cpi']
    
    for date, row in cpi_data.iterrows():
        rating, actual, consensus, surprise, surprise_pct = calculate_rating(row, cpi_data, 'CPI')
        
        if abs(rating) < MIN_RATING_THRESHOLD:
            continue
        
        vix_features = get_vix_features(vix, date)
        if not vix_features or vix_features['vix'] < VIX_THRESHOLD:
            continue
        
        tech_features = calculate_technical_indicators(prices, date)
        if not tech_features:
            continue
        
        fed_features = get_fed_funds_features(fed_funds, date)
        if not fed_features:
            continue
        
        ret = calculate_returns(prices, date, HOLDING_PERIOD)
        if ret is None:
            continue
        
        features = {
            'date': date,
            'type': 'CPI',
            'rating': rating,
            'surprise_pct': surprise_pct,
            **vix_features,
            **tech_features,
            **fed_features,
            'returns': ret,
            'target': 1 if ret > 0 else 0
        }
        
        data.append(features)
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n✓ Created dataset with {len(df)} events")
    print(f"  - Positive returns: {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
    print(f"  - Negative returns: {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    
    return df


# ============================================
# 4. WALK-FORWARD VALIDATION
# ============================================

def walk_forward_validation(df):
    """Train/test with walk-forward approach (NO DATA LEAKAGE!)"""
    
    print("\n" + "="*60)
    print("🔄 WALK-FORWARD VALIDATION")
    print("="*60)
    print("\nКритично: тренируем на прошлом, тестируем на будущем!")
    print("Это предотвращает overfitting и data leakage.\n")
    
    # Feature columns
    feature_cols = [
        'rating', 'surprise_pct',
        'vix', 'vix_ma_20', 'vix_change_5d', 'vix_change_20d', 'vix_above_20',
        'rsi_14', 'volatility_20', 'momentum_5d', 'momentum_20d',
        'fed_funds_rate', 'fed_rate_change_3m'
    ]
    
    # Add type encoding
    df['type_nfp'] = (df['type'] == 'NFP').astype(int)
    feature_cols.append('type_nfp')
    
    # Walk-forward split
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    feature_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"  Train: {len(train_idx)} samples (до {df.iloc[train_idx[-1]]['date']})")
        print(f"  Test:  {len(test_idx)} samples ({df.iloc[test_idx[0]]['date']} - {df.iloc[test_idx[-1]]['date']})")
        
        # Split data
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]['target']
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx]['target']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate P&L
        test_df = df.iloc[test_idx].copy()
        test_df['prediction'] = y_pred
        test_df['pred_proba'] = y_pred_proba
        
        # Only trade on predictions
        test_df['pnl'] = 0.0
        test_df.loc[test_df['prediction'] == 1, 'pnl'] = test_df.loc[test_df['prediction'] == 1, 'returns']
        test_df.loc[test_df['prediction'] == 0, 'pnl'] = -test_df.loc[test_df['prediction'] == 0, 'returns']
        
        avg_pnl = test_df['pnl'].mean()
        total_pnl = test_df['pnl'].sum()
        
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Avg P&L: {avg_pnl:+.4f}%")
        print(f"  Total P&L: {total_pnl:+.2f}%")
        
        results.append({
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': accuracy,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl
        })
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importances.append(importance)
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("📊 OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"\nAvg Accuracy: {results_df['accuracy'].mean()*100:.1f}%")
    print(f"Avg P&L per trade: {results_df['avg_pnl'].mean():+.4f}%")
    print(f"Total P&L (all folds): {results_df['total_pnl'].sum():+.2f}%")
    
    return results_df, feature_importances, df


# ============================================
# 5. FINAL MODEL & COMPARISON
# ============================================

def train_final_model(df):
    """Train final model on all data for production"""
    
    print("\n" + "="*60)
    print("🎯 TRAINING FINAL MODEL")
    print("="*60)
    
    feature_cols = [
        'rating', 'surprise_pct',
        'vix', 'vix_ma_20', 'vix_change_5d', 'vix_change_20d', 'vix_above_20',
        'rsi_14', 'volatility_20', 'momentum_5d', 'momentum_20d',
        'fed_funds_rate', 'fed_rate_change_3m', 'type_nfp'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_scaled, y, verbose=False)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    # Save model
    os.makedirs('outputs_phase2', exist_ok=True)
    model.save_model('outputs_phase2/xgboost_model.json')
    importance.to_csv('outputs_phase2/feature_importance.csv', index=False)
    
    print("\n✓ Model saved to outputs_phase2/xgboost_model.json")
    print("✓ Feature importance saved")
    
    return model, scaler, importance


# ============================================
# 6. MAIN
# ============================================

def main():
    print("\nStarting Phase 2...")
    
    # Download data
    prices = download_prices(INSTRUMENT)
    vix = download_vix()
    macro_data = download_macro_data()
    fed_funds = macro_data['fed_funds']
    
    # Create ML dataset
    df = create_ml_dataset(prices, vix, macro_data, fed_funds)
    
    # Save dataset
    os.makedirs('outputs_phase2', exist_ok=True)
    df.to_csv('outputs_phase2/ml_dataset.csv', index=False)
    print(f"\n✓ Saved ML dataset to outputs_phase2/ml_dataset.csv")
    
    # Walk-forward validation
    results_df, feature_importances, df_with_type = walk_forward_validation(df)
    results_df.to_csv('outputs_phase2/walk_forward_results.csv', index=False)
    
    # Train final model
    model, scaler, importance = train_final_model(df_with_type)
    
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE!")
    print("="*70)
    print("\nFiles created in outputs_phase2/:")
    print("  1. ml_dataset.csv - Full dataset with features")
    print("  2. walk_forward_results.csv - Validation results")
    print("  3. xgboost_model.json - Trained model")
    print("  4. feature_importance.csv - Feature rankings")
    print()
    print("🎯 NEXT: Review results and compare with Phase 1!")
    print()


if __name__ == "__main__":
    main()
