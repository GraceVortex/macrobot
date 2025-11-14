"""
PHASE 2.5 - HYBRID MODEL
========================
Комбинация Phase 1 (filters) + Phase 2 (ML predictions)

Подход:
1. Phase 1 filters отбирают quality события (VIX>15, strong rating, trend)
2. ML model предсказывает на filtered событиях
3. Trade только на high-confidence predictions (>60%)

Expected: Sharpe >1.3, Win Rate >55%, Avg P&L >0.12%
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

# ============================================
# CONFIGURATION
# ============================================

FRED_API_KEY = "ef636b6d99542d08f7d0ab6152932290"

START_DATE = "2014-01-01"
END_DATE = "2024-11-14"

INSTRUMENT = "IEF"
HOLDING_PERIOD = '2d'

# Phase 2.5: Hybrid parameters
VIX_THRESHOLD = 15
MIN_RATING_THRESHOLD = 2.0  # Stronger than Phase 2 (was 1.0)
TREND_MA_SHORT = 20
TREND_MA_LONG = 50
ML_CONFIDENCE_THRESHOLD = 0.60  # Only trade if ML confidence >60%

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
║            PHASE 2.5: HYBRID MODEL (Best of Both!)               ║
║         Phase 1 Filters + Phase 2 ML = Maximum Quality          ║
╚══════════════════════════════════════════════════════════════════╝

Стратегия:
  1. Phase 1 Filters: VIX>15 + Rating>2 + Trend alignment
  2. ML Model: XGBoost predictions на filtered events
  3. High Confidence: Trade только если ML confidence >60%

Target: Sharpe >1.3, Win Rate >55%, Avg P&L >0.12%
""")

# ============================================
# DATA LOADING (reuse)
# ============================================

def download_all_data():
    """Download all required data"""
    print("📊 Downloading data...")
    
    # Prices
    prices = yf.download(INSTRUMENT, start=START_DATE, end=END_DATE, progress=False)
    prices = prices[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    prices.columns = ['open', 'high', 'low', 'close', 'volume']
    prices.index = pd.to_datetime(prices.index).date
    
    # VIX
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    
    # Macro
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
    
    print(f"✓ IEF: {len(prices)} days, VIX: {len(vix)}, NFP: {len(nfp_df)}, CPI: {len(cpi_df)}")
    
    return prices, vix, {'nfp': nfp_df, 'cpi': cpi_df, 'fed_funds': fed_funds}


# ============================================
# PHASE 1 FILTERS (from phase1.py)
# ============================================

def calculate_trend(prices, date):
    """Phase 1: Trend filter"""
    if hasattr(date, 'date'):
        date = date.date()
    
    prices_to_date = prices[prices.index <= date]
    
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


def get_vix_value(vix, date):
    """Get VIX at date"""
    if hasattr(date, 'date'):
        date = date.date()
    
    if date in vix.index:
        return vix.loc[date, 'vix']
    
    past_dates = [d for d in vix.index if d <= date]
    if past_dates:
        return vix.loc[past_dates[-1], 'vix']
    return None


def apply_phase1_filters(date, rating, vix_value, trend):
    """Apply all Phase 1 filters"""
    # VIX filter
    if vix_value is None or vix_value < VIX_THRESHOLD:
        return False, "VIX too low"
    
    # Rating filter
    if abs(rating) < MIN_RATING_THRESHOLD:
        return False, "Rating too weak"
    
    # Trend alignment filter
    if rating > 0 and trend == 'uptrend':
        return False, "Hawkish but uptrend"
    if rating < 0 and trend == 'downtrend':
        return False, "Dovish but downtrend"
    
    return True, "Passed all filters"


# ============================================
# FEATURE ENGINEERING (from phase2.py)
# ============================================

def calculate_all_features(prices, vix, fed_funds, date):
    """Calculate all ML features"""
    if hasattr(date, 'date'):
        date = date.date()
    
    features = {}
    
    # Technical indicators
    prices_to_date = prices[prices.index <= date].copy()
    
    if len(prices_to_date) < 50:
        return None
    
    # RSI
    delta = prices_to_date['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features['rsi_14'] = rsi.iloc[-1] if len(rsi) > 0 else 50
    
    # Momentum
    features['momentum_5d'] = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-6] - 1) * 100 if len(prices_to_date) >= 6 else 0
    features['momentum_20d'] = (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-21] - 1) * 100 if len(prices_to_date) >= 21 else 0
    
    # Volatility
    returns = prices_to_date['close'].pct_change()
    features['volatility_20'] = returns.rolling(20).std().iloc[-1] * 100 if len(returns) >= 20 else 0
    
    # VIX features
    vix_to_date = vix[vix.index <= date].copy()
    
    if len(vix_to_date) < 20:
        return None
    
    features['vix'] = vix_to_date['vix'].iloc[-1]
    features['vix_ma_20'] = vix_to_date['vix'].rolling(20).mean().iloc[-1]
    features['vix_change_5d'] = vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-6] if len(vix_to_date) >= 6 else 0
    features['vix_change_20d'] = vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-21] if len(vix_to_date) >= 21 else 0
    features['vix_above_20'] = 1 if features['vix'] > 20 else 0
    
    # Fed features
    fed_dates = [d for d in fed_funds.index if pd.Timestamp(d).date() <= date]
    
    if len(fed_dates) == 0:
        return None
    
    features['fed_funds_rate'] = fed_funds.loc[fed_dates[-1]]
    
    three_months_ago = [d for d in fed_dates if pd.Timestamp(d).date() <= date - timedelta(days=90)]
    features['fed_rate_change_3m'] = features['fed_funds_rate'] - fed_funds.loc[three_months_ago[-1]] if three_months_ago else 0
    
    return features


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
    
    period = 2  # 2-day holding
    if period - 1 < len(future_dates):
        future_price = prices.loc[future_dates[period - 1], 'close']
        ret = (future_price - base_price) / base_price * 100
    else:
        return None
    
    return rating, surprise_pct, ret


# ============================================
# CREATE HYBRID DATASET
# ============================================

def create_hybrid_dataset(prices, vix, macro_data, fed_funds):
    """Create dataset with Phase 1 filters + Phase 2 features"""
    
    print("\n" + "="*70)
    print("🔧 CREATING HYBRID DATASET (Phase 1 Filters + ML Features)")
    print("="*70)
    
    data = []
    
    # Process NFP
    print("\n📊 Processing NFP events...")
    nfp_data = macro_data['nfp']
    
    filtered_count = {'vix': 0, 'rating': 0, 'trend': 0, 'features': 0}
    
    for date, row in nfp_data.iterrows():
        result = calculate_rating_and_returns(row, nfp_data, 'NFP', prices)
        if result is None:
            continue
        
        rating, surprise_pct, ret = result
        
        # Phase 1 Filter 1: VIX
        vix_value = get_vix_value(vix, date)
        if vix_value is None or vix_value < VIX_THRESHOLD:
            filtered_count['vix'] += 1
            continue
        
        # Phase 1 Filter 2: Rating strength
        if abs(rating) < MIN_RATING_THRESHOLD:
            filtered_count['rating'] += 1
            continue
        
        # Phase 1 Filter 3: Trend alignment
        trend = calculate_trend(prices, date)
        passed, reason = apply_phase1_filters(date, rating, vix_value, trend)
        if not passed:
            filtered_count['trend'] += 1
            continue
        
        # Calculate ML features
        ml_features = calculate_all_features(prices, vix, fed_funds, date)
        if ml_features is None:
            filtered_count['features'] += 1
            continue
        
        # Combine all
        features = {
            'date': date,
            'type': 'NFP',
            'rating': rating,
            'surprise_pct': surprise_pct,
            **ml_features,
            'returns': ret,
            'target': 1 if ret > 0 else 0
        }
        
        data.append(features)
    
    # Process CPI
    print("📊 Processing CPI events...")
    cpi_data = macro_data['cpi']
    
    for date, row in cpi_data.iterrows():
        result = calculate_rating_and_returns(row, cpi_data, 'CPI', prices)
        if result is None:
            continue
        
        rating, surprise_pct, ret = result
        
        vix_value = get_vix_value(vix, date)
        if vix_value is None or vix_value < VIX_THRESHOLD:
            filtered_count['vix'] += 1
            continue
        
        if abs(rating) < MIN_RATING_THRESHOLD:
            filtered_count['rating'] += 1
            continue
        
        trend = calculate_trend(prices, date)
        passed, reason = apply_phase1_filters(date, rating, vix_value, trend)
        if not passed:
            filtered_count['trend'] += 1
            continue
        
        ml_features = calculate_all_features(prices, vix, fed_funds, date)
        if ml_features is None:
            filtered_count['features'] += 1
            continue
        
        features = {
            'date': date,
            'type': 'CPI',
            'rating': rating,
            'surprise_pct': surprise_pct,
            **ml_features,
            'returns': ret,
            'target': 1 if ret > 0 else 0
        }
        
        data.append(features)
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n✅ Phase 1 Filtering Results:")
    print(f"   Filtered by VIX<{VIX_THRESHOLD}: {filtered_count['vix']}")
    print(f"   Filtered by rating<{MIN_RATING_THRESHOLD}: {filtered_count['rating']}")
    print(f"   Filtered by trend misalignment: {filtered_count['trend']}")
    print(f"   Filtered by missing features: {filtered_count['features']}")
    print(f"\n✓ Final dataset: {len(df)} HIGH-QUALITY events")
    print(f"  - Positive: {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
    print(f"  - Negative: {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    
    return df


# ============================================
# HYBRID MODEL TRAINING
# ============================================

def train_hybrid_model(df):
    """Train with walk-forward + confidence filtering"""
    
    print("\n" + "="*70)
    print("🤖 TRAINING HYBRID MODEL (ML on Filtered Data)")
    print("="*70)
    
    feature_cols = [
        'rating', 'surprise_pct',
        'vix', 'vix_ma_20', 'vix_change_5d', 'vix_change_20d', 'vix_above_20',
        'rsi_14', 'volatility_20', 'momentum_5d', 'momentum_20d',
        'fed_funds_rate', 'fed_rate_change_3m'
    ]
    
    df['type_nfp'] = (df['type'] == 'NFP').astype(int)
    feature_cols.append('type_nfp')
    
    # Walk-forward
    n_splits = min(5, len(df) // 15)  # Adaptive splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    all_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]['target']
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx]['target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predictions with probability
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Apply confidence filter
        test_df = df.iloc[test_idx].copy()
        test_df['prediction'] = y_pred
        test_df['confidence'] = np.maximum(y_pred_proba, 1 - y_pred_proba)  # Max of both classes
        
        # Only trade on high-confidence
        high_conf = test_df[test_df['confidence'] >= ML_CONFIDENCE_THRESHOLD].copy()
        
        if len(high_conf) > 0:
            # Calculate P&L
            high_conf['pnl'] = 0.0
            high_conf.loc[high_conf['prediction'] == 1, 'pnl'] = high_conf.loc[high_conf['prediction'] == 1, 'returns']
            high_conf.loc[high_conf['prediction'] == 0, 'pnl'] = -high_conf.loc[high_conf['prediction'] == 0, 'returns']
            
            accuracy = accuracy_score(high_conf['target'], high_conf['prediction'])
            avg_pnl = high_conf['pnl'].mean()
            total_pnl = high_conf['pnl'].sum()
            
            print(f"  High-conf trades: {len(high_conf)}/{len(test_df)} ({len(high_conf)/len(test_df)*100:.0f}%)")
            print(f"  Accuracy: {accuracy*100:.1f}%")
            print(f"  Avg P&L: {avg_pnl:+.4f}%")
            print(f"  Total P&L: {total_pnl:+.2f}%")
        else:
            accuracy, avg_pnl, total_pnl = 0, 0, 0
            print(f"  No high-confidence trades!")
        
        results.append({
            'fold': fold,
            'all_trades': len(test_df),
            'high_conf_trades': len(high_conf) if len(high_conf) > 0 else 0,
            'accuracy': accuracy,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl
        })
        
        all_predictions.append(test_df)
    
    results_df = pd.DataFrame(results)
    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\n{'='*70}")
    print("📊 HYBRID MODEL RESULTS")
    print(f"{'='*70}")
    print(f"\nAvg High-Conf Trades per fold: {results_df['high_conf_trades'].mean():.1f}")
    print(f"Avg Accuracy: {results_df['accuracy'].mean()*100:.1f}%")
    print(f"Avg P&L: {results_df['avg_pnl'].mean():+.4f}%")
    print(f"Total P&L: {results_df['total_pnl'].sum():+.2f}%")
    
    return results_df, all_pred_df


# ============================================
# COMPARISON & VISUALIZATION
# ============================================

def create_comparison_report(hybrid_results, df):
    """Compare Phase 1, 2, and 2.5"""
    
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON: Phase 1 vs 2 vs 2.5")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Phase': ['Phase 1 (Filters)', 'Phase 2 (ML All)', 'Phase 2.5 (Hybrid)'],
        'Trades': [75, 134, len(df)],
        'Avg_PnL': [0.076, -0.017, hybrid_results['avg_pnl'].mean()],
        'Accuracy': [45.3, 53.6, hybrid_results['accuracy'].mean() * 100],
        'Total_PnL': [5.74, -1.81, hybrid_results['total_pnl'].sum()]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    os.makedirs('outputs_phase2_5', exist_ok=True)
    comparison.to_csv('outputs_phase2_5/phase_comparison.csv', index=False)
    
    print("\n✓ Saved comparison to outputs_phase2_5/phase_comparison.csv")
    
    return comparison


# ============================================
# MAIN
# ============================================

def main():
    print("\nStarting Phase 2.5 (Hybrid)...")
    
    # Download
    prices, vix, macro_data = download_all_data()
    
    # Create hybrid dataset (Phase 1 filtered + Phase 2 features)
    df = create_hybrid_dataset(prices, vix, macro_data, macro_data['fed_funds'])
    
    if len(df) < 20:
        print("\n⚠️ Not enough data after filtering!")
        return
    
    # Save dataset
    os.makedirs('outputs_phase2_5', exist_ok=True)
    df.to_csv('outputs_phase2_5/hybrid_dataset.csv', index=False)
    
    # Train hybrid model
    results_df, predictions_df = train_hybrid_model(df)
    
    # Save results
    results_df.to_csv('outputs_phase2_5/hybrid_results.csv', index=False)
    predictions_df.to_csv('outputs_phase2_5/all_predictions.csv', index=False)
    
    # Comparison
    comparison = create_comparison_report(results_df, df)
    
    print("\n" + "="*70)
    print("✅ PHASE 2.5 COMPLETE!")
    print("="*70)
    print("\nFiles in outputs_phase2_5/:")
    print("  1. hybrid_dataset.csv")
    print("  2. hybrid_results.csv")
    print("  3. all_predictions.csv")
    print("  4. phase_comparison.csv")
    print()


if __name__ == "__main__":
    main()
