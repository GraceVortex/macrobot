"""
PHASE 5: PORTFOLIO COMBINATION
===============================
Testing combined deployment of Phase 3 + Phase 4.5 strategies

Approaches tested:
1. Pure Phase 3 (100% NFP+CPI)
2. Pure Phase 4.5 (100% 5 events)
3. 70/30 Blend (70% Phase 3, 30% Phase 4.5)
4. Equal Weight (50/50)
5. Dynamic allocation based on VIX regime

Goal: Optimize risk-adjusted returns through portfolio combination
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
                    PHASE 5: PORTFOLIO COMBINATION
================================================================================

Testing combined deployment of Phase 3 + Phase 4.5:

Portfolio Allocations:
  1. 100% Phase 3 (NFP+CPI only)
  2. 100% Phase 4.5 (5 events)
  3. 70/30 Phase 3/4.5
  4. 50/50 Equal weight
  5. Dynamic VIX-based allocation

Metrics: Returns, Sharpe, correlation, diversification benefit
""")

def download_prices(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    data = data[['Close']].copy()
    data.columns = ['close']
    data.index = pd.to_datetime(data.index).date
    return data

def download_vix():
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    return vix

def download_macro_data():
    fred = Fred(api_key=FRED_API_KEY)
    macro_data = {}
    
    nfp = fred.get_series('PAYEMS', observation_start=START_DATE, observation_end=END_DATE)
    nfp_df = pd.DataFrame({'value': nfp})
    nfp_df['change'] = nfp_df['value'].diff() * 1000
    macro_data['nfp'] = nfp_df.dropna()
    
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    cpi_df = pd.DataFrame({'value': cpi})
    cpi_df['yoy'] = cpi_df['value'].pct_change(12) * 100
    macro_data['cpi'] = cpi_df.dropna()
    
    gdp = fred.get_series('GDP', observation_start=START_DATE, observation_end=END_DATE)
    gdp_df = pd.DataFrame({'value': gdp})
    gdp_df['qoq'] = gdp_df['value'].pct_change() * 100
    macro_data['gdp'] = gdp_df.dropna()
    
    retail = fred.get_series('RSXFS', observation_start=START_DATE, observation_end=END_DATE)
    retail_df = pd.DataFrame({'value': retail})
    retail_df['mom'] = retail_df['value'].pct_change() * 100
    macro_data['retail'] = retail_df.dropna()
    
    pmi = fred.get_series('MANEMP', observation_start=START_DATE, observation_end=END_DATE)
    pmi_df = pd.DataFrame({'value': pmi})
    pmi_df['change'] = pmi_df['value'].diff()
    macro_data['pmi'] = pmi_df.dropna()
    
    macro_data['fed_funds'] = fred.get_series('FEDFUNDS', observation_start=START_DATE, observation_end=END_DATE)
    
    return macro_data

# Reuse helper functions (simplified for brevity)
def calculate_features(prices, vix, fed_funds, date):
    """Calculate all features - reuse from previous phases"""
    if hasattr(date, 'date'):
        date = date.date()
    
    prices_to_date = prices[prices.index < date]
    if len(prices_to_date) < 50:
        return None
    
    # Technical
    delta = prices_to_date['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    # VIX
    vix_to_date = vix[vix.index <= date]
    if len(vix_to_date) < 20:
        return None
    
    # Fed
    fed_dates = [d for d in fed_funds.index if pd.Timestamp(d).date() <= date]
    if not fed_dates:
        return None
    
    return {
        'rsi_14': rsi.iloc[-1] if len(rsi) > 0 else 50,
        'momentum_5d': (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-6] - 1) * 100 if len(prices_to_date) >= 6 else 0,
        'momentum_20d': (prices_to_date['close'].iloc[-1] / prices_to_date['close'].iloc[-21] - 1) * 100 if len(prices_to_date) >= 21 else 0,
        'volatility_20': prices_to_date['close'].pct_change().rolling(20).std().iloc[-1] * 100 if len(prices_to_date) >= 20 else 0,
        'vix': vix_to_date['vix'].iloc[-1],
        'vix_ma_20': vix_to_date['vix'].rolling(20).mean().iloc[-1],
        'vix_change_5d': vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-6] if len(vix_to_date) >= 6 else 0,
        'vix_change_20d': vix_to_date['vix'].iloc[-1] - vix_to_date['vix'].iloc[-21] if len(vix_to_date) >= 21 else 0,
        'vix_above_20': 1 if vix_to_date['vix'].iloc[-1] > 20 else 0,
        'fed_funds_rate': fed_funds.loc[fed_dates[-1]],
        'fed_rate_change_3m': fed_funds.loc[fed_dates[-1]] - fed_funds.loc[[d for d in fed_dates if pd.Timestamp(d).date() <= date - timedelta(days=90)][-1]] if len([d for d in fed_dates if pd.Timestamp(d).date() <= date - timedelta(days=90)]) > 0 else 0
    }

def load_phase_trades():
    """Load pre-generated trades from Phase 3 and 4.5"""
    print("\nLoading Phase 3 and Phase 4.5 results...")
    
    # Load Phase 3 trades
    phase3_trades = pd.read_csv('outputs_phase3/tmf_all_trades.csv')
    phase3_trades['strategy'] = 'Phase3'
    print(f"  Phase 3: {len(phase3_trades)} trades")
    
    # Load Phase 4.5 trades
    phase45_trades = pd.read_csv('outputs_phase4_5/all_trades.csv')
    phase45_trades['strategy'] = 'Phase4.5'
    print(f"  Phase 4.5: {len(phase45_trades)} trades")
    
    # Combine
    all_trades = pd.concat([phase3_trades, phase45_trades], ignore_index=True)
    all_trades['date'] = pd.to_datetime(all_trades['date'])
    all_trades = all_trades.sort_values('date').reset_index(drop=True)
    
    return phase3_trades, phase45_trades, all_trades

def calculate_portfolio_metrics(trades, weights={'Phase3': 1.0, 'Phase4.5': 0.0}):
    """Calculate portfolio performance with given weights"""
    
    portfolio_trades = []
    
    for _, trade in trades.iterrows():
        strategy = trade['strategy']
        weight = weights.get(strategy, 0)
        
        if weight > 0:
            weighted_trade = trade.copy()
            weighted_trade['weighted_pnl'] = trade['pnl'] * weight
            portfolio_trades.append(weighted_trade)
    
    if not portfolio_trades:
        return None
    
    portfolio_df = pd.DataFrame(portfolio_trades)
    
    total_pnl = portfolio_df['weighted_pnl'].sum()
    avg_pnl = portfolio_df['weighted_pnl'].mean()
    win_rate = (portfolio_df['weighted_pnl'] > 0).mean()
    
    # Sharpe
    if portfolio_df['weighted_pnl'].std() > 0:
        sharpe = portfolio_df['weighted_pnl'].mean() / portfolio_df['weighted_pnl'].std() * np.sqrt(52)
    else:
        sharpe = 0
    
    # Max DD
    cumulative = portfolio_df['weighted_pnl'].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    return {
        'trades': len(portfolio_df),
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'win_rate': win_rate * 100,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'volatility': portfolio_df['weighted_pnl'].std()
    }

def test_portfolio_allocations(all_trades):
    """Test different portfolio allocations"""
    
    print(f"\n{'='*70}")
    print("PORTFOLIO ALLOCATION TESTING")
    print(f"{'='*70}\n")
    
    allocations = {
        '100% Phase 3': {'Phase3': 1.0, 'Phase4.5': 0.0},
        '100% Phase 4.5': {'Phase3': 0.0, 'Phase4.5': 1.0},
        '70/30 P3/P4.5': {'Phase3': 0.7, 'Phase4.5': 0.3},
        '50/50 Equal': {'Phase3': 0.5, 'Phase4.5': 0.5},
        '30/70 P3/P4.5': {'Phase3': 0.3, 'Phase4.5': 0.7}
    }
    
    results = []
    
    for name, weights in allocations.items():
        metrics = calculate_portfolio_metrics(all_trades, weights)
        if metrics:
            metrics['allocation'] = name
            results.append(metrics)
            
            print(f"{name}:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Avg P&L: {metrics['avg_pnl']:+.4f}%")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Sharpe: {metrics['sharpe']:.2f}")
            print(f"  Max DD: {metrics['max_dd']:.2f}%")
            print()
    
    return pd.DataFrame(results)

def calculate_correlation(phase3_trades, phase45_trades):
    """Calculate correlation between strategies"""
    
    print(f"\n{'='*70}")
    print("STRATEGY CORRELATION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Align by date
    phase3_dates = set(pd.to_datetime(phase3_trades['date']).dt.date)
    phase45_dates = set(pd.to_datetime(phase45_trades['date']).dt.date)
    
    overlap_dates = phase3_dates & phase45_dates
    unique_phase3 = phase3_dates - phase45_dates
    unique_phase45 = phase45_dates - phase3_dates
    
    print(f"Overlapping trade dates: {len(overlap_dates)}")
    print(f"Unique to Phase 3: {len(unique_phase3)}")
    print(f"Unique to Phase 4.5: {len(unique_phase45)}")
    print(f"Total unique dates: {len(phase3_dates | phase45_dates)}")
    
    # Calculate correlation for overlapping dates
    if overlap_dates:
        p3_pnl = []
        p45_pnl = []
        
        for date in overlap_dates:
            p3_trades_date = phase3_trades[pd.to_datetime(phase3_trades['date']).dt.date == date]
            p45_trades_date = phase45_trades[pd.to_datetime(phase45_trades['date']).dt.date == date]
            
            if len(p3_trades_date) > 0 and len(p45_trades_date) > 0:
                p3_pnl.append(p3_trades_date['pnl'].mean())
                p45_pnl.append(p45_trades_date['pnl'].mean())
        
        if len(p3_pnl) > 1:
            correlation = np.corrcoef(p3_pnl, p45_pnl)[0, 1]
            print(f"\nP&L Correlation (overlapping dates): {correlation:.3f}")
            
            if correlation < 0.3:
                print("  Low correlation - good diversification potential")
            elif correlation < 0.7:
                print("  Moderate correlation - some diversification benefit")
            else:
                print("  High correlation - limited diversification benefit")
    
    return len(overlap_dates), len(unique_phase3), len(unique_phase45)

def dynamic_allocation_test(all_trades, vix):
    """Test dynamic VIX-based allocation"""
    
    print(f"\n{'='*70}")
    print("DYNAMIC VIX-BASED ALLOCATION")
    print(f"{'='*70}\n")
    
    portfolio_trades = []
    
    for _, trade in all_trades.iterrows():
        trade_date = pd.to_datetime(trade['date']).date()
        
        # Get VIX at trade date
        if trade_date in vix.index:
            vix_value = vix.loc[trade_date, 'vix']
        else:
            vix_value = 15  # default
        
        # Dynamic allocation
        if vix_value > 20:
            # High vol = prefer Phase 3 (quality)
            phase3_weight = 0.8
        else:
            # Normal vol = equal weight
            phase3_weight = 0.5
        
        weight = phase3_weight if trade['strategy'] == 'Phase3' else (1 - phase3_weight)
        
        weighted_trade = trade.copy()
        weighted_trade['weighted_pnl'] = trade['pnl'] * weight
        weighted_trade['vix'] = vix_value
        weighted_trade['weight'] = weight
        portfolio_trades.append(weighted_trade)
    
    portfolio_df = pd.DataFrame(portfolio_trades)
    
    metrics = {
        'trades': len(portfolio_df),
        'avg_pnl': portfolio_df['weighted_pnl'].mean(),
        'total_pnl': portfolio_df['weighted_pnl'].sum(),
        'win_rate': (portfolio_df['weighted_pnl'] > 0).mean() * 100,
        'sharpe': portfolio_df['weighted_pnl'].mean() / portfolio_df['weighted_pnl'].std() * np.sqrt(52) if portfolio_df['weighted_pnl'].std() > 0 else 0
    }
    
    print(f"Dynamic Allocation Results:")
    print(f"  Avg P&L: {metrics['avg_pnl']:+.4f}%")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Total P&L: {metrics['total_pnl']:+.2f}%")
    
    return metrics, portfolio_df

def main():
    print("\nStarting Phase 5...")
    
    # Load VIX
    vix = download_vix()
    
    # Load trades from previous phases
    phase3_trades, phase45_trades, all_trades = load_phase_trades()
    
    # Test fixed allocations
    allocation_results = test_portfolio_allocations(all_trades)
    
    # Correlation analysis
    calculate_correlation(phase3_trades, phase45_trades)
    
    # Dynamic allocation
    dynamic_metrics, dynamic_portfolio = dynamic_allocation_test(all_trades, vix)
    
    # Save results
    os.makedirs('outputs_phase5', exist_ok=True)
    allocation_results.to_csv('outputs_phase5/allocation_comparison.csv', index=False)
    dynamic_portfolio.to_csv('outputs_phase5/dynamic_portfolio.csv', index=False)
    
    # Find best allocation
    print(f"\n{'='*70}")
    print("OPTIMAL ALLOCATION")
    print(f"{'='*70}\n")
    
    best = allocation_results.loc[allocation_results['sharpe'].idxmax()]
    print(f"Best by Sharpe: {best['allocation']}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Avg P&L: {best['avg_pnl']:+.4f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    
    print(f"\n{'='*70}")
    print("PHASE 5 COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
