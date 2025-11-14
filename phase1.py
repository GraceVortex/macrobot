"""
PHASE 1 IMPROVEMENTS - MACRO TRADING BACKTEST
==============================================
Улучшения:
1. VIX Filter - торгуем только при VIX > 15
2. IEF Testing - тестируем 7-10 Year Treasuries вместо TLT
3. Trend Filter - торгуем только по тренду
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
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

FRED_API_KEY = "ef636b6d99542d08f7d0ab6152932290"

START_DATE = "2014-01-01"
END_DATE = "2024-11-14"

# Phase 1: Test multiple instruments
INSTRUMENTS = {
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF'
}

HOLDING_PERIODS = [1, 2, 3, 5]
RATING_THRESHOLD = 1.0  # Aggressive threshold

# Phase 1 improvements
VIX_THRESHOLD = 15  # Only trade when VIX > this
TREND_MA_SHORT = 20  # 20-day MA
TREND_MA_LONG = 50   # 50-day MA

# ============================================
# 1. DATA DOWNLOAD
# ============================================

def download_prices(ticker):
    """Download historical prices"""
    print(f"📊 Downloading {ticker} prices...")
    
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    data = data[['Close']].copy()
    data.columns = ['close']
    data.index = pd.to_datetime(data.index).date
    
    print(f"   ✓ Downloaded {len(data)} days")
    return data


def download_vix():
    """Download VIX (Volatility Index)"""
    print("📊 Downloading VIX...")
    
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    vix = vix[['Close']].copy()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index).date
    
    print(f"   ✓ Downloaded {len(vix)} days of VIX data")
    return vix


def download_macro_data():
    """Download macro events from FRED"""
    print("📈 Downloading macro data from FRED...")
    
    fred = Fred(api_key=FRED_API_KEY)
    
    # NFP
    print("   Downloading NFP data...")
    nfp = fred.get_series('PAYEMS', observation_start=START_DATE, observation_end=END_DATE)
    nfp_df = pd.DataFrame({'value': nfp})
    nfp_df['change'] = nfp_df['value'].diff() * 1000
    nfp_df = nfp_df.dropna()
    
    # CPI
    print("   Downloading CPI data...")
    cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE, observation_end=END_DATE)
    cpi_df = pd.DataFrame({'value': cpi})
    cpi_df['yoy'] = cpi_df['value'].pct_change(12) * 100
    cpi_df = cpi_df.dropna()
    
    print(f"   ✓ Downloaded {len(nfp_df)} NFP releases")
    print(f"   ✓ Downloaded {len(cpi_df)} CPI releases")
    
    return {'nfp': nfp_df, 'cpi': cpi_df}


# ============================================
# 2. RATING CALCULATION
# ============================================

def calculate_nfp_rating(row, nfp_data):
    """Calculate NFP surprise rating"""
    actual = row['change']
    
    # Consensus = rolling mean of last 6 months
    idx = nfp_data.index.get_loc(row.name)
    if idx < 6:
        return 0, actual, actual, 0, 0
    
    consensus = nfp_data.iloc[idx-6:idx]['change'].mean()
    surprise = actual - consensus
    surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0
    
    # Rating scale
    if abs(surprise_pct) < 5:
        rating = 0
    elif abs(surprise_pct) < 15:
        rating = np.sign(surprise_pct) * 2
    elif abs(surprise_pct) < 30:
        rating = np.sign(surprise_pct) * 3
    elif abs(surprise_pct) < 50:
        rating = np.sign(surprise_pct) * 4
    else:
        rating = np.sign(surprise_pct) * 5
    
    return rating, actual, consensus, surprise, surprise_pct


def calculate_cpi_rating(row, cpi_data):
    """Calculate CPI surprise rating"""
    actual = row['yoy']
    
    # Consensus = rolling mean of last 6 months
    idx = cpi_data.index.get_loc(row.name)
    if idx < 6:
        return 0, actual, actual, 0, 0
    
    consensus = cpi_data.iloc[idx-6:idx]['yoy'].mean()
    surprise = actual - consensus
    surprise_pct = (surprise / abs(consensus)) * 100 if consensus != 0 else 0
    
    # Rating scale
    if abs(surprise_pct) < 2:
        rating = 0
    elif abs(surprise_pct) < 5:
        rating = np.sign(surprise_pct) * 2
    elif abs(surprise_pct) < 10:
        rating = np.sign(surprise_pct) * 3
    elif abs(surprise_pct) < 20:
        rating = np.sign(surprise_pct) * 4
    else:
        rating = np.sign(surprise_pct) * 5
    
    return rating, actual, consensus, surprise, surprise_pct


# ============================================
# 3. TREND CALCULATION
# ============================================

def calculate_trend(prices, date):
    """Calculate if instrument is in uptrend or downtrend"""
    # Convert date to date if it's a Timestamp
    if hasattr(date, 'date'):
        date = date.date()
    
    # Get prices up to this date
    prices_to_date = prices[prices.index <= date]
    
    if len(prices_to_date) < TREND_MA_LONG:
        return 'neutral'
    
    # Calculate MAs
    ma_short = prices_to_date['close'].iloc[-TREND_MA_SHORT:].mean()
    ma_long = prices_to_date['close'].iloc[-TREND_MA_LONG:].mean()
    
    if ma_short > ma_long:
        return 'uptrend'
    elif ma_short < ma_long:
        return 'downtrend'
    else:
        return 'neutral'


# ============================================
# 4. BACKTEST ENGINE WITH FILTERS
# ============================================

def calculate_returns(prices, event_date, holding_periods):
    """Calculate returns for various holding periods"""
    
    # Convert event_date to date if it's a Timestamp
    if hasattr(event_date, 'date'):
        event_date = event_date.date()
    
    if event_date not in prices.index:
        # Find next available date
        future_dates = [d for d in prices.index if d > event_date]
        if not future_dates:
            return {}
        event_date = future_dates[0]
    
    base_price = prices.loc[event_date, 'close']
    returns = {}
    
    # Get future dates
    future_dates = [d for d in prices.index if d > event_date]
    
    for period in holding_periods:
        if period - 1 < len(future_dates):
            future_date = future_dates[period - 1]
            future_price = prices.loc[future_date, 'close']
            ret = (future_price - base_price) / base_price * 100
            returns[f'{period}d'] = ret
        else:
            returns[f'{period}d'] = None
    
    return returns


def get_vix_at_date(vix, date):
    """Get VIX value at specific date"""
    if hasattr(date, 'date'):
        date = date.date()
    
    if date in vix.index:
        return vix.loc[date, 'vix']
    
    # Find closest date
    past_dates = [d for d in vix.index if d <= date]
    if past_dates:
        return vix.loc[past_dates[-1], 'vix']
    
    return None


def run_backtest_with_filters(prices, vix, macro_data, holding_periods, instrument_name):
    """Run backtest with Phase 1 filters"""
    
    print(f"\n{'='*60}")
    print(f"🔄 RUNNING BACKTEST FOR {instrument_name}")
    print(f"{'='*60}")
    
    results = []
    
    # Process NFP events
    print(f"\n📊 Processing NFP events...")
    nfp_data = macro_data['nfp']
    
    for date, row in nfp_data.iterrows():
        rating, actual, consensus, surprise, surprise_pct = calculate_nfp_rating(row, nfp_data)
        
        if abs(rating) < RATING_THRESHOLD:
            continue
        
        # PHASE 1 FILTER 1: VIX Filter
        vix_value = get_vix_at_date(vix, date)
        if vix_value is None or vix_value < VIX_THRESHOLD:
            continue  # Skip trade if VIX too low
        
        # PHASE 1 FILTER 2: Trend Filter
        trend = calculate_trend(prices, date)
        # Only trade if signal aligns with trend
        if rating > 0 and trend == 'uptrend':
            continue  # Hawkish signal but price in uptrend, skip
        if rating < 0 and trend == 'downtrend':
            continue  # Dovish signal but price in downtrend, skip
        
        # Calculate returns
        returns = calculate_returns(prices, date, holding_periods)
        
        results.append({
            'date': date,
            'type': 'NFP',
            'actual': actual,
            'consensus': consensus,
            'surprise': surprise,
            'surprise_pct': surprise_pct,
            'rating': rating,
            'vix': vix_value,
            'trend': trend,
            **returns
        })
    
    # Process CPI events
    print(f"📊 Processing CPI events...")
    cpi_data = macro_data['cpi']
    
    for date, row in cpi_data.iterrows():
        rating, actual, consensus, surprise, surprise_pct = calculate_cpi_rating(row, cpi_data)
        
        if abs(rating) < RATING_THRESHOLD:
            continue
        
        # PHASE 1 FILTER 1: VIX Filter
        vix_value = get_vix_at_date(vix, date)
        if vix_value is None or vix_value < VIX_THRESHOLD:
            continue
        
        # PHASE 1 FILTER 2: Trend Filter
        trend = calculate_trend(prices, date)
        if rating > 0 and trend == 'uptrend':
            continue
        if rating < 0 and trend == 'downtrend':
            continue
        
        # Calculate returns
        returns = calculate_returns(prices, date, holding_periods)
        
        results.append({
            'date': date,
            'type': 'CPI',
            'actual': actual,
            'consensus': consensus,
            'surprise': surprise,
            'surprise_pct': surprise_pct,
            'rating': rating,
            'vix': vix_value,
            'trend': trend,
            **returns
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print(f"\n✓ Processed {len(df)} total trades")
        print(f"  - NFP: {len(df[df['type']=='NFP'])} trades")
        print(f"  - CPI: {len(df[df['type']=='CPI'])} trades")
        print(f"  - Avg VIX: {df['vix'].mean():.1f}")
        print(f"  - Uptrend: {len(df[df['trend']=='uptrend'])}")
        print(f"  - Downtrend: {len(df[df['trend']=='downtrend'])}")
    else:
        print(f"\n⚠️ No trades passed filters!")
    
    return df


# ============================================
# 5. ANALYSIS
# ============================================

def analyze_results(df, instrument_name, holding_period='2d'):
    """Analyze backtest results"""
    
    if len(df) == 0:
        print(f"\n⚠️ No data to analyze for {instrument_name}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📈 RESULTS FOR {instrument_name} ({holding_period} holding)")
    print(f"{'='*60}")
    
    returns_col = holding_period
    valid_trades = df[df[returns_col].notna()].copy()
    
    if len(valid_trades) == 0:
        print("No valid trades")
        return None
    
    # Calculate metrics
    total_pnl = valid_trades[returns_col].sum()
    avg_pnl = valid_trades[returns_col].mean()
    median_pnl = valid_trades[returns_col].median()
    std_pnl = valid_trades[returns_col].std()
    
    wins = valid_trades[valid_trades[returns_col] > 0]
    losses = valid_trades[valid_trades[returns_col] < 0]
    
    win_rate = len(wins) / len(valid_trades) * 100
    avg_win = wins[returns_col].mean() if len(wins) > 0 else 0
    avg_loss = losses[returns_col].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins[returns_col].sum() / losses[returns_col].sum()) if len(losses) > 0 else float('inf')
    
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252/2) if std_pnl > 0 else 0
    
    # Max drawdown
    cumsum = valid_trades[returns_col].cumsum()
    running_max = cumsum.expanding().max()
    drawdown = cumsum - running_max
    max_dd = drawdown.min()
    
    print(f"\nTrades: {len(valid_trades)}")
    print(f"Total P&L: {total_pnl:+.2f}%")
    print(f"Avg P&L: {avg_pnl:+.4f}%")
    print(f"Median P&L: {median_pnl:+.4f}%")
    print(f"Std Dev: {std_pnl:.4f}%")
    print(f"\nWin Rate: {win_rate:.2f}%")
    print(f"Avg Win: {avg_win:+.4f}%")
    print(f"Avg Loss: {avg_loss:+.4f}%")
    print(f"Profit Factor: {profit_factor:.3f}")
    print(f"\nSharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    
    # Compare with baseline (no filters)
    print(f"\n🔍 Filter Impact:")
    print(f"   VIX threshold: >{VIX_THRESHOLD}")
    print(f"   Avg VIX in trades: {valid_trades['vix'].mean():.1f}")
    print(f"   Trend alignment: Required")
    
    return {
        'instrument': instrument_name,
        'trades': len(valid_trades),
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'profit_factor': profit_factor
    }


# ============================================
# 6. COMPARISON & VISUALIZATION
# ============================================

def compare_instruments(results_dict):
    """Compare results across instruments"""
    
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON: TLT vs IEF (with Phase 1 filters)")
    print(f"{'='*70}")
    
    comparison = []
    for name, metrics in results_dict.items():
        if metrics:
            comparison.append(metrics)
    
    if len(comparison) == 0:
        print("No results to compare")
        return
    
    comp_df = pd.DataFrame(comparison)
    
    print(f"\n{comp_df.to_string(index=False)}")
    
    # Find best
    best = comp_df.loc[comp_df['sharpe'].idxmax()]
    print(f"\n🏆 BEST INSTRUMENT: {best['instrument']}")
    print(f"   Sharpe: {best['sharpe']:.3f}")
    print(f"   Win Rate: {best['win_rate']:.2f}%")
    print(f"   Avg P&L: {best['avg_pnl']:+.4f}%")
    
    # Save comparison
    os.makedirs('outputs_phase1', exist_ok=True)
    comp_df.to_csv('outputs_phase1/instrument_comparison.csv', index=False)
    print(f"\n✓ Saved: outputs_phase1/instrument_comparison.csv")
    
    return comp_df


def create_comparison_chart(all_results):
    """Create comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1 Results: TLT vs IEF (with VIX + Trend Filters)', fontsize=14, fontweight='bold')
    
    colors = {'TLT': '#e74c3c', 'IEF': '#3498db'}
    
    # Plot 1: Win Rate comparison
    ax1 = axes[0, 0]
    for name, df in all_results.items():
        if len(df) > 0:
            valid = df[df['2d'].notna()]
            win_rate = (valid['2d'] > 0).sum() / len(valid) * 100
            ax1.bar(name, win_rate, color=colors[name], alpha=0.7)
    
    ax1.axhline(y=50, color='red', linestyle='--', label='Random (50%)')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe Ratio
    ax2 = axes[0, 1]
    for name, df in all_results.items():
        if len(df) > 0:
            valid = df[df['2d'].notna()]
            avg_pnl = valid['2d'].mean()
            std_pnl = valid['2d'].std()
            sharpe = (avg_pnl / std_pnl) * np.sqrt(252/2) if std_pnl > 0 else 0
            ax2.bar(name, sharpe, color=colors[name], alpha=0.7)
    
    ax2.axhline(y=1.0, color='green', linestyle='--', label='Target (1.0)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Equity curves
    ax3 = axes[1, 0]
    for name, df in all_results.items():
        if len(df) > 0:
            valid = df[df['2d'].notna()].copy()
            valid = valid.sort_values('date')
            equity = valid['2d'].cumsum()
            ax3.plot(equity.values, label=name, color=colors[name], linewidth=2)
    
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative P&L (%)')
    ax3.set_title('Equity Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Number of trades
    ax4 = axes[1, 1]
    for name, df in all_results.items():
        if len(df) > 0:
            ax4.bar(name, len(df), color=colors[name], alpha=0.7)
    
    ax4.set_ylabel('Number of Trades')
    ax4.set_title('Trade Count (after filters)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('outputs_phase1', exist_ok=True)
    plt.savefig('outputs_phase1/phase1_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: outputs_phase1/phase1_comparison.png")
    
    plt.close()


# ============================================
# 7. MAIN
# ============================================

def main():
    print("\n" + "="*70)
    print("🚀 PHASE 1 IMPROVEMENTS - MACRO TRADING BACKTEST")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"\nPhase 1 Improvements:")
    print(f"  ✓ VIX Filter (only trade when VIX > {VIX_THRESHOLD})")
    print(f"  ✓ Trend Filter (align trades with {TREND_MA_SHORT}/{TREND_MA_LONG} MA trend)")
    print(f"  ✓ Testing IEF (7-10Y) vs TLT (20+Y)")
    print()
    
    # Download common data
    vix = download_vix()
    macro_data = download_macro_data()
    
    # Test each instrument
    all_results = {}
    comparison_metrics = {}
    
    for ticker, description in INSTRUMENTS.items():
        print(f"\n{'='*70}")
        print(f"Testing {ticker}: {description}")
        print(f"{'='*70}")
        
        prices = download_prices(ticker)
        
        df = run_backtest_with_filters(prices, vix, macro_data, HOLDING_PERIODS, ticker)
        
        if len(df) > 0:
            all_results[ticker] = df
            
            # Save individual results
            os.makedirs('outputs_phase1', exist_ok=True)
            df.to_csv(f'outputs_phase1/{ticker.lower()}_trades.csv', index=False)
            print(f"✓ Saved: outputs_phase1/{ticker.lower()}_trades.csv")
            
            # Analyze
            metrics = analyze_results(df, ticker, '2d')
            comparison_metrics[ticker] = metrics
    
    # Compare instruments
    if len(comparison_metrics) > 0:
        compare_instruments(comparison_metrics)
        create_comparison_chart(all_results)
    
    print(f"\n{'='*70}")
    print("✅ PHASE 1 COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated files in outputs_phase1/:")
    print("  1. tlt_trades.csv - TLT trades with filters")
    print("  2. ief_trades.csv - IEF trades with filters")
    print("  3. instrument_comparison.csv - Side-by-side comparison")
    print("  4. phase1_comparison.png - Visual comparison")
    print()
    
    # Final recommendation
    print("🎯 NEXT STEPS:")
    print("   1. Review comparison results")
    print("   2. If Sharpe > 0.8: Proceed to Phase 2 (ML model)")
    print("   3. If Sharpe < 0.8: Try adjusting VIX threshold or add more filters")
    print()


if __name__ == "__main__":
    main()
