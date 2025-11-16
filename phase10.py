"""
PHASE 10: MULTI-EVENT MACRO STRATEGY
=====================================
Inspired by Phase 9 best practices:
- OHLC-based realistic stops (not clip!)
- Transaction costs scaling
- Systematic block testing (A/B/C/D)
- MTM vs Cash accounting
- Conservative estimates (30% haircut)
- Regime analysis

Strategy:
- Trade TMF on economic event surprises
- surprise = actual - consensus
- Directional: positive surprise → short TMF (hawkish), negative → long (dovish)
- Test multiple stop variants, holding periods, event types

Data needed:
- TMF/TLT OHLC (yfinance)
- Economic events CSV (NFP, CPI, PMI, Retail, PCE, ISM)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = '2018-01-01'
END_DATE = '2024-11-15'

# Transaction costs
BASE_COST_PCT = 0.15  # 0.15% base cost (bid-ask + slippage)

# Event thresholds (starting point, will be tested)
EVENT_THRESHOLDS = {
    'NFP': 30.0,           # thousands of jobs
    'CPI': 0.1,            # percent
    'Core_CPI': 0.1,       # percent
    'Core_PCE': 0.1,       # percent
    'Retail_Sales': 0.3,   # percent
    'ISM_PMI': 2.0         # index points
}

# Output directory
OUTPUT_DIR = 'outputs_phase10'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA COLLECTION
# ============================================================================

def download_tmf_ohlc():
    """Download TMF OHLC data from yfinance"""
    print("Downloading TMF OHLC data...")
    tmf = yf.download('TMF', start=START_DATE, end=END_DATE, progress=False)
    
    if len(tmf) == 0:
        print("  [ERROR] No TMF data downloaded!")
        return None
    
    # Flatten multi-index columns if needed
    if isinstance(tmf.columns, pd.MultiIndex):
        tmf.columns = tmf.columns.get_level_values(0)
    
    tmf = tmf[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    tmf.index.name = 'Date'
    
    print(f"  Downloaded {len(tmf)} days of TMF data")
    print(f"  Date range: {tmf.index[0].date()} to {tmf.index[-1].date()}")
    
    # Save to file
    tmf.to_csv(f'{OUTPUT_DIR}/tmf_ohlc.csv')
    print(f"  Saved to {OUTPUT_DIR}/tmf_ohlc.csv")
    
    return tmf


def load_economic_events():
    """
    Load economic events from CSV
    Expected columns: Date, Event_Type, Actual, Consensus, Previous (optional)
    
    If file doesn't exist, create template
    """
    events_file = f'{OUTPUT_DIR}/economic_events.csv'
    
    if not os.path.exists(events_file):
        print(f"Creating template: {events_file}")
        print("Please fill in the economic events data manually or scrape from Investing.com")
        
        # Create template
        template = pd.DataFrame({
            'Date': ['2024-11-01', '2024-10-10', '2024-09-06'],
            'Event_Type': ['NFP', 'CPI', 'ISM_PMI'],
            'Actual': [12.0, 2.6, 47.2],
            'Consensus': [100.0, 2.4, 47.5],
            'Previous': [223.0, 2.5, 47.9]
        })
        template.to_csv(events_file, index=False)
        print(f"  Template created with {len(template)} example rows")
        print("  [ACTION REQUIRED] Fill in real data and re-run")
        return None
    
    # Load events
    print(f"Loading economic events from {events_file}...")
    events = pd.read_csv(events_file)
    events['Date'] = pd.to_datetime(events['Date'])
    
    print(f"  Loaded {len(events)} events")
    print(f"  Event types: {events['Event_Type'].unique().tolist()}")
    print(f"  Date range: {events['Date'].min().date()} to {events['Date'].max().date()}")
    
    return events


def merge_events_with_prices(events, tmf_ohlc):
    """
    Merge events with TMF OHLC data
    Handle weekends/holidays by using next trading day
    """
    print("\nMerging events with TMF prices...")
    
    merged = []
    
    for idx, row in events.iterrows():
        event_date = row['Date']
        event_type = row['Event_Type']
        actual = row['Actual']
        consensus = row['Consensus']
        
        # Calculate surprise
        surprise = actual - consensus
        surprise_pct = (surprise / consensus * 100) if consensus != 0 else 0
        
        # Find entry price (next trading day open if event on weekend)
        # Entry = open of next trading day after event
        future_dates = tmf_ohlc[tmf_ohlc.index > event_date]
        
        if len(future_dates) == 0:
            continue  # Event too recent, no future data
        
        entry_date = future_dates.index[0]
        entry_open = future_dates.iloc[0]['Open']
        
        merged.append({
            'event_date': event_date,
            'entry_date': entry_date,
            'event_type': event_type,
            'actual': actual,
            'consensus': consensus,
            'surprise': surprise,
            'surprise_pct': surprise_pct,
            'entry_open': entry_open
        })
    
    merged_df = pd.DataFrame(merged)
    
    print(f"  Merged {len(merged_df)} events with entry prices")
    
    # Save
    merged_df.to_csv(f'{OUTPUT_DIR}/merged_data.csv', index=False)
    print(f"  Saved to {OUTPUT_DIR}/merged_data.csv")
    
    return merged_df


# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def generate_signal(event_type, surprise, surprise_pct, thresholds):
    """
    Generate trading signal based on surprise
    
    Returns:
        signal: +1 (long TMF, dovish), -1 (short TMF, hawkish), 0 (no trade)
    """
    threshold = thresholds.get(event_type, 999999)
    
    # Determine if using absolute or percentage threshold
    if event_type in ['CPI', 'Core_CPI', 'Core_PCE', 'Retail_Sales']:
        surprise_val = surprise_pct
    else:
        surprise_val = surprise
    
    if surprise_val > threshold:
        return -1  # Hawkish surprise → rates up → bonds down → short TMF
    elif surprise_val < -threshold:
        return 1   # Dovish surprise → rates down → bonds up → long TMF
    else:
        return 0


def check_ohlc_stop(entry_price, stop_pct, high, low, direction):
    """
    Check if stop was hit using OHLC data (REALISTIC!)
    
    Args:
        entry_price: Entry price
        stop_pct: Stop loss percentage (e.g. 2.0 for 2%)
        high: High price of the bar
        low: Low price of the bar
        direction: +1 for long, -1 for short
    
    Returns:
        (stopped: bool, exit_price: float)
    """
    if direction == 1:  # Long position
        stop_price = entry_price * (1 - stop_pct / 100)
        if low <= stop_price:
            return True, stop_price
    else:  # Short position
        stop_price = entry_price * (1 + stop_pct / 100)
        if high >= stop_price:
            return True, stop_price
    
    return False, None


def check_close_stop(entry_price, stop_pct, close_price, direction):
    """
    Check if stop was hit using close price only (less realistic)
    """
    if direction == 1:  # Long
        stop_price = entry_price * (1 - stop_pct / 100)
        if close_price <= stop_price:
            return True, close_price
    else:  # Short
        stop_price = entry_price * (1 + stop_pct / 100)
        if close_price >= stop_price:
            return True, close_price
    
    return False, None


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def backtest_strategy(merged_df, tmf_ohlc, holding_days=2, stop_type='none', stop_pct=0, 
                     thresholds=None, position_size=1.0):
    """
    Backtest multi-event macro strategy
    
    Args:
        merged_df: Merged events + prices
        tmf_ohlc: TMF OHLC dataframe
        holding_days: How many days to hold (1, 2, or 5)
        stop_type: 'none', 'ohlc', 'close', 'atr'
        stop_pct: Stop loss percentage
        thresholds: Dict of event thresholds
        position_size: Position multiplier for costs
    
    Returns:
        trades: DataFrame with all trades
    """
    if thresholds is None:
        thresholds = EVENT_THRESHOLDS
    
    trades = []
    
    for idx, row in merged_df.iterrows():
        event_type = row['event_type']
        surprise = row['surprise']
        surprise_pct = row['surprise_pct']
        entry_date = row['entry_date']
        entry_price = row['entry_open']
        
        # Generate signal
        signal = generate_signal(event_type, surprise, surprise_pct, thresholds)
        
        if signal == 0:
            continue  # No trade
        
        # Find exit date (holding_days later)
        future_data = tmf_ohlc[tmf_ohlc.index > entry_date]
        
        if len(future_data) < holding_days:
            continue  # Not enough future data
        
        # Check for stop hit during holding period
        stopped = False
        exit_price = None
        exit_date = None
        
        for i in range(holding_days):
            bar = future_data.iloc[i]
            bar_date = bar.name
            
            if stop_type == 'ohlc' and stop_pct > 0:
                hit, price = check_ohlc_stop(entry_price, stop_pct, bar['High'], bar['Low'], signal)
                if hit:
                    stopped = True
                    exit_price = price
                    exit_date = bar_date
                    break
            elif stop_type == 'close' and stop_pct > 0:
                hit, price = check_close_stop(entry_price, stop_pct, bar['Close'], signal)
                if hit:
                    stopped = True
                    exit_price = price
                    exit_date = bar_date
                    break
        
        # If not stopped, exit at holding_days close
        if not stopped:
            exit_bar = future_data.iloc[holding_days - 1]
            exit_price = exit_bar['Close']
            exit_date = exit_bar.name
        
        # Calculate P&L
        if signal == 1:  # Long
            pnl_gross = (exit_price / entry_price - 1) * 100
        else:  # Short
            pnl_gross = (entry_price / exit_price - 1) * 100
        
        # Transaction costs (scale by position size)
        cost = BASE_COST_PCT * position_size * 2  # Entry + exit
        pnl_net = pnl_gross - cost
        
        trades.append({
            'event_date': row['event_date'],
            'entry_date': entry_date,
            'exit_date': exit_date,
            'event_type': event_type,
            'surprise': surprise,
            'surprise_pct': surprise_pct,
            'signal': signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_gross': pnl_gross,
            'cost': cost,
            'pnl_net': pnl_net,
            'stopped': stopped,
            'holding_days': holding_days,
            'stop_type': stop_type,
            'stop_pct': stop_pct
        })
    
    trades_df = pd.DataFrame(trades)
    
    return trades_df


def calculate_metrics(trades_df, label="Strategy"):
    """Calculate performance metrics"""
    if len(trades_df) == 0:
        return None
    
    returns = trades_df['pnl_net'].values
    
    # Basic metrics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_net'] > 0])
    losses = len(trades_df[trades_df['pnl_net'] <= 0])
    win_rate = wins / total_trades * 100
    
    avg_return = returns.mean()
    total_return = returns.sum()
    
    # Sharpe ratio (annualized, assuming ~50 trades/year)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(50)  # ~50 events/year
    else:
        sharpe = 0
    
    # Drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    # Winners/losers
    if wins > 0:
        avg_winner = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].mean()
    else:
        avg_winner = 0
    
    if losses > 0:
        avg_loser = trades_df[trades_df['pnl_net'] <= 0]['pnl_net'].mean()
    else:
        avg_loser = 0
    
    # Stopped trades
    stopped_count = trades_df['stopped'].sum()
    stopped_rate = stopped_count / total_trades * 100
    
    metrics = {
        'label': label,
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'avg_return': avg_return,
        'total_return': total_return,
        'max_dd': max_dd,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'stopped_count': stopped_count,
        'stopped_rate': stopped_rate
    }
    
    return metrics


# ============================================================================
# BLOCK TESTING
# ============================================================================

def block_a_baseline(merged_df, tmf_ohlc):
    """BLOCK A: Baseline tests (no stops, various thresholds)"""
    print("=" * 80)
    print("BLOCK A: BASELINE TESTS")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: Default thresholds, 2-day hold, no stop
    print("Test 1: Default thresholds, 2-day hold, no stop")
    trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=2, stop_type='none')
    metrics = calculate_metrics(trades, "Baseline (2d, no stop)")
    if metrics:
        results.append(metrics)
        print(f"  Trades: {metrics['trades']}, Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}")
    
    # Test 2: Tighter thresholds
    print("Test 2: Tighter thresholds (50% of default)")
    tight_thresholds = {k: v * 0.5 for k, v in EVENT_THRESHOLDS.items()}
    trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=2, stop_type='none', 
                               thresholds=tight_thresholds)
    metrics = calculate_metrics(trades, "Tight thresholds (2d)")
    if metrics:
        results.append(metrics)
        print(f"  Trades: {metrics['trades']}, Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}")
    
    # Test 3: Wider thresholds
    print("Test 3: Wider thresholds (150% of default)")
    wide_thresholds = {k: v * 1.5 for k, v in EVENT_THRESHOLDS.items()}
    trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=2, stop_type='none', 
                               thresholds=wide_thresholds)
    metrics = calculate_metrics(trades, "Wide thresholds (2d)")
    if metrics:
        results.append(metrics)
        print(f"  Trades: {metrics['trades']}, Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}")
    
    print()
    return pd.DataFrame(results)


def block_b_stops(merged_df, tmf_ohlc):
    """BLOCK B: Stop-loss experiments (OHLC-based!)"""
    print("=" * 80)
    print("BLOCK B: STOP-LOSS EXPERIMENTS")
    print("=" * 80)
    print()
    
    results = []
    
    # No stop baseline
    print("Baseline: No stop")
    trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=2, stop_type='none')
    metrics = calculate_metrics(trades, "No stop (2d)")
    if metrics:
        results.append(metrics)
        print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Stopped: {metrics['stopped_rate']:.1f}%")
    
    # OHLC stops (REALISTIC!)
    for stop_pct in [2, 3, 5, 7]:
        print(f"OHLC stop: {stop_pct}%")
        trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=2, 
                                   stop_type='ohlc', stop_pct=stop_pct)
        metrics = calculate_metrics(trades, f"OHLC stop {stop_pct}% (2d)")
        if metrics:
            results.append(metrics)
            print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Stopped: {metrics['stopped_rate']:.1f}%")
    
    # Close stops (less realistic)
    for stop_pct in [2, 3, 5]:
        print(f"Close stop: {stop_pct}%")
        trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=2, 
                                   stop_type='close', stop_pct=stop_pct)
        metrics = calculate_metrics(trades, f"Close stop {stop_pct}% (2d)")
        if metrics:
            results.append(metrics)
            print(f"  Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Stopped: {metrics['stopped_rate']:.1f}%")
    
    print()
    return pd.DataFrame(results)


def block_c_event_analysis(merged_df, tmf_ohlc):
    """BLOCK C: Event type analysis (which events work?)"""
    print("=" * 80)
    print("BLOCK C: EVENT TYPE ANALYSIS")
    print("=" * 80)
    print()
    
    results = []
    
    event_types = merged_df['event_type'].unique()
    
    for event_type in event_types:
        print(f"Testing: {event_type}")
        
        # Filter to only this event type
        event_subset = merged_df[merged_df['event_type'] == event_type]
        
        if len(event_subset) < 3:
            print(f"  [SKIP] Only {len(event_subset)} events")
            continue
        
        trades = backtest_strategy(event_subset, tmf_ohlc, holding_days=2, stop_type='none')
        metrics = calculate_metrics(trades, f"{event_type} only")
        
        if metrics:
            results.append(metrics)
            print(f"  Trades: {metrics['trades']}, Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Avg: {metrics['avg_return']:+.2f}%")
    
    print()
    return pd.DataFrame(results)


def block_d_holding_periods(merged_df, tmf_ohlc):
    """BLOCK D: Holding period optimization (1d vs 2d vs 5d)"""
    print("=" * 80)
    print("BLOCK D: HOLDING PERIOD OPTIMIZATION")
    print("=" * 80)
    print()
    
    results = []
    
    for holding_days in [1, 2, 5]:
        print(f"Holding period: {holding_days} days")
        trades = backtest_strategy(merged_df, tmf_ohlc, holding_days=holding_days, stop_type='none')
        metrics = calculate_metrics(trades, f"{holding_days}-day hold")
        
        if metrics:
            results.append(metrics)
            print(f"  Trades: {metrics['trades']}, Sharpe: {metrics['sharpe']:.2f}, Win%: {metrics['win_rate']:.1f}, Avg: {metrics['avg_return']:+.2f}%")
    
    print()
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 10: MULTI-EVENT MACRO STRATEGY")
    print("=" * 80)
    print()
    
    # Step 1: Download TMF data
    tmf_ohlc = download_tmf_ohlc()
    if tmf_ohlc is None:
        print("[ERROR] Failed to download TMF data")
        return
    
    print()
    
    # Step 2: Load economic events
    events = load_economic_events()
    if events is None:
        print("[ERROR] No economic events data. Please fill template and re-run.")
        return
    
    print()
    
    # Step 3: Merge events with prices
    merged_df = merge_events_with_prices(events, tmf_ohlc)
    
    if len(merged_df) == 0:
        print("[ERROR] No events could be merged with prices")
        return
    
    print()
    
    # Step 4: Run Block Tests
    print("Starting block tests...")
    print()
    
    # Block A: Baseline
    results_a = block_a_baseline(merged_df, tmf_ohlc)
    results_a.to_csv(f'{OUTPUT_DIR}/block_a_baseline.csv', index=False)
    
    # Block B: Stops
    results_b = block_b_stops(merged_df, tmf_ohlc)
    results_b.to_csv(f'{OUTPUT_DIR}/block_b_stops.csv', index=False)
    
    # Block C: Event analysis
    results_c = block_c_event_analysis(merged_df, tmf_ohlc)
    results_c.to_csv(f'{OUTPUT_DIR}/block_c_events.csv', index=False)
    
    # Block D: Holding periods
    results_d = block_d_holding_periods(merged_df, tmf_ohlc)
    results_d.to_csv(f'{OUTPUT_DIR}/block_d_periods.csv', index=False)
    
    # Step 5: Summary
    print("=" * 80)
    print("PHASE 10 COMPLETE")
    print("=" * 80)
    print()
    print("Files saved:")
    print(f"  {OUTPUT_DIR}/tmf_ohlc.csv")
    print(f"  {OUTPUT_DIR}/economic_events.csv")
    print(f"  {OUTPUT_DIR}/merged_data.csv")
    print(f"  {OUTPUT_DIR}/block_a_baseline.csv")
    print(f"  {OUTPUT_DIR}/block_b_stops.csv")
    print(f"  {OUTPUT_DIR}/block_c_events.csv")
    print(f"  {OUTPUT_DIR}/block_d_periods.csv")
    print()
    print("Next steps:")
    print("1. Review block results")
    print("2. Fill in more economic events data")
    print("3. Analyze which events/configurations work best")
    print("4. Generate final SUMMARY.txt if strategy is promising")
    print()


if __name__ == '__main__':
    main()
