PHASE 10: MULTI-EVENT MACRO STRATEGY
=====================================

Quick Start Guide

WHAT IS THIS?
-------------
Event-driven TMF trading strategy based on economic data surprises.
Tests whether trading on NFP, CPI, PMI, Retail Sales, and PCE works.

Uses Phase 9 best practices:
- OHLC-based realistic stops
- Transaction costs scaling
- Systematic block testing
- Conservative estimates

KEY FINDING: STOPS WORK (unlike Phase 9!)
------------------------------------------
OHLC 2% stop: Sharpe 1.32, +155% return
No stop:      Sharpe 0.13, +18% return

WHY? Event signals create trends, not noise.
Stops protect from wrong direction calls.

BEST EVENTS:
------------
1. ISM_PMI:  Sharpe 9.87, Win 100% (6 trades) - INCREDIBLE
2. CPI:      Sharpe 1.02, Win 59.5% (42 trades) - GOOD
3. NFP:      Sharpe -0.19 (doesn't work)
4. Retail:   Sharpe -0.54 (avoid)

RECOMMENDATION:
---------------
Focus on CPI + ISM_PMI only
Use OHLC 2-3% stop
Hold for 5 days

Expected (conservative):
- Sharpe: 0.7-0.8
- Annual: 10-15%
- Trades: 12-15/year

FILES:
------
tmf_ohlc.csv           - TMF OHLC data (2018-2024)
economic_events.csv    - 266 events (manual collection)
merged_data.csv        - Events + prices aligned
block_a_baseline.csv   - Threshold tests
block_b_stops.csv      - Stop-loss experiments (BEST: OHLC 2%)
block_c_events.csv     - Event type analysis (BEST: ISM_PMI, CPI)
block_d_periods.csv    - Holding period (BEST: 5-day)
SUMMARY.txt            - Complete analysis

NEXT STEPS:
-----------
1. Validate ISM_PMI results (small sample)
2. Paper trade CPI-only strategy
3. Combine with Phase 9 for diversification

STATUS: Promising, needs validation
