# final_strategy_group2(XAUandXAG).py
# we load the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
import warnings

warnings.filterwarnings('ignore')

# Group 2 quarters
quarters = ['2023_Q1', '2023_Q3', '2023_Q4',
            '2024_Q2', '2024_Q4',
            '2025_Q1', '2025_Q2']

# Strategy Parameters (From Original Code)
MOMENTUM_WINDOW = 276      # approx 1 day
REBALANCE_FREQ = 276       # approx 1 day
XAG_THRESHOLD = 0.05
ANNUALIZATION = 252

# Contract Specs
SPECS = {
    'XAU': {'point_val': 100.0, 'cost': 15.0},
    'XAG': {'point_val': 5000.0, 'cost': 10.0}
}

def mySR(x, scale):
    if np.nanstd(x) == 0:
        return 0
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

# Create an empty DataFrame to store summary for all quarters
summary_data2_all_quarters = pd.DataFrame()

for quarter in quarters:

    print(f'Processing quarter: {quarter}')

    try:
        # Load data
        data2 = pd.read_parquet(f'data/data2_{quarter}.parquet')
    except FileNotFoundError:
        print(f"File not found: data/data2_{quarter}.parquet. Skipping.")
        continue

    # Lets set the datetime index
    data2.set_index('datetime', inplace = True)

    # ============================================================
    # 1. Data Cleaning (Match Original Logic)
    # ============================================================
    # Original code removes 17:00-18:00 physically
    data2 = data2.between_time("18:00", "17:00", inclusive="left")

    if data2.empty:
        continue

    # ============================================================
    # 2. Strategy Calculation (Exact Loop Logic)
    # ============================================================
    
    if 'XAU' not in data2.columns or 'XAG' not in data2.columns:
        continue

    mom_xau = np.log(data2["XAU"] / data2["XAU"].shift(MOMENTUM_WINDOW)).values
    mom_xag = np.log(data2["XAG"] / data2["XAG"].shift(MOMENTUM_WINDOW)).values

    # Time filters for "Flat Zone" (Original Logic)
    times = data2.index.time
    t_exit = pd.to_datetime("16:50").time()
    t_resume = pd.to_datetime("18:10").time()
    
    is_flat_zone = (times >= t_exit) & (times <= t_resume)

    # Prepare arrays for loop
    n = len(data2)
    pos_xau = np.zeros(n, dtype=int)
    pos_xag = np.zeros(n, dtype=int)
    
    curr_pos_xau = 0
    curr_pos_xag = 0

    # --- MAIN LOOP (Identical to Original) ---
    for i in range(n):
        # 1. Mandatory Flat Rule (Priority #1)
        if is_flat_zone[i]:
            curr_pos_xau = 0
            curr_pos_xag = 0
            pos_xau[i] = 0
            pos_xag[i] = 0
            continue

        # 2. Check Data Availability
        m_xau = mom_xau[i]
        m_xag = mom_xag[i]

        if np.isnan(m_xau) or np.isnan(m_xag):
            # Hold current
            pos_xau[i] = curr_pos_xau
            pos_xag[i] = curr_pos_xag
            continue

        # 3. Rebalance Logic
        if (i % REBALANCE_FREQ) == 0:
            # XAU Logic
            if m_xau > m_xag:
                curr_pos_xau = 1
            else:
                curr_pos_xau = -1

            # XAG Logic
            if m_xag > (m_xau + XAG_THRESHOLD):
                curr_pos_xag = 1
            elif m_xag < (m_xau - XAG_THRESHOLD):
                curr_pos_xag = -1
            else:
                curr_pos_xag = 0

        # Assign
        pos_xau[i] = curr_pos_xau
        pos_xag[i] = curr_pos_xag

    # Shift positions to simulate execution on next bar
    s_pos_xau = pd.Series(pos_xau, index=data2.index).shift(1).fillna(0).astype(int)
    s_pos_xag = pd.Series(pos_xag, index=data2.index).shift(1).fillna(0).astype(int)

    # Re-apply Flat Rule to Shifted Series
    s_pos_xau[is_flat_zone] = 0
    s_pos_xag[is_flat_zone] = 0

    pos_mom_xau = s_pos_xau.values
    pos_mom_xag = s_pos_xag.values
                    
    # ============================================================
    # 3. PnL Calculation
    # ============================================================
    
    # Calculate gross pnl
    pnl_gross_xau = pos_mom_xau * data2["XAU"].diff() * SPECS['XAU']['point_val']
    pnl_gross_xag = pos_mom_xag * data2["XAG"].diff() * SPECS['XAG']['point_val']
    
    pnl_gross_xau = np.nan_to_num(pnl_gross_xau, nan=0.0)
    pnl_gross_xag = np.nan_to_num(pnl_gross_xag, nan=0.0)

    pnl_gross_total = pnl_gross_xau + pnl_gross_xag

    # Calculate number of transactions
    ntrans_xau = np.abs(np.diff(pos_mom_xau, prepend = 0))
    ntrans_xag = np.abs(np.diff(pos_mom_xag, prepend = 0))

    # Calculate net pnl
    cost_total = (ntrans_xau * SPECS['XAU']['cost']) + (ntrans_xag * SPECS['XAG']['cost'])
    pnl_net_total = pnl_gross_total - cost_total
    
    # Pct approximation (for Calmar)
    combined_price = data2["XAU"].shift(1) * SPECS['XAU']['point_val'] + \
                     data2["XAG"].shift(1) * SPECS['XAG']['point_val']
    
    # Avoid division by zero
    pnl_gross_pct = np.divide(pnl_gross_total, combined_price, out=np.zeros_like(pnl_gross_total), where=combined_price!=0)
    pnl_net_pct = np.divide(pnl_net_total, combined_price, out=np.zeros_like(pnl_net_total), where=combined_price!=0)
                    
    # Aggregate to daily data
    # Create Series with DateTimeIndex
    pnl_gross_s = pd.Series(pnl_gross_total, index=data2.index)
    pnl_gross_d = pnl_gross_s.resample('D').sum()
    
    # Filter out non-trading days
    valid_days = np.unique(data2.index.date)
    
    # FIX: Use np.isin because index.date is a numpy array
    pnl_gross_d = pnl_gross_d[np.isin(pnl_gross_d.index.date, valid_days)]

    pnl_net_s = pd.Series(pnl_net_total, index=data2.index)
    pnl_net_d = pnl_net_s.resample('D').sum()
    pnl_net_d = pnl_net_d[np.isin(pnl_net_d.index.date, valid_days)]

    pnl_gross_pct_s = pd.Series(pnl_gross_pct, index=data2.index)
    pnl_gross_pct_d = pnl_gross_pct_s.resample('D').sum()
    pnl_gross_pct_d = pnl_gross_pct_d[np.isin(pnl_gross_pct_d.index.date, valid_days)]

    pnl_net_pct_s = pd.Series(pnl_net_pct, index=data2.index)
    pnl_net_pct_d = pnl_net_pct_s.resample('D').sum()
    pnl_net_pct_d = pnl_net_pct_d[np.isin(pnl_net_pct_d.index.date, valid_days)]

    ntrans_total = ntrans_xau + ntrans_xag
    ntrans_s = pd.Series(ntrans_total, index=data2.index)
    ntrans_d = ntrans_s.resample('D').sum()
    ntrans_d = ntrans_d[np.isin(ntrans_d.index.date, valid_days)]

    # Calculate Sharpe Ratio and PnL
    gross_SR_mom = mySR(pnl_gross_d, scale=252)
    net_SR_mom = mySR(pnl_net_d, scale=252)
    gross_PnL_mom = pnl_gross_d.sum()
    net_PnL_mom = pnl_net_d.sum()
    gross_CR_mom = qs.stats.calmar(pnl_gross_pct_d.dropna())
    net_CR_mom = qs.stats.calmar(pnl_net_pct_d.dropna())

    av_daily_ntrans = ntrans_d.mean()

    stat = (net_SR_mom - 0.5) * np.maximum(0, np.log(np.abs(net_PnL_mom/1000)))

    # Collect necessary results into one object
    summary = pd.DataFrame({'quarter': quarter,
                            'gross_SR': gross_SR_mom,
                            'net_SR': net_SR_mom,
                            'gross_PnL': gross_PnL_mom,
                            'net_PnL': net_PnL_mom,
                            'gross_CR': gross_CR_mom,
                            'net_CR': net_CR_mom,
                            'av_daily_ntrans': av_daily_ntrans,
                            'stat': stat
                        }, index=[0])

    # Append results to the summary
    summary_data2_all_quarters = pd.concat([summary_data2_all_quarters, summary], ignore_index=True)

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(pnl_gross_d.fillna(0)), label = 'Gross PnL', color='blue')
    plt.plot(np.cumsum(pnl_net_d.fillna(0)), label = 'Net PnL', color='red')
    plt.title('Cumulative Gross and Net PnL (' + quarter + ')')
    plt.legend()
    plt.grid(axis='x')

    # UPDATED: Added (XAUandXAG) to PNG filename
    plt.savefig(f"data2_{quarter}(XAUandXAG).png", dpi = 300, bbox_inches = "tight")
    plt.close()

    # clean up
    del data2, mom_xau, mom_xag, pos_xau, pos_xag
    del s_pos_xau, s_pos_xag, pos_mom_xau, pos_mom_xag
    del summary

# save the summary for all quarters to a csv file
# UPDATED: Added (XAUandXAG) to CSV filename
summary_data2_all_quarters.to_csv('summary_data2_all_quarters(XAUandXAG).csv', index=False)