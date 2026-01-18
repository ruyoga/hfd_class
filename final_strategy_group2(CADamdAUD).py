# final_strategy_group2(CADandAUD).py
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

# Contract Specs for CAD & AUD
SPECS = {
    'CAD': {'point_val': 100000.0, 'cost': 10.0},
    'AUD': {'point_val': 100000.0, 'cost': 10.0}
}

def mySR(x, scale):
    if np.nanstd(x) == 0:
        return 0
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

# Create an empty DataFrame to store summary for all quarters
summary_data2_all_quarters = pd.DataFrame()

# ============================================================
# STRATEGY PARAMETERS (Add your parameters here)
# ============================================================
# Example:
# WINDOW = ...
# THRESHOLD = ...
ANNUALIZATION = 252

for quarter in quarters:

    print(f'Processing quarter: {quarter}')

    try:
        # Load data (Group 2 uses data2_*.parquet)
        data2 = pd.read_parquet(f'data/data2_{quarter}.parquet')
    except FileNotFoundError:
        print(f"File not found: data/data2_{quarter}.parquet. Skipping.")
        continue

    # Set datetime index
    data2.set_index('datetime', inplace=True)

    # ============================================================
    # 1. Data Cleaning
    # ============================================================
    # Remove break time (17:00-18:00)
    data2 = data2.between_time("18:00", "17:00", inclusive="left")

    if data2.empty:
        continue

    # Ensure columns exist
    if 'CAD' not in data2.columns or 'AUD' not in data2.columns:
        print(f"CAD or AUD missing in {quarter}")
        continue

    # ============================================================
    # 2. Strategy Preparation (Indicators)
    # ============================================================
    # --------------------------------------------------------
    # [TODO] Calculate your indicators here (outside the loop for speed)
    # --------------------------------------------------------
    # Example:
    # prices_cad = data2['CAD'].values
    # prices_aud = data2['AUD'].values
    # ma_cad = data2['CAD'].rolling(window=...).mean().values
    
    
    
    # ============================================================
    # 3. Strategy Loop (Execution Logic)
    # ============================================================
    
    # Time filters for "Flat Zone" (Mandatory for Group 2)
    times = data2.index.time
    t_exit = pd.to_datetime("16:50").time()
    t_resume = pd.to_datetime("18:10").time()
    
    is_flat_zone = (times >= t_exit) & (times <= t_resume)

    # Prepare arrays
    n = len(data2)
    pos_cad = np.zeros(n, dtype=int)
    pos_aud = np.zeros(n, dtype=int)
    
    curr_pos_cad = 0
    curr_pos_aud = 0

    # --- MAIN LOOP ---
    for i in range(n):
        # 1. Mandatory Flat Rule (Priority #1) - DO NOT REMOVE
        if is_flat_zone[i]:
            curr_pos_cad = 0
            curr_pos_aud = 0
            pos_cad[i] = 0
            pos_aud[i] = 0
            continue

        # ----------------------------------------------------
        # [TODO] INSERT YOUR STRATEGY LOGIC HERE
        # ----------------------------------------------------
        # Access data: p_cad = prices_cad[i], etc.
        # Logic:
        # if condition:
        #     curr_pos_cad = 1
        #     curr_pos_aud = -1
        # else:
        #     curr_pos_cad = 0
        #     curr_pos_aud = 0
        # ----------------------------------------------------

        # Assign calculated position
        pos_cad[i] = curr_pos_cad
        pos_aud[i] = curr_pos_aud

    # Shift positions to simulate execution on next bar
    s_pos_cad = pd.Series(pos_cad, index=data2.index).shift(1).fillna(0).astype(int)
    s_pos_aud = pd.Series(pos_aud, index=data2.index).shift(1).fillna(0).astype(int)

    # Re-apply Flat Rule to Shifted Series (Safety)
    s_pos_cad[is_flat_zone] = 0
    s_pos_aud[is_flat_zone] = 0

    pos_final_cad = s_pos_cad.values
    pos_final_aud = s_pos_aud.values
                    
    # ============================================================
    # 4. PnL Calculation
    # ============================================================
    
    # Calculate gross pnl
    pnl_gross_cad = pos_final_cad * data2["CAD"].diff() * SPECS['CAD']['point_val']
    pnl_gross_aud = pos_final_aud * data2["AUD"].diff() * SPECS['AUD']['point_val']
    
    pnl_gross_cad = np.nan_to_num(pnl_gross_cad, nan=0.0)
    pnl_gross_aud = np.nan_to_num(pnl_gross_aud, nan=0.0)

    pnl_gross_total = pnl_gross_cad + pnl_gross_aud

    # Calculate number of transactions
    ntrans_cad = np.abs(np.diff(pos_final_cad, prepend=0))
    ntrans_aud = np.abs(np.diff(pos_final_aud, prepend=0))

    # Calculate net pnl
    cost_total = (ntrans_cad * SPECS['CAD']['cost']) + (ntrans_aud * SPECS['AUD']['cost'])
    pnl_net_total = pnl_gross_total - cost_total
    
    # Pct approximation (for Calmar)
    # Using sum of prices as denominator for combined exposure
    combined_price = data2["CAD"].shift(1) * SPECS['CAD']['point_val'] + \
                     data2["AUD"].shift(1) * SPECS['AUD']['point_val']
    
    # Handle zero division safely
    with np.errstate(divide='ignore', invalid='ignore'):
        pnl_gross_pct = np.where(combined_price != 0, pnl_gross_total / combined_price, 0)
        pnl_net_pct = np.where(combined_price != 0, pnl_net_total / combined_price, 0)
                    
    # Aggregate to daily data
    # Create Series with DateTimeIndex
    pnl_gross_s = pd.Series(pnl_gross_total, index=data2.index)
    pnl_gross_d = pnl_gross_s.resample('D').sum()
    
    # Filter out non-trading days
    valid_days = np.unique(data2.index.date)
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

    ntrans_total = ntrans_cad + ntrans_aud
    ntrans_s = pd.Series(ntrans_total, index=data2.index)
    ntrans_d = ntrans_s.resample('D').sum()
    ntrans_d = ntrans_d[np.isin(ntrans_d.index.date, valid_days)]

    # Calculate Sharpe Ratio and PnL
    gross_SR_mom = mySR(pnl_gross_d, scale=ANNUALIZATION)
    net_SR_mom = mySR(pnl_net_d, scale=ANNUALIZATION)
    gross_PnL_mom = pnl_gross_d.sum()
    net_PnL_mom = pnl_net_d.sum()
    
    try:
        gross_CR_mom = qs.stats.calmar(pnl_gross_pct_d.dropna())
        net_CR_mom = qs.stats.calmar(pnl_net_pct_d.dropna())
    except:
        gross_CR_mom = 0
        net_CR_mom = 0

    av_daily_ntrans = ntrans_d.mean()

    # Professor's statistic
    stat = (net_SR_mom - 0.5) * np.maximum(0, np.log(np.abs(net_PnL_mom/1000))) if net_PnL_mom != 0 else 0

    # Collect results
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

    # Append results
    summary_data2_all_quarters = pd.concat([summary_data2_all_quarters, summary], ignore_index=True)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(pnl_gross_d.fillna(0)), label='Gross PnL', color='blue')
    plt.plot(np.cumsum(pnl_net_d.fillna(0)), label='Net PnL', color='red')
    plt.title('Cumulative Gross and Net PnL (' + quarter + ') - CAD & AUD')
    plt.legend()
    plt.grid(axis='x')

    plt.savefig(f"data2_{quarter}(CADandAUD).png", dpi=300, bbox_inches="tight")
    plt.close()

    # Clean up memory
    del data2, pos_cad, pos_aud
    del s_pos_cad, s_pos_aud, pos_final_cad, pos_final_aud
    del summary

# Save summary
summary_data2_all_quarters.to_csv('summary_data2_all_quarters(CADandAUD).csv', index=False)
print("Processing complete. Summary saved.")