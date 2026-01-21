# final_strategy_group2_clean.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # 講義ルールに基づく厳密な期間分割
    IS_QUARTERS = [
        '2023_Q1', '2023_Q3', '2023_Q4',
        '2024_Q2', '2024_Q4',
        '2025_Q1', '2025_Q2'
    ]

    OOS_QUARTERS = [
        '2023_Q2',
        '2024_Q1', '2024_Q3',
        '2025_Q3', '2025_Q4'
    ]

    # 時系列順に処理するためにソートして結合
    ALL_QUARTERS = sorted(IS_QUARTERS + OOS_QUARTERS)

    # Strategy Parameters (Group 2)
    MOMENTUM_WINDOW = 276  # approx 1 day
    REBALANCE_FREQ = 276  # approx 1 day
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


# ==========================================
# 2. Main Strategy Execution
# ==========================================
summary_list = []
all_daily_dfs = []

print("Starting Group 2 (XAU/XAG) Analysis with Strict IS/OOS Separation...")

for quarter in Config.ALL_QUARTERS:
    print(f'Processing {quarter}...', end=' ')

    try:
        # Load data
        data2 = pd.read_parquet(f'data/data2_{quarter}.parquet')
        print("Loaded.")
    except FileNotFoundError:
        print(f"Skipping (File not found).")
        continue

    # Lets set the datetime index
    data2.set_index('datetime', inplace=True)

    # --- Data Cleaning (Match Original Logic) ---
    # Original code removes 17:00-18:00 physically
    data2 = data2.between_time("18:00", "17:00", inclusive="left")

    if data2.empty:
        continue

    if 'XAU' not in data2.columns or 'XAG' not in data2.columns:
        continue

    # --- Strategy Calculation (Exact Loop Logic) ---
    mom_xau = np.log(data2["XAU"] / data2["XAU"].shift(Config.MOMENTUM_WINDOW)).values
    mom_xag = np.log(data2["XAG"] / data2["XAG"].shift(Config.MOMENTUM_WINDOW)).values

    # Time filters for "Flat Zone"
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

    # --- MAIN LOOP ---
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
        if (i % Config.REBALANCE_FREQ) == 0:
            # XAU Logic
            if m_xau > m_xag:
                curr_pos_xau = 1
            else:
                curr_pos_xau = -1

            # XAG Logic
            if m_xag > (m_xau + Config.XAG_THRESHOLD):
                curr_pos_xag = 1
            elif m_xag < (m_xau - Config.XAG_THRESHOLD):
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

    # --- PnL Calculation ---

    # Calculate gross pnl
    pnl_gross_xau = pos_mom_xau * data2["XAU"].diff() * Config.SPECS['XAU']['point_val']
    pnl_gross_xag = pos_mom_xag * data2["XAG"].diff() * Config.SPECS['XAG']['point_val']

    pnl_gross_total = np.nan_to_num(pnl_gross_xau, nan=0.0) + np.nan_to_num(pnl_gross_xag, nan=0.0)

    # Calculate number of transactions
    ntrans_xau = np.abs(np.diff(pos_mom_xau, prepend=0))
    ntrans_xag = np.abs(np.diff(pos_mom_xag, prepend=0))

    # Calculate net pnl
    cost_total = (ntrans_xau * Config.SPECS['XAU']['cost']) + (ntrans_xag * Config.SPECS['XAG']['cost'])
    pnl_net_total = pnl_gross_total - cost_total

    # Pct approximation for Calmar Ratio
    combined_price = data2["XAU"].shift(1) * Config.SPECS['XAU']['point_val'] + \
                     data2["XAG"].shift(1) * Config.SPECS['XAG']['point_val']

    pnl_net_pct = np.divide(pnl_net_total, combined_price, out=np.zeros_like(pnl_net_total), where=combined_price != 0)

    # --- Aggregation for Table ---
    # Create Series with DateTimeIndex
    pnl_gross_s = pd.Series(pnl_gross_total, index=data2.index)
    pnl_gross_d = pnl_gross_s.resample('D').sum()

    # Filter out non-trading days
    valid_days = np.unique(data2.index.date)
    pnl_gross_d = pnl_gross_d[np.isin(pnl_gross_d.index.date, valid_days)]

    pnl_net_s = pd.Series(pnl_net_total, index=data2.index)
    pnl_net_d = pnl_net_s.resample('D').sum()
    pnl_net_d = pnl_net_d[np.isin(pnl_net_d.index.date, valid_days)]

    pnl_net_pct_s = pd.Series(pnl_net_pct, index=data2.index)
    pnl_net_pct_d = pnl_net_pct_s.resample('D').sum()
    pnl_net_pct_d = pnl_net_pct_d[np.isin(pnl_net_pct_d.index.date, valid_days)]

    ntrans_total = ntrans_xau + ntrans_xag
    ntrans_s = pd.Series(ntrans_total, index=data2.index)
    ntrans_d = ntrans_s.resample('D').sum()
    ntrans_d = ntrans_d[np.isin(ntrans_d.index.date, valid_days)]

    # Metrics
    gross_SR_mom = mySR(pnl_gross_d, scale=Config.ANNUALIZATION)
    net_SR_mom = mySR(pnl_net_d, scale=Config.ANNUALIZATION)
    gross_PnL_mom = pnl_gross_d.sum()
    net_PnL_mom = pnl_net_d.sum()

    try:
        net_CR_mom = qs.stats.calmar(pnl_net_pct_d.dropna())
    except:
        net_CR_mom = 0

    av_daily_ntrans = ntrans_d.mean()
    stat = (net_SR_mom - 0.5) * np.maximum(0, np.log(np.abs(net_PnL_mom / 1000)))

    # Collect Summary Data (Row)
    row = {
        'quarter': quarter,
        'gross_SR': gross_SR_mom,
        'net_SR': net_SR_mom,
        'gross_PnL': gross_PnL_mom,
        'net_PnL': net_PnL_mom,
        'net_CR': net_CR_mom,
        'av_daily_ntrans': av_daily_ntrans,
        'stat': stat
    }
    summary_list.append(row)

    # Collect Daily Data for Plotting
    daily_df = pd.DataFrame({
        'Net_PnL': pnl_net_d,
        'Gross_PnL': pnl_gross_d,
        'Quarter': quarter
    })
    all_daily_dfs.append(daily_df)

    # clean up
    del data2, mom_xau, mom_xag, pos_xau, pos_xag

# ==========================================
# 3. Output Generation
# ==========================================

# 1. Summary Table CSV (Requested Format)
summary_df = pd.DataFrame(summary_list)
# Columns matching the G1 format
cols = ['quarter', 'gross_SR', 'net_SR', 'gross_PnL', 'net_PnL', 'net_CR', 'av_daily_ntrans', 'stat']
summary_df = summary_df[cols]

# CSV Output (No Header, No Index)
summary_df.to_csv('summary_data2_all_quarters(XAUandXAG).csv', index=False, header=False)
print("\nGenerated: summary_data2_all_quarters(XAUandXAG).csv")
print(summary_df)

# 2. Cumulative Equity Curve Plot
if all_daily_dfs:
    full_df = pd.concat(all_daily_dfs).sort_index()
    full_df['Cumulative_Net_PnL'] = full_df['Net_PnL'].cumsum()
    full_df['Cumulative_Gross_PnL'] = full_df['Gross_PnL'].cumsum()

    plt.figure(figsize=(15, 8))

    # Main Lines (Green for Group 2)
    plt.plot(full_df.index, full_df['Cumulative_Net_PnL'], label='Net PnL', color='darkgreen', linewidth=1.5)
    plt.plot(full_df.index, full_df['Cumulative_Gross_PnL'], label='Gross PnL', color='lightgreen', linestyle='--',
             linewidth=1, alpha=0.7)

    # OOS Highlighting
    added_label = False

    for quarter in Config.OOS_QUARTERS:
        q_data = full_df[full_df['Quarter'] == quarter]
        if not q_data.empty:
            start_date = q_data.index[0]
            end_date = q_data.index[-1]

            plt.axvspan(start_date, end_date, color='red', alpha=0.1,
                        label='Out-of-Sample' if not added_label else "")

            # Label on top
            plt.text(start_date, full_df['Cumulative_Net_PnL'].max(), quarter,
                     rotation=90, verticalalignment='top', fontsize=8, color='red', alpha=0.7)
            added_label = True

    plt.title('Group 2 (XAU/XAG): Cumulative Equity Curve (IS vs OOS)', fontsize=14)
    plt.ylabel('Cumulative PnL ($)')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig('G2_Final_Equity_Curve(XAUandXAG).png', dpi=300)
    plt.show()
    print("Generated: G2_Final_Equity_Curve(XAUandXAG).png")