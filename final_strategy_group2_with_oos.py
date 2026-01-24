import pandas as pd
import numpy as np
import quantstats as qs
import warnings
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Config:
    QUARTERS = [
        '2023_Q1', '2023_Q2', '2023_Q3', '2023_Q4',
        '2024_Q1', '2024_Q2', '2024_Q3', '2024_Q4',
        '2025_Q1', '2025_Q2', '2025_Q3', '2025_Q4'
    ]
    OOS_QUARTERS = ['2023_Q2', '2024_Q1', '2024_Q3', '2025_Q3', '2025_Q4']
    MOMENTUM_WINDOW = 276
    REBALANCE_FREQ = 276
    XAG_THRESHOLD = 0.05
    ANNUALIZATION = 252
    SPECS = {
        'XAU': {'point_val': 100.0, 'cost': 15.0},
        'XAG': {'point_val': 5000.0, 'cost': 10.0}
    }


def mySR(x, scale):
    if np.nanstd(x) == 0: return 0
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)


summary_list = []
all_daily_dfs = []

print("Starting Group 2 (XAU/XAG) Analysis...")

for quarter in Config.QUARTERS:
    print(f'Processing quarter: {quarter}')
    try:
        data2 = pd.read_parquet(f'data/data2_{quarter}.parquet')
    except FileNotFoundError:
        print(f"File not found: data/data2_{quarter}.parquet. Skipping.")
        continue

    data2.set_index('datetime', inplace=True)
    data2 = data2.between_time("18:00", "17:00", inclusive="left")
    if data2.empty or 'XAU' not in data2.columns or 'XAG' not in data2.columns:
        continue

    # Strategy
    mom_xau = np.log(data2["XAU"] / data2["XAU"].shift(Config.MOMENTUM_WINDOW)).values
    mom_xag = np.log(data2["XAG"] / data2["XAG"].shift(Config.MOMENTUM_WINDOW)).values

    times = data2.index.time
    t_exit = pd.to_datetime("16:50").time()
    t_resume = pd.to_datetime("18:10").time()
    is_flat_zone = (times >= t_exit) & (times <= t_resume)

    n = len(data2)
    pos_xau = np.zeros(n, dtype=int)
    pos_xag = np.zeros(n, dtype=int)
    curr_pos_xau = 0
    curr_pos_xag = 0

    for i in range(n):
        if is_flat_zone[i]:
            curr_pos_xau, curr_pos_xag = 0, 0
            pos_xau[i], pos_xag[i] = 0, 0
            continue

        m_xau, m_xag = mom_xau[i], mom_xag[i]
        if np.isnan(m_xau) or np.isnan(m_xag):
            pos_xau[i], pos_xag[i] = curr_pos_xau, curr_pos_xag
            continue

        if (i % Config.REBALANCE_FREQ) == 0:
            curr_pos_xau = 1 if m_xau > m_xag else -1
            if m_xag > (m_xau + Config.XAG_THRESHOLD):
                curr_pos_xag = 1
            elif m_xag < (m_xau - Config.XAG_THRESHOLD):
                curr_pos_xag = -1
            else:
                curr_pos_xag = 0

        pos_xau[i], pos_xag[i] = curr_pos_xau, curr_pos_xag

    s_pos_xau = pd.Series(pos_xau, index=data2.index).shift(1).fillna(0).astype(int)
    s_pos_xag = pd.Series(pos_xag, index=data2.index).shift(1).fillna(0).astype(int)
    s_pos_xau[is_flat_zone] = 0
    s_pos_xag[is_flat_zone] = 0

    pos_mom_xau = s_pos_xau.values
    pos_mom_xag = s_pos_xag.values

    # PnL
    pnl_gross_xau = pos_mom_xau * data2["XAU"].diff() * Config.SPECS['XAU']['point_val']
    pnl_gross_xag = pos_mom_xag * data2["XAG"].diff() * Config.SPECS['XAG']['point_val']
    pnl_gross_total = np.nan_to_num(pnl_gross_xau, nan=0.0) + np.nan_to_num(pnl_gross_xag, nan=0.0)

    ntrans_xau = np.abs(np.diff(pos_mom_xau, prepend=0))
    ntrans_xag = np.abs(np.diff(pos_mom_xag, prepend=0))
    cost_total = (ntrans_xau * Config.SPECS['XAU']['cost']) + (ntrans_xag * Config.SPECS['XAG']['cost'])
    pnl_net_total = pnl_gross_total - cost_total

    # Pct approximation
    combined_price = data2["XAU"].shift(1) * Config.SPECS['XAU']['point_val'] + \
                     data2["XAG"].shift(1) * Config.SPECS['XAG']['point_val']

    pnl_net_pct = np.divide(pnl_net_total, combined_price, out=np.zeros_like(pnl_net_total), where=combined_price != 0)
    pnl_gross_pct = np.divide(pnl_gross_total, combined_price, out=np.zeros_like(pnl_gross_total),
                              where=combined_price != 0)

    # Aggregation
    pnl_gross_d = pd.Series(pnl_gross_total, index=data2.index).resample('D').sum()
    pnl_net_d = pd.Series(pnl_net_total, index=data2.index).resample('D').sum()

    # Filter valid days
    valid_days = np.unique(data2.index.date)
    pnl_gross_d = pnl_gross_d[np.isin(pnl_gross_d.index.date, valid_days)]
    pnl_net_d = pnl_net_d[np.isin(pnl_net_d.index.date, valid_days)]

    # Resample Pct
    pnl_net_pct_d = pd.Series(pnl_net_pct, index=data2.index).resample('D').sum()
    pnl_net_pct_d = pnl_net_pct_d[np.isin(pnl_net_pct_d.index.date, valid_days)]

    pnl_gross_pct_d = pd.Series(pnl_gross_pct, index=data2.index).resample('D').sum()
    pnl_gross_pct_d = pnl_gross_pct_d[np.isin(pnl_gross_pct_d.index.date, valid_days)]

    ntrans_d = pd.Series(ntrans_xau + ntrans_xag, index=data2.index).resample('D').sum()
    ntrans_d = ntrans_d[np.isin(ntrans_d.index.date, valid_days)]

    # Metrics
    gross_SR_mom = mySR(pnl_gross_d, scale=Config.ANNUALIZATION)
    net_SR_mom = mySR(pnl_net_d, scale=Config.ANNUALIZATION)

    try:
        net_CR_mom = qs.stats.calmar(pnl_net_pct_d.dropna())
    except:
        net_CR_mom = 0

    try:
        gross_CR_mom = qs.stats.calmar(pnl_gross_pct_d.dropna())  # Added
    except:
        gross_CR_mom = 0

    stat = (net_SR_mom - 0.5) * np.maximum(0, np.log(np.abs(pnl_net_d.sum() / 1000)))

    row = {
        'quarter': quarter,
        'gross_SR': gross_SR_mom,
        'net_SR': net_SR_mom,
        'gross_PnL': pnl_gross_d.sum(),
        'net_PnL': pnl_net_d.sum(),
        'gross_CR': gross_CR_mom,  # Added
        'net_CR': net_CR_mom,
        'av_daily_ntrans': ntrans_d.mean(),
        'stat': stat
    }
    summary_list.append(row)
    all_daily_dfs.append(pd.DataFrame({'Net_PnL': pnl_net_d, 'Gross_PnL': pnl_gross_d, 'Quarter': quarter}))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(pnl_gross_d.fillna(0)), label='Gross PnL', color='blue')
    plt.plot(np.cumsum(pnl_net_d.fillna(0)), label='Net PnL', color='red')
    plt.title(f'Cumulative Gross and Net PnL ({quarter})')
    plt.legend()
    plt.grid(axis='x')
    plt.savefig(f"data2_{quarter}(XAUandXAG).png", dpi=300, bbox_inches="tight")
    plt.close()

# Summary Save (9 Columns)
summary_df = pd.DataFrame(summary_list)
# Included 'gross_CR'
cols = ['quarter', 'gross_SR', 'net_SR', 'gross_PnL', 'net_PnL', 'gross_CR', 'net_CR', 'av_daily_ntrans', 'stat']
summary_df = summary_df[cols]
summary_df.to_csv('summary_data2_all_quarters(XAUandXAG).csv', index=False, header=False)

print("\nGenerated: summary_data2_all_quarters(XAUandXAG).csv")

# Cumulative Plot
if all_daily_dfs:
    full_df = pd.concat(all_daily_dfs).sort_index()
    full_df['Cumulative_Net_PnL'] = full_df['Net_PnL'].cumsum()
    full_df['Cumulative_Gross_PnL'] = full_df['Gross_PnL'].cumsum()

    plt.figure(figsize=(15, 8))
    plt.plot(full_df.index, full_df['Cumulative_Net_PnL'], label='Net PnL', color='darkgreen', linewidth=1.5)
    plt.plot(full_df.index, full_df['Cumulative_Gross_PnL'], label='Gross PnL', color='lightgreen', linestyle='--',
             linewidth=1, alpha=0.7)

    added_label = False
    for quarter in Config.OOS_QUARTERS:
        q_data = full_df[full_df['Quarter'] == quarter]
        if not q_data.empty:
            start_date = q_data.index[0]
            end_date = q_data.index[-1]
            plt.axvspan(start_date, end_date, color='red', alpha=0.1, label='Out-of-Sample' if not added_label else "")
            plt.text(start_date, full_df['Cumulative_Net_PnL'].max(), quarter, rotation=90, verticalalignment='top',
                     fontsize=8, color='red', alpha=0.7)
            added_label = True

    plt.title('Group 2 (XAU/XAG): Cumulative Equity Curve (IS vs OOS)', fontsize=14)
    plt.ylabel('Cumulative PnL ($)')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('G2_Final_Equity_Curve(XAUandXAG).png', dpi=300)
    print("Generated: G2_Final_Equity_Curve(XAUandXAG).png")