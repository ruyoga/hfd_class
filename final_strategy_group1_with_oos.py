# final_strategy_group1_clean.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # 処理順序（時系列順に並べ替えて処理するためのリスト）
    ALL_QUARTERS = sorted(IS_QUARTERS + OOS_QUARTERS)

    # Contract Specifications
    POINT_VAL_SP = 50.0
    POINT_VAL_NQ = 20.0
    COST_SP = 12.0
    COST_NQ = 12.0

    # Trading Hours
    TRADE_START_TIME = pd.to_datetime("10:00").time()
    EXIT_TIME = pd.to_datetime("15:40").time()

    # Strategy Parameters
    WINDOW = 45
    BETA_WINDOW = 600
    VOL_WINDOW = 180
    VOL_BASELINE_WINDOW = 600
    BASE_Z_ENTRY = 3.0
    STOP_LOSS_Z = 5.0
    Z_EXIT = 0.0
    ADAPTIVE_SENSITIVITY = 0.5
    MAX_Z_ENTRY = 3.5
    ANNUALIZATION = 252
    COOLDOWN_MINUTES = 30


def mySR(x, scale):
    if np.nanstd(x) == 0: return 0
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)


# ==========================================
# 2. Main Strategy Execution
# ==========================================
summary_list = []
all_daily_dfs = []

print("Starting Group 1 Analysis...")

for quarter in Config.ALL_QUARTERS:
    print(f'Processing {quarter}...', end=' ')

    try:
        data1 = pd.read_parquet(f'data/data1_{quarter}.parquet')
        print("Loaded.")
    except FileNotFoundError:
        print(f"Skipping (File not found).")
        continue

    data1.set_index('datetime', inplace=True)

    # --- Mandatory Data Cleaning ---
    data1.loc[data1.between_time("9:31", "9:40").index] = np.nan
    data1.loc[data1.between_time("15:51", "16:00").index] = np.nan

    # --- Indicators Calculation ---
    ln_sp = np.log(data1['SP'])
    ln_nq = np.log(data1['NQ'])

    nq_ret = ln_nq.diff()
    current_vol = nq_ret.rolling(window=Config.VOL_WINDOW).std()
    baseline_vol = nq_ret.rolling(window=Config.VOL_BASELINE_WINDOW).std()
    vol_ratio = (current_vol / baseline_vol.replace(0, np.nan)).fillna(1.0)

    cov = ln_nq.rolling(window=Config.BETA_WINDOW).cov(ln_sp)
    var = ln_sp.rolling(window=Config.BETA_WINDOW).var()
    beta_lag = (cov / var).shift(1).fillna(1.0)

    spread = ln_nq - (beta_lag * ln_sp)
    z_score = (spread - spread.rolling(window=Config.WINDOW).mean()) / spread.rolling(
        window=Config.WINDOW).std().replace(0, np.nan)

    # --- Iterative Backtest Loop ---
    times = data1.index.time
    datetimes = data1.index
    z_vals = z_score.values
    vol_vals = vol_ratio.values

    pos_strategy = np.zeros(len(data1))
    curr_pos = 0
    last_exit_time = None

    for i in range(len(data1)):
        if times[i] < Config.TRADE_START_TIME or times[i] >= Config.EXIT_TIME:
            if curr_pos != 0: last_exit_time = datetimes[i]
            curr_pos = 0
            pos_strategy[i] = 0
            continue

        if np.isnan(z_vals[i]) or np.isnan(vol_vals[i]):
            pos_strategy[i] = curr_pos
            continue

        is_cooldown = False
        if last_exit_time is not None:
            if (datetimes[i] - last_exit_time).total_seconds() / 60 < Config.COOLDOWN_MINUTES:
                is_cooldown = True

        extra_thresh = max(0, (vol_vals[i] - 1.0) * Config.ADAPTIVE_SENSITIVITY)
        current_entry_z = min(Config.BASE_Z_ENTRY + extra_thresh, Config.MAX_Z_ENTRY)
        z = z_vals[i]

        if curr_pos == 0:
            if not is_cooldown:
                if z > current_entry_z:
                    curr_pos = -1
                elif z < -current_entry_z:
                    curr_pos = 1
        elif curr_pos == 1:
            if z >= Config.Z_EXIT or z < -Config.STOP_LOSS_Z:
                curr_pos = 0
                last_exit_time = datetimes[i]
        elif curr_pos == -1:
            if z <= -Config.Z_EXIT or z > Config.STOP_LOSS_Z:
                curr_pos = 0
                last_exit_time = datetimes[i]
        pos_strategy[i] = curr_pos

    # --- PnL Calculation ---
    pos_held = pd.Series(pos_strategy, index=data1.index).shift(1).fillna(0)
    pos_held[times >= Config.EXIT_TIME] = 0

    pnl_nq = pos_held * data1["NQ"].diff() * Config.POINT_VAL_NQ
    pnl_sp = (pos_held * -1) * data1["SP"].diff() * Config.POINT_VAL_SP
    pnl_gross = (pnl_nq + pnl_sp).fillna(0)

    ntrans = np.abs(pos_held.diff().fillna(0))
    pnl_net = pnl_gross - (ntrans * (Config.COST_NQ + Config.COST_SP))

    # Aggregation for Table
    pnl_gross_d = pnl_gross.resample('D').sum()
    pnl_net_d = pnl_net.resample('D').sum()
    ntrans_d = ntrans.resample('D').sum()
    pnl_net_pct_d = pnl_net_d / data1["NQ"].resample('D').first()

    net_sr = mySR(pnl_net_d, scale=Config.ANNUALIZATION)
    try:
        net_cr = qs.stats.calmar(pnl_net_pct_d.dropna())
    except:
        net_cr = 0

    stat_val = (net_sr - 0.5) * np.maximum(0, np.log(np.abs(pnl_net_d.sum() / 1000)))

    # Collect Summary Data (指定形式の順序)
    row = {
        'quarter': quarter,
        'gross_SR': mySR(pnl_gross_d, scale=Config.ANNUALIZATION),
        'net_SR': net_sr,
        'gross_PnL': pnl_gross_d.sum(),
        'net_PnL': pnl_net_d.sum(),
        'net_CR': net_cr,
        'av_daily_ntrans': ntrans_d.mean(),
        'stat': stat_val
    }
    summary_list.append(row)

    # Collect Daily Data for Plotting
    daily_df = pd.DataFrame({
        'Net_PnL': pnl_net_d,
        'Gross_PnL': pnl_gross_d,
        'Quarter': quarter
    })
    # Remove days with 0 PnL (weekends/holidays) to make chart continuous
    daily_df = daily_df[daily_df['Gross_PnL'] != 0]
    all_daily_dfs.append(daily_df)

# ==========================================
# 3. Output Generation
# ==========================================

# 1. Summary Table CSV (Requested Format)
summary_df = pd.DataFrame(summary_list)
# カラム順序を固定
cols = ['quarter', 'gross_SR', 'net_SR', 'gross_PnL', 'net_PnL', 'net_CR', 'av_daily_ntrans', 'stat']
summary_df = summary_df[cols]
# CSV出力（ヘッダーなし、インデックスなし）
summary_df.to_csv('summary_data1_all_quarters.csv', index=False, header=False)
print("\nGenerated: summary_data1_all_quarters.csv")
print(summary_df)

# 2. Cumulative Equity Curve Plot
if all_daily_dfs:
    full_df = pd.concat(all_daily_dfs).sort_index()
    full_df['Cumulative_Net_PnL'] = full_df['Net_PnL'].cumsum()
    full_df['Cumulative_Gross_PnL'] = full_df['Gross_PnL'].cumsum()

    plt.figure(figsize=(15, 8))

    # Main Lines
    plt.plot(full_df.index, full_df['Cumulative_Net_PnL'], label='Net PnL', color='darkblue', linewidth=1.5)
    plt.plot(full_df.index, full_df['Cumulative_Gross_PnL'], label='Gross PnL', color='lightblue', linestyle='--',
             linewidth=1, alpha=0.7)

    # OOS Highlighting
    added_label = False
    y_min, y_max = plt.ylim()

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

    plt.title('Group 1: Cumulative Equity Curve (IS vs OOS)', fontsize=14)
    plt.ylabel('Cumulative PnL ($)')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig('G1_Final_Equity_Curve.png', dpi=300)
    plt.show()
    print("Generated: G1_Final_Equity_Curve.png")