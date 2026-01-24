# final_strategy_group1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration (Final Validated Logic)
# ==========================================
class Config:
    # Submission Quarter List
    QUARTERS = ['2023_Q1', '2023_Q3', '2023_Q4', '2024_Q2', '2024_Q4', '2025_Q1', '2025_Q2']
    
    # Contract Specifications
    POINT_VAL_SP = 50.0
    POINT_VAL_NQ = 20.0
    COST_SP = 12.0
    COST_NQ = 12.0

    # Trading Hours Rules (Start 10:00, Exit 15:40)
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
summary_data1_all_quarters = pd.DataFrame()

print("Starting Group 1 Strategy Backtest with Final Logic...")

# Modified to refer to QUARTERS in Config class
for quarter in Config.QUARTERS:
    print(f'Processing quarter: {quarter}')

    try:
        data1 = pd.read_parquet(f'data/data1_{quarter}.parquet')
    except FileNotFoundError:
        print(f"File not found: data/data1_{quarter}.parquet. Skipping.")
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
    z_score = (spread - spread.rolling(window=Config.WINDOW).mean()) / spread.rolling(window=Config.WINDOW).std().replace(0, np.nan)
    
    # --- Iterative Backtest Loop ---
    times = data1.index.time
    datetimes = data1.index
    z_vals = z_score.values
    vol_vals = vol_ratio.values
    
    pos_strategy = np.zeros(len(data1))
    curr_pos = 0
    last_exit_time = None
    
    for i in range(len(data1)):
        # Trading Hours & Forced Exit
        if times[i] < Config.TRADE_START_TIME or times[i] >= Config.EXIT_TIME:
            if curr_pos != 0:
                last_exit_time = datetimes[i]
            curr_pos = 0
            pos_strategy[i] = 0
            continue
            
        if np.isnan(z_vals[i]) or np.isnan(vol_vals[i]):
            pos_strategy[i] = curr_pos
            continue
            
        # Cooldown check
        is_cooldown = False
        if last_exit_time is not None:
            if (datetimes[i] - last_exit_time).total_seconds() / 60 < Config.COOLDOWN_MINUTES:
                is_cooldown = True

        # Adaptive Threshold
        extra_thresh = max(0, (vol_vals[i] - 1.0) * Config.ADAPTIVE_SENSITIVITY)
        current_entry_z = min(Config.BASE_Z_ENTRY + extra_thresh, Config.MAX_Z_ENTRY)
        z = z_vals[i]
        
        if curr_pos == 0:
            if not is_cooldown:
                if z > current_entry_z: curr_pos = -1
                elif z < -current_entry_z: curr_pos = 1
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
    
    # Aggregation
    pnl_gross_d = pnl_gross.resample('D').sum()
    pnl_net_d = pnl_net.resample('D').sum()
    ntrans_d = ntrans.resample('D').sum()
    
    pnl_net_pct_d = pnl_net_d / data1["NQ"].resample('D').first()

    # Metrics
    net_sr = mySR(pnl_net_d, scale=Config.ANNUALIZATION)
    try:
        net_cr = qs.stats.calmar(pnl_net_pct_d.dropna())
    except:
        net_cr = 0

    stat_val = (net_sr - 0.5) * np.maximum(0, np.log(np.abs(pnl_net_d.sum()/1000)))

    # --- Collect Summary ---
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
    summary_data1_all_quarters = pd.concat([summary_data1_all_quarters, pd.DataFrame([row])], ignore_index=True)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_gross_d.index, pnl_gross_d.cumsum(), label='Gross PnL', color='blue')
    plt.plot(pnl_net_d.index, pnl_net_d.cumsum(), label='Net PnL', color='red')
    plt.title(f'Cumulative P&L ({quarter})')
    plt.legend()
    plt.savefig(f"data1_{quarter}.png", dpi=300, bbox_inches="tight")
    plt.close()

# --- Final Save ---
summary_data1_all_quarters.to_csv('summary_data1_all_quarters.csv', index=False)
print(summary_data1_all_quarters)