import os
import pandas as pd
import numpy as np
import sys

# Define paths
DATA_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\data"

# Add the paths to python path
SCRIPTS_DIR = r"c:\Users\liuqi\quant_system_v2\etf-valuation-strategy\scripts"
sys.path.append(SCRIPTS_DIR)

# Load daily stock PNL
df_stock = pd.read_csv(os.path.join(DATA_DIR, 'daily_stock_pnl.csv'))
df_stock['trade_date'] = pd.to_datetime(df_stock['trade_date'].astype(str))

import step11_risk_parity_strategy as rp
df_unified = rp.load_data_8assets(ma_window=200, val_window=1400, vol_lookback=60)

# Merge daily stock strategy PNL
df_all = pd.merge(df_unified, df_stock, on='trade_date', how='inner')
print(f"Merged dataset shape: {df_all.shape}, Dates: {df_all['trade_date'].min().strftime('%Y-%m-%d')} to {df_all['trade_date'].max().strftime('%Y-%m-%d')}")

# 1. Compute correlations
assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond']
data_rets = {}
for a in assets:
    data_rets[a] = df_all[f'ret_{a}']
data_rets['daily_stock_opt'] = df_all['pnl_opt']
data_rets['daily_stock_base'] = df_all['pnl_base']

df_rets = pd.DataFrame(data_rets)
corr = df_rets.corr()
print("\nCorrelation matrix including Daily Stock Strategy:")
print(corr[['daily_stock_opt', 'daily_stock_base']].round(3))

# 2. Run Comparison Backtests (2022-2026)
print("\nRunning backtests over the 2022-2026 period...")

# Backtest 1: Baseline 8-asset Risk Parity Strategy (vol_target = 0.10, val_tilt = 0.0, strike_ratio = 0.95)
print("1. Running Baseline 8-asset Risk Parity...")
df_nav_base, _, _ = rp.run_backtest_risk_parity(
    df_all, vol_target=0.10, val_tilt=0.0, strike_ratio=0.95, buy_put=True
)
metrics_base = rp.compute_metrics(df_nav_base['nav'])

# Backtest 2: Replacement Portfolio
# We replace hs300, zz500, chinext, div returns with daily_stock_opt in the backtest
# In this backtest, the equity return is determined by the active Daily Stock Strategy
def run_backtest_replaced_equity(df_period, vol_target=0.10, mult=1.0):
    df_period = df_period.copy().reset_index(drop=True)
    
    # We replace stock returns in df_period with pnl_opt * mult
    # and set their volatilities equal to pnl_opt's volatility
    # This represents a portfolio containing: active stock picker, gold, nasdaq, bond, cbond
    active_assets = ['active_stock', 'gold', 'nasdaq', 'bond', 'cbond']
    
    # Calculate rolling volatility for the active stock picker
    df_period['ret_active_stock'] = df_period['pnl_opt'] * mult
    df_period['vol_active_stock'] = df_period['ret_active_stock'].rolling(window=60, min_periods=20).std().bfill()
    
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    ret_cols = ['ret_active_stock', 'ret_gold', 'ret_nasdaq', 'ret_bond', 'ret_cbond']
    df_returns = df_period[ret_cols].copy()
    df_returns.columns = active_assets
    
    val = {a: 0.0 for a in active_assets}
    val_cash = 1000000.0
    nav_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            for a in active_assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + 0.02 / 242.0)
            
        nav = sum(val.values()) + val_cash
        
        # Risk Parity weights
        vols = np.array([row[f'vol_{a}'] for a in active_assets])
        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        w_rp = inv_vols / inv_vols.sum()
        w_target = {a: w_rp[i] for i, a in enumerate(active_assets)}
        
        # Vol target layer
        if idx >= 60:
            cov_matrix = df_returns.iloc[idx - 60 + 1:idx + 1].cov().values
        else:
            cov_matrix = df_returns.iloc[0:idx + 1].cov().values if idx > 5 else np.eye(len(active_assets)) * (0.01 / 252.0)
            
        w_vector = np.array([w_target[a] for a in active_assets])
        port_variance = np.dot(w_vector, np.dot(cov_matrix, w_vector))
        port_vol = np.sqrt(port_variance * 252.0)
        
        sf = min(1.0, vol_target / max(port_vol, 1e-6))
        w_target_final = {a: w_target[a] * sf for a in active_assets}
        w_target_final['cash'] = 1.0 - sf
        
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in active_assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        devs = [abs(w_curr[a] - w_target_final[a]) for a in active_assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_check_dates) or (max_dev > 0.05) or (idx == 0)
        
        if is_rebal_day:
            val_target = {a: nav * w_target_final[a] for a in active_assets}
            val_target_cash = nav * w_target_final['cash']
            trade_vol = sum(abs(val_target[a] - val[a]) for a in active_assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * 0.0005
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in active_assets:
                val[a] = nav * w_target_final[a]
                
        nav_history.append({'trade_date': dt, 'nav': nav})
        
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    return df_nav

print("2. Running Replacement Active Equity Portfolio (Unscaled)...")
df_nav_replace_1 = run_backtest_replaced_equity(df_all, vol_target=0.10, mult=1.0)
metrics_replace_1 = rp.compute_metrics(df_nav_replace_1['nav'])

# 3. 9-Asset Satellite Portfolio Backtest
def run_backtest_9assets(
    df_period,
    vol_target=0.10,
    val_tilt=0.0,
    q_threshold=0.20,
    strike_ratio=0.95,
    trend_reduce_factor=0.5,
    vol_lookback=60,
    dev_threshold=0.05,
    initial_capital=1000000.0,
    rf_rate=0.02,
    buy_put=True,
    mult=1.0
):
    if len(df_period) == 0:
        return None
        
    df_period = df_period.copy().reset_index(drop=True)
    
    # 9 Assets list
    assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond', 'active_stock']
    equity_assets = ['hs300', 'zz500', 'chinext', 'div']
    
    # Calculate rolling volatility for the active stock picker (scaled by mult)
    df_period['ret_active_stock'] = df_period['pnl_opt'] * mult
    df_period['vol_active_stock'] = df_period['ret_active_stock'].rolling(window=vol_lookback, min_periods=20).std().bfill()
    
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebalance_check_dates = set(df_period.groupby('year_week')['trade_date'].first())
    
    ret_cols = [f'ret_{a}' for a in assets]
    df_returns = df_period[ret_cols].copy()
    df_returns.columns = assets
    
    val = {a: 0.0 for a in assets}
    val_cash = initial_capital
    options_held = []
    nav_history = []
    
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        
        if idx > 0:
            for a in assets:
                val[a] *= (1.0 + row[f'ret_{a}'])
            val_cash *= (1.0 + rf_rate / 242.0)
            
        payoff_today = 0.0
        active_options = []
        for opt in options_held:
            if idx >= opt['expiry_idx']:
                asset = opt['asset']
                close_price = row[f'close_{asset}']
                purchase_price = opt['purchase_price']
                strike_price = opt['strike_price']
                payoff = opt['purchase_val'] * max(strike_price / purchase_price - close_price / purchase_price, 0.0)
                payoff_today += payoff
            else:
                active_options.append(opt)
        options_held = active_options
        val_cash += payoff_today
        
        nav = sum(val.values()) + val_cash
        
        # Risk Parity weights
        vols = np.array([row[f'vol_{a}'] for a in assets])
        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        w_rp = inv_vols / inv_vols.sum()
        w_target = {a: w_rp[i] for i, a in enumerate(assets)}
        
        # Valuation Timing (only A-shares)
        for a in equity_assets:
            val_q = row[f'val_q_{a}']
            if not pd.isna(val_q):
                w_target[a] *= (1.0 - val_tilt * (val_q - 0.5))
                
        # Trend filter (skip for active_stock)
        for a in assets:
            if a == 'active_stock':
                continue
            close_px = row[f'close_{a}']
            ma_px = row[f'ma_{a}']
            trend_up = close_px >= ma_px if not pd.isna(ma_px) else True
            
            if not trend_up:
                if a in equity_assets:
                    val_q = row[f'val_q_{a}']
                    if pd.isna(val_q) or val_q > q_threshold:
                        w_target[a] *= trend_reduce_factor
                else:
                    if a in ['gold', 'nasdaq']:
                        w_target[a] *= trend_reduce_factor
                        
        # Normalize weights
        w_sum = sum(w_target.values())
        for a in assets:
            w_target[a] /= w_sum
            
        # Vol target layer
        if idx >= vol_lookback:
            cov_matrix = df_returns.iloc[idx - vol_lookback + 1:idx + 1].cov().values
        else:
            cov_matrix = df_returns.iloc[0:idx + 1].cov().values if idx > 5 else np.eye(len(assets)) * (0.01 / 252.0)
            
        w_vector = np.array([w_target[a] for a in assets])
        port_variance = np.dot(w_vector, np.dot(cov_matrix, w_vector))
        port_vol = np.sqrt(port_variance * 252.0)
        
        sf = min(1.0, vol_target / max(port_vol, 1e-6))
        w_target_final = {a: w_target[a] * sf for a in assets}
        w_target_final['cash'] = 1.0 - sf
        
        w_curr = {a: val[a] / nav if nav > 0 else 0.0 for a in assets}
        w_curr['cash'] = val_cash / nav if nav > 0 else 0.0
        devs = [abs(w_curr[a] - w_target_final[a]) for a in assets] + [abs(w_curr['cash'] - w_target_final['cash'])]
        max_dev = max(devs)
        
        is_rebal_day = (dt in rebalance_check_dates) or (max_dev > dev_threshold) or (idx == 0)
        
        if is_rebal_day:
            val_target = {a: nav * w_target_final[a] for a in assets}
            val_target_cash = nav * w_target_final['cash']
            trade_vol = sum(abs(val_target[a] - val[a]) for a in assets) + abs(val_target_cash - val_cash)
            cost = trade_vol * 0.0005
            nav -= cost
            val_cash = nav * w_target_final['cash']
            for a in assets:
                val[a] = nav * w_target_final[a]
                
        # Put options buying
        if buy_put and (idx % 20 == 0):
            T_years = 20.0 / 252.0
            for a in equity_assets:
                val_q = row[f'val_q_{a}']
                val_holding = val[a]
                if (not pd.isna(val_q)) and (val_q > 0.70) and (val_holding > 0.0):
                    S0 = row[f'close_{a}']
                    K = S0 * strike_ratio
                    r = (row[f'yield_10y_{a}'] / 100.0) if not pd.isna(row[f'yield_10y_{a}']) else 0.025
                    current_iv = (row['qvix'] / 100.0) if 'qvix' in row else row[f'vol_{a}'] * np.sqrt(252.0)
                    if pd.isna(current_iv) or current_iv <= 0:
                        current_iv = 0.20
                    put_price_per_share = rp.bs_put_price(S0, K, T_years, r, current_iv)
                    pct_premium = put_price_per_share / S0
                    opt_premium_cost = val_holding * pct_premium
                    val_cash -= opt_premium_cost
                    nav -= opt_premium_cost
                    options_held.append({
                        'expiry_idx': idx + 20,
                        'asset': a,
                        'purchase_val': val_holding,
                        'purchase_price': S0,
                        'strike_price': K
                    })
                    
        nav_history.append({'trade_date': dt, 'nav': nav})
        
    df_nav = pd.DataFrame(nav_history).set_index('trade_date')
    return df_nav

print("3. Running Active Satellite 9-asset Risk Parity (Unscaled)...")
df_nav_satellite_1 = run_backtest_9assets(
    df_all, vol_target=0.10, val_tilt=0.0, strike_ratio=0.95, buy_put=True, mult=1.0
)
metrics_satellite_1 = rp.compute_metrics(df_nav_satellite_1['nav'])

# 4. Print Standalone Volatilities
print("\n" + "="*80)
print("  STANDALONE ASSET PERFORMANCE (2022-01-04 to 2026-03-11)")
print("="*80)
all_test_assets = ['hs300', 'zz500', 'chinext', 'div', 'gold', 'nasdaq', 'bond', 'cbond', 'pnl_opt']
for a in all_test_assets:
    ret_series = df_all[f'ret_{a}'] if f'ret_{a}' in df_all else df_all[a]
    ann_ret = (ret_series + 1.0).prod() ** (252.0 / len(ret_series)) - 1.0
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    print(f"Asset: {a:12s} | Ann Return: {ann_ret:7.2%} | Ann Volatility: {ann_vol:7.2%} | Sharpe: {sharpe:5.2f}")
print("="*80)

# 5. Perform Multiplier Sweep
print("\n" + "="*80)
print("  PERFORMANCE SWEEP OVER STOCK PICKER MULTIPLIER (LEVERAGE)")
print("="*80)
multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
sweep_results = []
for m in multipliers:
    # Option B
    df_b = run_backtest_replaced_equity(df_all, vol_target=0.10, mult=m)
    m_b = rp.compute_metrics(df_b['nav'])
    # Option C
    df_c = run_backtest_9assets(df_all, vol_target=0.10, mult=m, buy_put=True)
    m_c = rp.compute_metrics(df_c['nav'])
    
    sweep_results.append({
        'multiplier': m,
        'b_cagr': m_b['CAGR'], 'b_vol': m_b['Volatility'], 'b_sharpe': m_b['Sharpe'], 'b_mdd': m_b['Max Drawdown'], 'b_calmar': m_b['Calmar'],
        'c_cagr': m_c['CAGR'], 'c_vol': m_c['Volatility'], 'c_sharpe': m_c['Sharpe'], 'c_mdd': m_c['Max Drawdown'], 'c_calmar': m_c['Calmar']
    })

print(f"{'Mult':<5s} | {'Repl CAGR':<9s} {'Repl Sharpe':<11s} {'Repl MDD':<8s} {'Repl Cal':<8s} | {'Sat CAGR':<8s} {'Sat Sharpe':<10s} {'Sat MDD':<8s} {'Sat Cal':<8s}")
print("-" * 95)
for r in sweep_results:
    print(f"{r['multiplier']:<5.1f} | {r['b_cagr']:<9.2%} {r['b_sharpe']:<11.2f} {r['b_mdd']:<8.2%} {r['b_calmar']:<8.2f} | {r['c_cagr']:<8.2%} {r['c_sharpe']:<10.2f} {r['c_mdd']:<8.2%} {r['c_calmar']:<8.2f}")
print("="*80)

# 6. Run Optimal Multiplier Configurations
# Let's find the best multiplier for Replacement and Satellite
# Criterion: Maximize CAGR subject to Max Drawdown >= -12.0%
valid_b = [r for r in sweep_results if r['b_mdd'] >= -0.12]
best_b_row = max(valid_b, key=lambda x: x['b_cagr']) if valid_b else sweep_results[0]
best_b_mult = best_b_row['multiplier']

valid_c = [r for r in sweep_results if r['c_mdd'] >= -0.12]
best_c_row = max(valid_c, key=lambda x: x['c_cagr']) if valid_c else sweep_results[0]
best_c_mult = best_c_row['multiplier']

print(f"\nOptimal Replacement Multiplier (MDD >= -12%): {best_b_mult:.1f}x (CAGR: {best_b_row['b_cagr']:.2%}, Sharpe: {best_b_row['b_sharpe']:.2f}, MDD: {best_b_row['b_mdd']:.2%})")
print(f"Optimal Satellite Multiplier   (MDD >= -12%): {best_c_mult:.1f}x (CAGR: {best_c_row['c_cagr']:.2%}, Sharpe: {best_c_row['c_sharpe']:.2f}, MDD: {best_c_row['c_mdd']:.2%})")

# Re-run optimal configs for saving and plotting
df_nav_replace_opt = run_backtest_replaced_equity(df_all, vol_target=0.10, mult=best_b_mult)
metrics_replace_opt = rp.compute_metrics(df_nav_replace_opt['nav'])

df_nav_satellite_opt = run_backtest_9assets(df_all, vol_target=0.10, mult=best_c_mult, buy_put=True)
metrics_satellite_opt = rp.compute_metrics(df_nav_satellite_opt['nav'])

# 7. Core-Satellite Portfolios (Option D)
# Mix Baseline daily returns with daily_stock_opt returns
base_rets = df_nav_base['nav'].pct_change().fillna(0)
stock_rets = df_all.set_index('trade_date')['pnl_opt']

df_nav_cs10 = pd.DataFrame(index=df_nav_base.index)
df_nav_cs10['nav'] = (0.90 * base_rets + 0.10 * stock_rets).add(1.0).cumprod() * 1000000.0
metrics_cs10 = rp.compute_metrics(df_nav_cs10['nav'])

df_nav_cs20 = pd.DataFrame(index=df_nav_base.index)
df_nav_cs20['nav'] = (0.80 * base_rets + 0.20 * stock_rets).add(1.0).cumprod() * 1000000.0
metrics_cs20 = rp.compute_metrics(df_nav_cs20['nav'])

# 8. Print final comparison report
print("\n" + "="*80)
print(f"  FINAL OPTIMIZED STRATEGY COMPARISON (2022-01-04 to 2026-03-11)")
print("="*80)
print(f"Baseline Portfolio (8-Asset)       | CAGR: {metrics_base['CAGR']:.2%}  Vol: {metrics_base['Volatility']:.2%}  Sharpe: {metrics_base['Sharpe']:.2f}  MDD: {metrics_base['Max Drawdown']:.2%}  Calmar: {metrics_base['Calmar']:.2f}")
print(f"Active Replacement (5-Asset, {best_b_mult:.1f}x) | CAGR: {metrics_replace_opt['CAGR']:.2%}  Vol: {metrics_replace_opt['Volatility']:.2%}  Sharpe: {metrics_replace_opt['Sharpe']:.2f}  MDD: {metrics_replace_opt['Max Drawdown']:.2%}  Calmar: {metrics_replace_opt['Calmar']:.2f}")
print(f"Active Satellite   (9-Asset, {best_c_mult:.1f}x) | CAGR: {metrics_satellite_opt['CAGR']:.2%}  Vol: {metrics_satellite_opt['Volatility']:.2%}  Sharpe: {metrics_satellite_opt['Sharpe']:.2f}  MDD: {metrics_satellite_opt['Max Drawdown']:.2%}  Calmar: {metrics_satellite_opt['Calmar']:.2f}")
print(f"Core-Satellite 10% (90/10 Fixed)   | CAGR: {metrics_cs10['CAGR']:.2%}  Vol: {metrics_cs10['Volatility']:.2%}  Sharpe: {metrics_cs10['Sharpe']:.2f}  MDD: {metrics_cs10['Max Drawdown']:.2%}  Calmar: {metrics_cs10['Calmar']:.2f}")
print(f"Core-Satellite 20% (80/20 Fixed)   | CAGR: {metrics_cs20['CAGR']:.2%}  Vol: {metrics_cs20['Volatility']:.2%}  Sharpe: {metrics_cs20['Sharpe']:.2f}  MDD: {metrics_cs20['Max Drawdown']:.2%}  Calmar: {metrics_cs20['Calmar']:.2f}")
print("="*80)

# Save NAV data to CSV
df_nav_base.rename(columns={'nav': 'nav_baseline'}).to_csv(os.path.join(DATA_DIR, 'nav_baseline_22_26.csv'))
df_nav_replace_opt.rename(columns={'nav': 'nav_replace_opt'}).to_csv(os.path.join(DATA_DIR, 'nav_replace_opt_22_26.csv'))
df_nav_satellite_opt.rename(columns={'nav': 'nav_satellite_opt'}).to_csv(os.path.join(DATA_DIR, 'nav_satellite_opt_22_26.csv'))
df_nav_cs10.rename(columns={'nav': 'nav_cs10'}).to_csv(os.path.join(DATA_DIR, 'nav_cs10_22_26.csv'))
df_nav_cs20.rename(columns={'nav': 'nav_cs20'}).to_csv(os.path.join(DATA_DIR, 'nav_cs20_22_26.csv'))
print("Daily NAV curves saved to ETF data folder.")

# 9. Plot comparison curves
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(df_nav_base.index, df_nav_base['nav'] / 1e6, label='Baseline Portfolio (8-Asset)', color='#1e88e5', linewidth=1.5)
plt.plot(df_nav_satellite_opt.index, df_nav_satellite_opt['nav'] / 1e6, label=f'Active Satellite (9-Asset, {best_c_mult:.1f}x)', color='#00897b', linewidth=2.0)
plt.plot(df_nav_replace_opt.index, df_nav_replace_opt['nav'] / 1e6, label=f'Active Replacement (5-Asset, {best_b_mult:.1f}x)', color='#8e24aa', linewidth=1.5, alpha=0.8)
plt.plot(df_nav_cs10.index, df_nav_cs10['nav'] / 1e6, label='Core-Satellite 10% (90/10 Fixed)', color='#fb8c00', linewidth=1.5, alpha=0.8)
plt.plot(df_nav_cs20.index, df_nav_cs20['nav'] / 1e6, label='Core-Satellite 20% (80/20 Fixed)', color='#f4511e', linewidth=1.5, alpha=0.8)
plt.title("Integrated Strategy NAV Comparison (2022-01-04 to 2026-03-11)", fontsize=12, fontweight='bold')
plt.ylabel("Normalized NAV")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(2, 1, 2)
dd_base = (df_nav_base['nav'] - df_nav_base['nav'].cummax()) / df_nav_base['nav'].cummax() * 100.0
dd_sat_opt = (df_nav_satellite_opt['nav'] - df_nav_satellite_opt['nav'].cummax()) / df_nav_satellite_opt['nav'].cummax() * 100.0
dd_rep_opt = (df_nav_replace_opt['nav'] - df_nav_replace_opt['nav'].cummax()) / df_nav_replace_opt['nav'].cummax() * 100.0
dd_cs10 = (df_nav_cs10['nav'] - df_nav_cs10['nav'].cummax()) / df_nav_cs10['nav'].cummax() * 100.0
dd_cs20 = (df_nav_cs20['nav'] - df_nav_cs20['nav'].cummax()) / df_nav_cs20['nav'].cummax() * 100.0

plt.fill_between(dd_sat_opt.index, dd_sat_opt, 0, label='Active Satellite Drawdown (Opt)', color='#00897b', alpha=0.3)
plt.plot(dd_base.index, dd_base, label='Baseline Drawdown', color='#1e88e5', alpha=0.5, linewidth=1.0)
plt.plot(dd_rep_opt.index, dd_rep_opt, label='Replacement Drawdown (Opt)', color='#8e24aa', alpha=0.5, linewidth=1.0)
plt.plot(dd_cs10.index, dd_cs10, label='CS 10% Drawdown', color='#fb8c00', alpha=0.5, linewidth=1.0)
plt.plot(dd_cs20.index, dd_cs20, label='CS 20% Drawdown', color='#f4511e', alpha=0.5, linewidth=1.0)
plt.ylabel("Drawdown (%)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save plot to results and artifacts folders
plot_path_results = os.path.join(rp.RESULTS_DIR, 'nav_integrated_comparison.png')
plot_path_artifacts = r"C:\Users\liuqi\.gemini\antigravity\brain\aedf743b-815e-4a43-a730-13a66c11d107\nav_integrated_comparison.png"
plt.savefig(plot_path_results, dpi=300)
plt.savefig(plot_path_artifacts, dpi=300)
plt.close()
print("NAV Comparison plot saved.")

