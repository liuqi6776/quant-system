"""
step2_factor_evaluation.py
计算多个候选因子的日频 Rank IC 与 ICIR，并进行分位数（Decile）组合回测，绘制净值曲线。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')

def calculate_ic_metrics(df, factors, target_col):
    print(f"\n--- Calculating Rank IC metrics against {target_col} ---")
    ic_results = []
    
    # 过滤包含有效 target_col 的行
    valid_df = df[df[target_col].notna()].copy()
    
    # 计算每日 Rank IC
    for factor in factors:
        if factor not in valid_df.columns:
            print(f"⚠️ Factor {factor} not found in features.")
            continue
            
        print(f"Processing factor: {factor}...")
        # 计算 Spearman correlation
        daily_ic = valid_df.groupby('trade_date').apply(
            lambda x: x[factor].corr(x[target_col], method='spearman')
        )
        
        # 移除 NaN 值
        daily_ic = daily_ic.dropna()
        if len(daily_ic) == 0:
            print(f"⚠️ No valid daily IC for {factor}.")
            continue
            
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ic_ir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else 0
        ic_pos_ratio = (daily_ic > 0).mean()
        
        ic_results.append({
            'factor': factor,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_pos_ratio': ic_pos_ratio,
            'valid_days': len(daily_ic)
        })
        
    ic_df = pd.DataFrame(ic_results)
    ic_df = ic_df.sort_values(by='ic_ir', key=abs, ascending=False)
    print(ic_df.to_string(index=False))
    
    out_csv = os.path.join(RESULTS_DIR, f'factor_ic_summary_{target_col}.csv')
    ic_df.to_csv(out_csv, index=False)
    print(f"Saved IC summary to {out_csv}")
    return ic_df

def run_decile_backtest(df, factor, target_ret_col, holding_days=10, ascending=True):
    print(f"\n--- Running Decile Backtest for Factor: {factor} (Holding: {holding_days}d) ---")
    # ascending=True 表示因子越小收益越好（如 turnover_rate、circ_mv 在A股通常越小越好）
    # ascending=False 表示因子越大收益越好
    
    # 获取所有的交易日期
    trade_dates = sorted(df['trade_date'].unique())
    
    # 每 holding_days 天进行一次调仓
    rebalance_dates = trade_dates[::holding_days]
    print(f"Total trading days: {len(trade_dates)}, Rebalance occurrences: {len(rebalance_dates)}")
    
    decile_returns = {f"Decile_{i+1}": [] for i in range(10)}
    decile_returns['Long_Short'] = []
    rebalance_records = []
    
    for idx, dt in enumerate(rebalance_dates):
        # 取调仓当天的截面数据
        day_df = df[df['trade_date'] == dt].copy()
        # 必须在调仓当天包含有效的因子值以及未来持有期的收益率
        day_df = day_df[day_df[factor].notna() & day_df[target_ret_col].notna()]
        
        if len(day_df) < 100:
            continue
            
        # 按照因子排序，分成10组
        day_df['rank'] = day_df[factor].rank(method='first', ascending=ascending)
        day_df['decile'] = pd.qcut(day_df['rank'], 10, labels=[f"Decile_{i+1}" for i in range(10)])
        
        # 计算每一组在持有期内的平均收益率
        group_rets = day_df.groupby('decile')[target_ret_col].mean()
        
        for decile_name in decile_returns.keys():
            if decile_name == 'Long_Short':
                # 多头组（第10组）减空头组（第1组）的超额收益率
                ls_ret = group_rets['Decile_10'] - group_rets['Decile_1']
                decile_returns['Long_Short'].append(ls_ret)
            else:
                decile_returns[decile_name].append(group_rets[decile_name])
                
        rebalance_records.append(dt)
        
    # 构建净值序列
    nav_df = pd.DataFrame(index=rebalance_records)
    for col, rets in decile_returns.items():
        if len(rets) == len(rebalance_records):
            if col == 'Long_Short':
                # 多空组合一般是纯Alpha，通过累加收益率或累乘 1 + 收益率 表现
                nav_df[col] = np.cumprod(1 + np.array(rets))
            else:
                nav_df[col] = np.cumprod(1 + np.array(rets))
                
    # 增加基准净值（全市场等权）
    all_rets = []
    for dt in rebalance_records:
        day_df = df[(df['trade_date'] == dt) & df[target_ret_col].notna()]
        all_rets.append(day_df[target_ret_col].mean() if len(day_df) > 0 else 0)
    nav_df['Benchmark'] = np.cumprod(1 + np.array(all_rets))
    
    # 计算绩效指标
    print("\n--- Performance Metrics by Decile ---")
    metrics = []
    years_held = (len(trade_dates) / 252.0)
    
    for col in nav_df.columns:
        final_nav = nav_df[col].iloc[-1]
        cagr = (final_nav) ** (1.0 / years_held) - 1 if final_nav > 0 else -1
        
        # 计算最大回撤
        cum_max = nav_df[col].cummax()
        drawdowns = (nav_df[col] - cum_max) / cum_max
        max_dd = drawdowns.min()
        
        # 计算 Sharpe Ratio (基于期间收益波动)
        rets = nav_df[col].pct_change().dropna()
        ann_std = rets.std() * np.sqrt(252 / holding_days)
        ann_mean = rets.mean() * (252 / holding_days)
        sharpe = ann_mean / ann_std if ann_std > 0 else 0
        
        metrics.append({
            'Portfolio': col,
            'Final_NAV': final_nav,
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Max_DD': max_dd
        })
        
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))
    
    # 保存绩效指标
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f'decile_metrics_{factor}.csv'), index=False)
    nav_df.to_csv(os.path.join(RESULTS_DIR, f'decile_navs_{factor}.csv'))
    
    # 绘制净值图
    plot_navs(nav_df, factor)

def plot_navs(nav_df, factor):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制 Decile 1 - 10
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))
    for i in range(10):
        name = f"Decile_{i+1}"
        ax.plot(nav_df.index, nav_df[name], label=name, color=colors[i], linewidth=1.5, alpha=0.8)
        
    ax.plot(nav_df.index, nav_df['Benchmark'], label='Benchmark (Equal-Weight)', color='blue', linewidth=2, linestyle='--')
    ax.plot(nav_df.index, nav_df['Long_Short'], label='Long-Short (D10 - D1)', color='black', linewidth=2.5)
    
    ax.set_title(f'Decile Portfolio NAV - Factor: {factor}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Rebalance Date', fontsize=12)
    ax.set_ylabel('Net Asset Value (NAV)', fontsize=12)
    
    # 每 15 个点显示一个横坐标刻度，防止重叠
    tick_spacing = max(1, len(nav_df) // 10)
    plt.xticks(nav_df.index[::tick_spacing], rotation=45)
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    out_png = os.path.join(RESULTS_DIR, f'decile_navs_{factor}.png')
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")
    plt.close()

def run_evaluation():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df)} rows.")
    
    # 候选因子列表
    factors = [
        'turnover_rate', 
        'circ_mv', 
        'macd', 
        'rsi_14', 
        'volume_ratio', 
        'ths_hot_rank'
    ]
    
    # 1. 计算不同周期收益的 Rank IC
    calculate_ic_metrics(df, factors, 'mkt_excess_ret_5d')
    calculate_ic_metrics(df, factors, 'mkt_excess_ret_10d')
    calculate_ic_metrics(df, factors, 'mkt_excess_ret_20d')
    
    # 2. 对最优代表因子进行 10d 分层组合回测
    # turnover_rate (换手率) 在A股是极强的负向因子 (越小越好)，因此 ascending=True
    run_decile_backtest(df, 'turnover_rate', 'ret_10d', holding_days=10, ascending=True)

if __name__ == '__main__':
    run_evaluation()
