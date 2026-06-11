"""
step3_analyze_relationship.py
统计检验“最大痛点阻力/引力”假说：
1. 价格收敛检验：检验标的价格与最大痛点的偏差（Deviation）的方差是否随着到期日（DTE）临近而显著缩小。
2. 收益预测性检验：运行 OLS 回归 R_{t->Expiry} = a + b * Dev_t + e，检验系数 b 是否显著为负，并分析其在不同 DTE 窗口下的显著性。
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 目录配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'max_pain_history.csv')
RESULTS_CSV = os.path.join(DATA_DIR, 'regression_results.csv')

def run_regression(x, y):
    """
    运行简单的一元 OLS 回归，返回系数、截距、t统计量、p值和R2
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    # 计算 t 统计量
    # t = slope / std_err
    t_stat = slope / std_err if std_err != 0 else np.nan
    return {
        'slope': slope,
        'intercept': intercept,
        't_stat': t_stat,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_obs': len(x)
    }

def main():
    print(">>> Starting Convergence & Predictability Analysis...", flush=True)
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found. Please run step2 first.")
        
    df = pd.read_csv(INPUT_FILE)
    df['trade_date'] = df['trade_date'].astype(str)
    df['expiry_date'] = df['expiry_date'].astype(str)
    
    # 提取标的价格映射表，以便找到到期日的 ETF 收盘价
    price_lookup = df[['trade_date', 'underlying_code', 'underlying_close']].drop_duplicates()
    price_map = price_lookup.set_index(['trade_date', 'underlying_code'])['underlying_close'].to_dict()
    
    # 获取到期日的 ETF 价格
    df['underlying_close_at_expiry'] = df.apply(
        lambda r: price_map.get((r['expiry_date'], r['underlying_code']), np.nan), axis=1
    )
    
    # 过滤掉无法获取到期日价格的记录（主要是尚未过期的最后一期合约）
    df_clean = df.dropna(subset=['underlying_close_at_expiry', 'days_to_expiry']).copy()
    # 过滤掉 DTE 为负的异常值
    df_clean = df_clean[df_clean['days_to_expiry'] >= 0]
    
    # 计算 Deviation (偏差) 和 Return to Expiry (到期回报)
    # Deviation 定义：(标的收盘价 - 最大痛点) / 标的收盘价
    df_clean['deviation'] = (df_clean['underlying_close'] - df_clean['max_pain_price']) / df_clean['underlying_close']
    # Return to Expiry 定义：(到期收盘价 - 当前收盘价) / 当前收盘价
    df_clean['return_to_expiry'] = (df_clean['underlying_close_at_expiry'] - df_clean['underlying_close']) / df_clean['underlying_close']
    
    # 保存包含特征的文件
    feature_file = os.path.join(DATA_DIR, 'max_pain_features.csv')
    df_clean.to_csv(feature_file, index=False)
    print(f"Saved cleaned features with expiry returns to {feature_file}")
    
    # ==================== 1. 价格收敛分析 ====================
    print("\n--- 1. Convergence Analysis (Volatility of Deviation vs DTE) ---")
    # 将 DTE 划分为区间
    bins = [0, 2, 5, 10, 15, 20, 30, 100]
    labels = ['0-2 days', '3-5 days', '6-10 days', '11-15 days', '16-20 days', '21-30 days', '30+ days']
    df_clean['dte_bin'] = pd.cut(df_clean['days_to_expiry'], bins=bins, labels=labels, include_lowest=True)
    
    convergence_stats = []
    for und_code in ['510050.SH', '510300.SH']:
        print(f"\nUnderlying Asset: {und_code}")
        df_sub = df_clean[df_clean['underlying_code'] == und_code]
        
        # 统计每个 bin 里的偏差标准差
        grouped = df_sub.groupby('dte_bin', observed=False)['deviation'].agg(['count', 'mean', 'std', 'var'])
        print(grouped)
        
        for idx, row in grouped.iterrows():
            convergence_stats.append({
                'underlying_code': und_code,
                'dte_bin': idx,
                'count': row['count'],
                'mean_dev': row['mean'],
                'std_dev': row['std'],
                'var_dev': row['var']
            })
            
    # ==================== 2. 回归预测分析 ====================
    print("\n--- 2. Return Predictability Regressions (R_{t->Expiry} = a + b * Dev_t + e) ---")
    
    regression_records = []
    
    # 检验窗口
    dte_filters = [
        ('All DTE', lambda dte: True),
        ('DTE <= 20', lambda dte: dte <= 20),
        ('DTE <= 10', lambda dte: dte <= 10),
        ('DTE <= 5', lambda dte: dte <= 5),
        ('DTE <= 3', lambda dte: dte <= 3),
        ('DTE <= 1', lambda dte: dte <= 1)
    ]
    
    for und_code in ['510050.SH', '510300.SH']:
        print(f"\nRegression results for {und_code}:")
        df_sub = df_clean[df_clean['underlying_code'] == und_code]
        
        for name, filter_fn in dte_filters:
            df_reg = df_sub[df_sub['days_to_expiry'].apply(filter_fn)]
            if len(df_reg) < 10:
                continue
                
            res = run_regression(df_reg['deviation'].values, df_reg['return_to_expiry'].values)
            res['underlying_code'] = und_code
            res['window'] = name
            regression_records.append(res)
            
            print(f"  Window: {name:<12} | Obs: {res['n_obs']:<4} | Slope: {res['slope']:.4f} | t-stat: {res['t_stat']:.2f} | p-val: {res['p_value']:.4f} | R2: {res['r_squared']:.4f}")
            
    df_reg_results = pd.DataFrame(regression_records)
    df_reg_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved regression results to {RESULTS_CSV}")
    
    # ==================== 3. 绘图 ====================
    print("\n>>> Generating Analysis Plots...", flush=True)
    
    # Plot 1: 收敛图（标准差 vs DTE Bin）
    fig, ax = plt.subplots(figsize=(10, 6))
    df_conv = pd.DataFrame(convergence_stats)
    
    colors = {'510050.SH': '#1f77b4', '510300.SH': '#ff7f0e'}
    for und_code, group in df_conv.groupby('underlying_code'):
        # 颠倒顺序以展示从远期到临期的变化
        group_sorted = group.iloc[::-1]
        ax.plot(group_sorted['dte_bin'], group_sorted['std_dev'] * 100, marker='o', linewidth=2, label=f"{und_code} Deviation Std (%)", color=colors[und_code])
        
    ax.set_title("Price Convergence to Max Pain as Expiry Approaches", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Standard Deviation of Price Deviation (%)", fontsize=12)
    ax.set_xlabel("Days to Expiry (DTE) Bins", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plot_conv_path = os.path.join(PLOTS_DIR, 'max_pain_convergence.png')
    plt.savefig(plot_conv_path, dpi=300)
    plt.close()
    print(f"Saved convergence plot to {plot_conv_path}")
    
    # Plot 2: 散点图与回归线 (DTE <= 5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for i, und_code in enumerate(['510050.SH', '510300.SH']):
        ax = axes[i]
        df_sub = df_clean[(df_clean['underlying_code'] == und_code) & (df_clean['days_to_expiry'] <= 5)]
        
        # 散点
        ax.scatter(df_sub['deviation'] * 100, df_sub['return_to_expiry'] * 100, alpha=0.5, color=colors[und_code], edgecolors='none', label='Obs (DTE <= 5)')
        
        # 回归线
        res = df_reg_results[(df_reg_results['underlying_code'] == und_code) & (df_reg_results['window'] == 'DTE <= 5')].iloc[0]
        x_vals = np.linspace(df_sub['deviation'].min(), df_sub['deviation'].max(), 100)
        y_vals = res['intercept'] + res['slope'] * x_vals
        ax.plot(x_vals * 100, y_vals * 100, color='red', linestyle='-', linewidth=2, 
                label=f"OLS: Return = {res['intercept']:.4f} + {res['slope']:.2f}*Dev\nt-stat: {res['t_stat']:.2f}")
        
        ax.set_title(f"{und_code} (DTE <= 5)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Price Deviation from Max Pain (%)", fontsize=11)
        if i == 0:
            ax.set_ylabel("ETF Return to Expiry (%)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        
    plt.suptitle("ETF Return to Expiry vs Price Deviation from Max Pain (DTE <= 5)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_scatter_path = os.path.join(PLOTS_DIR, 'max_pain_predictability.png')
    plt.savefig(plot_scatter_path, dpi=300)
    plt.close()
    print(f"Saved predictability plot to {plot_scatter_path}")
    
    # 总结与判定
    print("\n--- 3. Hypothesis Verification Summary ---")
    for und_code in ['510050.SH', '510300.SH']:
        res_all = df_reg_results[(df_reg_results['underlying_code'] == und_code) & (df_reg_results['window'] == 'All DTE')].iloc[0]
        res_5 = df_reg_results[(df_reg_results['underlying_code'] == und_code) & (df_reg_results['window'] == 'DTE <= 5')].iloc[0]
        
        print(f"\nAsset: {und_code}")
        print(f"  All-DTE Slope: {res_all['slope']:.4f} (t-stat: {res_all['t_stat']:.2f})")
        print(f"  DTE <= 5 Slope: {res_5['slope']:.4f} (t-stat: {res_5['t_stat']:.2f})")
        
        is_convergent = res_5['slope'] < 0 and res_5['t_stat'] < -1.96
        if is_convergent:
            print("  ==> CONCLUSION: Max Pain Gravitational Hypothesis is VALID! Price deviation has statistically significant negative correlation with future returns near expiry.")
        else:
            print("  ==> CONCLUSION: Max Pain Gravitational Hypothesis is INVALID or statistically insignificant. No strong evidence of price convergence near expiry.")
            
if __name__ == '__main__':
    main()
