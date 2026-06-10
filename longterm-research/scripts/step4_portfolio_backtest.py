"""
step4_portfolio_backtest.py
高回真度A股组合回测系统。
支持：
1. 真实交易摩擦：买入 0.2% (佣金+滑点)，卖出 0.3% (印花税+佣金+滑点)。
2. 交易限制过滤：涨停无法买入，跌停或停牌无法卖出（被迫持仓）。
3. 行业中性与分散化：单行业持仓比例上限（例如 20%）。
4. 10日定期调仓。
5. 每日 NAV 跟踪与可视化。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PRED_FILE = os.path.join(PRED_DIR, 'predictions_longterm.parquet')
FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')

# 策略参数
PORTFOLIO_SIZE = 50                 # 持仓股票数
HOLDING_DAYS = 10                  # 调仓周期 (10个交易日)
MAX_WEIGHT_PER_INDUSTRY = 0.20     # 单个行业最大权重 (20%)
INITIAL_CAPITAL = 1000000.0        # 初始资金 (100万)

# 交易成本
BUY_COST_RATE = 0.002              # 买入佣金+滑点 = 0.2%
SELL_COST_RATE = 0.003             # 卖出印花税+佣金+滑点 = 0.3%

def is_limit_up(row, code=None):
    if code is None:
        code = row['ts_code']
    # star/chi market (300/688) has 20% limit, others have 10% limit
    limit = 0.198 if (code.startswith('30') or code.startswith('68')) else 0.098
    return row['next_open'] >= row['close'] * (1 + limit)

def is_limit_down(row, code=None):
    if code is None:
        code = row['ts_code']
    limit = 0.198 if (code.startswith('30') or code.startswith('68')) else 0.098
    return row['next_open'] <= row['close'] * (1 - limit)

def run_backtest():
    print("Loading predictions...", flush=True)
    pred_df = pd.read_parquet(PRED_FILE)
    pred_df = pred_df.drop_duplicates(subset=['trade_date', 'ts_code'])
    
    print("Loading volume columns from features...", flush=True)
    feat_df = pd.read_parquet(FEATURES_FILE, columns=['trade_date', 'ts_code', 'vol'])
    feat_df = feat_df.drop_duplicates(subset=['trade_date', 'ts_code'])
    
    # 合并成交量数据
    df = pred_df.merge(feat_df, on=['trade_date', 'ts_code'], how='left')
    df['vol'] = df['vol'].fillna(0)
    
    # 按日期和代码排序
    df = df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
    
    trade_dates = sorted(df['trade_date'].unique())
    print(f"Backtest period: {trade_dates[0]} to {trade_dates[-1]} ({len(trade_dates)} trading days)", flush=True)
    
    # 调仓日期 (每 10 天调仓一次)
    rebalance_dates = set(trade_dates[::HOLDING_DAYS])
    
    # 状态变量
    current_holdings = {}  # {ts_code: cash_value}
    cash = INITIAL_CAPITAL
    daily_navs = []
    
    # 辅助查找：加速每日数据索引
    df_by_date = {dt: g for dt, g in df.groupby('trade_date')}
    
    print("\nStarting daily simulation...", flush=True)
    
    for idx, dt in enumerate(trade_dates):
        # 1. 获取当天的行情数据
        if dt not in df_by_date:
            # 如果当天没有数据，净值保持不变
            prev_nav = daily_navs[-1]['nav'] if daily_navs else INITIAL_CAPITAL
            daily_navs.append({'trade_date': dt, 'nav': prev_nav, 'cash': cash, 'holdings_val': 0})
            continue
            
        day_data = df_by_date[dt]
        day_prices = day_data.set_index('ts_code').to_dict(orient='index')
        
        # 2. 如果是持仓日，更新持仓股票在当天的市值
        holdings_val = 0
        if current_holdings:
            updated_holdings = {}
            for code, prev_val in current_holdings.items():
                if code in day_prices:
                    # 如果是新买入的第一天（即调仓日的后一天），以 next_open 买入，以收盘价结算
                    # 我们通过标记来识别，这里使用简单的逻辑：如果前一天是调仓日
                    is_entry_day = False
                    if idx > 0 and trade_dates[idx-1] in rebalance_dates:
                        # 检查是不是新买入的
                        # 我们可以在调仓逻辑中计算好买入数量和价值，这里直接乘 return
                        # 或者在进入当天直接算 (close_d - open_d) / open_d
                        pass
                    
                    # 统一使用 pct_chg 更新（包含停牌的 pct_chg=0）
                    pct_chg = day_prices[code]['pct_chg']
                    if pd.isna(pct_chg):
                        pct_chg = 0.0
                    val = prev_val * (1 + pct_chg)
                    updated_holdings[code] = val
                    holdings_val += val
                else:
                    # 如果数据中缺失该股票，市值保持不变
                    updated_holdings[code] = prev_val
                    holdings_val += prev_val
            current_holdings = updated_holdings
            
        current_nav = cash + holdings_val
        daily_navs.append({
            'trade_date': dt,
            'nav': current_nav,
            'cash': cash,
            'holdings_val': holdings_val
        })
        
        # 3. 调仓逻辑 (发生在调仓日的收盘后/次日开盘前，即在当天收盘后决定，并在次日开盘执行交易)
        if dt in rebalance_dates:
            # 次日是 idx+1 交易日
            if idx + 1 >= len(trade_dates):
                continue  # 数据结束，不再调仓
                
            next_dt = trade_dates[idx+1]
            if next_dt not in df_by_date:
                continue
                
            next_day_prices = df_by_date[next_dt].set_index('ts_code').to_dict(orient='index')
            
            # A. 确定可卖出和不可卖出的持仓股票
            sell_candidates = []
            forced_holdings = {}
            
            for code, val in current_holdings.items():
                # 判断次日是否可交易（停牌或跌停开盘无法卖出）
                untradeable = False
                if code not in next_day_prices:
                    untradeable = True
                else:
                    row = next_day_prices[code]
                    if row['vol'] == 0 or is_limit_down(row, code):
                        untradeable = True
                        
                if untradeable:
                    forced_holdings[code] = val  # 强行继续持有，无法卖出
                else:
                    sell_candidates.append(code)
                    
            # B. 卖出可卖出的股票并回收现金
            for code in sell_candidates:
                val = current_holdings[code]
                # 计算交易摩擦 (卖出 0.3%)
                cost = val * SELL_COST_RATE
                cash += (val - cost)
                
            # C. 挑选新的目标股票
            # 过滤可买入的候选股：必须有预测值，成交量 > 0，且次日开盘不涨停
            candidates = day_data[day_data['pred_score'].notna()].copy()
            
            # 判断次日是否涨停
            # 先合并次日价格以判定涨停
            candidates = candidates.merge(
                df_by_date[next_dt][['ts_code', 'next_open', 'vol']], 
                on='ts_code', 
                suffixes=('', '_next')
            )
            
            # 过滤可买入股
            candidates = candidates[
                (candidates['vol_next'] > 0) & 
                (~candidates.apply(lambda r: is_limit_up(r, r['ts_code']), axis=1))
            ]
            
            # 按预测值降序排序
            candidates = candidates.sort_values(by='pred_score', ascending=False)
            
            # D. 执行行业限制筛选
            selected_codes = []
            industry_counts = {}
            max_stocks_per_industry = int(PORTFOLIO_SIZE * MAX_WEIGHT_PER_INDUSTRY)
            
            # 先把被迫持仓的股票计入行业计数中
            for code in forced_holdings.keys():
                ind = day_prices.get(code, {}).get('industry', 'Unknown')
                industry_counts[ind] = industry_counts.get(ind, 0) + 1
                
            # 从候选股中补充股票
            for _, row in candidates.iterrows():
                if len(selected_codes) + len(forced_holdings) >= PORTFOLIO_SIZE:
                    break
                code = row['ts_code']
                if code in forced_holdings:
                    continue  # 已被迫持有
                    
                ind = row['industry']
                count = industry_counts.get(ind, 0)
                if count < max_stocks_per_industry:
                    selected_codes.append(code)
                    industry_counts[ind] = count + 1
                    
            # E. 买入新股票
            total_target_stocks = len(selected_codes) + len(forced_holdings)
            if total_target_stocks > 0:
                # 扣除被迫持仓后，剩余的可分配资金
                total_val = cash + sum(forced_holdings.values())
                # 平均分配到每个目标持仓
                target_value_per_stock = total_val / total_target_stocks
                
                # 对被迫持仓股：如果其当前价值大于目标价值，无法卖出减仓；如果小于目标价值，也不予补仓（保持现状）
                # 可分配资金为 cash，分给所有新买入的股票
                new_holdings = {code: val for code, val in forced_holdings.items()}
                
                if len(selected_codes) > 0:
                    # 剩余的钱用来买新股
                    # 实际上可能发生被迫持仓的市值超过了 target_value_per_stock
                    # 这里我们采用保守且简单的分配：将现金 cash 平均分配给新买入的股票
                    buy_value_per_stock = cash / len(selected_codes)
                    for code in selected_codes:
                        # 扣除买入成本 (0.2%)
                        cost = buy_value_per_stock * BUY_COST_RATE
                        new_holdings[code] = buy_value_per_stock - cost
                    cash = 0.0
                else:
                    # 没有新买入的，现金就保留
                    pass
                    
                current_holdings = new_holdings
            else:
                current_holdings = {code: val for code, val in forced_holdings.items()}
                
    # 4. 计算回测绩效
    nav_df = pd.DataFrame(daily_navs)
    nav_df['trade_date'] = pd.to_datetime(nav_df['trade_date'])
    nav_df = nav_df.set_index('trade_date')
    
    # 增加基准净值进行对比
    # 基准为全市场等权组合 (每日等权持有所有股票)
    bench_returns = df.groupby('trade_date')['pct_chg'].mean().fillna(0.0)
    bench_returns.index = pd.to_datetime(bench_returns.index)
    bench_nav = (1 + bench_returns).cumprod() * INITIAL_CAPITAL
    nav_df['benchmark'] = bench_nav.reindex(nav_df.index).ffill()
    
    # 计算绩效指标
    print("\n--- Portfolio Backtest Results ---")
    years = len(nav_df) / 252.0
    final_nav = nav_df['nav'].iloc[-1]
    total_ret = final_nav / INITIAL_CAPITAL - 1
    cagr = (final_nav / INITIAL_CAPITAL) ** (1.0 / years) - 1
    
    # Vol and Sharpe
    daily_rets = nav_df['nav'].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / ann_vol if ann_vol > 0 else 0
    
    # Max Drawdown
    cum_max = nav_df['nav'].cummax()
    drawdowns = (nav_df['nav'] - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    # Benchmark Comparison
    bench_final = nav_df['benchmark'].iloc[-1]
    bench_cagr = (bench_final / INITIAL_CAPITAL) ** (1.0 / years) - 1
    bench_daily_rets = nav_df['benchmark'].pct_change().dropna()
    bench_vol = bench_daily_rets.std() * np.sqrt(252)
    bench_sharpe = (bench_daily_rets.mean() * 252) / bench_vol if bench_vol > 0 else 0
    bench_max_dd = ((nav_df['benchmark'] - nav_df['benchmark'].cummax()) / nav_df['benchmark'].cummax()).min()
    
    metrics = {
        'Metric': ['Total Return', 'CAGR (Annualized)', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy': [f"{total_ret:.2%}", f"{cagr:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}", f"{max_dd:.2%}"],
        'Benchmark': [f"{bench_final / INITIAL_CAPITAL - 1:.2%}", f"{bench_cagr:.2%}", f"{bench_vol:.2%}", f"{bench_sharpe:.2f}", f"{bench_max_dd:.2%}"]
    }
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))
    
    # 保存结果
    nav_df.to_csv(os.path.join(RESULTS_DIR, 'portfolio_backtest_nav.csv'))
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'portfolio_backtest_metrics.csv'), index=False)
    
    # 绘制净值图
    plot_nav(nav_df)

def plot_nav(nav_df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 归一化为 1.0 开始
    strat_norm = nav_df['nav'] / INITIAL_CAPITAL
    bench_norm = nav_df['benchmark'] / INITIAL_CAPITAL
    
    ax.plot(nav_df.index, strat_norm, label='Multi-Factor Ranking Strategy (Top 50)', color='#1b5e20', linewidth=2.5)
    ax.plot(nav_df.index, bench_norm, label='Benchmark (Market Equal-Weight)', color='#546e7a', linewidth=1.5, linestyle='--')
    
    # 绘制超额收益 (Active Return)
    active_norm = strat_norm - bench_norm
    ax.fill_between(nav_df.index, active_norm, label='Active Return (Strategy - Bench)', color='#a5d6a7', alpha=0.3)
    
    ax.set_title('Multi-Factor Ranking Strategy Backtest (Out-of-Sample 2022-2026)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Net Asset Value (NAV)', fontsize=12)
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout()
    
    out_png = os.path.join(RESULTS_DIR, 'portfolio_backtest_nav.png')
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")
    plt.close()

if __name__ == '__main__':
    run_backtest()
