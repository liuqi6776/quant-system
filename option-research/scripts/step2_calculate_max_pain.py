"""
step2_calculate_max_pain.py
从每日下载的期权数据中计算 50ETF & 300ETF 期权的日度最大痛点价格 (Max Pain Price)。
数学原理：
对于某一日期的某一到期月期权链，设定标的价格格点 S，寻找最小化期权买方行权总价值的 S*：
Pain(S) = sum( Call_OI * max(S - K, 0) ) + sum( Put_OI * max(K - S, 0) )
S* = argmin Pain(S)
"""
import os
import glob
import pandas as pd
import numpy as np

# 目录配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
DAILY_DIR = os.path.join(DATA_DIR, 'daily')

BASIC_FILE = os.path.join(DATA_DIR, 'opt_basic.parquet')
UNDERLYING_FILE = os.path.join(DATA_DIR, 'underlying_daily.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'max_pain_history.csv')

def calculate_max_pain_for_chain(df_chain):
    """
    向量化计算某一期权链的最大痛点价格
    """
    calls = df_chain[df_chain['call_put'] == 'C']
    puts = df_chain[df_chain['call_put'] == 'P']
    
    if len(calls) == 0 or len(puts) == 0:
        return np.nan, 0, 0
        
    # 提取行权价与持仓量 (OI)
    call_strikes = calls['exercise_price'].values
    call_oi = calls['oi'].fillna(0.0).values
    
    put_strikes = puts['exercise_price'].values
    put_oi = puts['oi'].fillna(0.0).values
    
    # 设定标的价格搜寻格点 S
    all_strikes = np.concatenate([call_strikes, put_strikes])
    min_s = max(0.5, all_strikes.min() - 0.2)
    max_s = all_strikes.max() + 0.2
    
    # 构建 0.005 步长的精细格点 (A股ETF价格通常在2.0-5.0之间)
    s_grid = np.arange(min_s, max_s, 0.005)
    
    # 转换为 2D 矩阵进行向量化计算以极大提升速度
    # s_grid: (M, 1), strikes: (1, N)
    S = s_grid[:, np.newaxis]
    
    # 计算 Calls 价值: max(S - K, 0) * Call_OI
    call_payoffs = np.maximum(S - call_strikes[np.newaxis, :], 0.0)
    call_pain = np.dot(call_payoffs, call_oi)
    
    # 计算 Puts 价值: max(K - S, 0) * Put_OI
    put_payoffs = np.maximum(put_strikes[np.newaxis, :] - S, 0.0)
    put_pain = np.dot(put_payoffs, put_oi)
    
    total_pain = call_pain + put_pain
    
    # 寻找 Pain 最小的格点价格
    best_idx = np.argmin(total_pain)
    max_pain_price = s_grid[best_idx]
    
    total_call_oi = np.sum(call_oi)
    total_put_oi = np.sum(put_oi)
    
    return max_pain_price, total_call_oi, total_put_oi

def main():
    print(">>> Starting Max Pain Calculation Process...", flush=True)
    
    if not os.path.exists(BASIC_FILE) or not os.path.exists(UNDERLYING_FILE):
        raise FileNotFoundError("Basic contract info or underlying daily prices not found. Please run step1 first.")
        
    df_basic = pd.read_parquet(BASIC_FILE)
    df_und = pd.read_parquet(UNDERLYING_FILE)
    
    # 建立 underlying mapping
    # 510050 开头的合约对应 510050.SH，510300 开头的合约对应 510300.SH
    df_basic['underlying_code'] = np.where(df_basic['symbol'].str.startswith('510050'), '510050.SH', '510300.SH')
    
    # 按 trade_date, ts_code 建立标的证券价格对照字典
    df_und['trade_date'] = df_und['trade_date'].astype(str)
    und_price_map = df_und.set_index(['trade_date', 'ts_code'])['close'].to_dict()
    
    # 搜寻所有每日数据文件
    daily_files = glob.glob(os.path.join(DAILY_DIR, "opt_daily_*.parquet"))
    daily_files = sorted(daily_files)
    print(f"Found {len(daily_files)} daily options data files to process.", flush=True)
    
    records = []
    
    # 缓存 contract 基础信息
    contract_info = df_basic.set_index('ts_code')[['call_put', 'exercise_price', 'last_edate', 'underlying_code']].to_dict(orient='index')
    
    for file_path in daily_files:
        filename = os.path.basename(file_path)
        date = filename.split('_')[-1].split('.')[0]
        
        # 加载每日数据
        try:
            df_day = pd.read_parquet(file_path)
        except Exception:
            continue
            
        if df_day.empty:
            continue
            
        # 匹配合约基本属性
        df_day['call_put'] = df_day['ts_code'].map(lambda x: contract_info.get(x, {}).get('call_put', None))
        df_day['exercise_price'] = df_day['ts_code'].map(lambda x: contract_info.get(x, {}).get('exercise_price', np.nan))
        df_day['last_edate'] = df_day['ts_code'].map(lambda x: contract_info.get(x, {}).get('last_edate', None))
        df_day['underlying_code'] = df_day['ts_code'].map(lambda x: contract_info.get(x, {}).get('underlying_code', None))
        
        # 剔除未匹配到基本信息的合约
        df_day = df_day.dropna(subset=['call_put', 'exercise_price', 'last_edate', 'underlying_code'])
        if df_day.empty:
            continue
            
        # 对每一个标的分别计算 Max Pain
        for und_code in ['510050.SH', '510300.SH']:
            df_und_day = df_day[df_day['underlying_code'] == und_code]
            if df_und_day.empty:
                continue
                
            # 确定该标的在这一天所有活跃合约中，各到期日的未平仓总量
            # A股期权有四个到期月：当月、下月、下季、隔季
            # 我们要寻找的是当期【最靠近到期日】的 dominant 主力到期系列
            # 过滤已过期或当天过期的到期日（last_edate 必须 >= 当前日期 date）
            active_dates = df_und_day[df_und_day['last_edate'] >= date]['last_edate'].unique()
            if len(active_dates) == 0:
                continue
                
            # 最近的到期日就是 dominant 周期
            expiry_date = min(active_dates)
            
            # 筛选出属于该到期日的所有合约链
            df_chain = df_und_day[df_und_day['last_edate'] == expiry_date].copy()
            
            # 计算 Max Pain
            max_pain, oi_call, oi_put = calculate_max_pain_for_chain(df_chain)
            
            if not pd.isna(max_pain):
                # 获取标的当日收盘价
                und_close = und_price_map.get((date, und_code), np.nan)
                
                # 计算 days to expiry (DTE) - 用交易日序列计算或者日历日计算
                # 这里使用日历日差值
                try:
                    dt_current = pd.to_datetime(date)
                    dt_expiry = pd.to_datetime(expiry_date)
                    dte = (dt_expiry - dt_current).days
                except Exception:
                    dte = np.nan
                
                records.append({
                    'trade_date': date,
                    'underlying_code': und_code,
                    'underlying_close': und_close,
                    'expiry_date': expiry_date,
                    'days_to_expiry': dte,
                    'max_pain_price': max_pain,
                    'total_oi_call': oi_call,
                    'total_oi_put': oi_put
                })
                
    if records:
        df_out = pd.DataFrame(records)
        df_out = df_out.sort_values(['underlying_code', 'trade_date']).reset_index(drop=True)
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"[SUCCESS] Calculated Max Pain history! Saved {len(df_out)} records to {OUTPUT_FILE}")
    else:
        print("[WARNING] No Max Pain records calculated.")

if __name__ == '__main__':
    main()
