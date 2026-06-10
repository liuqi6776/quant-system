"""
step1_prepare_labels.py
基于 004 基础特征文件构建长期多因子研究的收益率标签与超额收益率（Market-relative & Industry-relative）。
支持的持有期限：5日、10日、20日。
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# 原始 004 数据路径
SRC_FEAT = r'C:\Users\liuqi\quant_system_v2\research\study_004_1d_release\data\all_features_v2.parquet'
OUT_FEAT = os.path.join(DATA_DIR, 'features_longterm.parquet')

def prepare_labels():
    print(f"Loading base features from {SRC_FEAT}...")
    df = pd.read_parquet(SRC_FEAT)
    df['trade_date'] = df['trade_date'].astype(str)
    
    # 排序以保证 shift 逻辑的正确性
    print("Sorting data by ts_code and trade_date...")
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 加载行业映射
    ind_file = os.path.join(DATA_DIR, 'industry_map.csv')
    if os.path.exists(ind_file):
        print("Merging industry map...")
        ind_df = pd.read_csv(ind_file)
        if 'industry' in df.columns:
            df = df.drop(columns=['industry'])
        df = df.merge(ind_df[['ts_code', 'industry']], on='ts_code', how='left')
        df['industry'] = df['industry'].fillna('Unknown')
    else:
        print("⚠️ Industry map file not found, using 'Unknown' for all stocks.")
        df['industry'] = 'Unknown'
        
    print("Calculating forward returns for 5d, 10d, and 20d...")
    # T+1 开盘买入
    df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
    
    for N in [5, 10, 20]:
        print(f"Calculating {N}-day forward return...")
        # T+N 收盘卖出
        df[f'close_T{N}'] = df.groupby('ts_code')['close'].shift(-N)
        
        # 收益率计算： (close_t+N - open_t+1) / open_t+1
        df[f'ret_{N}d'] = (df[f'close_T{N}'] - df['next_open']) / df['next_open']
        
        # 过滤掉无法计算收益率的边界行 (如临近数据集末尾的行)
        # 如果 return_1d_open 为空，通常表示股票在该日期后已停牌或退市或已到数据集末尾
        # 这里用 next_open 或 close_TN 的空值来判定
        
        # 1. 市场超额收益率：相对于每日所有股票平均收益率的超额
        print(f"Calculating market-relative excess return for {N}d...")
        mkt_mean = df.groupby('trade_date')[f'ret_{N}d'].transform('mean')
        df[f'mkt_excess_ret_{N}d'] = df[f'ret_{N}d'] - mkt_mean
        
        # 2. 行业超额收益率：相对于每日同行业股票平均收益率的超额
        print(f"Calculating industry-relative excess return for {N}d...")
        ind_mean = df.groupby(['trade_date', 'industry'])[f'ret_{N}d'].transform('mean')
        df[f'ind_excess_ret_{N}d'] = df[f'ret_{N}d'] - ind_mean
        
        # 打印部分统计数据
        valid_rows = df[f'ret_{N}d'].notna().sum()
        print(f"  {N}d valid rows: {valid_rows}, Mean return: {df[f'ret_{N}d'].mean():.4f}, Std: {df[f'ret_{N}d'].std():.4f}")
        
    print("Saving enriched feature file to", OUT_FEAT)
    df.to_parquet(OUT_FEAT, index=False)
    print("Enriched feature file saved successfully!")

if __name__ == '__main__':
    prepare_labels()
