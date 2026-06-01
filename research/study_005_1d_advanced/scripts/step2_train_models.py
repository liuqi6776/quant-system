"""
step2_train_models.py
基于 features_005.parquet，进行按月滚动的 Walk-Forward 训练。
1. 样本时间衰减: sample_weight = exp(-ln(2) * (days_diff / 252))
2. 双模型:
   - xgb_up: predict target_up (> 1%)
   - xgb_crash: predict target_crash_bin (T+1 intraday < -3%)
输出 -> predictions/predictions_005_wf.parquet
"""
import os, sys, time, gc, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
FEATURES_FILE = os.path.join(STUDY_DIR, 'data', 'features_005.parquet')
OUTPUT = os.path.join(STUDY_DIR, 'predictions', 'predictions_005_wf.parquet')
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

TRAIN_START = '20200101'
HALF_LIFE_DAYS = 252.0  # 半衰期 1 年 (约252个交易日)

def get_feature_cols(df):
    exclude_cols = {
        'ts_code', 'trade_date', 'ds', 'industry',
        'open', 'high', 'low', 'close', 'pre_close',
        'entry_price', 'next_open',
        'exit_price_1d', 'return_1d', 'return_1d_open',
        'exit_price_5d', 'return_5d', 'return_5d_open',
        'exit_price_28d', 'return_28d', 'return_28d_open',
        'exit_28d_close',
        'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
        'entry_vs_close',
        'return_1d_open_old', 'actual_return',
        't1_intraday_return', 'target_crash_bin', 'target_up_bin',
        'index_ma20_bias'
    }
    return [c for c in df.columns
            if c not in exclude_cols
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

def run():
    t0 = time.time()
    print("Loading features 005...", flush=True)
    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)
    
    # 构造 label_up
    df['label_up'] = (df['return_1d_open'] > 0.01).astype(np.int8)
    
    # 如果 target_crash_bin 有 NaN，处理一下
    df['label_crash'] = df['target_crash_bin'].fillna(0).astype(np.int8)

    feature_cols = get_feature_cols(df)
    print(f"Rows: {len(df)}, Features: {len(feature_cols)}", flush=True)

    months = sorted(df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= '202201']
    print(f"Predict months: {len(pred_months)} ({pred_months[0]}-{pred_months[-1]})", flush=True)

    all_preds = []
    
    for i, month in enumerate(pred_months):
        mt0 = time.time()
        train_end = str(int(month) - 1)
        if train_end.endswith('00'):
            train_end = f"{int(train_end[:4])-1}12"

        train_mask = (df['ds'] >= TRAIN_START) & (df['ds'].str[:6] <= train_end) & df['return_1d_open'].notna()
        pred_mask = df['ds'].str[:6] == month

        train_df = df.loc[train_mask, feature_cols + ['ds', 'label_up', 'label_crash']].copy()
        pred_df = df.loc[pred_mask, feature_cols + ['trade_date', 'ts_code', 'return_1d_open', 'next_open', 'industry']].copy()

        if len(train_df) < 50000 or len(pred_df) == 0:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_up = train_df['label_up']
        y_crash = train_df['label_crash']
        
        # --- Time-decayed Sample Weight 计算 ---
        # 假设当前预测月的第一天，或者简单用 train_end 的最后一天作为参考点
        # 为了计算简单，给每个样本分配一个相对日期序号 (基于行索引或简单按时间戳)
        train_dates = pd.to_datetime(train_df['ds'], format='%Y%m%d')
        ref_date = train_dates.max()
        days_diff = (ref_date - train_dates).dt.days
        # 指数衰减权重: exp(-lambda * t) where lambda = ln(2) / half_life
        decay_lambda = np.log(2) / HALF_LIFE_DAYS
        sample_weights = np.exp(-decay_lambda * days_diff).values
        # Normalize weights to have mean=1 so learning rate behaves consistently
        sample_weights = sample_weights / sample_weights.mean()

        # Model 1: UP Model
        model_up = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, eval_metric='logloss',
            tree_method='hist'
        )
        model_up.fit(X_train, y_up, sample_weight=sample_weights, verbose=False)
        
        # Model 2: CRASH Model
        model_crash = XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1, # Crash模型稍微浅一点防过拟合
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.0, # 虽然极不平衡，但我们看绝对概率，先不平衡
            random_state=43, n_jobs=4, eval_metric='logloss',
            tree_method='hist'
        )
        model_crash.fit(X_train, y_crash, sample_weight=sample_weights, verbose=False)

        X_pred = pred_df[feature_cols].fillna(0)
        prob_up = model_up.predict_proba(X_pred)[:, 1]
        prob_crash = model_crash.predict_proba(X_pred)[:, 1]

        mp = pd.DataFrame({
            'trade_date': pred_df['trade_date'].values,
            'ts_code': pred_df['ts_code'].values,
            'industry': pred_df['industry'].values,
            'prob_up': prob_up,
            'prob_crash': prob_crash,
            'actual_return': pred_df['return_1d_open'].values,
            'entry_price': pred_df['next_open'].values if 'next_open' in pred_df.columns else np.nan,
        })
        all_preds.append(mp)

        elapsed = time.time() - mt0
        total = time.time() - t0
        remaining = (total / (i + 1)) * (len(pred_months) - i - 1)
        print(f"[{i+1}/{len(pred_months)}] {month}: train={len(train_df)}, "
              f"up_pos={y_up.mean():.1%}, crash_pos={y_crash.mean():.1%}, "
              f"elapsed={elapsed:.0f}s, rem~{remaining/60:.0f}m", flush=True)

        del train_df, pred_df, X_train, y_up, y_crash, model_up, model_crash, X_pred, mp
        gc.collect()

    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_parquet(OUTPUT)
        print(f"\nSaved: {OUTPUT} ({len(combined)} rows)", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    run()
