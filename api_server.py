"""
api_server.py — 本地量化引擎 API 服务
连接云端 GitHub Pages Dashboard 与本地计算环境

启动方式:
    pip install fastapi uvicorn pandas pyarrow
    python api_server.py

或用 reload 模式（开发调试）:
    uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
"""
import os, json, subprocess, sys, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ───────────────────────────── 路径配置 ─────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

PREDICTIONS_FILE   = BASE_DIR / "research/study_004_1d_release/predictions/predictions_1d_open_wf_monthly.parquet"
FEATURES_FILE      = BASE_DIR / "research/study_004_1d_release/data/all_features_v2.parquet"
STEP1_SCRIPT       = BASE_DIR / "research/study_004_1d_release/scripts/step1_build_features.py"
FIX_TARGET_SCRIPT  = BASE_DIR / "research/study_004_1d_release/scripts/fix_1d_target.py"
TRAIN_SCRIPT       = BASE_DIR / "research/study_004_1d_release/scripts/train_1d_open.py"
NEWS_DIR           = Path("D:/iquant_data/data_v2/news_major1")

# 实盘参数（与 backtest 中最优 combo 一致）
PROB_THRESHOLD  = 0.50
MAX_POSITIONS   = 3
GAP_UP_LOW      = 0.02
GAP_UP_HIGH     = 0.06
STOP_LOSS       = -0.05
TAKE_PROFIT     = 0.05

# ───────────────────────────── FastAPI App ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quant Local Engine API",
    description="本地量化引擎，供 GitHub Pages Dashboard 远程调用",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://liuqi6776.github.io",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",   # VS Code Live Server
        "http://127.0.0.1:5500",
        "null",                    # 本地直接打开 HTML 文件
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────── 数据模型 ─────────────────────────────
class PipelineRequest(BaseModel):
    target_date: str          # 格式 YYYY-MM-DD，对应 T 日
    news_data: Optional[str] = None   # news_major1 JSON 字符串（可选）


class SignalRequest(BaseModel):
    target_date: str          # 格式 YYYYMMDD 或 YYYY-MM-DD


# ───────────────────────────── 工具函数 ─────────────────────────────
def normalize_date(date_str: str) -> str:
    """统一日期格式为 YYYYMMDD"""
    return date_str.replace("-", "")


def get_signals_for_date(date_str: str) -> list[dict]:
    """
    从预测文件中取指定日期的 top 候选标的
    date_str: YYYYMMDD 格式
    """
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"预测文件不存在: {PREDICTIONS_FILE}")

    df = pd.read_parquet(PREDICTIONS_FILE, columns=["trade_date", "ts_code", "prob", "entry_price"])
    df["trade_date"] = df["trade_date"].astype(str)

    today = df[df["trade_date"] == date_str].copy()
    if today.empty:
        return []

    # 按概率降序，取 top-N
    candidates = (
        today[today["prob"] >= PROB_THRESHOLD]
        .nlargest(MAX_POSITIONS, "prob")
        .reset_index(drop=True)
    )

    results = []
    for _, row in candidates.iterrows():
        entry_price = row["entry_price"] if pd.notna(row["entry_price"]) else None
        results.append({
            "ts_code":     row["ts_code"],
            "prob":        round(float(row["prob"]), 4),
            "entry_price": round(float(entry_price), 2) if entry_price else None,
            "action":      f"T+1 集合竞价后判断高开幅度：若 {GAP_UP_LOW*100:.0f}% < 高开 < {GAP_UP_HIGH*100:.0f}%，以开盘价买入",
            "stop_loss":   f"{STOP_LOSS*100:.0f}%",
            "take_profit": f"+{TAKE_PROFIT*100:.0f}%",
        })

    return results


def get_data_status() -> dict:
    """返回本地数据文件的状态信息"""
    status = {}

    if PREDICTIONS_FILE.exists():
        df = pd.read_parquet(PREDICTIONS_FILE, columns=["trade_date"])
        dates = df["trade_date"].astype(str)
        status["predictions"] = {
            "exists": True,
            "rows": len(df),
            "date_min": dates.min(),
            "date_max": dates.max(),
        }
    else:
        status["predictions"] = {"exists": False}

    if FEATURES_FILE.exists():
        import os
        mtime = os.path.getmtime(FEATURES_FILE)
        status["features"] = {
            "exists": True,
            "last_modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
        }
    else:
        status["features"] = {"exists": False}

    return status


# ───────────────────────────── 接口定义 ─────────────────────────────
@app.get("/health")
def health_check():
    """连通性测试 — Dashboard 的 Test Ping 按钮调用此接口"""
    return {
        "status": "ok",
        "msg": "Local Quant Engine is Online ✓",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_status": get_data_status(),
    }


@app.get("/api/v1/signals")
def get_signals(date: str):
    """
    获取指定日期的交易信号（GET 方式，便于测试）
    参数: ?date=20260509 或 ?date=2026-05-09
    """
    date_norm = normalize_date(date)
    try:
        signals = get_signals_for_date(date_norm)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "status": "ok",
        "date": date_norm,
        "threshold": PROB_THRESHOLD,
        "max_positions": MAX_POSITIONS,
        "signals": signals,
        "signal_count": len(signals),
    }


@app.post("/api/v1/pipeline/run")
async def run_pipeline(payload: PipelineRequest):
    """
    主流程接口（POST）
    1. 保存 news JSON 到本地
    2. 读取当日预测信号并返回
    （月度 retrain 由定时任务或手动触发，不在此实时运行）
    """
    date_raw  = payload.target_date                  # YYYY-MM-DD
    date_norm = normalize_date(date_raw)             # YYYYMMDD
    date_fmt  = f"{date_raw[:4]}-{date_raw[5:7]}-{date_raw[8:10]}"  # YYYY-MM-DD

    steps = []

    # Step 1: 保存 news JSON
    if payload.news_data:
        try:
            news_path = NEWS_DIR / f"analysis_{date_fmt}.json"
            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            news_path.write_text(payload.news_data, encoding="utf-8")
            steps.append(f"✓ 新闻数据已保存: {news_path}")
            logger.info(f"Saved news for {date_fmt} → {news_path}")
        except Exception as e:
            steps.append(f"⚠ 新闻保存失败: {e}")
            logger.warning(f"News save failed: {e}")
    else:
        steps.append("ℹ 本次未附带新闻数据")

    # Step 2: 读取当日预测信号
    try:
        signals = get_signals_for_date(date_norm)
        steps.append(f"✓ 信号读取完成: {len(signals)} 个候选标的")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"预测文件缺失: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信号读取失败: {e}")

    return {
        "status": "success",
        "date": date_norm,
        "steps": steps,
        "signal_count": len(signals),
        "results": signals,
        "note": "月度模型 retrain 需手动触发 train_1d_open.py，不在每日请求中执行（耗时 30-60 分钟）",
    }


@app.post("/api/v1/retrain/trigger")
async def trigger_retrain(background: bool = True):
    """
    手动触发月度 retrain（仅在月底需要时调用）
    background=True: 后台异步执行（推荐）
    background=False: 同步等待（会超时，仅用于测试）
    """
    if background:
        # 非阻塞后台执行
        subprocess.Popen(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(BASE_DIR),
            stdout=open(BASE_DIR / "retrain.log", "w"),
            stderr=subprocess.STDOUT,
        )
        return {"status": "started", "msg": "Retrain 已在后台启动，预计 30-60 分钟完成。查看 retrain.log 获取进度。"}
    else:
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=3600,
        )
        return {"status": "done", "returncode": result.returncode, "stdout": result.stdout[-2000:]}


# ───────────────────────────── 启动入口 ─────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  Quant Local Engine API Server")
    print("  http://127.0.0.1:8000")
    print("  Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
