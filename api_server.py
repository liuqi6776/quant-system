"""
api_server.py — 本地量化引擎 API 服务
连接云端 GitHub Pages Dashboard 与本地计算环境

启动方式 (外网可访问):
    python api_server.py
    # 或
    uvicorn api_server:app --host 0.0.0.0 --port 8000

密码验证: 所有接口需在 Header 中带 X-API-Key: liuqe
测试:
    curl -H "X-API-Key: liuqe" http://localhost:8000/health
"""
import os, json, sys, subprocess, logging, re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ───────────────────────────── 路径配置 ─────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
JIAYO_DIR       = BASE_DIR / "jiayo-analysis"
STUDY_DIR       = BASE_DIR / "research/study_004_1d_release"
PREDICTIONS_FILE = STUDY_DIR / "predictions/predictions_1d_open_wf_monthly.parquet"
TRAIN_SCRIPT    = STUDY_DIR / "scripts/train_1d_open.py"
NEWS_DIR        = Path("D:/iquant_data/data_v2/news_major1")

# ───────────────────────────── 参数 ─────────────────────────────────
API_PASSWORD    = "liuqe"
PROB_THRESHOLD  = 0.50
MAX_POSITIONS   = 3
GAP_UP_LOW      = 0.02
GAP_UP_HIGH     = 0.06
STOP_LOSS       = -0.05
TAKE_PROFIT     = 0.05

# ───────────────────────────── App ──────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Quant Local Engine API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://liuqi6776.github.io",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5500", "http://127.0.0.1:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────── 密码验证依赖 ───────────────────────────────
def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != API_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid API Key. Please set X-API-Key header.")
    return x_api_key


# ─────────────────────── 数据模型 ───────────────────────────────────
class PipelineRequest(BaseModel):
    target_date: str          # YYYY-MM-DD
    news_json: Optional[str] = None  # 已经解析好的 JSON 字符串

class ParseHtmlRequest(BaseModel):
    target_date: str          # YYYY-MM-DD
    raw_html: Optional[str] = None    # HTML 原文
    raw_markdown: Optional[str] = None  # Markdown 原文（二选一）
    save_to_disk: bool = True  # 是否保存到 D:/iquant_data/data_v2/news_major1/


# ─────────────────────── 工具函数 ───────────────────────────────────
def normalize_date(date_str: str) -> str:
    return date_str.replace("-", "")


def get_signals_for_date(date_norm: str) -> list:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"预测文件不存在: {PREDICTIONS_FILE}")
    df = pd.read_parquet(PREDICTIONS_FILE, columns=["trade_date", "ts_code", "prob", "entry_price"])
    df["trade_date"] = df["trade_date"].astype(str)
    today = df[df["trade_date"] == date_norm].copy()
    if today.empty:
        return []
    candidates = today[today["prob"] >= PROB_THRESHOLD].nlargest(MAX_POSITIONS, "prob").reset_index(drop=True)
    results = []
    for _, row in candidates.iterrows():
        ep = row["entry_price"] if pd.notna(row["entry_price"]) else None
        results.append({
            "ts_code":     row["ts_code"],
            "prob":        round(float(row["prob"]), 4),
            "entry_price": round(float(ep), 2) if ep else None,
            "action":      f"T+1 集合竞价后判断：若高开幅度在 {GAP_UP_LOW*100:.0f}%~{GAP_UP_HIGH*100:.0f}%，以开盘价买入",
            "stop_loss":   f"{STOP_LOSS*100:.0f}%",
            "take_profit": f"+{TAKE_PROFIT*100:.0f}%",
        })
    return results


def extract_content_from_html(html: str) -> tuple[str, str]:
    """从 HTML 中提取 title 和纯文本 content"""
    title_m = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else "Unknown"
    title = re.sub(r'\s*[-_|]\s*韭研公社\s*$', '', title)

    # 优先从 text-box 抽取正文
    body_m = re.search(r'<div[^>]*class="[^"]*text-box[^"]*"[^>]*>(.*?)</div>\s*<div[^>]*class="[^"]*reward', html, re.DOTALL)
    if body_m:
        body = body_m.group(1)
    else:
        body_m2 = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
        body = body_m2.group(1) if body_m2 else html

    # strip scripts/styles
    body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.DOTALL)
    body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL)
    body = re.sub(r'<[^>]+>', ' ', body)
    body = re.sub(r'&nbsp;', ' ', body)
    body = re.sub(r'&amp;', '&', body)
    body = re.sub(r'\s+', ' ', body).strip()

    return title, body


def analyze_with_zhipu(title: str, content: str) -> Optional[dict]:
    """调用 ZhipuAI GLM 分析文章，复用 jiayo-analysis/analyzer.py 的逻辑"""
    try:
        from zhipuai import ZhipuAI
        API_KEY = "7c406ccb126c48e28758c255b9aede76.nTXKzG8O0EKO9YE9"
        client = ZhipuAI(api_key=API_KEY)
    except ImportError:
        raise HTTPException(status_code=500, detail="zhipuai 未安装，请运行: pip install zhipuai")

    prompt = f"""你是一个专业的A股市场分析师。请分析以下文章内容，评估其对市场、板块和个股的影响。

文章标题：{title}

文章内容：
{content[:5000]}

请以JSON格式返回分析结果，格式如下：
{{
    "market_impact": 利好利空程度(-5到+5，负数表示利空，正数表示利好，0表示中性),
    "market_analysis": "对大盘影响的简要说明（50字以内）",
    "sectors": [
        {{
            "sector_name": "板块名称",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }}
    ],
    "stocks": [
        {{
            "stock_name": "股票名称",
            "stock_code": "股票代码（如有）",
            "impact": 利好利空程度(-5到+5),
            "analysis": "简要说明（30字以内）"
        }}
    ]
}}

只返回JSON，不要其他内容。"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error attempt {attempt+1}: {e}")
        except Exception as e:
            logger.warning(f"ZhipuAI error attempt {attempt+1}: {e}")
    return None


def get_data_status() -> dict:
    status = {}
    if PREDICTIONS_FILE.exists():
        df = pd.read_parquet(PREDICTIONS_FILE, columns=["trade_date"])
        dates = df["trade_date"].astype(str)
        status["predictions"] = {"exists": True, "rows": len(df), "date_min": dates.min(), "date_max": dates.max()}
    else:
        status["predictions"] = {"exists": False}
    return status


# ─────────────────────── 接口定义 ───────────────────────────────────
@app.get("/health")
def health_check(api_key: str = Depends(verify_api_key)):
    """连通性测试 — 需要 X-API-Key 验证"""
    return {
        "status": "ok",
        "msg": "Local Quant Engine is Online ✓",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_status": get_data_status(),
    }


@app.get("/api/v1/signals")
def get_signals(date: str, api_key: str = Depends(verify_api_key)):
    """获取指定日期的交易信号。date 格式: 20260509 或 2026-05-09"""
    date_norm = normalize_date(date)
    try:
        signals = get_signals_for_date(date_norm)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "ok", "date": date_norm, "signals": signals, "signal_count": len(signals)}


@app.post("/api/v1/parse/html")
async def parse_html_to_json(payload: ParseHtmlRequest, api_key: str = Depends(verify_api_key)):
    """
    将 raw HTML 或 Markdown 原文解析成 news_major1 JSON 格式。
    流程: 提取正文 → 调用 ZhipuAI GLM 分析 → 返回 JSON（可选保存到本地）
    """
    date_fmt = payload.target_date  # YYYY-MM-DD

    if not payload.raw_html and not payload.raw_markdown:
        raise HTTPException(status_code=400, detail="raw_html 或 raw_markdown 必须提供其中一个")

    # 1. 提取正文
    if payload.raw_html:
        title, content = extract_content_from_html(payload.raw_html)
    else:
        # Markdown: 直接去掉 markdown 标记符作为 content
        md = payload.raw_markdown
        title_m = re.search(r'^#+ (.+)$', md, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else "盘前纪要"
        content = re.sub(r'#+ ', '', md)
        content = re.sub(r'\*+', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

    logger.info(f"Parsing article: '{title}', content length={len(content)}")

    if len(content) < 100:
        raise HTTPException(status_code=422, detail=f"内容提取失败或过短（{len(content)}字），请检查输入格式")

    # 2. ZhipuAI 分析
    result = analyze_with_zhipu(title, content)
    if not result:
        raise HTTPException(status_code=502, detail="ZhipuAI 分析失败（3次重试均未能返回有效JSON）")

    result["article_title"] = title
    result["article_date"] = date_fmt

    # 3. 保存到本地
    saved_path = None
    if payload.save_to_disk:
        try:
            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = NEWS_DIR / f"analysis_{date_fmt}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            saved_path = str(out_path)
            logger.info(f"Saved analysis to: {saved_path}")
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    return {
        "status": "ok",
        "date": date_fmt,
        "title": title,
        "content_length": len(content),
        "saved_to": saved_path,
        "analysis": result,
    }


@app.post("/api/v1/pipeline/run")
async def run_pipeline(payload: PipelineRequest, api_key: str = Depends(verify_api_key)):
    """
    主流程接口:
    1. 保存 news JSON 到本地（如果提供）
    2. 读取当日预测信号并返回
    """
    date_raw  = payload.target_date
    date_norm = normalize_date(date_raw)
    date_fmt  = f"{date_raw[:4]}-{date_raw[5:7]}-{date_raw[8:10]}"

    steps = []

    if payload.news_json:
        try:
            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            news_path = NEWS_DIR / f"analysis_{date_fmt}.json"
            news_path.write_text(payload.news_json, encoding="utf-8")
            steps.append(f"✓ 新闻JSON已保存: {news_path}")
        except Exception as e:
            steps.append(f"⚠ 新闻保存失败: {e}")
    else:
        steps.append("ℹ 本次未附带新闻JSON")

    try:
        signals = get_signals_for_date(date_norm)
        steps.append(f"✓ 信号读取完成: {len(signals)} 个候选标的")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"预测文件缺失: {e}")

    return {
        "status": "success",
        "date": date_norm,
        "steps": steps,
        "signal_count": len(signals),
        "results": signals,
    }


@app.post("/api/v1/retrain/trigger")
async def trigger_retrain(api_key: str = Depends(verify_api_key)):
    """后台触发月度 retrain（异步，预计 30-60 分钟）"""
    subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(BASE_DIR),
        stdout=open(BASE_DIR / "retrain.log", "w"),
        stderr=subprocess.STDOUT,
    )
    return {"status": "started", "msg": "Retrain 已在后台启动，查看 retrain.log 获取进度"}


# ─────────────────────── 启动入口 ───────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Quant Local Engine  |  0.0.0.0:8000")
    print("  Password: X-API-Key: liuqe")
    print("  Docs:  http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
