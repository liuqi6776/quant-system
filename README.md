# 量化交易系统 V2

基于 Python 的 A股量化交易系统，重构自 quant_trading_system。

## 功能特性

- 📈 **数据获取**: 从 Tushare 获取 A股行情、技术指标、筹码分布等数据
- 📊 **特征工程**: Alpha 101因子、技术指标、筹码分析
- 🎯 **机器学习**: XGBoost 模型训练与预测
- 🔄 **回测系统**: 策略验证与绩效分析

## 项目结构

```
quant_system_v2/
├── config/         # 配置管理
├── data/           # 数据获取与存储
├── features/       # 特征工程
├── processing/     # 数据处理
├── models/         # 模型训练与预测
├── backtesting/    # 回测引擎
├── utils/          # 工具函数
└── main.py         # 主入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

创建 `.env` 文件或设置环境变量：

```bash
TUSHARE_TOKEN=your_token_here
TUSHARE_API_URL=http://tsapi.majors.ltd:7000
DATA_PATH=D:/iquant_data/data_v2
```

### 3. 运行

```python
from data.fetcher import DataFetcher
from processing.pipeline import DataPipeline
from models.trainer import ModelTrainer

# 获取数据
fetcher = DataFetcher()
fetcher.fetch_all(end_date="20251118")

# 处理数据
pipeline = DataPipeline()
df = pipeline.run(start_date="20200103", end_date="20251118")

# 训练模型
trainer = ModelTrainer()
trainer.train(df)
```

## 依赖

- Python 3.7+
- pandas, numpy
- tushare
- xgboost, scikit-learn
- joblib, tqdm

## License

MIT
