# ETF Valuation-based Sizing & Trend-filtered Asset Allocation Strategy

This research implements a systematic asset allocation framework using **HS300 ETF** (represented by `000300.SH`), **ZZ500 ETF`** (represented by `000905.SH`), and **Treasury Bond ETF** (represented by `511010.SH`). 

Its core philosophy is to capture index beta with strict valuation discipline and trend-following rules, focusing entirely on **drawdown compression** and **bear market survival** rather than maximizing active return.

---

## 🧭 Strategy Design & Allocation Rules

1. **Core Assets**:
   - **HS300 ETF**: Large-cap core beta.
   - **ZZ500 ETF**: Mid-cap core beta.
   - **Treasury Bond ETF (Safe Asset)**: Safe-haven allocation for idle capital.
2. **Valuation-based Sizing (估值纪律)**:
   - Calculate the 5-year rolling quantile of PE and PB daily:
     $$Q_{i, t} = \frac{rank\_5y(PE_{i, t}) + rank\_5y(PB_{i, t})}{2}$$
   - The base weight allocated to each index is inversely proportional to its valuation quantile:
     $$W_{val, i, t} = 0.5 \times (1.0 - Q_{i, t})$$
   - This ensures that we naturally buy more when indices are cheap and scale back/profit-take when they are expensive.
3. **Trend Filter (趋势大铁闸)**:
   - Use the 250-day moving average (MA250) of close price as the long-term trend indicator.
   - If $Close_{i, t} < MA250_{i, t}$, the index is in a downtrend (catching falling knives). We set its weight to 0:
     $$W_{target, i, t} = W_{val, i, t} \times I(Close_{i, t} \ge MA250_{i, t})$$
4. **Rebalancing Logic & Friction Cost**:
   - We check the portfolio weights **weekly** (first trading day of each ISO week).
   - We only trigger execution if the absolute deviation of either index's actual holding weight from the target weight is greater than **10%**:
     $$|W_{current, i} - W_{target, i}| > 0.10$$
   - A transaction cost of **0.05%** is charged on all traded volume.
   - Idle capital is parked in the Treasury Bond ETF (yielding `511010.SH` daily returns, or 3.0% annualized before its listing).

---

## 📊 Backtest Performance (2015 - 2026)

### 1. 2015-2016 Stress Period (Circuit Breakers & Crash)
*A-shares experienced a historic bubble and crash, with indices falling over 50%.*

- **Valuation+Trend Strategy**: Total Return **+12.01%** | CAGR **5.88%** | Max Drawdown **-4.59%** | Volatility **4.53%**
- **HS300 Buy & Hold**: Total Return **-6.33%** | CAGR **-3.24%** | Max Drawdown **-46.70%** | Volatility **31.94%**
- **ZZ500 Buy & Hold**: Total Return **+17.68%** | CAGR **8.55%** | Max Drawdown **-54.35%** | Volatility **38.23%**
- **Static 50/50**: Total Return **+3.49%** | CAGR **1.74%** | Max Drawdown **-50.31%** | Volatility **33.90%**

### 2. Full Period (2015-2026)
*Full cycle coverage of A-share volatility.*

- **Valuation+Trend Strategy**: Total Return **+38.26%** | CAGR **2.95%** | Max Drawdown **-15.70%** | Volatility **6.60%**
- **HS300 Buy & Hold**: Total Return **+33.31%** | CAGR **2.61%** | Max Drawdown **-46.70%** | Volatility **21.56%**
- **ZZ500 Buy & Hold**: Total Return **+62.67%** | CAGR **4.46%** | Max Drawdown **-65.20%** | Volatility **25.34%**
- **Static 50/50**: Total Return **+47.65%** | CAGR **3.56%** | Max Drawdown **-54.98%** | Volatility **22.58%**

---

## 🔍 Behavior Audit: Why the Strategy Succeeds

- **Downtrend Defense**: During downtrends (Close < MA250), the average equity weight is compressed to just **7.60%**, shielding the portfolio from the index's -50% crashes.
- **Uptrend tracking**: During uptrends (Close >= MA250), the average equity weight rises to **35.77%**. The correlation with market index returns is **0.7217** (Beta of **0.2609**), capturing the uptrend beta smoothly.
- **Drawdown Capped**: Across the entire 11-year history, the maximum drawdown was restricted to **-15.70%**, creating a portfolio that investors can hold comfortably even in the deepest bear markets.

---

## 🛠️ How to Run
1. **Download Data**:
   ```bash
   python etf-valuation-strategy/scripts/step1_download_data.py
   ```
2. **Run Backtest**:
   ```bash
   python etf-valuation-strategy/scripts/step2_backtest.py
   ```
3. **Analyze Strategy Behavior**:
   ```bash
   python etf-valuation-strategy/scripts/step3_analyze_beta.py
   ```
