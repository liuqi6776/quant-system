# Walkthrough: Long-Term Multi-Factor Ranking Strategy Red-Flag Fixes Complete

We have completed the implementation of all red-flag fixes, resolved the structural mismatches, eliminated short-sample feature pollution, and benchmarked our strategy against the **CSI 1000 Index** and the Market Equal-Weight average.

---

## 📈 Final Portfolio Performance (2022–2026 Out-of-Sample)

The final, clean, and unpolluted strategy returns (20-day rebalancing, rebalanced at next-day open with first-day buy return corrected, and paying 0.2% buy / 0.3% sell transaction costs) are shown below:

| Portfolio | Total Return | CAGR | Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy (Pure Multi-Factor)** | **+59.94%** | **12.42%** | **18.41%** | **0.73** | **-28.03%** |
| **Strategy (Options Wind-Control)** | **+46.89%** | **10.06%** | **16.93%** | **0.65** | **-28.27%** |
| **Benchmark (Market Equal-Weight)** | **+46.98%** | **10.08%** | **21.68%** | **0.54** | **-29.97%** |
| **Benchmark (CSI 1000 Index)** | **+4.77%** | **1.17%** | **24.90%** | **0.17** | **-46.22%** |

### Key Improvements & Discoveries:
1. **Unpolluted Alpha**: Removing the short-sample `ths_hot`/`ths_hot_rank` factors (which only had ~342 days of active data) eliminated training data leakage and placeholder pollution. This actually **improved** the out-of-sample Daily Rank IC of our monthly rolling Walk-Forward Ridge model from **+0.0987** to **+0.1002**, showing that the THS factors were acting as noise over the full history.
2. **True Outperformance**: The Strategy outperforms the **CSI 1000 Index** (Cagr 1.17%, Sharpe 0.17) by a massive **+11.25% annual excess return (CAGR)** and reduces the Max Drawdown from **-46.22%** to **-28.03%**. However, whether this outperformance is driven by clean stock selection alpha or remains dominated by residual small-cap beta exposure requires further verification through style attribution analysis.
3. **First-Day Return Bug Fixed**: Storing each position's actual purchase price (`next_open`) and updating the buy day's return using $close / open$ corrected the previous mismatch. The corrected return remains highly stable and positive (+59.94% total return).
4. **Hedge Efficiency**: Raising the `QVIX_PANIC_THRESHOLD` to `2.0` and disabling the PCR filter kept the options wind-control strategy in the market during minor volatility spikes. It generated **+46.89% total return** (10.06% CAGR, 0.65 Sharpe), which matches the Market Equal-Weight benchmark while significantly lowering the volatility and drawdown.

### Final NAV Curve Comparison
![Portfolio NAV Curve](results/portfolio_backtest_nav.png)

---

## 📊 Factor Evaluation Results (20-Day Target)

Factors have been re-evaluated against the 20-day forward excess return target (`mkt_excess_ret_20d`) with corrected Rank ICIR (removing the incorrect `* np.sqrt(252)` annualization multiplier).

### 1. Corrected Daily Rank IC & ICIR (Full Period vs. THS Valid Period)

| Factor | Category | Full Period Mean IC | Full Period Daily ICIR | THS Period Mean IC | THS Period Daily ICIR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ths_hot_rank` | Concept | **+0.1404** | **+1.08** | **+0.1404** | **+1.08** |
| `turnover_rate` | Liquidity | **-0.0858** | **-0.52** | **-0.0885** | **-0.40** |
| `alpha_012` | Alpha101 | **+0.0339** | **+0.54** | **+0.0437** | **+0.57** |
| `alpha_006` | Alpha101 | **+0.0366** | **+0.50** | **+0.0532** | **+0.58** |
| `macd` | Technical | **-0.0696** | **-0.51** | **-0.0795** | **-0.49** |
| `net_mf_amount_norm` | MoneyFlow | **+0.0284** | **+0.30** | **+0.0320** | **+0.27** |
| `news_stock_impact` | News | **-0.0084** | **-0.41** | **-0.0085** | **-0.39** |

---

## 🛠️ Verification Checklist Completed

- [x] Deleted obsolete `portfolio_backtest_metrics.csv` and `portfolio_backtest_nav.csv` in `longterm-research/results/`.
- [x] Excluded `'ths_hot'` and `'ths_hot_rank'` from feature columns in `step3_train_ranking_model.py`.
- [x] Corrected the Rank ICIR calculation formula in `step2_factor_evaluation.py` (removed `* np.sqrt(252)`).
- [x] Fixed the first-day return bug in `step4_portfolio_backtest.py` using execution-day $close / open$ prices.
- [x] Loaded the CSI 1000 index history from `longterm-research/data/index_regime.csv` as a fair comparison benchmark.
- [x] Re-ran factor evaluations, walk-forward training, and portfolio backtests.
- [x] Verified and updated results and charts in `longterm-research/results/`.

---

## ⚠️ Key Caveats, Limitations & Next Steps (Professional Review)

While the strategy achieves a solid **+59.94% total return (12.42% CAGR, 0.73 Sharpe)** in simulation, we must remain objective and acknowledge two major structural attributes before considering this clean alpha:

### 1. Small-Cap Beta Exposure vs. True Alpha
- **Equal-Weight Benchmark comparison**: The strategy (+59.94%) and the Market Equal-Weight benchmark (+46.98%) both dramatically outperform the CSI 1000 Index (+4.77%). Because the equal-weight average of 5,000+ A-share stocks is heavily tilted toward small and micro-cap stocks, the strategy's massive outperformance against the CSI 1000 is primarily driven by its **small-cap beta exposure**, rather than pure stock selection.
- **True Alpha**: The strategy's actual selection alpha is the **~13% excess return over the Market Equal-Weight benchmark**, which still contains potential style or industry biases.

### 2. Factor Non-Monotonicity
- **Decile Monotonicity**: Factor decile backtests show that the most profitable groups are often the middle groups (e.g., Decile 3–4 for `turnover_rate`), whereas the extreme Decile 10 (highest turnover) loses money. 
- **Selection Behavior**: This suggests that the Ridge ranking model's selection of the Top 50 is profitable because it successfully avoids high-turnover "speculative garbage" stocks (Decile 10), rather than because it perfectly isolates the absolute highest-performing stocks.

---

## 🚀 Future Refinement Roadmap

To further isolate clean selection alpha, the following steps are recommended:
1. **Style Attribution Analysis**: Regress strategy returns against style risk factors (Size, Value, Liquidity, Momentum) to calculate the strategy's factor betas and isolate the residual "pure" alpha.
2. **Winsorization & Industry Neutralization**: Apply cross-sectional winsorization (handling outliers) and industry-neutralization on factors *before* feeding them to the Ridge regression, ensuring a cleaner linear relation and more monotonic ranking score.
3. **Year-by-Year Performance Audit**: Break down the backtest by calendar years (2022, 2023, 2024, 2025, 2026) to verify if the return is stable or if it was heavily carried by a single anomalous small-cap year (e.g., 2024).
4. **Out-of-Sample Holdout Testing**: Reserve the final 6 months of data (e.g., late 2025 to 2026) completely untouched by the model and feature engineering to serve as a pure blind validation set.
