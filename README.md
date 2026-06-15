# Systematic Macro-Momentum Strategy: ML + Inverse Volatility Risk Parity

> Predicting cross-sectional stock relative strength with a Random Forest trained on price technicals + macroeconomic news sentiment, then allocating capital with an inverse-volatility risk parity engine — backtested and deployed as a live Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project is an end-to-end systematic equity trading pipeline that combines **machine learning** with **risk-parity portfolio construction** to isolate downside-risk-adjusted alpha.

It works in two deliberately separate stages:

1. **Alpha Generation** — A Random Forest Regressor (300 trees) is trained on a zero-centered percentile-rank target, learning non-linear interactions between cross-sectional technical z-scores and macroeconomic NLP sentiment to forecast each stock's relative forward strength.
2. **Risk Parity Allocation** — The model only decides *which* assets to hold. *How much* capital each one gets is decided independently using **Inverse Volatility Sizing**, so every position contributes roughly equal marginal risk to the portfolio.

In out-of-sample backtesting (2024), the Top-5 portfolio achieved a **30.07% CAGR** and a **1.48 Sortino Ratio**, outperforming an equal-weight benchmark while running at **lower annualized volatility**.

---

## Pipeline Architecture

```
[ Raw Data Source ] ──> [ Feature Engineering ] ──> [ Walk-Forward ML Engine ]
                                                             │
                                                             ▼
[ Interactive Dashboard ] <── [ Vectorized Backtester ] <── [ Predictions Matrix ]
```

| Stage | What it does |
|---|---|
| **Raw Data Source** | Ingests daily OHLCV price data for a fixed basket of large-cap, liquid equities, plus a daily NLP-derived macroeconomic sentiment series. |
| **Feature Engineering** | Cleans, aligns, and transforms raw data into log returns, cross-sectional z-scores, macro-sentiment interaction terms, and the zero-centered rank target. |
| **Walk-Forward ML Engine** | Trains/validates the Random Forest with a 60-day embargo window to prevent data leakage between train and test sets. |
| **Predictions Matrix** | Daily cross-sectional rank forecast for every asset in the universe. |
| **Vectorized Backtester** | Selects the Top-N ranked assets and sizes them via inverse volatility weighting. |
| **Interactive Dashboard** | Streamlit + Plotly app for live PnL tracking and equity curve visualization. |

---

## Data Pipeline & Feature Engineering

### Data Sources
- **Price data:** Daily OHLCV for a fixed universe of large-cap, highly liquid equities (to avoid micro-cap volatility skew).
- **Sentiment data:** Daily macroeconomic sentiment scores derived from financial news via NLP.

### Cleaning Protocol
- **Timestamp normalization** — all series standardized to a uniform daily frequency.
- **Forward-fill imputation** — missing values are filled using `ffill` only, ensuring the model never sees information that wasn't actually available at that point in time (no look-ahead bias).
- **Left-join merge** — the per-asset price panel is merged with the global sentiment series on the date key, preserving each asset's full history.

### Feature Transformations

**1. Logarithmic returns** (for stationarity):

$$R_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

**2. Cross-sectional z-scores** (relative strength vs. the universe):

$$Z_{i,t} = \frac{R_{i,t} - \mu_t}{\sigma_t}$$

**3. Macro-interaction features** — sentiment × volatility and sentiment × z-score, so the model can learn regime-dependent behavior (e.g., how high-beta names behave in negative macro environments).

**4. Zero-centered rank target** — the model predicts relative rank, not absolute return:

$$y_{i,t} = \frac{\text{Rank}(R_{i,t+1})}{N} - 0.5$$

---

## Model: Random Forest Regressor

- **300 decision trees**, trained via bagging + feature subsampling (decorrelating individual trees and reducing ensemble variance).
- **60-day embargo window** between training and out-of-sample test data to prevent leakage and serial correlation.

### Daily Prediction Mechanism

Each morning, the pipeline ingests the previous day's closing prices and the latest macro sentiment, recomputes the feature set for every stock, and runs it through all 300 trees. The final prediction is the average of all tree outputs:

$$\hat{y}_{i,t} = \frac{1}{300}\sum_{k=1}^{300}\hat{y}_{i,t}^{(k)}$$

```
                [ New Daily Stock Data Input ]
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    [ Cross-Sectional Z          [ Sentiment × Z-Score
        > 1.5? ]                       > 0.05? ]
        ┌───┴───┐                     ┌───┴───┐
        ▼       ▼                     ▼       ▼
      [Yes]    [No]                 [Yes]    [No]
        │       │                     │       │
        ▼       ▼                     ▼       ▼
   (Check MACD) ...              (Check Lags) ...
        │                             │
        └──────────────┬──────────────┘
                       ▼
[ Average of 300 Trees ] ──> Final Forward Return Prediction
```

---

## Portfolio Construction: Inverse Volatility Risk Parity

Each trading day, the Random Forest ranks the universe and the backtester selects the **Top N** assets. Instead of equal weighting (which lets one volatile asset dominate portfolio risk), capital is allocated **inversely proportional to each asset's trailing 14-day volatility**:

$$w_i = \frac{\frac{1}{\sigma_i}}{\sum_{j=1}^{N}\frac{1}{\sigma_j}}$$

This equalizes each position's marginal contribution to portfolio risk — calmer assets get larger allocations, volatile ones get sized down.

---

## Out-of-Sample Results (2024)

| Portfolio Configuration | Total Return | CAGR | Ann. Volatility | Sharpe | Sortino |
|---|---|---|---|---|---|
| Benchmark (Equal-Weight) | 101.53% | 27.24% | 27.00% | 0.89 | 1.28 |
| Aggressive Alpha (Top 3) | 129.24% | 33.00% | 28.50% | 0.99 | 1.38 |
| **Optimized Parity (Top 5)** | **114.82%** | **30.07%** | **25.45%** | **1.03** | **1.48** |
| Index Hugger (Top 10) | 87.33% | 24.08% | 26.50% | 0.93 | 1.30 |

**Top 5** is the sweet spot: a strong 30.07% CAGR at the *lowest* volatility of any configuration (1.55 pts below the benchmark), driving the best Sortino ratio of the sweep. Top 10 underperforming the benchmark confirms the edge comes from the model's selectivity, not the weighting scheme alone.

---

## Live Dashboard

The strategy is deployed as an interactive **Streamlit** app (built with **Plotly** for charts):

- **Live PnL tracking** — converts strategy returns into absolute dollar P&L based on a user-defined starting capital.
- **Equity curve visualization** — interactive chart comparing the strategy's compounded growth vs. the benchmark, especially useful for spotting outperformance during drawdowns.

> 🔗 **Live demo:** https://raj-srivastav28-stock-market-news-prediction-app-4erwcz.streamlit.app/
---

## Repository Structure

```
STOCK-MARKET-NEWS-PREDICTION/
├── data/            # Raw and processed datasets (price + sentiment)
├── models/          # Trained model artifacts
├── notebooks/        # Jupyter notebooks (EDA, feature engineering, modeling, backtesting)
├── src/             # Core pipeline source code
│   ├── feature_engineering.py   # Computes z-scores, log returns, sentiment interactions, rank target
│   └── ...                       # (additional modules — model training, backtester, risk parity engine)
├── app.py           # Streamlit dashboard entry point
├── requirements.txt # Python dependencies
├── LICENSE          # MIT License
└── README.md
```

> ⚠️ *Update the `src/` file listing above to match your actual module names if they differ.*

---

## Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/raj-srivastav28/STOCK-MARKET-NEWS-PREDICTION.git
cd STOCK-MARKET-NEWS-PREDICTION
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Explore the Notebooks

The `notebooks/` directory contains the exploratory data analysis, feature engineering, model training, and backtesting workflows used to build this pipeline. Open them with:

```bash
jupyter notebook
```

---

## Methodology Summary

| Component | Approach |
|---|---|
| **Universe** | Fixed basket of large-cap, liquid equities |
| **Target** | Zero-centered cross-sectional rank of next-period return |
| **Model** | Random Forest Regressor (B = 300 trees) |
| **Leakage control** | Forward-fill imputation + 60-day train/test embargo |
| **Selection** | Top-N assets by predicted rank |
| **Sizing** | Inverse volatility (risk parity), 14-day trailing σ |
| **Evaluation** | Sharpe, Sortino, CAGR, annualized volatility vs. equal-weight benchmark |

---

## Future Work

- Test the other ML Model 
- Test the pipeline across other asset universes (mid-caps, sectors, international equities)
- Extend the walk-forward framework to support periodic model retraining as new data arrives

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Raj Srivastav**
[GitHub](https://github.com/raj-srivastav28) · Project: [STOCK-MARKET-NEWS-PREDICTION](https://github.com/raj-srivastav28/STOCK-MARKET-NEWS-PREDICTION)
