# Getting Started

This project uses `uv` to manage Python dependencies.

## 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Initialize uv

```bash
uv init
```

## 3. Install dependencies Or Sync the dependancies to match what everyone is using

```bash
uv sync
```

## 4. Run the project

```bash
uv run script.py
```

## 5. Add a new library (Example)

```bash
uv add matlplotlib 
```

## 6. Activate the virtual environment

```bash
.venv/Scripts/activate 
```

## 7. USe the environment in Jupyter

Just select the Python version that is inside of your .venv folder and Jupyter will run with all of the correct dependancies


That's it! `uv` handles the virtual environment for you automatically.

# Belgian Electricity Price Forecasting 
## Project Overview / What have been done as to now:)

This project develops and compares multiple machine learning and statistical models to forecast day-ahead electricity prices in the Belgian market (Belpex). The goal is to predict 24 hourly prices for the next day using historical price data, solar generation, wind generation, and electricity load as inputs.

---

## Dataset

- **Source:** `Full_Dataset.csv`
- **Market:** Belgian electricity market (Belpex)
- **Frequency:** Hourly
- **Sample period:** January 2021 – May 2025 (~38,391 observations)
- **Features used:**
  - `Price_BE` — Belgian day-ahead electricity price (€/MWh)
  - `Solar_BE` — Solar generation (MW)
  - `Wind_BE` — Wind generation (MW)
  - `Load_BE` — Electricity load/demand (MW)

---

## Data Preparation

- Datetime index set with `dayfirst=True`
- Duplicate timestamps resolved by averaging (`groupby().mean()`)
- Resampled to strict 1-hour intervals with forward-fill for missing values
- **Feature engineering for ANN:**
  - Lag-1: previous day's 24 hourly prices
  - Lag-7: same weekday's prices from 7 days ago
  - Same-day solar, wind, and load forecasts (24 hours each)
  - Total input vector: 120 features per day
- **Feature engineering for LSTM:**
  - 3D input array: (days, timesteps, features)
  - Lookback window: past 7 days of hourly data (168 timesteps × 4 features)

### Train / Validation / Test Split
All splits are **chronological** (`shuffle=False`) to respect time-series structure:

| Set | Proportion | Purpose |
|---|---|---|
| Training | 70% | Model learning |
| Validation | 15% | Monitor overfitting during training |
| Test | 15% | Final honest evaluation |

### Scaling
- `StandardScaler` applied separately to features (X) and targets (Y)
- Scalers fitted **only on training data** to prevent data leakage
- Predictions inverse-transformed back to real €/MWh for evaluation

---

## Models Developed

### 1. Feedforward Artificial Neural Network (ANN)

A grid search was performed over 16 configurations:

| Hyperparameter | Options Tested |
|---|---|
| Hidden layers | 1, 2 |
| Nodes per layer | 64, 128 |
| Activation function | ReLU, Tanh, Sigmoid, Linear |

- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE
- **Epochs:** 50, Batch size: 32
- **Best configuration:** 2 layers, 64 nodes, linear activation
- **Key finding:** Linear activation winning suggests the price-feature relationship is largely linear in this dataset — the ANN effectively reduces to a linear regression

---

### 2. SARIMAX (Seasonal ARIMA with Exogenous Variables)

**Parameter selection:**
- ADF test confirmed the series is **stationary** (ADF = −7.31, p = 0.000)
- However, `auto_arima` preferred `d=1` based on lower AIC — series exhibits near unit-root behaviour
- `auto_arima` with stepwise search used to identify optimal orders
- Best model identified: `SARIMAX(0,1,1)(2,0,0,24)`

**Model specification:**
```
SARIMAX(0, 1, 1)(2, 0, 0)[24]
  d=1    → one differencing (near non-stationary behaviour)
  q=1    → MA term captures residual autocorrelation
  P=2    → seasonal AR: same hour yesterday and 2 days ago
  s=24   → daily seasonality confirmed
```

**Exogenous variables:** solar, wind, load

**Coefficient results:**
| Variable | Coefficient | Interpretation |
|---|---|---|
| solar | −0.0217 | Renewables suppress prices ✅ |
| wind | −0.0111 | Renewables suppress prices ✅ |
| load | +0.0274 | Demand raises prices ✅ |
| ma.L1 | −0.0159 | Error correction term ✅ |
| ar.S.L24 | +0.2590 | Same hour yesterday ✅ |
| ar.S.L48 | +0.1724 | Same hour 2 days ago ✅ |

**Diagnostics:**
- Ljung-Box: p=0.87 ✅ — residuals are clean (no autocorrelation)
- Heteroskedasticity: p=0.00 ⚠️ — variance changes over time (common in electricity markets)
- Kurtosis: 27.78 ⚠️ — fat tails from price spikes (expected)

**Model saved using:**
```python
fitted.save('sarimax_fitted.pkl')       # full model
pickle.dump(fitted.params, ...)         # parameters only (lightweight)
```

---

### 3. LSTM (Long Short-Term Memory)

Designed to capture **temporal dependencies** that the ANN misses.

**Architecture:**
```
Input: (168 timesteps × 4 features)
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32, return_sequences=False)
  → Dropout(0.2)
  → Dense(24, linear)
```

- **Loss function:** Huber (delta=1.0) — robust to price spikes and troughs
- **Optimizer:** Adam (lr=0.001)
- **Callbacks:** EarlyStopping (patience=15), ReduceLROnPlateau (factor=0.5)

**Improvements applied:**
- Switched from MSE to **Huber loss** to reduce trough-smoothing behaviour
- Investigated **regime-switching approach**: separate LSTM models trained for high, normal, and low demand periods to improve trough prediction accuracy

**Demand regime classification:**
| Regime | Threshold | Specialist Model |
|---|---|---|
| Low demand | Bottom 33% of daily avg price | `model_lstm_low.keras` |
| Normal demand | Middle 34% | `model_lstm_normal.keras` |
| High demand | Top 33% | `model_lstm_high.keras` |

**Models saved:**
```
model_lstm_high.keras
model_lstm_low.keras
model_lstm_normal.keras
demand_thresholds.pkl
scaler_lstm_x.pkl
scaler_lstm_y.pkl
```

---

## Evaluation Metrics

All models evaluated on the same held-out test set using:

| Metric | Formula | Unit |
|---|---|---|
| MAE | mean(|actual − forecast|) | €/MWh |
| RMSE | √mean((actual − forecast)²) | €/MWh |
| MAPE | mean(|actual − forecast| / |actual|) × 100 | % |

> Note: MAPE computed with zero-price masking to avoid division by zero, which occurs during periods of excess renewable generation in Belgium.

---

## Key Findings

1. **Linear activation won the ANN grid search** — suggesting the price-feature relationship is largely linear, making SARIMAX and XGBoost inherently well-suited for this problem

2. **ADF test showed stationarity** (p=0.000) but `auto_arima` preferred d=1 based on AIC — Belgian electricity prices exhibit near unit-root behaviour with slow mean reversion

3. **SARIMAX(0,1,1)(2,0,0,24) passed the Ljung-Box test** (p=0.87) after adding the MA(1) term — residuals are clean white noise

4. **LSTM consistently smoothed price troughs** — a known limitation of MSE-trained sequential models on imbalanced price distributions. Addressed with Huber loss and regime-switching

5. **Near-zero and negative prices** are fundamentally difficult to predict for any model — caused by sudden unexpected renewable surges that are not fully captured by available features

---

## File Structure

```
project/
│
├── Full_Dataset.csv              ← raw input data
│
├── model_2.keras                 ← best ANN model
├── scaler_x.pkl                  ← ANN feature scaler
├── scaler_y.pkl                  ← ANN price scaler
│
├── sarimax_fitted.pkl            ← fitted SARIMAX model
├── sarimax_params.pkl            ← SARIMAX parameters (lightweight)
├── sarimax_results.csv           ← SARIMAX forecast vs actual
│
├── model_lstm.keras              ← base LSTM model
├── model_lstm_high.keras         ← high demand specialist
├── model_lstm_normal.keras       ← normal demand specialist
├── model_lstm_low.keras          ← low demand specialist
├── demand_thresholds.pkl         ← low/high demand thresholds
├── scaler_lstm_x.pkl             ← LSTM feature scaler
├── scaler_lstm_y.pkl             ← LSTM price scaler
│
└── model_comparison.csv          ← final metrics comparison table
```


## References

- Belpex Belgian Power Exchange data
- Statsmodels SARIMAX documentation
- TensorFlow/Keras LSTM documentation
- pmdarima auto_arima documentation