
# ── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
#  1. LOAD DATA
#     Exact column names from Full_Dataset.csv:
#       Date, Price_BE, Load_FR, Gen_FR, Price_CH, Wind_BE, Solar_BE, Load_BE
# =============================================================================
DATA_PATH  = "Full_Dataset.csv"     # 12 may SWAP
TARGET_COL = "Price_BE"
DATE_COL   = "Date"

df = pd.read_csv(
    DATA_PATH,
    parse_dates=[DATE_COL],
    dayfirst=True,
)
df = df.sort_values(DATE_COL).reset_index(drop=True)

print(f"Dataset shape  : {df.shape}")
print(f"Date range     : {df[DATE_COL].iloc[0]}  →  {df[DATE_COL].iloc[-1]}")
print(f"\nNull counts before filling:\n{df.isnull().sum()}")

# =============================================================================
#  2. HANDLE MISSING VALUES
#     Gen_FR: 292 nulls | Wind_BE / Solar_BE: 5 | Load_FR: 3
# =============================================================================
EXOG_COLS = ["Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE"]

df[EXOG_COLS] = df[EXOG_COLS].ffill().bfill()

print(f"\nNull counts after filling:\n{df.isnull().sum()}")

# =============================================================================
#  3. FEATURE ENGINEERING
# =============================================================================
# --- Calendar features -------------------------------------------------------
df["hour"]        = df[DATE_COL].dt.hour
df["day_of_week"] = df[DATE_COL].dt.dayofweek   # 0=Mon … 6=Sun
df["month"]       = df[DATE_COL].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

# --- Lagged target (Price_BE) ------------------------------------------------
for lag in [24, 48, 168]:          # t-24h, t-48h, same hour last week
    df[f"Price_BE_lag{lag}"] = df[TARGET_COL].shift(lag)

# Rolling statistics (shifted 24h to prevent data leakage)
df["Price_BE_roll24_mean"]  = df[TARGET_COL].shift(24).rolling(24).mean()
df["Price_BE_roll24_std"]   = df[TARGET_COL].shift(24).rolling(24).std()
df["Price_BE_roll168_mean"] = df[TARGET_COL].shift(168).rolling(168).mean()

# --- Lagged exogenous features -----------------------------------------------
for col in EXOG_COLS:
    df[f"{col}_lag24"] = df[col].shift(24)

# Drop NaN rows introduced by lags (first ~168 rows)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nShape after feature engineering: {df.shape}")

# =============================================================================
#  4. DEFINE FEATURE SET
# =============================================================================
FEATURE_COLS = [
    # Calendar
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
    # Lagged target
    "Price_BE_lag24", "Price_BE_lag48", "Price_BE_lag168",
    "Price_BE_roll24_mean", "Price_BE_roll24_std", "Price_BE_roll168_mean",
    # Current exogenous (available at day-ahead forecast time)
    "Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE",
    # Lagged exogenous
    "Load_FR_lag24", "Gen_FR_lag24", "Price_CH_lag24",
    "Wind_BE_lag24", "Solar_BE_lag24", "Load_BE_lag24",
]

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# =============================================================================
#  5. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT  (60 / 20 / 20)
# =============================================================================
n         = len(df)
train_end = int(n * 0.60)
val_end   = int(n * 0.80)

X_train, y_train = X[:train_end],          y[:train_end]
X_val,   y_val   = X[train_end:val_end],   y[train_end:val_end]
X_test,  y_test  = X[val_end:],            y[val_end:]

print(f"\nSplit sizes → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train : {df[DATE_COL].iloc[0]}  →  {df[DATE_COL].iloc[train_end-1]}")
print(f"Val   : {df[DATE_COL].iloc[train_end]}  →  {df[DATE_COL].iloc[val_end-1]}")
print(f"Test  : {df[DATE_COL].iloc[val_end]}  →  {df[DATE_COL].iloc[-1]}")

# =============================================================================
#  6. TRAIN XGBOOST MODEL
# =============================================================================
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_COLS)
dval72  = xgb.DMatrix(X_val,  label=y_val,  feature_names=FEATURE_COLS)
dtest72 = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLS)

params = {
    "objective":        "reg:squarederror",   # directly minimises MSE
    "eval_metric":      "rmse",
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "seed":             42,
    "n_jobs":           -1,
}

print("\nTraining XGBoost …")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)
print(f"\nBest iteration : {model.best_iteration}")
print(f"Best val RMSE  : {model.best_score:.4f}")

# =============================================================================
#  7. EVALUATE ON VALIDATION AND TEST SETS
# =============================================================================

def evaluate(model, dmat, y_true, label="Set"):
    preds = model.predict(dmat, iteration_range=(0, model.best_iteration + 1))
    """
    mse   = mean_squared_error(y_true, preds)
    rmse  = np.sqrt(mse)
    mae   = np.mean(np.abs(y_true - preds))
    mape  = np.mean(np.abs((y_true - preds) / (np.abs(y_true) + 1e-8))) * 100
    """
    # Only last 72 hrs
    mse   = mean_squared_error(y_true[-72:], preds[-72:])
    rmse  = np.sqrt(mse)
    mae   = np.mean(np.abs(y_true[-72:] - preds[-72:]))
    mape  = np.mean(np.abs((y_true[-72:] - preds[-72:]) / (np.abs(y_true[-72:]) + 1e-8))) * 100
    print(f"  {label:6s} → MSE: {mse:8.2f}  RMSE: {rmse:6.2f}  MAE: {mae:6.2f}  MAPE: {mape:.2f}%")
    return preds

print("\n── Performance Metrics ──────────────────────────────────────────────")

preds_train = evaluate(model, dtrain, y_train, "Train")
preds_val  = evaluate(model, dval,  y_val,  "Val")
preds_test = evaluate(model, dtest, y_test, "Test")

# =============================================================================
#  8. PLOTS  (save for your report)
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

eval_range = 72

ax = axes[0]
ax.plot(y_test[-72:],     label="Actual",    color="steelblue", lw=1.5)
ax.plot(preds_test[-72:], label="Predicted", color="tomato",    lw=1.5, alpha=0.85)
ax.set_title("Actual vs Predicted — Test Set")
ax.set_xlabel("Hour")
ax.set_ylabel("Price (€/MWh)")
ax.legend()

# (b) Residuals
ax = axes[1]
ax.plot(y_test - preds_test, color="darkorange", lw=0.6, alpha=0.7)
ax.axhline(0, color="black", lw=1)
ax.set_title("Residuals — Test Set")
ax.set_xlabel("Hour")
ax.set_ylabel("Error (€/MWh)")

# (c) Feature importance (top 15 by gain)
ax = axes[2]
imp = pd.DataFrame({
    "feature": list(model.get_score(importance_type="gain").keys()),
    "gain":    list(model.get_score(importance_type="gain").values()),
}).sort_values("gain", ascending=False).head(15)
ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="teal")
ax.set_title("Top-15 Feature Importance (Gain)")
ax.set_xlabel("Gain")

plt.tight_layout()
#plt.savefig("model_performance.png", dpi=150)
plt.show()
#print("\nPlot saved → model_performance.png")

