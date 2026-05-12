import os
import sys

# --- PRE-IMPORT GPU LINKING ---
venv_base = os.path.dirname(sys.executable)
# Path to your nvidia site-packages
nvidia_path = os.path.join(venv_base, "..", "lib", "python3.13", "site-packages", "nvidia")

if os.path.exists(nvidia_path):
    cuda_libs = []
    for root, dirs, files in os.walk(nvidia_path):
        if 'lib' in dirs:
            cuda_libs.append(os.path.join(root, 'lib'))
    
    # Prepend to LD_LIBRARY_PATH so TF sees them immediately
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ":".join(cuda_libs) + ":" + current_ld

import tensorflow as tf
# Verify
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"SUCCESS: Found {len(gpus)} GPU(s). Ready to train on RTX 4050.")
    # Optional: Prevent TF from hogging all VRAM immediately
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("WARNING: GPU still not detected. Check 'pip list | grep nvidia'")
# ── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg') # MUST come before importing plt
import matplotlib.pyplot as plt
# =============================================================================
#  1. LOAD DATA
#     Exact column names from Full_Dataset.csv:
#       Date, Price_BE, Load_FR, Gen_FR, Price_CH, Wind_BE, Solar_BE, Load_BE
# =============================================================================
DATA_PATH  = "Full_Dataset_Updated.csv"     # 12 may SWAP
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
# The "Net Load" is the biggest driver of price dips
df["Net_Load_BE"] = df["Load_BE"] - df["Wind_BE"] - df["Solar_BE"]
# Lag it as well to capture the momentum of the dip
df["Net_Load_BE_lag24"] = df["Net_Load_BE"].shift(24)


# Tells the model how the hour interacts with the weekend
df["hour_weekend_interaction"] = df["hour"] * df["is_weekend"]

# Cyclical encoding
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

# --- Lagged target (Price_BE) ------------------------------------------------
# --- Lagged target (Price_BE) ------------------------------------------------
# ADDED 2 to the list below so the column is created for training
for lag in [2, 24, 48, 168]:          # t-2h, t-24h, t-48h, same hour last week
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

# Make sure to add these to your imports at the top:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
#  4. DEFINE FEATURE SET & SCALE FOR NEURAL NETWORKS
# =============================================================================
FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
    "Price_BE_lag24", "Price_BE_lag48", "Price_BE_lag168", "Price_BE_lag2", # Added lag2 here
    "Price_BE_roll24_mean", "Price_BE_roll24_std", "Price_BE_roll168_mean",
    "Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE",
    "Load_FR_lag24", "Gen_FR_lag24", "Price_CH_lag24",
    "Wind_BE_lag24", "Solar_BE_lag24", "Load_BE_lag24",
    "Net_Load_BE", "Net_Load_BE_lag24", "hour_weekend_interaction"
]

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# Neural Networks (LSTM) are highly sensitive to unscaled data.
# We must scale the features to a [0, 1] range. XGBoost is scale-invariant, 
# so it will still perform perfectly fine with scaled data.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
#  5. PERIODIC TIME-SERIES SPLIT & SHAPE FOR LSTM
# =============================================================================
# Instead of a random or single chronological split, TimeSeriesSplit respects 
# the periodic nature of the data by creating expanding windows of train/test folds.
# For evaluating our final model before the assignment submission, we'll hold out 
# the absolute last 72 hours as our true "Test" set to mimic the assignment.

horizon = 72
X_train_val = X_scaled[:-horizon]
y_train_val = y[:-horizon]

X_test = X_scaled[-horizon:]
y_test = y[-horizon:]

# Split the train_val into train and val for early stopping
tscv = TimeSeriesSplit(n_splits=3)
for train_index, val_index in tscv.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

# LSTM expects 3D input: (samples, time_steps, features)
# We treat each row as a single time step with multiple features.
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm   = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples ( mimicking the 72h assignment )")

# =============================================================================
#  6. TRAIN XGBOOST MODEL
# =============================================================================
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_COLS)

xgb_params = {
    "objective":        "reg:squarederror",   
    "eval_metric":      "rmse",
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             42,
    "n_jobs":           -1,
}

print("\nTraining XGBoost …")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# =============================================================================
#  7. TRAIN LSTM MODEL
# =============================================================================
print("\nTraining LSTM …")
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse') # Optimizing directly for MSE

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

lstm_model.fit(
    X_train_lstm, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val_lstm, y_val),
    callbacks=[early_stop],
    verbose=0
)

# =============================================================================
#  8. EVALUATE ENSEMBLE (HYBRID)
# =============================================================================
# Get predictions for the 72-hour test set
xgb_preds = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1))
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

# Ensemble: simple average of both models
ensemble_preds = (xgb_preds + lstm_preds) / 2

# Calculate final assignment metric
mse_xgb = mean_squared_error(y_test, xgb_preds)
mse_lstm = mean_squared_error(y_test, lstm_preds)
mse_ensemble = mean_squared_error(y_test, ensemble_preds)

print("\n── Performance Metrics (MSE) ────────────────────────────────────────")
print(f"  XGBoost MSE  : {mse_xgb:.2f}")
print(f"  LSTM MSE     : {mse_lstm:.2f}")
print(f"  Ensemble MSE : {mse_ensemble:.2f}")

# =============================================================================
#  9. EXPORT PREDICTIONS.CSV (ASSIGNMENT FORMAT)
# =============================================================================
# The assignment requires EXACTLY 72 rows, 1 column, no headers, no index.
output_df = pd.DataFrame(ensemble_preds)
output_df.to_csv("predictions.csv", index=False, header=False)
print("\nSuccess: 'predictions.csv' generated with 72 rows and no headers.")

# =============================================================================
#  10. PLOTS & VISUALIZATION
# =============================================================================
# Initialize the figure and axes properly on their own line
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# (a) Actual vs Predicted (Last 72 Hours)
ax = axes[0]
ax.plot(y_test, label="Actual (Belpex)", color="steelblue", lw=2)
ax.plot(ensemble_preds, label="Hybrid Prediction (XGB+LSTM)", color="tomato", lw=2, linestyle='--')
ax.set_title("Actual vs Predicted — Final 72-Hour Test Set")
ax.set_xlabel("Hour")
ax.set_ylabel("Price (€/MWh)")
ax.grid(True, alpha=0.3)
ax.legend()

# (b) Residuals (Errors)
ax = axes[1]
residuals = y_test - ensemble_preds
ax.plot(residuals, color="darkorange", lw=1.5)
ax.axhline(0, color="black", lw=1, linestyle='-')
ax.set_title("Residuals (Actual - Predicted)")
ax.set_xlabel("Hour")
ax.set_ylabel("Error (€/MWh)")
ax.grid(True, alpha=0.3)

# (c) Feature Importance (From the XGBoost component)
ax = axes[2]
# Using xgb_model as defined in the training step
importance_dict = xgb_model.get_score(importance_type="gain")
imp = pd.DataFrame({
    "feature": list(importance_dict.keys()),
    "gain":    list(importance_dict.values()),
}).sort_values("gain", ascending=False).head(15)

ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="teal")
ax.set_title("Top-15 Feature Importance (XGBoost Gain)")
ax.set_xlabel("Gain")

plt.tight_layout()
plt.savefig("model_performance.png", dpi=300)
print("Plot saved as model_performance.png")

# =============================================================================
#  11. PERSISTENCE BASELINES (BENCHMARKS)
# =============================================================================
# We need to reach back into the original dataframe 'df' to get historical prices 
# for the test period (the last 72 hours).

# 1. Weekly Persistence (t-168)
# Since FEATURE_COLS already has Price_BE_lag168, we can pull it directly
persistence_weekly = df["Price_BE_lag168"].values[-72:]

# 2. Daily Persistence (t-24)
persistence_daily = df["Price_BE_lag24"].values[-72:]

# 3. 7-Day Seasonal Moving Average 
# Average of the same hour for the last 7 days (t-24, t-48, ..., t-168)
lags_7d = [24, 48, 72, 96, 120, 144, 168]
persistence_ma7 = np.mean([df[TARGET_COL].shift(l).values[-72:] for l in lags_7d], axis=0)

# Calculate MSE for benchmarks
mse_weekly = mean_squared_error(y_test, persistence_weekly)
mse_daily  = mean_squared_error(y_test, persistence_daily)
mse_ma7    = mean_squared_error(y_test, persistence_ma7)

print("\n── Baseline Metrics (MSE) ──────────────────────────────────────────")
print(f"  Naive 24h MSE    : {mse_daily:.2f}")
print(f"  Weekly Pers. MSE : {mse_weekly:.2f}")
print(f"  7-Day MA MSE     : {mse_ma7:.2f}")

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ... [rest of your code] ...

import matplotlib
matplotlib.use('Agg') # Prevents crashes on Linux/Gnome by disabling the GUI window
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
#  12. FINAL FUTURE FORECAST (May 12 - May 14) - REPAIRED
# =============================================================================
print("\nStarting repaired sophisticated final forecast...")

# 1. Prepare history and future dates
df_history = df[df[DATE_COL] < "2026-05-12 00:00:00"].copy()
future_dates = pd.date_range(start="2026-05-12 00:00:00", periods=72, freq="h")
df_future = pd.concat([df_history, pd.DataFrame({DATE_COL: future_dates})], ignore_index=True)

# 2. Re-run FULL Calendar Engineering (No placeholders!)
df_future["hour"] = df_future[DATE_COL].dt.hour
df_future["day_of_week"] = df_future[DATE_COL].dt.dayofweek
df_future["month"] = df_future[DATE_COL].dt.month
df_future["is_weekend"] = (df_future["day_of_week"] >= 5).astype(int)

df_future["hour_sin"] = np.sin(2 * np.pi * df_future["hour"] / 24)
df_future["hour_cos"] = np.cos(2 * np.pi * df_future["hour"] / 24)
df_future["dow_sin"]  = np.sin(2 * np.pi * df_future["day_of_week"] / 7)
df_future["dow_cos"]  = np.cos(2 * np.pi * df_future["day_of_week"] / 7)
df_future["month_sin"] = np.sin(2 * np.pi * df_future["month"] / 12)
df_future["month_cos"] = np.cos(2 * np.pi * df_future["month"] / 12)

# 3. SMARTER WEATHER: 3-Week Seasonal Average
for col in EXOG_COLS:
    future_mask = df_future[DATE_COL] >= "2026-05-12 00:00:00"
    for i in df_future.index[future_mask]:
        df_future.loc[i, col] = (df_future.loc[i-168, col] + 
                                 df_future.loc[i-336, col] + 
                                 df_future.loc[i-504, col]) / 3

# 4. The Autoregressive Loop
future_idx_start = df_future.index[df_future[DATE_COL] == "2026-05-12 00:00:00"][0]

for i in range(future_idx_start, len(df_future)):
    # Feature Updates
    df_future.loc[i, "Net_Load_BE"] = df_future.loc[i, "Load_BE"] - df_future.loc[i, "Wind_BE"] - df_future.loc[i, "Solar_BE"]
    df_future.loc[i, "Net_Load_BE_lag24"] = df_future.loc[i - 24, "Net_Load_BE"]
    df_future.loc[i, "hour_weekend_interaction"] = df_future.loc[i, "hour"] * df_future.loc[i, "is_weekend"]
    
    # Price Lags
    df_future.loc[i, "Price_BE_lag2"]   = df_future.loc[i - 2, "Price_BE"] # The new lag
    df_future.loc[i, "Price_BE_lag24"]  = df_future.loc[i - 24, "Price_BE"]
    df_future.loc[i, "Price_BE_lag48"]  = df_future.loc[i - 48, "Price_BE"]
    df_future.loc[i, "Price_BE_lag168"] = df_future.loc[i - 168, "Price_BE"]

    # Rolling stats (Crucial: Added roll168_mean to prevent NaNs)
    df_future.loc[i, "Price_BE_roll24_mean"]  = df_future.loc[i-24:i-1, "Price_BE"].mean()
    df_future.loc[i, "Price_BE_roll24_std"]   = df_future.loc[i-24:i-1, "Price_BE"].std()
    df_future.loc[i, "Price_BE_roll168_mean"] = df_future.loc[i-168:i-1, "Price_BE"].mean()
    
    # Exogenous Lags
    for col in EXOG_COLS:
        df_future.loc[i, f"{col}_lag24"] = df_future.loc[i - 24, col]

    # Predict
    row_features = df_future.loc[[i], FEATURE_COLS]
    
    # Safety Check: If there's still a NaN, this will tell us exactly which column
    if row_features.isnull().values.any():
        print(f"NaN found at index {i} in columns: {row_features.columns[row_features.isnull().any()].tolist()}")
        row_features = row_features.fillna(0) # Emergency fill

    row_scaled = scaler.transform(row_features)
    
    xp = xgb_model.predict(xgb.DMatrix(row_scaled, feature_names=FEATURE_COLS))[0]
    lp = lstm_model.predict(row_scaled.reshape(1, 1, -1), verbose=0).flatten()[0]
    
    df_future.loc[i, "Price_BE"] = (float(xp) + float(lp)) / 2.0

# 5. Final Export
final_72 = df_future.loc[future_idx_start:, "Price_BE"].values
pd.DataFrame(final_72).to_csv("72predictions.csv", index=False, header=False)
print(f"\nSUCCESS: predictions.csv generated. First value: {final_72[0]:.2f}")
# =============================================================================
#  PLOTTING & PERFORMANCE METRICS
# =============================================================================

# 1. Define a helper function to calculate metrics for the table
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return [f"{mse:.2f}", f"{rmse:.2f}", f"{mae:.2f}", f"{mape:.2f}%"]

# 2. Compile metrics for all models/benchmarks
# (Assumes these variables exist from your training logic)
results = {
    "Hybrid Ensemble": get_metrics(y_test, ensemble_preds),
    "Weekly Persist": get_metrics(y_test, persistence_weekly),
    "7-Day MA":       get_metrics(y_test, persistence_ma7),
    "Daily Persist":  get_metrics(y_test, persistence_daily)
}

# 3. Create the multi-pane figure
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])

# --- Top Plot: Forecast vs. Actuals ---
ax0 = fig.add_subplot(gs[0])
ax0.plot(y_test, label="Actual Price (Belpex)", color="black", lw=2.5, zorder=5)
ax0.plot(ensemble_preds, label="Hybrid Model Forecast", color="#e74c3c", lw=2, zorder=6)
ax0.plot(persistence_weekly, label="Weekly Persistence (t-168)", color="#2ecc71", alpha=0.5, linestyle='--')
ax0.plot(persistence_ma7, label="7-Day Moving Average", color="#3498db", alpha=0.5, linestyle=':')

ax0.set_title("Price Forecast Comparison (Final 72-Hour Window)", fontsize=16, pad=20)
ax0.set_xlabel("Hour", fontsize=12)
ax0.set_ylabel("Electricity Price (€/MWh)", fontsize=12)
ax0.legend(loc='upper left', frameon=True, shadow=True)
ax0.grid(True, which='both', linestyle='--', alpha=0.4)

# --- Bottom Plot: Performance Metrics Table ---
ax1 = fig.add_subplot(gs[1])
ax1.axis('off')

# Assignment Evaluation Note: Evaluation is strictly based on Mean Squared Error (MSE)
# Formula: $MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_{i} - \hat{Y}_{i})^2$
header = ["Forecast Method", "MSE (Target)", "RMSE", "MAE", "MAPE"]
table_rows = [[k] + v for k, v in results.items()]

the_table = ax1.table(
    cellText=[header] + table_rows,
    loc='center',
    cellLoc='center',
    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
)

# Table Styling
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1, 3) # Increase row height for readability

# Bold the header row
for i in range(len(header)):
    the_table[(0, i)].get_text().set_weight('bold')
    the_table[(0, i)].set_facecolor('#f2f2f2')

plt.tight_layout()
plt.savefig("assignment_metrics_plot.png", dpi=300, bbox_inches='tight')
print("\nSuccess: 'assignment_metrics_plot.png' generated for your report.")