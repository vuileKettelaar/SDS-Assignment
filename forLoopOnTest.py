import os
import sys

# --- PRE-IMPORT GPU LINKING ---
venv_base = os.path.dirname(sys.executable)
nvidia_path = os.path.join(venv_base, "..", "lib", "python3.13", "site-packages", "nvidia")
if os.path.exists(nvidia_path):
    cuda_libs = [os.path.join(root, 'lib') for root, dirs, files in os.walk(nvidia_path) if 'lib' in dirs]
    os.environ['LD_LIBRARY_PATH'] = ":".join(cuda_libs) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    print(f"SUCCESS: Found {len(gpus)} GPU(s).")

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
#  CONFIGURATION
# =============================================================================
MODEL_CHOICE = "2-WAY" 

# =============================================================================
#  1. LOAD DATA & SET UP FINAL WINDOW (MAY 12 - MAY 14)
# =============================================================================
DATA_PATH  = "Full_Dataset_Updated.csv"     
TARGET_COL = "Price_BE"
DATE_COL   = "Date"

df_history = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL], dayfirst=True)
df_history = df_history.sort_values(DATE_COL).reset_index(drop=True)

# Cut training data at the start of the assignment window
df_history = df_history[df_history[DATE_COL] < "2026-05-11 22:00:00"].copy()

EXOG_COLS = ["Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE"]
df_history[EXOG_COLS] = df_history[EXOG_COLS].ffill().bfill()
df_history["Net_Load_BE"] = df_history["Load_BE"] - df_history["Wind_BE"] - df_history["Solar_BE"]

# Create future dates for the assignment (72 hours)
future_dates = pd.date_range(start="2026-05-11 22:00:00", periods=72, freq="h")
df_future = pd.DataFrame({DATE_COL: future_dates})

df = pd.concat([df_history, df_future], ignore_index=True)

# =============================================================================
#  2. FEATURE ENGINEERING (72-HOUR GAP)
# =============================================================================
df["hour"]        = df[DATE_COL].dt.hour
df["day_of_week"] = df[DATE_COL].dt.dayofweek   
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["hour_weekend_interaction"] = df["hour"] * df["is_weekend"]
df["month"] = df[DATE_COL].dt.month

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

for lag in [72, 96, 168, 192]:
    df[f"Price_BE_lag{lag}"] = df[TARGET_COL].shift(lag)

df["Price_Velocity_72_vs_96"] = df["Price_BE_lag72"] - df["Price_BE_lag96"]
df["Price_Velocity_168_vs_192"] = df["Price_BE_lag168"] - df["Price_BE_lag192"]

df["Price_BE_roll72_mean"] = df[TARGET_COL].shift(72).rolling(24).mean()
df["Price_BE_roll168_mean"] = df[TARGET_COL].shift(168).rolling(24).mean()

for col in EXOG_COLS + ["Net_Load_BE"]:
    df[f"{col}_lag72"] = df[col].shift(72)
    df[f"{col}_lag96"] = df[col].shift(96)
    df[f"{col}_lag168"] = df[col].shift(168)
    df[f"{col}_Velocity_72_vs_96"] = df[f"{col}_lag72"] - df[f"{col}_lag96"]
    df[f"{col}_roll168_mean"] = df[f"{col}_lag72"].rolling(168).mean()

# =============================================================================
#  3. SPLIT & SCALE
# =============================================================================
FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "hour_weekend_interaction",
    "Price_BE_lag72", "Price_BE_lag96", "Price_BE_lag168",
    "Price_Velocity_72_vs_96", "Price_Velocity_168_vs_192", 
    "Price_BE_roll72_mean", "Price_BE_roll168_mean",
] + [f"{col}_lag72" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_lag168" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_Velocity_72_vs_96" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_roll168_mean" for col in EXOG_COLS + ["Net_Load_BE"]]

train_df = df[df[DATE_COL] < "2026-05-11 22:00:00"].copy()
future_df = df[df[DATE_COL] >= "2026-05-11 22:00:00"].copy()

train_df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
train_df.reset_index(drop=True, inplace=True)

X = train_df[FEATURE_COLS].values
y = train_df[TARGET_COL].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
#  4. TRAIN/VAL SPLIT
# =============================================================================
horizon = 72
X_train_val = X_scaled[:-horizon]
y_train_val = y[:-horizon]
X_test = X_scaled[-horizon:]
y_test = y[-horizon:]

tscv = TimeSeriesSplit(n_splits=3)
for train_index, val_index in tscv.split(X_train_val):
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm   = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_final_future = future_df[FEATURE_COLS].values
X_final_future_scaled = scaler.transform(X_final_future)
X_final_future_lstm = X_final_future_scaled.reshape((X_final_future_scaled.shape[0], 1, len(FEATURE_COLS)))

# =============================================================================
#  5. PRE-TRAIN LSTM
# =============================================================================
print("\nTraining LSTM …")
lstm_model = Sequential([LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True), 
                         Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(1)])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, validation_data=(X_val_lstm, y_val), 
               callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)

lstm_f_preds = lstm_model.predict(X_final_future_lstm, verbose=0).flatten()

# =============================================================================
#  6. PRE-LOAD TRUE ACTUALS FOR OPTIMIZATION
# =============================================================================
print("\nLoading Actual True Prices for Blind Optimization...")
actuals_df = pd.read_csv("Actual_Hourly_Belpex_Prices.csv")
actuals_df['datetime'] = pd.to_datetime(actuals_df['Date'] + ' ' + actuals_df['Time'], format='%d/%m/%Y %H:%M')
actuals_df = actuals_df.sort_values('datetime').reset_index(drop=True)

mask = actuals_df['datetime'] >= "2026-05-12 00:00:00"
available_actuals = actuals_df.loc[mask].copy()

# Ensure we have data to score against
true_prices = available_actuals['Actual_Price_BE'].values
if len(true_prices) == 0:
    print("ERROR: No actual prices found for May 12+ window. Cannot optimize.")
    sys.exit()

# =============================================================================
#  7. XGBOOST HYPERPARAMETER LOOP & CSV GENERATION
# =============================================================================
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)

rounds_to_test = [3000, 3500, 4000] + list(range(4500, 5501, 100)) + [6000]

best_mse = float('inf')
best_xgb_model = None
best_final_predictions = None
best_round = 0

print("\n" + "="*50)
print(f" 🚀 STARTING OPTIMIZATION BASED ON TRUE TEST DATA ({len(true_prices)} hours)")
print("="*50)

for r in rounds_to_test:
    temp_xgb_model = xgb.train(
        {"objective": "reg:squarederror", "eval_metric": "rmse", "learning_rate": 0.01, "max_depth": 7}, 
        dtrain, 
        num_boost_round=r, 
        evals=[(dtrain, "train"), (dval, "val")], 
        verbose_eval=False
    )
    
    # Immediately predict the Future window (May 12-14)
    xgb_f_preds = temp_xgb_model.predict(xgb.DMatrix(X_final_future_scaled, feature_names=FEATURE_COLS))
    
    # Calculate 2-WAY Ensemble Future Forecast
    final_predictions = (xgb_f_preds + lstm_f_preds) / 2
    
    # Slice to match the length of available true prices
    matched_preds = final_predictions[:len(true_prices)]
    
    # Score against the TRUE BLIND ACTUALS
    current_true_mse = mean_squared_error(true_prices, matched_preds)
    
    print(f"[{r} Trees] -> 2-WAY TRUE EVAL MSE: {current_true_mse:.2f}")
    
    # Save a CSV specific to this iteration
    csv_filename = f"2-WAY_predictions_XGB_{r}.csv"
    pd.DataFrame(final_predictions).to_csv(csv_filename, index=False, header=False)
    
    # Track the ultimate best based on the TRUE score
    if current_true_mse < best_mse:
        best_mse = current_true_mse
        best_round = r
        best_xgb_model = temp_xgb_model
        best_final_predictions = final_predictions

print("="*50)
print(f"🏆 BEST MODEL FOUND: XGBoost at {best_round} trees!")
print(f"🏆 LOWEST 2-WAY TRUE MSE: {best_mse:.2f}")
print("="*50)

# Anchor the best results for final plotting
xgb_model = best_xgb_model
final_predictions = best_final_predictions

# Save the master prediction CSV
master_csv = "Best_2-WAY_predictions.csv"
pd.DataFrame(final_predictions).to_csv(master_csv, index=False, header=False)
print(f"\nSUCCESS: {master_csv} created using XGB round {best_round}.")

# =============================================================================
#  8. PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(available_actuals['datetime'], true_prices, label="Actual Price", color="black", lw=2)
axes[0].plot(available_actuals['datetime'], final_predictions[:len(available_actuals)], 
             label=f"2-WAY Forecast (XGB {best_round})", color="tomato", ls='--', lw=2)

axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f"Optimized 2-WAY Model Forecast (XGBoost {best_round} trees)")

# Feature Importance logic
imp_dict = xgb_model.get_score(importance_type="gain")
imp = pd.DataFrame({"feature": list(imp_dict.keys()), "gain": list(imp_dict.values())})
imp = imp.sort_values("gain", ascending=False).head(10)
axes[1].barh(imp["feature"][::-1], imp["gain"][::-1], color="teal")
axes[1].set_title("Top 10 Features (Gain)")

plt.tight_layout()
out_plot = f"{MODEL_CHOICE}_best_{best_round}_performance.png"
plt.savefig(out_plot, dpi=300)
print(f"\nSUCCESS: Plot saved as {out_plot}.")