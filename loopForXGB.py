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
import lightgbm as lgb
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
#  1. LOAD DATA & SET UP FINAL WINDOW (MAY 12 - MAY 14)
# =============================================================================
DATA_PATH  = "Full_Dataset_Updated.csv"     
TARGET_COL = "Price_BE"
DATE_COL   = "Date"

df_history = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL], dayfirst=True)
df_history = df_history.sort_values(DATE_COL).reset_index(drop=True)

df_history = df_history[df_history[DATE_COL] < "2026-05-11 22:00:00"].copy()

EXOG_COLS = ["Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE"]
df_history[EXOG_COLS] = df_history[EXOG_COLS].ffill().bfill()
df_history["Net_Load_BE"] = df_history["Load_BE"] - df_history["Wind_BE"] - df_history["Solar_BE"]

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

# =============================================================================
#  5. TRAIN TUNED MODELS
# =============================================================================
print("\nTraining TUNED LightGBM …")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_COLS)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': 6,             # Slightly shallower trees to prevent overfitting
    'num_leaves': 31,           # Limits tree complexity
    'feature_fraction': 0.8,    # Uses 80% of features per tree (robustness)
    'bagging_fraction': 0.8,    # Uses 80% of data per tree
    'bagging_freq': 1,
    'min_data_in_leaf': 20,
    'verbose': -1
}
# Increased rounds heavily, relying on early stopping to find the exact perfect point
lgb_model = lgb.train(lgb_params, train_data, num_boost_round=5000, 
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'], 
                      callbacks=[lgb.early_stopping(200, verbose=False)])

print("Training TUNED XGBoost …")
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)

xgb_params = {
    "objective": "reg:squarederror", 
    "eval_metric": "rmse", 
    "learning_rate": 0.01, 
    "max_depth": 6, 
    "min_child_weight": 5,      # Forces conservative splits
    "subsample": 0.8,           # Stochastic boosting
    "colsample_bytree": 0.8     # Feature sampling
}
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=5000, 
                      evals=[(dtrain, "train"), (dval, "val")], 
                      early_stopping_rounds=200, verbose_eval=False)

print("Training LSTM (Standard) …")
lstm_model = Sequential([LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(1)])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, validation_data=(X_val_lstm, y_val), callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

# =============================================================================
#  6. HISTORICAL VALIDATION
# =============================================================================
lgb_preds = lgb_model.predict(X_test)
xgb_preds = xgb_model.predict(xgb.DMatrix(X_test, feature_names=FEATURE_COLS))
lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()

print("\n── Historical Validation (May 9-11) ──")
print(f"  TUNED LGBM MSE           : {mean_squared_error(y_test, lgb_preds):.2f}")
print(f"  TUNED XGBoost MSE        : {mean_squared_error(y_test, xgb_preds):.2f}")
print(f"  LSTM MSE                 : {mean_squared_error(y_test, lstm_preds):.2f}")
print(f"  2-WAY Ensemble MSE       : {mean_squared_error(y_test, (xgb_preds + lstm_preds) / 2):.2f}")
print(f"  3-WAY Ensemble MSE       : {mean_squared_error(y_test, (lgb_preds + xgb_preds + lstm_preds) / 3):.2f}")

# =============================================================================
#  7. ASSIGNMENT FORECAST (MAY 12-14) - SAVING ALL MODELS
# =============================================================================
X_final_future = future_df[FEATURE_COLS].values
X_final_future_scaled = scaler.transform(X_final_future)

lgb_f_preds = lgb_model.predict(X_final_future_scaled)
xgb_f_preds = xgb_model.predict(xgb.DMatrix(X_final_future_scaled, feature_names=FEATURE_COLS))
lstm_f_preds = lstm_model.predict(X_final_future_scaled.reshape((X_final_future_scaled.shape[0], 1, len(FEATURE_COLS))), verbose=0).flatten()

all_predictions = {
    "TUNED_LGBM": lgb_f_preds,
    "TUNED_XGB": xgb_f_preds,
    "TUNED_LSTM": lstm_f_preds,
    "TUNED_2-WAY": (xgb_f_preds*0.4 + lstm_f_preds*0.6),
    "TUNED_3-WAY": (lgb_f_preds*0.2 + xgb_f_preds*0.2 + lstm_f_preds*0.6)
}

print("\n── Generating CSV Files ──")
for model_name, preds in all_predictions.items():
    csv_filename = f"{model_name}_predictions.csv"
    pd.DataFrame(preds).to_csv(csv_filename, index=False, header=False)
    print(f"  SUCCESS: Saved {csv_filename}")

# =============================================================================
#  8. TRUE BLIND EVALUATION (May 12 - May 14) - SCORING ALL MODELS
# =============================================================================
print("\n── TRUE BLIND EVALUATION (May 12 - May 14) ──")
actuals_df = pd.read_csv("Actual_Hourly_Belpex_Prices.csv")
actuals_df['datetime'] = pd.to_datetime(actuals_df['Date'] + ' ' + actuals_df['Time'], format='%d/%m/%Y %H:%M')
actuals_df = actuals_df.sort_values('datetime').reset_index(drop=True)

mask = actuals_df['datetime'] >= "2026-05-12 00:00:00"
available_actuals = actuals_df.loc[mask].copy()
true_prices = available_actuals['Actual_Price_BE'].values

if len(true_prices) > 0:
    print(f"  Evaluating {len(true_prices)} hours of real data...")
    for model_name, preds in all_predictions.items():
        matched_preds = preds[:len(true_prices)]
        print(f"  TRUE {model_name} MSE: {mean_squared_error(true_prices, matched_preds):.2f}")

# =============================================================================
#  9. PLOTS - SAVING ALL MODELS
# =============================================================================
print("\n── Generating Plots ──")
for model_name, preds in all_predictions.items():
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(available_actuals['datetime'], true_prices, label="Actual Price", color="black", lw=2)
    axes[0].plot(available_actuals['datetime'], preds[:len(available_actuals)], 
                 label=f"{model_name} Forecast", color="tomato", ls='--', lw=2)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    axes[0].set_title(f"Belpex Forecast vs Actuals: {model_name} Model", fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    if model_name in ["TUNED_LGBM", "TUNED_3-WAY"]:
        imp = pd.DataFrame({"feature": lgb_model.feature_name(), "gain": lgb_model.feature_importance(importance_type='gain')})
        ref_model_name = "LGBM"
    else:
        imp_dict = xgb_model.get_score(importance_type="gain")
        imp = pd.DataFrame({"feature": list(imp_dict.keys()), "gain": list(imp_dict.values())})
        ref_model_name = "XGBoost"

    imp = imp.sort_values("gain", ascending=False).head(10)
    axes[1].barh(imp["feature"][::-1], imp["gain"][::-1], color="teal")
    axes[1].set_title(f"Top 10 Features (Gain) - Reference: {ref_model_name}")
    
    plt.tight_layout()
    out_name = f"{model_name}_performance.png"
    plt.savefig(out_name, dpi=300)
    plt.close(fig) 
    print(f"  SUCCESS: Saved {out_name}")

lstm_model.save("best_lstm_model.keras")

# Save the XGBoost model
xgb_model.save_model("best_xgb_model.json")

# Save the LightGBM model
lgb_model.save_model("best_lgb_model.txt")

print("\nCHECKPOINT: All models have been saved to the current directory!")
# =============================================================================
# 10. RELOAD & VERIFY (FINAL CHECK)
# =============================================================================
print("\n── Final Verification: Reloading Saved Model ──")

# Load the model back from the disk
reloaded_model = tf.keras.models.load_model("best_lstm_model.keras")

# Prepare input (reshape exactly like Section 7)
X_verify = X_final_future_scaled.reshape((X_final_future_scaled.shape[0], 1, X_final_future_scaled.shape[1]))

# Generate predictions with the reloaded model
reloaded_preds = reloaded_model.predict(X_verify, verbose=0).flatten()

# Compare with the original "live" predictions
difference = np.abs(all_predictions["TUNED_LSTM"] - reloaded_preds).sum()

if difference < 1e-5:
    print("  VERIFICATION SUCCESS: Reloaded model matches live predictions exactly!")
    print(f"  Reloaded MSE (Blind Test): {mean_squared_error(true_prices, reloaded_preds[:len(true_prices)]):.2f}")
else:
    print(f"  VERIFICATION WARNING: Difference of {difference:.6f} detected.")
    print("  (Minor differences are normal on GPUs, but values should be nearly identical.)")

# Save a separate verification CSV just to be safe
pd.DataFrame(reloaded_preds).to_csv("VERIFIED_LSTM_predictions.csv", index=False, header=False)
print("  SUCCESS: VERIFIED_LSTM_predictions.csv saved.")