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
# Remove: import xgboost as xgb
import lightgbm as lgb
# =============================================================================
#  1. LOAD DATA & APPEND FUTURE DATES
# =============================================================================
DATA_PATH  = "Full_Dataset_Updated.csv"     
TARGET_COL = "Price_BE"
DATE_COL   = "Date"

df_history = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL], dayfirst=True)
df_history = df_history.sort_values(DATE_COL).reset_index(drop=True)

# Handle Missing Values in History
EXOG_COLS = ["Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE"]
df_history[EXOG_COLS] = df_history[EXOG_COLS].ffill().bfill()
df_history["Net_Load_BE"] = df_history["Load_BE"] - df_history["Wind_BE"] - df_history["Solar_BE"]

# Create the future 72 hours for May 12-14
future_dates = pd.date_range(start="2026-05-12 00:00:00", periods=72, freq="h")
df_future = pd.DataFrame({DATE_COL: future_dates})

# Combine them so we can calculate lags flawlessly in one step
df = pd.concat([df_history, df_future], ignore_index=True)

# =============================================================================
#  2. FEATURE ENGINEERING (MOMENTUM + 72-HOUR GAP)
# =============================================================================
# Calendar features
df["hour"]        = df[DATE_COL].dt.hour
df["day_of_week"] = df[DATE_COL].dt.dayofweek   
df["month"]       = df[DATE_COL].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["hour_weekend_interaction"] = df["hour"] * df["is_weekend"]

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Target Lags (Added 192 for weekly momentum)
for lag in [72, 96, 168, 192]:
    df[f"Price_BE_lag{lag}"] = df[TARGET_COL].shift(lag)

# Target Momentum (Is the price rising or falling compared to the previous day/week?)
df["Price_Velocity_72_vs_96"] = df["Price_BE_lag72"] - df["Price_BE_lag96"]
df["Price_Velocity_168_vs_192"] = df["Price_BE_lag168"] - df["Price_BE_lag192"]

# Rolling stats
df["Price_BE_roll72_mean"] = df[TARGET_COL].shift(72).rolling(24).mean()
df["Price_BE_roll168_mean"] = df[TARGET_COL].shift(168).rolling(24).mean()

# Exogenous Lags & Momentum
for col in EXOG_COLS + ["Net_Load_BE"]:
    df[f"{col}_lag72"] = df[col].shift(72)
    df[f"{col}_lag96"] = df[col].shift(96)
    df[f"{col}_lag168"] = df[col].shift(168)
    
    # Exogenous Momentum
    df[f"{col}_Velocity_72_vs_96"] = df[f"{col}_lag72"] - df[f"{col}_lag96"]
# =============================================================================
#  3. SPLIT HISTORY AND FUTURE
# =============================================================================
# The features we train on. Notice NO current-hour weather or short lags!
# The features we train on (Now includes Velocity/Momentum)
FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "hour_weekend_interaction",
    "Price_BE_lag72", "Price_BE_lag96", "Price_BE_lag168",
    "Price_Velocity_72_vs_96", "Price_Velocity_168_vs_192", # <--- NEW
    "Price_BE_roll72_mean", "Price_BE_roll168_mean",
] + [f"{col}_lag72" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_lag168" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_Velocity_72_vs_96" for col in EXOG_COLS + ["Net_Load_BE"]] # <--- NEW

# Separate the dataset back into History and Future
train_df = df[df[DATE_COL] < "2026-05-12 00:00:00"].copy()
future_df = df[df[DATE_COL] >= "2026-05-12 00:00:00"].copy()

# Drop rows with NaNs caused by the 168h shift in the training set
train_df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
train_df.reset_index(drop=True, inplace=True)

X = train_df[FEATURE_COLS].values
y = train_df[TARGET_COL].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
#  4. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
horizon = 72
X_train_val = X_scaled[:-horizon]
y_train_val = y[:-horizon]

# Held out test set (May 9 - May 11) to measure true assignment accuracy
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
#  5. TRAIN XGBOOST & LSTM
# =============================================================================
# print("\nTraining XGBoost …")
# dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
# dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)
# dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_COLS)

# xgb_params = {
#     "objective": "reg:squarederror", 
#     "eval_metric": "rmse", 
#     "learning_rate": 0.01,    # LOWER learning rate (slower, but more precise)
#     "max_depth": 8,           # DEEPER trees (allows it to find more complex interactions)
#     "min_child_weight": 3,    # Lets the tree create slightly more specific leaf nodes
#     "subsample": 0.8, 
#     "colsample_bytree": 0.8
# }

# # Because we lowered the learning rate to 0.01, we need to let it build MORE trees to compensate
# xgb_model = xgb.train(
#     xgb_params, dtrain, 
#     num_boost_round=3000,     # Increased from 1000
#     evals=[(dtrain, "train"), (dval, "val")], 
#     early_stopping_rounds=100, # Increased patience
#     verbose_eval=False
# )
print("\nTraining LightGBM …")
# LightGBM has its own highly optimized Dataset format
train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_COLS)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,    # Slowed down for precision
    'max_depth': 8,
    'num_leaves': 63,         # Crucial for LightGBM (2^max_depth is the absolute max)
    'feature_fraction': 0.8,  # Same as colsample_bytree
    'bagging_fraction': 0.8,  # Same as subsample
    'bagging_freq': 1,
    'verbose': -1,            # Keeps the terminal clean
    'random_state': 42
}

# Train the model with early stopping
lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=3000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
)

print("Training LSTM …")
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    Dropout(0.2), LSTM(32), Dropout(0.2),
    Dense(16, activation='relu'), Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, validation_data=(X_val_lstm, y_val), 
               callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)

# =============================================================================
#  6. EVALUATE ON MAY 9-11
# =============================================================================
# xgb_preds = xgb_model.predict(dtest)
# lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()
# ensemble_preds = (xgb_preds + lstm_preds) / 2
# Change xgb_preds to lgb_preds
lgb_preds = lgb_model.predict(X_test)
lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()
ensemble_preds = (lgb_preds + lstm_preds) / 2

# Update your print statements to say "LightGBM MSE" instead of XGBoost
print("\n── Validation Metrics (Strict >72h Gap) ──")
print(f"  XGBoost MSE  : {mean_squared_error(y_test, lgb_preds):.2f}")
print(f"  LSTM MSE     : {mean_squared_error(y_test, lstm_preds):.2f}")
print(f"  Ensemble MSE : {mean_squared_error(y_test, ensemble_preds):.2f}")

# =============================================================================
#  7. THE ASSIGNMENT FORECAST (MAY 12 - MAY 14)
# =============================================================================
# Because our features are shifted by 72+ hours, the future data is ALREADY populated!
X_final_future = future_df[FEATURE_COLS].values
X_final_future_scaled = scaler.transform(X_final_future)

# # Predict all 72 hours instantly. NO loops. NO error compounding.
# xgb_future_preds = xgb_model.predict(xgb.DMatrix(X_final_future_scaled, feature_names=FEATURE_COLS))
# lstm_future_preds = lstm_model.predict(X_final_future_scaled.reshape((72, 1, len(FEATURE_COLS))), verbose=0).flatten()
# final_predictions = (xgb_future_preds + lstm_future_preds) / 2
# Change xgb_future_preds to lgb_future_preds
lgb_future_preds = lgb_model.predict(X_final_future_scaled)
lstm_future_preds = lstm_model.predict(X_final_future_scaled.reshape((72, 1, len(FEATURE_COLS))), verbose=0).flatten()
final_predictions = (lgb_future_preds + lstm_future_preds) / 2

# Export
pd.DataFrame(final_predictions).to_csv("72predictions.csv", index=False, header=False)
print(f"\nSUCCESS: 72predictions.csv generated. First value: {final_predictions[0]:.2f}")
# =============================================================================
#  8. PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(y_test, label="Actual (May 9-11)", color="black", lw=2)
axes[0].plot(ensemble_preds, label="Hybrid Pred", color="tomato", linestyle='--', lw=2)
axes[0].set_title("Test Set Evaluation (Strict 72h Gap)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Extract LightGBM Feature Importances
features = lgb_model.feature_name()
importances = lgb_model.feature_importance(importance_type='gain')

# Create DataFrame and sort
imp = pd.DataFrame({
    "feature": features,
    "gain": importances
}).sort_values("gain", ascending=False).head(10)

axes[1].barh(imp["feature"][::-1], imp["gain"][::-1], color="teal")
axes[1].set_title("Top 10 Features (Notice how it relies heavily on lag72 and lag168)")

plt.tight_layout()
plt.savefig("model_performance.png", dpi=300)
print("Plot saved as model_performance.png")