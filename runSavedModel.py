import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. SETUP PATHS
MODEL_PATH = "best_lstm_model.keras"
DATA_PATH = "Full_Dataset_Updated.csv"
ACTUALS_PATH = "Actual_Hourly_Belpex_Prices.csv"
TARGET_COL = "Price_BE"
DATE_COL = "Date"

# 2. LOAD MODEL & DATA
print("Loading model and data...")
model = tf.keras.models.load_model(MODEL_PATH)
df_history = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL], dayfirst=True)
df_history = df_history.sort_values(DATE_COL).reset_index(drop=True)

# 3. CONCAT FUTURE WINDOW (Ensure we cover the full May 12-14 period)
# We start the future dates at the exact point history ends to keep lags consistent
last_hist_date = df_history[DATE_COL].max()
future_dates = pd.date_range(start=last_hist_date + pd.Timedelta(hours=1), periods=96, freq="h")
df_future = pd.DataFrame({DATE_COL: future_dates})
df = pd.concat([df_history, df_future], ignore_index=True)

# 4. FEATURE ENGINEERING
EXOG_COLS = ["Load_FR", "Gen_FR", "Price_CH", "Wind_BE", "Solar_BE", "Load_BE"]
df[EXOG_COLS] = df[EXOG_COLS].ffill().bfill()
df["Net_Load_BE"] = df["Load_BE"] - df["Wind_BE"] - df["Solar_BE"]

df["hour"] = df[DATE_COL].dt.hour
df["day_of_week"] = df[DATE_COL].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["hour_weekend_interaction"] = df["hour"] * df["is_weekend"]
df["month"] = df[DATE_COL].dt.month

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

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

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "hour_weekend_interaction",
    "Price_BE_lag72", "Price_BE_lag96", "Price_BE_lag168",
    "Price_Velocity_72_vs_96", "Price_Velocity_168_vs_192", 
    "Price_BE_roll72_mean", "Price_BE_roll168_mean",
] + [f"{col}_lag72" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_lag168" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_Velocity_72_vs_96" for col in EXOG_COLS + ["Net_Load_BE"]] \
  + [f"{col}_roll168_mean" for col in EXOG_COLS + ["Net_Load_BE"]]

# 5. SCALING (Fit on history, transform the target window)
train_history = df[df[DATE_COL] < "2026-05-12 00:00:00"].dropna(subset=FEATURE_COLS + [TARGET_COL])
scaler = MinMaxScaler()
scaler.fit(train_history[FEATURE_COLS].values)

# Align target window EXACTLY with the blind test: May 12 00:00 to May 14 23:00
target_window = df[(df[DATE_COL] >= "2026-05-12 00:00:00") & (df[DATE_COL] <= "2026-05-14 23:00:00")].copy()

# Critical: Check if we have 72 rows. If not, ffill the remaining features.
target_window[FEATURE_COLS] = target_window[FEATURE_COLS].ffill().bfill()

X_input = scaler.transform(target_window[FEATURE_COLS].values)
X_input_lstm = X_input.reshape((X_input.shape[0], 1, X_input.shape[1]))

# 6. PREDICT
print(f"Generating predictions for {len(X_input_lstm)} hours...")
predictions = model.predict(X_input_lstm).flatten()

# 7. SAVE RESULTS
output_filename = "Reloaded_LSTM_predictions.csv"
pd.DataFrame(predictions).to_csv(output_filename, index=False, header=False)
print(f"SUCCESS: Forecast saved to {output_filename}")

# 8. VALIDATION
print("Loading actual prices for validation...")
actuals_df = pd.read_csv(ACTUALS_PATH)
actuals_df['datetime'] = pd.to_datetime(actuals_df['Date'] + ' ' + actuals_df['Time'], format='%d/%m/%Y %H:%M')
actuals_df = actuals_df.sort_values('datetime').reset_index(drop=True)

mask = (actuals_df['datetime'] >= "2026-05-12 00:00:00") & (actuals_df['datetime'] <= "2026-05-14 23:00:00")
true_prices = actuals_df.loc[mask, 'Actual_Price_BE'].values

if len(true_prices) == len(predictions):
    mse = mean_squared_error(true_prices, predictions)
    print(f"\n── RELOADED MODEL MSE: {mse:.4f} ──")
else:
    print(f"Length mismatch: Actuals ({len(true_prices)}) vs Preds ({len(predictions)})")