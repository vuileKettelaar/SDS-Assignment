import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA & FEATURE ENGINEERING
# ==========================================
print("Loading data and engineering features...")
df_hourly = pd.read_csv('Full_Dataset.csv')
df_hourly['Date'] = pd.to_datetime(df_hourly['Date'], dayfirst=True)
df_hourly = df_hourly.set_index('Date').sort_index()

# Calendar Features
df_hourly['Hour'] = df_hourly.index.hour
df_hourly['DayOfWeek'] = df_hourly.index.dayofweek
df_hourly['Is_Weekend'] = df_hourly['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Cyclical encoding for the hour
df_hourly['Hour_Sin'] = np.sin(df_hourly['Hour'] * (2. * np.pi / 24))
df_hourly['Hour_Cos'] = np.cos(df_hourly['Hour'] * (2. * np.pi / 24))

# ==========================================
# 2. UNIFIED FEATURE SELECTION
# ==========================================
# Unified list (11 features) used for both Training and Benchmarking
features = [
    'Price_BE', 'Load_FR', 'Gen_FR', 'Price_CH', 'Wind_BE', 'Solar_BE', 'Load_BE',
    'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'Is_Weekend'
]

# Create training/scaling data
data = df_hourly[features].dropna()

# ==========================================
# 3. SCALING
# ==========================================
print("Scaling features...")
scaler_x = StandardScaler()
scaler_y = StandardScaler()

data_scaled = data.copy()
data_scaled[features] = scaler_x.fit_transform(data[features])
price_scaled = scaler_y.fit_transform(data['Price_BE'].values.reshape(-1, 1))

# ==========================================
# 4. SEQUENCE GENERATION
# ==========================================
print("Generating 168-hour sequences...")
lookback = 7 
hours_lookback = lookback * 24
days = len(data) // 24

X_lstm, Y_lstm = [], []
for d in range(lookback, days - 1): 
    past = data_scaled[features].values[(d-lookback)*24 : d*24]
    future = price_scaled[d*24 : (d+1)*24]
    
    if len(past) == hours_lookback and len(future) == 24:
        X_lstm.append(past)
        Y_lstm.append(future.flatten())

X_lstm = np.array(X_lstm)
Y_lstm = np.array(Y_lstm)

# ==========================================
# 5. DATA SPLIT (Chronological)[cite: 2]
# ==========================================
X_tv, X_test, Y_tv, Y_test = train_test_split(X_lstm, Y_lstm, test_size=0.15, shuffle=False)
X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=0.176, shuffle=False)

# ==========================================
# 6. MODEL ARCHITECTURE & TRAINING
# ==========================================
actual_features = X_train.shape[2] 
print(f"Building Bidirectional LSTM with {actual_features} features...")

model_lstm = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(168, actual_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(24, activation='linear')
])

model_lstm.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=0.001), metrics=['mae', 'mse'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

print("Starting training...")
model_lstm.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1)

# ==========================================
# 7. SAVING
# ==========================================
save_dir = "./trained_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_lstm.save(os.path.join(save_dir, "LSTM_model_1.keras"))
with open(os.path.join(save_dir, "LSTM_scaler_x.pkl"), 'wb') as f: pickle.dump(scaler_x, f)
with open(os.path.join(save_dir, "LSTM_scaler_y.pkl"), 'wb') as f: pickle.dump(scaler_y, f)

# ==========================================
# 8. BENCHMARKING (Last 72 Hours)
# ==========================================
print("\nRunning Benchmark on last 72 hours...")
test_actuals = df_hourly['Price_BE'].tail(72)
baseline_results = pd.DataFrame(index=test_actuals.index)
baseline_results['Actual'] = test_actuals.values

# Calculate Persistence Baselines[cite: 2]
naive_weekly, avg_hourly_week = [], []
for timestamp in test_actuals.index:
    t_minus_7 = timestamp - pd.Timedelta(days=7)
    naive_weekly.append(df_hourly.loc[t_minus_7, 'Price_BE'] if t_minus_7 in df_hourly.index else np.nan)
    past_7_days = [df_hourly.loc[timestamp - pd.Timedelta(days=d), 'Price_BE'] for d in range(1, 8) if (timestamp - pd.Timedelta(days=d)) in df_hourly.index]
    avg_hourly_week.append(np.mean(past_7_days) if past_7_days else np.nan)

baseline_results['Naive_Weekly'] = naive_weekly
baseline_results['Avg_Hour_Last_Week'] = avg_hourly_week

# Generate LSTM Predictions
df_features = df_hourly[features].copy()
scaled_features = scaler_x.transform(df_features.ffill()) # ffill handles small gaps
df_scaled = pd.DataFrame(scaled_features, index=df_features.index, columns=features)

start_times = [test_actuals.index[0], test_actuals.index[24], test_actuals.index[48]]
lstm_preds = []

for start_time in start_times:
    input_seq = df_scaled.loc[start_time - pd.Timedelta(hours=168) : start_time - pd.Timedelta(hours=1)].values
    input_seq = input_seq.reshape(1, 168, actual_features)
    pred_real = scaler_y.inverse_transform(model_lstm.predict(input_seq, verbose=0))
    lstm_preds.extend(pred_real[0])

baseline_results['LSTM_Predict'] = lstm_preds

# ==========================================
# 9. METRICS & PLOT[cite: 1]
# ==========================================
def get_metrics(actual, predicted, name):
    mask = ~np.isnan(predicted)
    mse = mean_squared_error(actual[mask], predicted[mask])
    mae = mean_absolute_error(actual[mask], predicted[mask])
    return {'Model': name, 'MSE': round(mse, 2), 'MAE': round(mae, 2)}

metrics_summary = [
    get_metrics(baseline_results['Actual'], baseline_results['Naive_Weekly'], 'Naive Weekly'),
    get_metrics(baseline_results['Actual'], baseline_results['Avg_Hour_Last_Week'], '7-Day Hourly Average'),
    get_metrics(baseline_results['Actual'], baseline_results['LSTM_Predict'], 'Upgraded LSTM')
]

print("\n--- Model Performance Comparison (Last 72 Hours) ---")
print(pd.DataFrame(metrics_summary).to_string(index=False))

plt.figure(figsize=(15, 8))
plt.plot(baseline_results.index, baseline_results['Actual'], label='Actual Price', color='black', linewidth=2)
plt.plot(baseline_results.index, baseline_results['LSTM_Predict'], label='Upgraded LSTM Forecast', color='blue', linewidth=2)
plt.title('Upgraded Day-Ahead Electricity Price Forecast Comparison')
plt.ylabel('Price [EUR/MWh]')
plt.legend()
plt.show()