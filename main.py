
# import os
# import sys

# # 1. Point Python to the NVIDIA libraries we just installed in the venv
# venv_base = os.path.dirname(sys.executable)
# lib_path = os.path.join(venv_base, "..", "lib", "python3.13", "site-packages", "nvidia")

# # Add all the subdirectories (cudnn, cublas, etc.) to the search path
# if os.path.exists(lib_path):
#     for root, dirs, files in os.walk(lib_path):
#         if 'lib' in dirs:
#             lib_dir = os.path.join(root, 'lib')
#             os.environ['LD_LIBRARY_PATH'] = lib_dir + ":" + os.environ.get('LD_LIBRARY_PATH', '')

# # 2. Now import tensorflow
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# if len(tf.config.list_physical_devices('GPU')) == 0:
#     print("WARNING: Running on CPU. Training will be slower.")
# else:
#     print("SUCCESS: RTX 4050 detected and ready.")

# import pandas as pd

# def load_and_clean_data(file_path):
#     df = pd.read_csv(file_path)
#     # 2. Convert 'Date' column to actual Python datetime objects
#     df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
#     # 3. Set the Date as the index
#     df.set_index('Date', inplace=True)
    
#     # 4. Handle missing values
#     df = df.ffill() #if a value is missing it will be copied to the next time stamp as the value of that timestamp
    
#     return df

# data = load_and_clean_data("Full_Dataset.csv")
# print(data.head())

# def create_lags(df, target_col, lags=[1, 2, 24, 168]):
# #This is the lagging part, makes sure our neural network has memory
#     for lag in lags:
#         df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
#     df.dropna(inplace=True)
#     return df

# data_with_memory = create_lags(data, 'Price_BE')

# def add_calendar_features(df):
#     # Electricity prices are highly cyclical (e.g., peak hours, weekdays vs weekends)
#     df['hour'] = df.index.hour
#     df['day_of_week'] = df.index.dayofweek
#     return df
# #Before we are actually creating the lags that we will use we will first
# #creat these calendar features so the model knows what day it is


# def create_lags(df, target_col='Price_BE', lags=[1, 2, 24, 168]):
#     # This creates the 'Autoregressive' (AR) part of your ARX model
#     # We use historical values to help the model see patterns over time.
#     for lag in lags:
#         df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

#     df.dropna(inplace=True)
#     return df

# # Usage in your script:
# data = load_and_clean_data("Full_Dataset.csv")
# data = add_calendar_features(data)
# data = create_lags(data)

# def split_data(df):
#     # We will split chronologically: Train (80%), Validation (10%), Test (10%)[cite: 1]
#     n = len(df)
#     train_df = df[0:int(n*0.8)]
#     val_df = df[int(n*0.8):int(n*0.9)]
#     test_df = df[int(n*0.9):]
    
#     return train_df, val_df, test_df


# train, val, test = split_data(data)

# #it's important to scale the data, it will center around 0 with a standard scaler now
# #point of improvement is looking for ohter scalement maybe :)
# from sklearn.preprocessing import StandardScaler

# def scale_data(train_df, val_df, test_df):
#     # 1. Initialize the scaler
#     scaler = StandardScaler()
    
#     # 2. Fit the scaler ONLY on the training data
#     scaler.fit(train_df)
    
#     # 3. Transform all three sets using the training parameters
#     train_scaled = scaler.transform(train_df)
#     val_scaled = scaler.transform(val_df)
#     test_scaled = scaler.transform(test_df)
    
#     return train_scaled, val_scaled, test_scaled, scaler

# # Usage in your script:
# train_s, val_s, test_s, scaler_obj = scale_data(train, val, test)

# def prepare_xy(scaled_array, target_index):
#     # X = all columns
#     # y = just the target column (Price_BE)
#     X = scaled_array
#     y = scaled_array[:, target_index]
    
#     return X, y

# # Example: If Price_BE is the 1st column (index 0)
# X_train, y_train = prepare_xy(train_s, 0)

# import numpy as np

# def create_sequences(data, target_index, look_back=168, forecast_horizon=72):
#     """
#     Chops data into windows.
#     look_back: 168 hours (1 week of past data)
#     forecast_horizon: 72 hours (3 days of future prediction)
#     """
#     X, y = [], []
    
#     # We loop through the data, stopping before we run out of future rows to predict
#     for i in range(len(data) - look_back - forecast_horizon + 1):
#         # 1. Grab the 'past' features (the window)
#         X.append(data[i : (i + look_back), :])
        
#         # 2. Grab the 'future' targets (the 72 hours of Price_BE)
#         y.append(data[(i + look_back) : (i + look_back + forecast_horizon), target_index])
        
#     return np.array(X), np.array(y)

# # Usage:
# target_idx = train.columns.get_loc('Price_BE')
# X_train, y_train = create_sequences(train_s, target_idx)
# X_val, y_val = create_sequences(val_s, target_idx)
# X_test, y_test = create_sequences(test_s, target_idx)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# def build_lstm_model(input_shape, output_steps):
#     model = Sequential([
#         # 1. LSTM Layer: Processes the sequences and extracts temporal patterns
#         LSTM(units=64, input_shape=input_shape, return_sequences=False),
        
#         # 2. Dropout: A regularization technique mentioned in slides to prevent overfitting
#         Dropout(0.2),
        
#         # 3. Dense Layer: The final layer that produces the 72-hour forecast[cite: 1, 2]
#         Dense(units=output_steps)
#     ])
    
#     # Use 'adam' optimizer and 'mse' loss as required by assignment evaluation[cite: 2]
#     model.compile(optimizer='adam', loss='mse')
#     return model
# # input_shape is (time_steps, features)
# # output_steps is the 72-hour forecast horizon
# my_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_steps=72)
# from tensorflow.keras.callbacks import EarlyStopping

# def train_nn_model(model, X_train, y_train, X_val, y_val):
#     # 1. Setup Early Stopping to prevent overfitting
#     # patience=10 means if the error doesn't improve for 10 'epochs', we stop.
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
#     # 2. Fit the model
#     # epochs: The maximum number of times the model sees the entire dataset.
#     # batch_size: Number of samples processed before the model updates its weights[cite: 1].
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=100, 
#         batch_size=32,
#         callbacks=[early_stop],
#         verbose=1
#     )
    
#     return history

# # Usage in your script:
# history = train_nn_model(my_model, X_train, y_train, X_val, y_val)

# def generate_final_submission(model, full_scaled_data, target_scaler, target_idx, look_back=168):
#     # 1. Get the last window of data (the 168 hours before the forecast starts)
#     # Full_Dataset.csv ends on 11/05/2026 23:00[cite: 2]
#     last_window = full_scaled_data[-look_back:] 
#     last_window = np.expand_dims(last_window, axis=0) # Shape: (1, 168, features)
    
#     # 2. Predict the scaled prices
#     scaled_prediction = model.predict(last_window)
    
#     # 3. Inverse scale to get actual EUR/MWh prices
#     # Note: Ensure you have a scaler specifically for the Price_BE column
#     final_prices = target_scaler.inverse_transform(scaled_prediction).flatten()
    
#     # 4. Save to CSV in the exact required format[cite: 2]
#     # No header, no index, 72 rows
#     pd.DataFrame(final_prices).to_csv('predictions.csv', index=False, header=False)
#     print("Submission file 'predictions.csv' generated successfully!")
#     return final_prices
# import matplotlib.pyplot as plt

# import os

# def play_finished_sound():
#     # Fedora uses PipeWire/PulseAudio. 'paplay' is the standard command-line player.
#     # We use a standard GNOME alert sound. 
#     sound_file = "/usr/share/sounds/gnome/default/alerts/glass.ogg"
    
#     if os.path.exists(sound_file):
#         os.system(f"paplay {sound_file}")
#     else:
#         # Fallback: a simple terminal beep (might not work in all terminals)
#         print('\a')
#         print("Training complete! (Sound file not found, terminal beep attempted)")

# # --- At the end of your script ---
# # generate_final_submission(...)
# play_finished_sound()

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Bidirectional, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib

# --- 1. GPU LINKING (DO NOT REMOVE) ---
venv_base = os.path.dirname(sys.executable)
lib_path = os.path.join(venv_base, "..", "lib", "python3.13", "site-packages", "nvidia")
if os.path.exists(lib_path):
    for root, dirs, files in os.walk(lib_path):
        if 'lib' in dirs:
            lib_dir = os.path.join(root, 'lib')
            os.environ['LD_LIBRARY_PATH'] = lib_dir + ":" + os.environ.get('LD_LIBRARY_PATH', '')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("WARNING: Running on CPU. Training will be slower.")
else:
    print("SUCCESS: RTX 4050 detected and ready.")

# --- 2. DATA PROCESSING FUNCTIONS ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # FIX: Remove rows with the same timestamp before setting the index
    df = df.drop_duplicates(subset='Date')
    
    df.set_index('Date', inplace=True)
    df = df.sort_index() # Essential for time-series shifting
    df = df.ffill() 
    return df

def add_engineered_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['Net_Load_BE'] = df['Load_BE'] - (df['Wind_BE'] + df['Solar_BE'])
    df['FR_Balance'] = df['Load_FR'] - df['Gen_FR']
    
    # Momentum features
    df['Price_Diff'] = df['Price_BE'].diff().bfill() 
    df['Price_MA_6'] = df['Price_BE'].rolling(window=6).mean().bfill() 
    df['Price_MA_24'] = df['Price_BE'].rolling(window=24).mean().bfill()
    
    return df

def create_lags(df, target_col='Price_BE', lags=[1, 2, 24]):
    cols_to_lag = [c for c in df.columns if 'lag' not in c and c != target_col]
    for col in cols_to_lag:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

# --- MODIFIED: EXACT 72H TEST SPLIT ---
def split_data_final_72h(df, look_back=168, forecast_horizon=72):
    """
    Ensures the test set is exactly the final 72-hour sequence.
    """
    # The last sequence requires the last 168h of input + 72h of target
    test_df = df.iloc[-(look_back + forecast_horizon):]
    
    # Training and Validation come from everything BEFORE those final 72 targets
    train_val_df = df.iloc[:-(forecast_horizon)]
    
    n = len(train_val_df)
    train_df = train_val_df.iloc[:int(n*0.9)]
    val_df = train_val_df.iloc[int(n*0.9):]
    
    return train_df, val_df, test_df

def scale_data(train_df, val_df, test_df, target_col='Price_BE'):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(train_df)
    train_s = scaler_X.transform(train_df)
    val_s = scaler_X.transform(val_df)
    test_s = scaler_X.transform(test_df)
    scaler_y.fit(train_df[[target_col]])
    return train_s, val_s, test_s, scaler_y

def create_sequences(data, target_idx, look_back=168, forecast_horizon=72):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i : (i + look_back), :])
        y.append(data[(i + look_back) : (i + look_back + forecast_horizon), target_idx]) 
    return np.array(X), np.array(y)

# --- 3. MODEL ARCHITECTURE ---

def build_lstm_model(input_shape, output_steps):
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.1),
        RepeatVector(output_steps),
        LSTM(64, return_sequences=True),
        Dropout(0.05),
        TimeDistributed(Dense(1)),
        Flatten()
    ])
    model.compile(optimizer=Adam(learning_rate=0.00075), loss='mse')
    return model

# --- 4. VISUALIZATION & UTILITIES ---

def save_trained_model(model, filename='sds_lstm_model.keras'):
    model.save(filename)
    print(f"Model saved successfully as '{filename}'")

def load_existing_model(filename='sds_lstm_model.keras'):
    if os.path.exists(filename):
        return tf.keras.models.load_model(filename)
    return None
    
# Headless Plotting Fix
matplotlib.use('Agg') 
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_learning_curve(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Learning Curve')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, 'learning_curve.png')); plt.close()

def plot_detailed_evaluation(model, X_data, y_data_scaled, scaler_y, set_name="Validation"):
    scaled_predictions = model.predict(X_data)
    actual_prices = scaler_y.inverse_transform(y_data_scaled)
    predicted_prices = scaler_y.inverse_transform(scaled_predictions)
    
    plt.figure(figsize=(15, 6))
    plt.plot(actual_prices[:168, 0], label='Actual', color='black', alpha=0.7)
    plt.plot(predicted_prices[:168, 0], label='Predicted', color='blue', linestyle='--')
    plt.title(f'{set_name}: Forecast Comparison')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f'forecast_{set_name}.png')); plt.close()

    mse = np.mean((actual_prices - predicted_prices)**2)
    print(f"--- {set_name} MSE: {mse:.4f} ---")

from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_baselines(df, test_df):
    # Ensure the full price series has a unique index to prevent reindexing errors
    full_prices = df['Price_BE'][~df.index.duplicated(keep='first')]
    
    # We use the ENTIRE test set to match the standard baseline calculation
    test_actual = test_df['Price_BE']
    test_index = test_actual.index.drop_duplicates(keep='first')
    
    # Reindex using the unique test index
    daily_pred = full_prices.shift(24).reindex(test_index)
    weekly_pred = full_prices.shift(168).reindex(test_index)
    avg_pred = full_prices.rolling(window=168).mean().shift(1).reindex(test_index)
    
    # Create a DataFrame to easily drop any remaining NaNs
    results = pd.DataFrame({
        'Actual': test_actual[~test_actual.index.duplicated(keep='first')],
        'Daily': daily_pred,
        'Weekly': weekly_pred,
        'Avg': avg_pred
    }).dropna()

    def get_metrics(true, pred):
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        return mse, mae

    mse_d, mae_d = get_metrics(results['Actual'], results['Daily'])
    mse_w, mae_w = get_metrics(results['Actual'], results['Weekly'])
    mse_a, mae_a = get_metrics(results['Actual'], results['Avg'])

    print("\n--- Standard Baseline Scores (Full Test Set) ---")
    print(f"Daily persistence 72h -> MSE: {mse_d:.4f}, MAE: {mae_d:.4f}")
    print(f"Weekly persistence 72h -> MSE: {mse_w:.4f}, MAE: {mae_w:.4f}")
    print(f"Recent weekly average 72h -> MSE: {mse_a:.4f}, MAE: {mae_a:.4f}")

# --- 5. MAIN EXECUTION FLOW ---

data = load_and_clean_data("Full_Dataset.csv")
data = add_engineered_features(data)
data = create_lags(data)

# Splitting specifically for exactly 72h test sequence
train, val, test = split_data_final_72h(data)
target_idx = train.columns.get_loc('Price_BE')
train_s, val_s, test_s, scaler_y = scale_data(train, val, test)

X_train, y_train = create_sequences(train_s, target_idx)
X_val, y_val = create_sequences(val_s, target_idx)
X_test, y_test = create_sequences(test_s, target_idx) # Should produce 1 sequence

model_file = 'sds_lstm_model.keras'
if os.path.exists(model_file):
    print("Skipping training, loading existing model...")
    my_model = load_existing_model(model_file)
else:
    my_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_steps=72)
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    
    print("Starting fresh training...")
    history = my_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                           epochs=100, batch_size=32, callbacks=[early_stop, lr_scheduler])
    plot_learning_curve(history)
    save_trained_model(my_model, model_file)

# Evaluate
plot_detailed_evaluation(my_model, X_val, y_val, scaler_y, "Validation")
# For the test set, we only have 1 sequence, so we print its specific MSE
test_preds = my_model.predict(X_test)
test_actual = scaler_y.inverse_transform(y_test)
test_predicted = scaler_y.inverse_transform(test_preds)
print(f"--- FINAL 72H TEST MSE: {np.mean((test_actual - test_predicted)**2):.4f} ---")

calculate_baselines(data, test)
print("All tasks complete! Check the 'plots' folder for results.")