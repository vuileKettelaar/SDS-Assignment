
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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
import matplotlib.pyplot as plt

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
    df.set_index('Date', inplace=True)
    df = df.ffill() #
    return df

def add_calendar_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    return df

def create_lags(df, target_col='Price_BE', lags=[1, 2, 24, 168]):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag) #
    df.dropna(inplace=True)
    return df

def split_data(df):
    n = len(df)
    train_df = df[0:int(n*0.8)]
    val_df = df[int(n*0.8):int(n*0.9)]
    test_df = df[int(n*0.9):] #
    return train_df, val_df, test_df

def scale_data(train_df, val_df, test_df, target_col='Price_BE'):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Scale features
    scaler_X.fit(train_df)
    train_s = scaler_X.transform(train_df)
    val_s = scaler_X.transform(val_df)
    test_s = scaler_X.transform(test_df)
    
    # Scale target separately for easy inverse transform later
    scaler_y.fit(train_df[[target_col]])
    
    return train_s, val_s, test_s, scaler_y

def create_sequences(data, target_idx, look_back=168, forecast_horizon=72):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i : (i + look_back), :])
        y.append(data[(i + look_back) : (i + look_back + forecast_horizon), target_idx]) #
    return np.array(X), np.array(y)

# --- 3. MODEL ARCHITECTURE ---

def build_lstm_model(input_shape, output_steps):
    model = Sequential([
        LSTM(units=64, input_shape=input_shape, return_sequences=False), #[cite: 1]
        Dropout(0.2),
        Dense(units=output_steps) # Output 72 hours at once[cite: 2]
    ])
    model.compile(optimizer='adam', loss='mse') # Evaluated on MSE[cite: 2]
    return model

# --- 4. VISUALIZATION POPUP ---

def visualize_test_performance(model, X_test, y_test, scaler_y):
    # Pick a random 72-hour window from the test set to visualize
    idx = np.random.randint(0, len(X_test))
    sample_X = np.expand_dims(X_test[idx], axis=0)
    
    # Predict and inverse scale
    scaled_pred = model.predict(sample_X)
    actual_prices = scaler_y.inverse_transform(y_test[idx].reshape(1, -1)).flatten()
    pred_prices = scaler_y.inverse_transform(scaled_pred).flatten()
    
    # Create the Popup Graph
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price (BE)', color='black', linewidth=2)
    plt.plot(pred_prices, label='LSTM 72h Forecast', color='blue', linestyle='--', linewidth=2)
    plt.title(f'72-Hour Forecast vs Actual Data (Test Set Sample)')
    plt.xlabel('Hours into Future')
    plt.ylabel('Price [EUR/MWh]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show() # This creates the popup!

def play_finished_sound():
    sound_file = "/usr/share/sounds/gnome/default/alerts/glass.ogg"
    if os.path.exists(sound_file):
        os.system(f"paplay {sound_file}")
    else:
        print('\a Training complete!')

def save_trained_model(model, filename='sds_lstm_model.keras'):
    # Saves the model architecture, weights, and training configuration
    model.save(filename)
    print(f"Model saved successfully as '{filename}'")

def load_existing_model(filename='sds_lstm_model.keras'):
    # Loads the model back into memory exactly as it was
    if os.path.exists(filename):
        model = tf.keras.models.load_model(filename)
        print(f"Model '{filename}' loaded successfully!")
        return model
    else:
        print(f"Error: '{filename}' not found.")
        return None
    
import os
import matplotlib
# CRITICAL: Use 'Agg' to avoid the display connection crash on Fedora
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Create the directory for your report pictures if it doesn't exist
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Updated Visualization Functions (Saving to Directory) ---

def visualize_test_performance(model, X_test, y_test, scaler_y):
    idx = np.random.randint(0, len(X_test))
    sample_X = np.expand_dims(X_test[idx], axis=0)
    
    scaled_pred = model.predict(sample_X)
    actual_prices = scaler_y.inverse_transform(y_test[idx].reshape(1, -1)).flatten()
    pred_prices = scaler_y.inverse_transform(scaled_pred).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price (BE)', color='black', linewidth=2)
    plt.plot(pred_prices, label='LSTM 72h Forecast', color='blue', linestyle='--', linewidth=2)
    plt.title(f'Sample 72-Hour Forecast vs Actual')
    plt.xlabel('Hours into Future')
    plt.ylabel('Price [EUR/MWh]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to the new directory
    save_path = os.path.join(PLOT_DIR, 'sample_test_forecast.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close() # Always close to free up GPU/System memory

def plot_learning_curve(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='orange')
    plt.title('Model Learning Curve: Training vs. Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(PLOT_DIR, 'learning_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def plot_detailed_evaluation(model, X_data, y_data_scaled, scaler_y, set_name="Validation"):
    scaled_predictions = model.predict(X_data)
    actual_prices = scaler_y.inverse_transform(y_data_scaled)
    predicted_prices = scaler_y.inverse_transform(scaled_predictions)
    
    # 1. Plot: Actual vs Predicted
    plt.figure(figsize=(15, 6))
    plt.plot(actual_prices[:168, 0], label='Actual Price', color='black', alpha=0.7)
    plt.plot(predicted_prices[:168, 0], label='Predicted Price', color='blue', linestyle='--')
    plt.title(f'{set_name} Set: 1-Week Forecast Comparison')
    plt.xlabel('Hours')
    plt.ylabel('Price [EUR/MWh]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(PLOT_DIR, f'forecast_{set_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    # 2. Plot: Error Distribution
    errors = actual_prices - predicted_prices
    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=50, color='red', alpha=0.6)
    plt.title(f'{set_name} Set: Error Distribution')
    plt.xlabel('Price Error [EUR/MWh]')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(PLOT_DIR, f'residuals_{set_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    mse = np.mean((actual_prices - predicted_prices)**2)
    print(f"--- {set_name} MSE: {mse:.4f} ---")



# --- 5. MAIN EXECUTION FLOW ---

# Load and Prepare
data = load_and_clean_data("Full_Dataset.csv")
data = add_calendar_features(data)
data = create_lags(data)

# Split and Scale
train, val, test = split_data(data)
target_idx = train.columns.get_loc('Price_BE')
train_s, val_s, test_s, scaler_y = scale_data(train, val, test)

# Create Sequences
X_train, y_train = create_sequences(train_s, target_idx)
X_val, y_val = create_sequences(val_s, target_idx)
X_test, y_test = create_sequences(test_s, target_idx)

model_file = 'sds_lstm_model.keras'

if os.path.exists(model_file):
    # Skip training and load the existing one
    print("-------------------skipping training--------------------------")
    my_model = load_existing_model(model_file)
else:
    # Build and train because no saved model exists
    my_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_steps=72)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Starting fresh training...")
    history = my_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                           epochs=50, batch_size=32, callbacks=[early_stop])
    
    plot_learning_curve(history)
    # Save it immediately after training finishes
    save_trained_model(my_model, model_file)

# Now you can visualize or predict instantly
visualize_test_performance(my_model, X_test, y_test, scaler_y)
# --- Call this in your main flow ---
plot_detailed_evaluation(my_model, X_val, y_val, scaler_y, "Validation")
plot_detailed_evaluation(my_model, X_test, y_test, scaler_y, "Test")
play_finished_sound()