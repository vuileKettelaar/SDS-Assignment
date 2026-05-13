import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

try:
    # --- 1. Load data ---
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    ens2_df   = pd.read_csv('FinalCode/finalPrediction.csv', header=None) 

    # --- 2. Parse timestamps ---
    actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
    
    # --- 3. Extract values ---
    ens2_vals = np.array([float(v) for v in ens2_df.iloc[:, 0].values]) 

    # Start predictions from the first timestamp in the actual_df
    pred_start = actual_df['Timestamp'].iloc[0]
    
    # Create the timeline for the LSTM model
    preds_timestamps = pd.date_range(start=pred_start, periods=len(ens2_vals), freq='h')

    # Calculate MSE and Residuals for overlapping hours
    n_eval = min(len(actual_df), len(ens2_vals))
    if n_eval > 0:
        actual_prices = actual_df['Actual_Price_BE'].values[:n_eval]
        pred_prices = ens2_vals[:n_eval]
        eval_timestamps = actual_df['Timestamp'].iloc[:n_eval]
        
        # Calculate residuals (Actual - Predicted)
        residuals = actual_prices - pred_prices
        mse = mean_squared_error(actual_prices, pred_prices)
        
        print(f"\nEvaluating on {n_eval} overlapping hours...")
        print(f"LSTM Model MSE: {mse:.2f}\n")

    # --- 4. Plot Setup (Two panels) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # --- TOP PANEL: Prices ---
    ax1.plot(actual_df['Timestamp'], actual_df['Actual_Price_BE'],
            label='Actual (May 12+)', color='royalblue', linewidth=2.5)

    ax1.plot(preds_timestamps, ens2_vals,
            label='LSTM Forecast', color='purple', linestyle='--', linewidth=2.2)

    ax1.set_title('Actual vs Predicted Hourly Belpex Prices (LSTM Model)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (€/MWh)', fontsize=11)
    ax1.legend(fontsize=12, facecolor='white', framealpha=0.9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- BOTTOM PANEL: Residuals ---
    if n_eval > 0:
        ax2.bar(eval_timestamps, residuals, width=0.03, color='slategray', alpha=0.8, label='Residuals (Actual - Predicted)')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        
    ax2.set_ylabel('Error (€/MWh)', fontsize=11)
    ax2.legend(fontsize=10, facecolor='white', framealpha=0.9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # --- 5. Formatting ---
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    fig.autofmt_xdate(rotation=0, ha='center')
    ax2.set_xlabel('Timestamp', fontsize=11)

    plt.tight_layout()

    # --- 6. Save ---
    out = 'belpex_comparison_lstm_with_residuals.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved to: {out}")

except Exception as e:
    import traceback
    traceback.print_exc()