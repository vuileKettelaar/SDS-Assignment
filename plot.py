import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    # --- 1. Load actual data ---
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    
    # --- 1b. Load prediction data for ALL FOUR models ---
    # Ensure you have generated all four CSVs from your main script!
    lgbm_df  = pd.read_csv('LGBM_predictions.csv', header=None)
    xgb_df   = pd.read_csv('XGB_predictions.csv', header=None)
    ens_df   = pd.read_csv('3-WAY_predictions.csv', header=None)
    ens2_df  = pd.read_csv('2-WAY_predictions.csv', header=None) # NEW: 2-WAY data

    # --- 2. Parse timestamps ---
    actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
    
    # --- 3. Build predictions timestamps ---
    # Extract values for all four models
    lgbm_vals = np.array([float(v) for v in lgbm_df.iloc[:, 0].values])
    xgb_vals  = np.array([float(v) for v in xgb_df.iloc[:, 0].values])
    ens_vals  = np.array([float(v) for v in ens_df.iloc[:, 0].values])
    ens2_vals = np.array([float(v) for v in ens2_df.iloc[:, 0].values]) # NEW: 2-WAY values

    # Start predictions from the first timestamp in the actual_df
    pred_start = actual_df['Timestamp'].iloc[0]
    preds_timestamps = pd.date_range(start=pred_start, periods=len(lgbm_vals), freq='h')

    # --- 4. Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Actuals
    ax.plot(actual_df['Timestamp'], actual_df['Actual_Price_BE'],
            label='Actual (May 12+)', color='royalblue', linewidth=2.5)

    # Plot Predictions
    ax.plot(preds_timestamps, lgbm_vals,
            label='LGBM Forecast', color='orange', linestyle='--', linewidth=1.8)
            
    ax.plot(preds_timestamps, xgb_vals,
            label='XGB Forecast', color='mediumseagreen', linestyle=':', linewidth=2.2)
            
    ax.plot(preds_timestamps, ens_vals,
            label='3-WAY Ensemble', color='crimson', linestyle='-.', linewidth=1.8)
            
    # NEW: Plot 2-WAY predictions
    ax.plot(preds_timestamps, ens2_vals,
            label='2-WAY (XGB+LSTM)', color='purple', linestyle='--', dashes=(5, 2, 1, 2), linewidth=1.8)

    # --- 5. Formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.set_title('Actual vs Predicted Hourly Belpex Prices (All Models)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp (UTC)', fontsize=11)
    ax.set_ylabel('Price (€/MWh)', fontsize=11)
    
    # Moved legend slightly to avoid overlapping with data lines
    ax.legend(fontsize=10, facecolor='white', framealpha=0.9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- 6. Save ---
    out = 'belpex_comparison_all_models.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved to: {out}")

except Exception as e:
    import traceback
    traceback.print_exc()