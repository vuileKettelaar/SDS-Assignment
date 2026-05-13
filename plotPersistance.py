import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

try:
    # =========================================================================
    # --- 1. LOAD PREDICTIONS ---
    # =========================================================================
    # Baselines
    weekly_df  = pd.read_csv('Baseline_Weekly_predictions.csv', header=None)
    daily_df   = pd.read_csv('Baseline_Daily_predictions.csv', header=None)
    avg7_df    = pd.read_csv('Baseline_7DayAvg_predictions.csv', header=None)
    
    # Your Tuned Model (Make sure the filename perfectly matches what you have!)
    lstm_df    = pd.read_csv('ffTUNED_LSTM_predictions.csv', header=None)

    # Store them in a dictionary
    models = {
        "Baseline: Weekly (168h lag)": np.array([float(v) for v in weekly_df.iloc[:, 0].values]),
        "Baseline: Daily (24h lag)": np.array([float(v) for v in daily_df.iloc[:, 0].values]),
        "Baseline: 7-Day Average": np.array([float(v) for v in avg7_df.iloc[:, 0].values]),
        "Tuned LSTM Forecast": np.array([float(v) for v in lstm_df.iloc[:, 0].values])
    }

    # =========================================================================
    # --- 2. LOAD & FILTER ACTUAL DATA ---
    # =========================================================================
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    
    # Format timestamps
    if 'Timestamp' in actual_df.columns:
        actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
    else:
        # Fallback just in case your file uses Date/Time columns instead
        actual_df['Timestamp'] = pd.to_datetime(actual_df['Date'] + ' ' + actual_df['Time'], format='%d/%m/%Y %H:%M')

    start_dt = pd.to_datetime("2026-05-12 00:00:00")
    end_dt   = pd.to_datetime("2026-05-14 23:00:00")
    
    mask = (actual_df['Timestamp'] >= start_dt) & (actual_df['Timestamp'] <= end_dt)
    actual_filtered = actual_df.loc[mask].copy()
    actual_prices = actual_filtered['Actual_Price_BE'].values
    actual_timeline = actual_filtered['Timestamp']

    # =========================================================================
    # --- 3. CALCULATE AND PRINT MSE ---
    # =========================================================================
    print("\n" + "="*55)
    print(" 🥊 MODEL VS. BASELINE: TRUE BLIND EVALUATION")
    print("="*55)
    
    if len(actual_prices) == 0:
        print("Waiting on actual data to be published for the 12th...")
    else:
        print(f"Evaluating the first {len(actual_prices)} hours of the 72h window:\n")
        for name, preds in models.items():
            # Slice the prediction to match how many actual hours we have so far
            matched_preds = preds[:len(actual_prices)]
            current_mse = mean_squared_error(actual_prices, matched_preds)
            
            # Print the LSTM a bit differently to make it stand out
            if "LSTM" in name:
                print(f"  ⭐ {name:<26} | MSE: {current_mse:.2f}")
            else:
                print(f"  -> {name:<26} | MSE: {current_mse:.2f}")
    print("="*55 + "\n")

    # =========================================================================
    # --- 4. PLOTTING ---
    # =========================================================================
    # We create a full 72-hour timeline for the predictions
    target_timeline = pd.date_range(start=start_dt, end=end_dt, freq='h')

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot Actuals (Thick Black Line)
    ax.plot(actual_timeline, actual_prices,
            label='Actual Belpex Price', color='black', linewidth=3.5, zorder=10)

    # Plot Baselines (Thinner, dotted/dashed lines in cooler colors)
    ax.plot(target_timeline, models["Baseline: Weekly (168h lag)"],
            label='Baseline: Weekly Persistence', color='steelblue', linestyle=':', linewidth=2)
            
    ax.plot(target_timeline, models["Baseline: Daily (24h lag)"],
            label='Baseline: Daily Persistence', color='mediumaquamarine', linestyle='-.', linewidth=2)

    ax.plot(target_timeline, models["Baseline: 7-Day Average"],
            label='Baseline: 7-Day Hourly Avg', color='gray', linestyle='--', linewidth=2)

    # Plot LSTM (Thick Red Line to stand out)
    ax.plot(target_timeline, models["Tuned LSTM Forecast"],
            label='⭐ Tuned LSTM Forecast', color='crimson', linestyle='-', linewidth=2.5, zorder=5)

    # Formatting
    ax.set_xlim(start_dt, end_dt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12])) 
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.set_title('Baseline Check: Has the Machine Actually Learned?', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Price (€/MWh)', fontsize=12)
    
    # Legend outside to the right so it doesn't block the lines
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True, fontsize=11)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out = 'Baseline_vs_LSTM_Comparison.png'
    plt.savefig(out, dpi=300)
    print(f"✅ Success! Comparative plot saved as {out}.")

except Exception as e:
    import traceback
    traceback.print_exc()