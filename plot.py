import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from collections import Counter

# =========================================================================
# --- OPTIMIZATION FUNCTION ---
# =========================================================================
def find_best_shift(actuals, preds, max_shift=6):
    results = []
    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            n_eval = min(len(actuals) - shift, len(preds))
            if n_eval <= 0: continue
            a_slice = actuals[shift : shift + n_eval]
            p_slice = preds[:n_eval]
        elif shift < 0:
            s = abs(shift)
            n_eval = min(len(actuals), len(preds) - s)
            if n_eval <= 0: continue
            a_slice = actuals[:n_eval]
            p_slice = preds[s : s + n_eval]
        else:
            n_eval = min(len(actuals), len(preds))
            if n_eval <= 0: continue
            a_slice = actuals[:n_eval]
            p_slice = preds[:n_eval]
            
        mse = mean_squared_error(a_slice, p_slice)
        results.append((shift, mse, n_eval))
    results.sort(key=lambda x: x[1])
    return results[0]

try:
    # --- 1. Load data ---
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    lgbm_df  = pd.read_csv('LGBM_predictions.csv', header=None)
    xgb_df   = pd.read_csv('XGB_predictions.csv', header=None)
    lstm_df = pd.read_csv('fTUNED_LSTM_predictions.csv', header= None)
    ens_df   = pd.read_csv('3-WAY_predictions.csv', header=None)
    ens2_df  = pd.read_csv('2-WAY_predictions.csv', header=None) 

    # --- 2. Define strict Assignment Window (May 12 - May 14) ---
    start_dt = pd.to_datetime("2026-05-12 00:00:00")
    end_dt   = pd.to_datetime("2026-05-14 23:00:00")
    
    actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
    mask = (actual_df['Timestamp'] >= start_dt) & (actual_df['Timestamp'] <= end_dt)
    actual_filtered = actual_df.loc[mask].copy()
    actual_prices = actual_filtered['Actual_Price_BE'].values
    
    # --- 3. Extract prediction values ---
    models = {
        "LGBM Forecast": np.array([float(v) for v in lgbm_df.iloc[:, 0].values]),
        "XGB Forecast": np.array([float(v) for v in xgb_df.iloc[:, 0].values]),
        "LSTM Forecast": np.array([float(v) for v in lstm_df.iloc[:,0].values]),
        "3-WAY Ensemble": np.array([float(v) for v in ens_df.iloc[:, 0].values]),
        "2-WAY (XGB+LSTM)": np.array([float(v) for v in ens2_df.iloc[:, 0].values])
    }
    # =========================================================================
    # --- NEW: CALCULATE AND PRINT MSE ---
    # =========================================================================
    print("\n" + "="*50)
    print(" 📊 MEAN SQUARED ERROR (MSE) EVALUATION")
    print("="*50)
    for name, preds in models.items():
        # Match lengths in case actuals file doesn't have the full 72 hours yet
        n_eval = min(len(actual_prices), len(preds))
        if n_eval > 0:
            current_mse = mean_squared_error(actual_prices[:n_eval], preds[:n_eval])
            print(f"  {name:<18} | MSE: {current_mse:.2f}")
    print("="*50 + "\n")
    
    # --- 4. RUN ALIGNMENT OPTIMIZER ---
    best_shifts_found = []
    for name, preds in models.items():
        best_shift, min_mse, n_eval = find_best_shift(actual_prices, preds, max_shift=6)
        best_shifts_found.append(best_shift)
        
    consensus_shift = Counter(best_shifts_found).most_common(1)[0][0]

    # --- 5. Plotting ---
    target_timeline = pd.date_range(start=start_dt, end=end_dt, freq='h')
    shifted_timeline = target_timeline + pd.Timedelta(hours=consensus_shift)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(actual_filtered['Timestamp'], actual_filtered['Actual_Price_BE'],
            label='Actual Belpex Price', color='royalblue', linewidth=3, zorder=5)

    ax.plot(shifted_timeline, models["LGBM Forecast"],
            label='LGBM Forecast', color='orange', linestyle='--', alpha=0.8)
    
    ax.plot(shifted_timeline, models["LSTM Forecast"],
            label='LSTM Forecast', color='red', linestyle='solid', alpha=0.8)    
        
    ax.plot(shifted_timeline, models["XGB Forecast"],
            label='XGB Forecast', color='mediumseagreen', linestyle=':', linewidth=2)
            
    ax.plot(shifted_timeline, models["3-WAY Ensemble"],
            label='3-WAY Ensemble', color='crimson', linestyle='-.')
            
    ax.plot(target_timeline, models["2-WAY (XGB+LSTM)"],
            label='2-WAY (XGB+LSTM) [Anchor]', color='purple', linestyle='--', dashes=(5, 2))

    # Formatting
    ax.set_xlim(start_dt, end_dt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12])) 
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.set_title(f'Belpex Price Forecast: May 12 – May 14, 2026', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time of Day', fontsize=11)
    ax.set_ylabel('Price (€/MWh)', fontsize=11)
    
    # NEW: Position legend at the bottom right corner of the axes
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10, facecolor='white', framealpha=0.9)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out = 'belpex_forecast_bottom_right_legend.png'
    plt.savefig(out, dpi=200)
    print(f"✅ Success! Plot saved as {out} with the legend in the bottom right corner.")

except Exception as e:
    import traceback
    traceback.print_exc()