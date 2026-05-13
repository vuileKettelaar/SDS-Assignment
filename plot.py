import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from collections import Counter

# =========================================================================
# --- NEW: OPTIMIZATION FUNCTION ---
# =========================================================================
def find_best_shift(actuals, preds, max_shift=6):
    """
    Tests shifts from -max_shift to +max_shift hours.
    Returns: (best_shift, lowest_mse, n_evaluated)
    """
    results = []
    
    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            # Shift Forward: Pred[0] aligns with Actual[shift]
            n_eval = min(len(actuals) - shift, len(preds))
            if n_eval <= 0: continue
            a_slice = actuals[shift : shift + n_eval]
            p_slice = preds[:n_eval]
        elif shift < 0:
            # Shift Backward: Pred[|shift|] aligns with Actual[0]
            s = abs(shift)
            n_eval = min(len(actuals), len(preds) - s)
            if n_eval <= 0: continue
            a_slice = actuals[:n_eval]
            p_slice = preds[s : s + n_eval]
        else:
            # No Shift
            n_eval = min(len(actuals), len(preds))
            if n_eval <= 0: continue
            a_slice = actuals[:n_eval]
            p_slice = preds[:n_eval]
            
        mse = mean_squared_error(a_slice, p_slice)
        results.append((shift, mse, n_eval))
        
    # Sort results by MSE to find the lowest
    results.sort(key=lambda x: x[1])
    return results[0] # Returns the tuple of the best result

try:
    # --- 1. Load actual data ---
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    
    # --- 1b. Load prediction data for ALL FOUR models ---
    lgbm_df  = pd.read_csv('LGBM_predictions.csv', header=None)
    xgb_df   = pd.read_csv('XGB_predictions.csv', header=None)
    ens_df   = pd.read_csv('3-WAY_predictions.csv', header=None)
    ens2_df  = pd.read_csv('2-WAY_predictions.csv', header=None) 

    # --- 2. Parse timestamps ---
    actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
    actual_prices = actual_df['Actual_Price_BE'].values
    
    # --- 3. Extract values ---
    models = {
        "LGBM Forecast": np.array([float(v) for v in lgbm_df.iloc[:, 0].values]),
        "XGB Forecast": np.array([float(v) for v in xgb_df.iloc[:, 0].values]),
        "3-WAY Ensemble": np.array([float(v) for v in ens_df.iloc[:, 0].values]),
        "2-WAY (XGB+LSTM)": np.array([float(v) for v in ens2_df.iloc[:, 0].values])
    }

    # =========================================================================
    # --- 4. RUN ALIGNMENT OPTIMIZER ---
    # =========================================================================
    print("\n" + "="*60)
    print(" 🎯 AUTO-ALIGNMENT OPTIMIZER (Testing shifts -6h to +6h)")
    print("="*60)
    print(f"{'Model':<18} | {'Optimal Shift':<15} | {'Lowest MSE':<10} | {'Overlap'}")
    print("-" * 60)
    
    best_shifts_found = []
    
    for name, preds in models.items():
        best_shift, min_mse, n_eval = find_best_shift(actual_prices, preds, max_shift=6)
        best_shifts_found.append(best_shift)
        
        # Format the shift string nicely
        shift_str = f"+{best_shift} hours" if best_shift > 0 else f"{best_shift} hours"
        if best_shift == 0: shift_str = "No Shift"
            
        print(f"{name:<18} | {shift_str:<15} | {min_mse:<10.2f} | {n_eval}h")
        
    print("="*60)

    # Find the consensus (most common) best shift across all models
    consensus_shift = Counter(best_shifts_found).most_common(1)[0][0]
    
    shift_str = f"+{consensus_shift}h" if consensus_shift > 0 else f"{consensus_shift}h"
    if consensus_shift == 0: shift_str = "0h"
    
    print(f"✅ Consensus Optimal Shift: {shift_str}. Applying to plot...\n")

    # =========================================================================
    # --- 5. Plot (Automatically using consensus shift) ---
    # =========================================================================
    pred_start = actual_df['Timestamp'].iloc[0]
    
    # --- CREATE TIMELINES ---
    # Shifted timeline for the first 3 models
    preds_timestamps = pd.date_range(start=pred_start, periods=len(models["LGBM Forecast"]), freq='h') 
    preds_timestamps += pd.Timedelta(hours=consensus_shift)
    
    # Unshifted timeline specifically for the 2-WAY model
    unshifted_timestamps = pd.date_range(start=pred_start, periods=len(models["2-WAY (XGB+LSTM)"]), freq='h')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Actuals
    ax.plot(actual_df['Timestamp'], actual_df['Actual_Price_BE'],
            label='Actual (May 12+)', color='royalblue', linewidth=2.5)

    # Plot Predictions (Shifted)
    ax.plot(preds_timestamps, models["LGBM Forecast"],
            label='LGBM Forecast', color='orange', linestyle='--', linewidth=1.8)
            
    ax.plot(preds_timestamps, models["XGB Forecast"],
            label='XGB Forecast', color='mediumseagreen', linestyle=':', linewidth=2.2)
            
    ax.plot(preds_timestamps, models["3-WAY Ensemble"],
            label='3-WAY Ensemble', color='crimson', linestyle='-.', linewidth=1.8)
            
    # Plot 2-WAY Prediction (Unshifted)
    ax.plot(unshifted_timestamps, models["2-WAY (XGB+LSTM)"],
            label='2-WAY (XGB+LSTM) [Unshifted]', color='purple', linestyle='--', dashes=(5, 2, 1, 2), linewidth=1.8)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.set_title(f'Actual vs Predicted Hourly Belpex Prices (All Models) [{shift_str} Optimized Shift]', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Price (€/MWh)', fontsize=11)
    
    ax.legend(fontsize=10, facecolor='white', framealpha=0.9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = 'belpex_comparison_all_models.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved to: {out}")

except Exception as e:
    import traceback
    traceback.print_exc()