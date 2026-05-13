import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    # --- 1. Load data ---
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    ens2_df   = pd.read_csv('2-WAY_predictions.csv', header=None) 

    # --- 2. Extract values ---
    ens2_vals = np.array([float(v) for v in ens2_df.iloc[:, 0].values]) 

    # --- 3. Calculate Residuals ---
    n_eval = min(len(actual_df), len(ens2_vals))
    
    if n_eval > 0:
        actual_prices = actual_df['Actual_Price_BE'].values[:n_eval]
        pred_prices = ens2_vals[:n_eval]
        
        # Residuals = Actual - Predicted
        residuals = actual_prices - pred_prices
        
        print(f"Calculating distribution for {n_eval} residuals...")

        # --- 4. Plot Setup ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot 1: The Histogram of your actual residuals
        # density=True turns the Y-axis into a probability scale so we can draw the curve over it
        ax.hist(residuals, bins=15, density=True, alpha=0.6, color='purple', edgecolor='black', label='Actual Residual Histogram')

        # Plot 2: The Theoretical Normal Distribution Curve
        # This calculates the Mean (mu) and Standard Deviation (std) of your errors
        mu, std = norm.fit(residuals)
        
        # Generate the smooth bell curve line
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        
        ax.plot(x, p, 'k', linewidth=2.5, label=f'Normal Curve\n(Mean: {mu:.1f}, Std: {std:.1f})')

        # Plot 3: A red dashed line at EXACTLY 0 error for visual reference
        ax.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Perfect Prediction (0 Error)')

        # --- 5. Formatting ---
        ax.set_title('Distribution of 2-WAY Model Errors (Residuals)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Error Magnitude (€/MWh)', fontsize=12)
        ax.set_ylabel('Density (Probability)', fontsize=12)
        
        ax.legend(fontsize=11, facecolor='white', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # --- 6. Save ---
        out = 'belpex_residuals_distribution.png'
        plt.savefig(out, dpi=150)
        print(f"SUCCESS: Plot saved to {out}")

    else:
        print("Error: No overlapping hours to calculate residuals.")

except Exception as e:
    import traceback
    traceback.print_exc()