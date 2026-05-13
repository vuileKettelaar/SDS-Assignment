import pandas as pd
import numpy as np

# =============================================================================
#  1. LOAD DATA & SET UP WINDOW
# =============================================================================
DATA_PATH  = "Full_Dataset_Updated.csv"     
TARGET_COL = "Price_BE"
DATE_COL   = "Date"

print("Loading historical data...")
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL], dayfirst=True)
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Cut data exactly where our training script cuts it (Start of May 12)
cutoff_date = pd.to_datetime("2026-05-12 00:00:00")
df_history = df[df[DATE_COL] < cutoff_date].copy()

# Ensure we have the last 168 hours (7 days) for our baselines
last_168h = df_history.tail(168).copy()
last_24h  = df_history.tail(24).copy()

# Create future dates for the assignment (72 hours)
future_dates = pd.date_range(start="2026-05-12 00:00:00", periods=72, freq="h")
future_df = pd.DataFrame({DATE_COL: future_dates})
future_df['hour'] = future_df[DATE_COL].dt.hour

# =============================================================================
#  2. CALCULATE BASELINE 1: WEEKLY PERSISTENCE
# =============================================================================
# Shift by exactly 168 hours (1 week). 
# May 12 gets May 5. May 13 gets May 6. May 14 gets May 7.
# Since we need 72 hours, we take the first 72 hours of our last_168h block.
baseline_weekly = last_168h[TARGET_COL].values[:72]

# =============================================================================
#  3. CALCULATE BASELINE 2: DAILY PERSISTENCE (REPEATED)
# =============================================================================
# Take the last available 24 hours (May 11) and tile it 3 times to get 72 hours
last_day_profile = last_24h[TARGET_COL].values
baseline_daily = np.tile(last_day_profile, 3)

# =============================================================================
#  4. CALCULATE BASELINE 3: 7-DAY HOURLY AVERAGE
# =============================================================================
# Group the last 168 hours by hour (0-23) and get the mean for each hour
last_168h['hour'] = last_168h[DATE_COL].dt.hour
hourly_avg_profile = last_168h.groupby('hour')[TARGET_COL].mean().reset_index()

# Map this average profile to our 72-hour future dataframe
baseline_7day_avg = future_df.merge(hourly_avg_profile, on='hour', how='left')[TARGET_COL].values

# =============================================================================
#  5. SAVE TO CSV
# =============================================================================
# Save them exactly like the ML models (no header, no index)
pd.DataFrame(baseline_weekly).to_csv("Baseline_Weekly_predictions.csv", index=False, header=False)
pd.DataFrame(baseline_daily).to_csv("Baseline_Daily_predictions.csv", index=False, header=False)
pd.DataFrame(baseline_7day_avg).to_csv("Baseline_7DayAvg_predictions.csv", index=False, header=False)

print("\nSUCCESS: Baseline models generated!")
print(" -> Saved: Baseline_Weekly_predictions.csv")
print(" -> Saved: Baseline_Daily_predictions.csv")
print(" -> Saved: Baseline_7DayAvg_predictions.csv")