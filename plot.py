import pandas as pd
import matplotlib
# Use the 'Agg' backend to avoid the Qt/X11 display error
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

try:
    actual_df = pd.read_csv('Actual_Hourly_Belpex_Prices.csv')
    preds_df = pd.read_csv('72predictions.csv')

    # Identify price columns
    actual_col = actual_df.select_dtypes(include=['number']).columns[0]
    preds_col = preds_df.select_dtypes(include=['number']).columns[0]

    plt.figure(figsize=(12, 6))
    plt.plot(actual_df[actual_col], label='Actual Belpex Price', color='blue')
    plt.plot(preds_df[preds_col], label='Predicted Price', color='orange', linestyle='--')

    plt.title('Actual vs Predicted Hourly Belpex Prices')
    plt.xlabel('Hour')
    plt.ylabel('Price (€/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # This will now work without a display
    plt.savefig('belpex_comparison_plot.png')
    print("Plot successfully saved to: belpex_comparison_plot.png")

except Exception as e:
    print(f"Error: {e}")