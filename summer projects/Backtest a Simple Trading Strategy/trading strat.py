import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- Step 1: Download historical data ---
symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2025-01-01"

data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
data = data[['Close']]

# --- Step 2: Define momentum strategy function ---
def momentum_strategy(data, lookback):
    """
    Calculates strategy returns using momentum.
    Buy if past return over lookback period is positive, else cash.
    """
    data = data.copy()
    data['Momentum'] = data['Close'].pct_change(lookback)
    data['Signal'] = np.where(data['Momentum'] > 0, 1, 0)  # long-only
    data['Position'] = data['Signal'].shift(1)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Position'] * data['Returns']
    data.dropna(inplace=True)
    
    sharpe_ratio = (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()) * np.sqrt(252)
    cumulative_return = (1 + data['Strategy_Returns']).cumprod().iloc[-1] - 1
    return sharpe_ratio, cumulative_return, data

# --- Step 3: Optimize lookback period ---
lookback_periods = range(5, 251, 5)  # Test 5 to 250 days
results = []

for lb in lookback_periods:
    sharpe, cum_return, _ = momentum_strategy(data, lb)
    results.append((lb, sharpe, cum_return))

results_df = pd.DataFrame(results, columns=['Lookback', 'Sharpe', 'Cumulative_Return'])
best_row = results_df.loc[results_df['Sharpe'].idxmax()]

print(f"Best Lookback: {best_row['Lookback']} days")
print(f"Sharpe Ratio: {best_row['Sharpe']:.2f}")
print(f"Cumulative Return: {best_row['Cumulative_Return']:.2%}")

# --- Step 4: Plot optimized strategy performance ---
_, _, best_data = momentum_strategy(data, int(best_row['Lookback']))

plt.figure(figsize=(12,6))
plt.plot((1 + best_data['Returns']).cumprod(), label='Market')
plt.plot(best_data['Cumulative_Returns'] if 'Cumulative_Returns' in best_data else (1 + best_data['Strategy_Returns']).cumprod(), label='Momentum Strategy')
plt.title(f"{symbol} Optimized Momentum Strategy (Lookback={int(best_row['Lookback'])} days)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()
