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

# --- Step 2: Calculate moving averages ---
short_window = 50
long_window = 200

data['SMA50'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['SMA200'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

# --- Step 3: Generate long-only signals ---
# 1 = long, 0 = cash
data['Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, 0)

# Shift by 1 day to avoid lookahead bias
data['Position'] = data['Signal'].shift(1)

# --- Step 4: Calculate returns ---
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Position'] * data['Returns']

# Drop NaN rows
data.dropna(inplace=True)

# --- Step 5: Evaluate performance metrics ---
# Annualized Sharpe Ratio
sharpe_ratio = (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()) * np.sqrt(252)

# Cumulative returns
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
data['Cumulative_Market'] = (1 + data['Returns']).cumprod()

# Max Drawdown
cumulative_max = data['Cumulative_Strategy'].cummax()
drawdown = (data['Cumulative_Strategy'] - cumulative_max) / cumulative_max
max_drawdown = drawdown.min()

# Total Profit
total_profit = data['Cumulative_Strategy'].iloc[-1] - 1

# --- Step 6: Print performance ---
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Total Profit: {total_profit:.2%}")

# --- Step 7: Plot strategy vs market ---
plt.figure(figsize=(12,6))
plt.plot(data['Cumulative_Market'], label='Market')
plt.plot(data['Cumulative_Strategy'], label='Strategy')
plt.title(f"{symbol} SMA50/SMA200 Long-Only Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()
