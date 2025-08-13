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

# --- Step 2: Define SMA50/SMA200 Long-Only Strategy ---
def sma_strategy(data, short_window=50, long_window=200):
    df = data.copy()
    df['SMA50'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA200'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    df['Signal'] = np.where(df['SMA50'] > df['SMA200'], 1, 0)  # long-only
    df['Position'] = df['Signal'].shift(1)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df.dropna(inplace=True)
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Market'] = (1 + df['Returns']).cumprod()
    
    sharpe_ratio = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
    cumulative_return = df['Cumulative_Strategy'].iloc[-1] - 1
    cumulative_max = df['Cumulative_Strategy'].cummax()
    drawdown = (df['Cumulative_Strategy'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return df, sharpe_ratio, cumulative_return, max_drawdown

# --- Step 3: Define Momentum Strategy ---
def momentum_strategy(data, lookback=60):
    df = data.copy()
    df['Momentum'] = df['Close'].pct_change(lookback)
    df['Signal'] = np.where(df['Momentum'] > 0, 1, 0)  # long-only
    df['Position'] = df['Signal'].shift(1)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df.dropna(inplace=True)
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    
    sharpe_ratio = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
    cumulative_return = df['Cumulative_Strategy'].iloc[-1] - 1
    cumulative_max = df['Cumulative_Strategy'].cummax()
    drawdown = (df['Cumulative_Strategy'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return df, sharpe_ratio, cumulative_return, max_drawdown

# --- Step 4: Optimize Momentum Lookback ---
lookback_periods = range(5, 251, 5)
best_sharpe = -np.inf
best_lookback = 20

for lb in lookback_periods:
    _, sharpe, _, _ = momentum_strategy(data, lb)
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_lookback = lb

# --- Step 5: Run Both Strategies ---
sma_df, sma_sharpe, sma_cum, sma_dd = sma_strategy(data)
mom_df, mom_sharpe, mom_cum, mom_dd = momentum_strategy(data, best_lookback)

# --- Step 6: Print Comparison ---
print("=== Strategy Comparison ===")
print(f"SMA50/SMA200 Strategy: Sharpe={sma_sharpe:.2f}, Max Drawdown={sma_dd:.2%}, Total Profit={sma_cum:.2%}")
print(f"Momentum Strategy (Lookback={best_lookback}): Sharpe={mom_sharpe:.2f}, Max Drawdown={mom_dd:.2%}, Total Profit={mom_cum:.2%}")

# --- Step 7: Plot cumulative returns ---
plt.figure(figsize=(12,6))
plt.plot(sma_df['Cumulative_Strategy'], label='SMA50/SMA200')
plt.plot(mom_df['Cumulative_Strategy'], label=f'Momentum (Lookback={best_lookback})')
plt.plot((1 + data['Close'].pct_change().fillna(0)).cumprod(), label='Market', linestyle='--', alpha=0.7)
plt.title(f"{symbol} Strategy Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()
