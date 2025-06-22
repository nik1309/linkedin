#arithmetic avrg
import numpy as np
import matplotlib.pyplot as plt

def plot_price_paths(paths, title='Simulated Asset Price Paths'):
    plt.figure(figsize=(12, 6))
    for path in paths:
        plt.plot(path, lw=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.grid(True)
    plt.show()

def asian_arithmetic_call_mc(S0, K, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    payoffs = []
    all_paths = []

    for _ in range(n_simulations):
        path = [S0]
        for _ in range(n_steps):
            z = np.random.normal()
            St = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(St)
        all_paths.append(path)
        avg_price = np.mean(path[1:])  
        payoff = max(avg_price - K, 0)
        payoffs.append(payoff)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    plot_price_paths(all_paths[:100], title='First 100 of 10000 simulated Asian Arithmetic Call Option Paths')
    return price


def asian_arithmetic_put_mc(S0, K, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    payoffs = []
    all_paths = []

    for _ in range(n_simulations):
        path = [S0]
        for _ in range(n_steps):
            z = np.random.normal()
            St = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(St)
        all_paths.append(path)
        avg_price = np.mean(path[1:])  
        payoff = max(K - avg_price, 0)
        payoffs.append(payoff)

    price_put = np.exp(-r * T) * np.mean(payoffs)
    plot_price_paths(all_paths[:100], title='First 100 of 10000 simulated Asian Arithmetic Put Option Paths')
    return price_put


price_call = asian_arithmetic_call_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=10000, n_steps=50)
print(f"Asian Arithmetic Call Option Price: {price_call:.4f}")
price_put = asian_arithmetic_put_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=10000, n_steps=50)
print(f"Asian Arithmetic Put Option Price: {price_put:.4f}")
