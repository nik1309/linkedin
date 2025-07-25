import numpy as np
import matplotlib.pyplot as plt

def plot_price_paths(paths, title='Simulated Asset Price Paths', num_paths=1000):
    plt.figure(figsize=(10, 5))
    for i in range(min(num_paths, len(paths))):
        plt.plot(paths[i], lw=1)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.grid(True)
    plt.show()


def asian_arithmetic_call_mc(S0, K, T, r, sigma, n_simulations, n_steps, plot_paths=True):
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
        avg_price = np.mean(path[1:])  # exclude S0
        payoff = max(avg_price - K, 0)
        payoffs.append(payoff)

    if plot_paths:
        plot_price_paths(all_paths, title='Asian Arithmetic Call Option Paths')

    price = np.exp(-r * T) * np.mean(payoffs)
    return price


def asian_arithmetic_put_mc(S0, K, T, r, sigma, n_simulations, n_steps, plot_paths=True):
    dt = T / n_steps
    payoffs = []
    all_paths = []

    for _ in range(n_simulations):
        path = [S0]
        for _ in range(n_simulations):
            z = np.random.normal()
            St = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(St)
        all_paths.append(path)
        avg_price = np.mean(path[1:])  # exclude S0
        payoff = max(K - avg_price, 0)
        payoffs.append(payoff)

    if plot_paths:
        plot_price_paths(all_paths, title='Asian Arithmetic Put Option Paths')

    price = np.exp(-r * T) * np.mean(payoffs)
    return price

# Example usage
put_price = asian_arithmetic_put_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=10000, n_steps=50)
call_price = asian_arithmetic_call_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=10000, n_steps=50)

print(f"Asian Arithmetic Put Option Price: {put_price:.4f}")
print(f"Asian Arithmetic Call Option Price: {call_price:.4f}")
