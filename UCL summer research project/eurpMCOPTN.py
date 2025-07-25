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


def european_call_mc(S0, K, T, r, sigma, n_simulations):
    Z = np.random.normal(size=n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price



def european_put_mc(S0, K, T, r, sigma, n_simulations):
    Z = np.random.normal(size=n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


price = european_call_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=100000)
print(f"European Call Option Price: {price:.4f}")
price = european_put_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=100000)
print(f"European Put Option Price: {price:.4f}")












