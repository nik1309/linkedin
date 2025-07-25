import numpy as np

def european_put_mc(S0, K, T, r, sigma, n_simulations):
    Z = np.random.normal(size=n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


price = european_put_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=100000)
print(f"European Put Option Price: {price:.4f}")
