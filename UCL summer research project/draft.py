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


def european_call_mc(S0, K, T, r, sigma, n_simulations, n_steps=50, plot_paths=True):
    dt = T / n_steps
    all_paths = np.zeros((n_simulations, n_steps + 1))
    all_paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=n_simulations)
        all_paths[:, t] = all_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    if plot_paths:
        plot_price_paths(all_paths, title='European Call Option Price Paths')

    ST = all_paths[:, -1]
    payoffs = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


def european_put_mc(S0, K, T, r, sigma, n_simulations, n_steps=50, plot_paths=True):
    dt = T / n_steps
    all_paths = np.zeros((n_simulations, n_steps + 1))
    all_paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=n_simulations)
        all_paths[:, t] = all_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    if plot_paths:
        plot_price_paths(all_paths, title='European Put Option Price Paths')

    ST = all_paths[:, -1]
    payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


# Example usage
n_simulations = 500  # use 100000 for final result, but plotting this many is not recommended
n_steps = 50

put_price = european_put_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                             n_simulations=n_simulations, n_steps=n_steps, plot_paths=True)

call_price = european_call_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                              n_simulations=n_simulations, n_steps=n_steps, plot_paths=True)

print(f"European Put Option Price:  {put_price:.4f}")
print(f"European Call Option Price: {call_price:.4f}")
