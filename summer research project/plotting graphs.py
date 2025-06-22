import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def plot_price_paths(S_paths, title='Simulated Price Paths', num_paths=1000):
    plt.figure(figsize=(10, 5))
    for i in range(min(num_paths, S_paths.shape[0])):
        plt.plot(S_paths[i], lw=0.8)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.grid(True)
    plt.show()

def european_option_mc(S0, K, T, r, sigma, N=10000, M=50, option_type='call', plot=True):
    dt = T / M
    S_paths = np.zeros((N, M + 1))
    S_paths[:, 0] = S0

    for t in range(1, M + 1):
        Z = np.random.normal(size=N)
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if plot:
        plot_price_paths(S_paths[:20], title='European Option Simulated Paths (GBM)')

    ST = S_paths[:, -1]
    payoff = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

def european_option_mc_2(S0, K, T, r, sigma, N=10000, M=1, option_type='call', plot=True):
    dt = T
    Z = np.random.normal(size=N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Simulate paths for plotting
    S_paths = np.zeros((20, 2))
    S_paths[:, 0] = S0
    for i in range(20):
        S_paths[i, 1] = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i])
    
    if plot:
        plot_price_paths(S_paths, title='European Option Simulated Paths')

    payoff = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

def asian_arithmetic_option_mc(S0, K, T, r, sigma, N=10000, M=50, option_type='call', plot=True):
    dt = T / M
    S_paths = np.zeros((N, M + 1))
    S_paths[:, 0] = S0

    for t in range(1, M + 1):
        Z = np.random.normal(size=N)
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if plot:
        plot_price_paths(S_paths[:20], title='Asian Option Simulated Paths')

    avg_prices = np.mean(S_paths[:, 1:], axis=1)
    payoff = np.maximum(avg_prices - K, 0) if option_type == 'call' else np.maximum(K - avg_prices, 0)
    return np.exp(-r * T) * np.mean(payoff)

def american_option_lsm_mc(S0, K, T, r, sigma, N=10000, M=50, option_type='put', plot=True):
    dt = T / M
    discount = np.exp(-r * dt)
    S = np.zeros((N, M + 1))
    S[:, 0] = S0

    for t in range(1, M + 1):
        Z = np.random.normal(size=N)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if plot:
        plot_price_paths(S[:20], title='American Option Simulated Paths')

    V = np.maximum(S[:, -1] - K, 0) if option_type == 'call' else np.maximum(K - S[:, -1], 0)

    for t in range(M - 1, 0, -1):
        if option_type == 'call':
            itm = np.where(S[:, t] > K)[0]
            exercise = S[:, t] - K
        else:
            itm = np.where(S[:, t] < K)[0]
            exercise = K - S[:, t]

        if len(itm) == 0:
            continue

        X = S[itm, t].reshape(-1, 1)
        Y = V[itm] * discount

        model = LinearRegression().fit(X, Y)
        continuation = model.predict(X)

        V[itm] = np.where(exercise[itm] > continuation, exercise[itm], V[itm] * discount)

    return discount * np.mean(V)

# Parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

# Run with plotting
european_option_mc(S0, K, T, r, sigma, option_type='call')
asian_arithmetic_option_mc(S0, K, T, r, sigma, option_type='put')
american_option_lsm_mc(S0, K, T, r, sigma, option_type='put')
