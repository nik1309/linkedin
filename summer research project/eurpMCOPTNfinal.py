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

def european_option_mc_visual(S0, K, T, r, sigma, N=10000, M=50, option_type='call', plot=True):
    """
    Parameters:
    S0 - initial stock price
    K - strike price
    T - time to maturity (in years)
    r - risk-free rate
    sigma - volatility
    N - number of simulations
    M - number of time steps
    option_type - 'call' or 'put'
    plot - whether to plot the price paths

    """
    dt = T / M 
    

    Z = np.random.normal(size=(N, M))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_returns = np.cumsum(increments, axis=1)
    

    log_returns = np.hstack([np.zeros((N, 1)), log_returns])
    S_paths = S0 * np.exp(log_returns)
    
    if plot:
        plot_price_paths(S_paths[:100], title=f'first 100 of 10000 simulated European {option_type} Option Simulated Paths')
    

    ST = S_paths[:, -1]
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    

    option_price = np.exp(-r * T) * np.mean(payoff)
    
    return option_price
 

call_price = european_option_mc_visual(S0=100, K=100, T=1, r=0.05, sigma=0.2, N=10000, M=50, option_type='call')
print(f"European Call Option Price: {call_price:.4f}")

put_price = european_option_mc_visual(S0=100, K=100, T=1, r=0.05, sigma=0.2,N=10000, M=50, option_type='put')
print(f"European Put Option Price: {put_price:.4f}")
