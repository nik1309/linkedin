#arithmatic avrg
import numpy as np


def asian_arithmetic_call_mc(S0, K, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    payoffs = []


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
    return price


price = asian_arithmetic_call_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=10000, n_steps=50)
print(f"Asian Arithmetic Call Option Price: {price:.4f}")
