import numpy as np
from scipy.stats import norm

def asian_geometric_bs(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes-like formula for geometric average Asian options (European-style).

    Parameters:
        S : float        - Initial stock price
        K : float        - Strike price
        T : float        - Time to maturity
        r : float        - Risk-free rate
        sigma : float    - Volatility
        option_type : str - 'call' or 'put'

    Returns:
        float: Option price
    """
    sigma_hat = sigma / np.sqrt(3)
    mu_hat = 0.5 * (r - 0.5 * sigma**2) + (sigma_hat**2) / 2

    d1 = (np.log(S / K) + (mu_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)

    if option_type == 'call':
        price = np.exp(-r * T) * (S * np.exp(mu_hat * T) * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'put':
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - S * np.exp(mu_hat * T) * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price
S = 100       # Current price
K = 100       # Strike price
T = 1         # Time to maturity (1 year)
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility

call_price = asian_geometric_bs(S, K, T, r, sigma, 'call')
put_price = asian_geometric_bs(S, K, T, r, sigma, 'put')

print(f"Geometric Asian Call Option Price: {call_price:.4f}")
print(f"Geometric Asian Put Option Price:  {put_price:.4f}")
