import numpy as np
from scipy.stats import norm

def black_scholes_european(S, K, T, r, sigma, option_type='call'):
    """
    parameters:
        S : float       - current stock price
        K : float       - strike price
        T : float       - time to maturity (in years)
        r : float       - risk-free interest rate (annualized)
        sigma : float   - volatility of the underlying asset (annualized)
        option_type : str - 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price


S = 100       # current stock price
K = 100       # strike price
T = 1         # time to maturity (1 year)
r = 0.05      # risk-free interest rate
sigma = 0.2   # volatility

call_price = black_scholes_european(S, K, T, r, sigma, 'call')
put_price = black_scholes_european(S, K, T, r, sigma, 'put')

print(f"European Call Option Price: {call_price:.4f}")
print(f"European Put Option Price:  {put_price:.4f}")
