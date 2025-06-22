#least squares
from sklearn.linear_model import LinearRegression
import numpy as np
def american_call_lsm_mc(S0, K, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    discount = np.exp(-r * dt)
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=n_simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    V = np.maximum(S[:, -1] - K, 0)

    for t in range(n_steps - 1, 0, -1):
        itm = np.where(S[:, t] > K)[0]
        if len(itm) == 0:
            continue
        X = S[itm, t].reshape(-1, 1)
        Y = V[itm] * discount

        # Regression to estimate continuation value
        model = LinearRegression().fit(X, Y)
        continuation = model.predict(X)

        exercise = S[itm, t] - K
        V[itm] = np.where(exercise > continuation, exercise, V[itm] * discount)

    price = np.exp(-r * dt) * np.mean(V)
    return price

price = american_call_lsm_mc(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_simulations=10000, n_steps=50)
print(f"American Call Option Price (LSM): {price:.4f}")
