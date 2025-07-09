from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        tickers = [t.strip().upper() for t in data['tickers'].split(',')]
        start_date = data['start_date']
        end_date = data['end_date']


        prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)


        if 'Adj Close' not in prices.columns:
            return jsonify({'error': 'Adj Close prices not found'}), 400

        adj_close = prices['Adj Close']


        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame()


        adj_close = adj_close[tickers]


        returns = adj_close.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(tickers)
        trading_days = 252

        def neg_sharpe(weights, mean_returns, cov_matrix, rf=0.01):
            port_return = np.dot(weights, mean_returns) * trading_days
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))
            return -(port_return - rf) / port_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]

        result = minimize(neg_sharpe, init_guess,
                          args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            return jsonify({'error': 'Optimization failed'}), 500

        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, mean_returns) * trading_days
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix * trading_days, optimal_weights)))
        sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility

        weights_dict = dict(zip(tickers, np.round(optimal_weights, 4).tolist()))

        # Prepare last 5 days of adjusted close prices with formatted dates
        recent_prices = adj_close.tail(5).round(2).copy()
        recent_prices.index = recent_prices.index.strftime('%Y-%m-%d')
        recent_prices.reset_index(inplace=True)
        recent_prices.rename(columns={'index': 'Date'}, inplace=True)
        adj_close_dict = recent_prices.to_dict(orient='records')

        return jsonify({
            'optimal_weights': weights_dict,
            'expected_annual_return': round(portfolio_return, 4),
            'annual_volatility': round(portfolio_volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'adj_close_prices': adj_close_dict
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
