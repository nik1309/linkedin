import yfinance as yf
import numpy as np

# RANDOM WALK symmetric h1
#gen random walk, cumsum - sums next gen value to sum of prev, rand.ch gen new num either -1,1 equl p of 0.5
y1 = np.cumsum(np.random.choice([-1,1],1000,[0.5,0.5]))
plt.plot(y1)
plt.title("rand wlk")
plt.xlabel("step")
plt.ylabel("pos")
plt.grid()
plt.show()

# RANDOM WALK non symmetric h2
#gen random walk, cumsum - sums next gen value to sum of prev, rand.ch gen new num either -1,1 equl p of 0.5
y1a = np.cumsum(np.random.choice([-1,1],1000,p=[0.1,0.9]))
plt.plot(y1a)
plt.title("rand wlk non symmetric")
plt.xlabel("step")
plt.ylabel("pos")
plt.grid()
plt.show()

#brwnim mtn using sn/sqrt(n) against n/n h3

y2 = y1/(np.sqrt(len(y1)))
x2 = np.linspace(0,1,len(y1))
plt.plot(x2,y2)
plt.title("brwn mtn")
plt.xlabel("step")
plt.ylabel("pos")
plt.grid()
plt.show()


#  brwn mtn formal defn h4
t = 1
n = 1000
dt = t/n

timeax = np.linspace(0,1,n+1)
dB = np.random.normal(0,np.sqrt(dt), n)

B = np.zeros(n+1)
B[1:] = np.cumsum(dB)
plt.plot(timeax,B)
plt.title("brwn mtn formal defn")
plt.grid()
plt.show()



#drift and volatilty
import yfinance as yf
import numpy as np

# historical data
meta = yf.Ticker("META")
start_date = '2018-01-01'
end_date = '2018-01-06'
data = meta.history(start=start_date, end=end_date)

# get closing prices and calc log returns
prices = data['Close'].values
z = np.log(prices[1:] / prices[:-1])  # Z_i = log(S_i / S_{i-1})

#  α̂ (mean of log returns, one-day drift for log prices)
alpha_hat = np.mean(z)

# σ̂² (sample variance with N-2 denominator)
variance = np.sum((z - alpha_hat)**2) / (len(z) - 2)
sigma_hat = np.sqrt(variance)

# μ̂ (drift for GBM)
mu_hat = alpha_hat + 0.5 * variance
#print results
print(variance)
print(f"alpha_hat (log-price drift): {alpha_hat:.8f}")
print(f"sigma_hat (daily volatility): {sigma_hat:.8f}")
print(f"mu_hat (GBM drift): {mu_hat:.8f}")