# import libraries for data handling, math, visualization, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------------------
# step 1: load and prepare data
# ------------------------------

# for demonstration, we simulate stock prices as a noisy sine wave
# in practice you would load real data, e.g., pd.read_csv('stock.csv')
np.random.seed(42)
time = np.arange(0, 200, 0.1)
prices = np.sin(0.05 * time) + 0.5 * np.random.normal(size=len(time))

# convert to dataframe
df = pd.DataFrame(prices, columns=['price'])

# normalize prices to [0,1] range for stable lstm training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df[['price']])

# ------------------------------
# step 2: create time series dataset
# ------------------------------

# function to create input-output pairs
# x = past look_back values, y = next value
def create_dataset(series, look_back=20):
    x, y = [], []
    for i in range(len(series) - look_back):
        x.append(series[i:(i+look_back), 0])
        y.append(series[i + look_back, 0])
    return np.array(x), np.array(y)

look_back = 20
x, y = create_dataset(scaled_prices, look_back)

# reshape x to fit lstm input: [samples, timesteps, features]
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# split into train and test sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ------------------------------
# step 3: build lstm model
# ------------------------------

# sequential model means layers are stacked one after another
model = Sequential()
# lstm layer with 50 hidden units
model.add(LSTM(50, input_shape=(look_back, 1)))
# dense layer with 1 output (the predicted next value)
model.add(Dense(1))

# compile model with adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mse')

# ------------------------------
# step 4: train model
# ------------------------------

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)

# ------------------------------
# step 5: make predictions
# ------------------------------

y_pred = model.predict(x_test)

# invert scaling back to original price values
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# ------------------------------
# step 6: benchmark methods
# ------------------------------

# naive forecast: tomorrow = today
naive_pred = y_test_inv[:-1]  # shift by one
naive_true = y_test_inv[1:]
naive_rmse = np.sqrt(mean_squared_error(naive_true, naive_pred))

# simple moving average forecast with window=3
sma_series = pd.Series(y_test_inv.flatten())
sma_pred = sma_series.rolling(window=3).mean().shift(1).values

# drop first few nans caused by rolling and shifting
valid_idx = ~np.isnan(sma_pred)
sma_pred = sma_pred[valid_idx]
sma_true = y_test_inv.flatten()[valid_idx]

# calculate sma rmse
sma_rmse = np.sqrt(mean_squared_error(sma_true, sma_pred))

# lstm rmse
lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print("naive forecast rmse:", naive_rmse)
print("sma forecast rmse:", sma_rmse)
print("lstm forecast rmse:", lstm_rmse)

# ------------------------------
# step 7: plot results
# ------------------------------

plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='true prices')
plt.plot(y_pred_inv, label='lstm predictions')
plt.legend()
plt.show()
