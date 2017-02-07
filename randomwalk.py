""" Random Walk Explorations

Code based heavily on post found at:
http://machinelearningmastery.com/
gentle-introduction-random-walk-times-series-forecasting-python/

Some ways to check if your time series is a random walk are as follows:

1) The time series shows a strong temporal dependence that decays linearly or in
   a similar pattern.
2) The time series is non-stationary and making it stationary shows no obviously
   learnable structure in the data.
3) The persistence model provides the best source of reliable predictions.

This last point is key for time series forecasting. Baseline forecasts with the
persistence model quickly flesh out whether you can do significantly better. If
you can’t, you’re probably working with a random walk.

"""

import random
from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot as ac_plot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error as mse

# Random Number Series Creation
random.seed(1)
series = [random.randrange(10) for i in range(1000)]

# Random Walk Series Creation
random.seed(1)
random_walk = list([0])
lag = len(random_walk)

for i in range(lag, 1000):
	movement = -1 if random.random() < 0.5 else 1
	value = random_walk[i-lag] + movement
	random_walk.append(value)

# Difference our random walk series to create stationarity.
diff = list()
for i in range(1,len(random_walk)):
    value = random_walk[i] - random_walk[i-1]
    diff.append(value)

# Split Test/Train Sets
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]

# Persistence (Naive Forecast)
predictions = list()
history = train[-1]
for i in range(len(test)):
    yhat = history
    predictions.append(yhat)
    history = test[i]
error = mse(test, predictions)
print('Persistence MSE: %.3f' % error)

# Random Prediction
""" Showing randomly picking +/-1 has worse MSE than picking last value. """
predictions_rand = list()
history = train[-1]
for i in range(len(test)):
    yhat_rand = history + (-1 if random.random() < 0.5 else 1)
    predictions_rand.append(yhat_rand)
    history = test[i]
error_rand = mse(test, predictions_rand)
print('Random Choice MSE: %.3f' % error_rand)

# Statistical Test
result = adfuller(random_walk)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# Plots
acrand_fig = plt.figure(1)
ac_plot(series)
plt.xscale('log')
acrand_fig.suptitle('Autocorrellation of Random Numbers')

rwfig = plt.figure(2)
plt.plot(random_walk)
rwfig.suptitle('Random Walk Sequence')
plt.xlabel('Period')
plt.ylabel('Measurement')

acrw_fig = plt.figure(3)
ac_plot(random_walk)
acrw_fig.suptitle('Autocorrellation of Random Walk')

diff_fig = plt.figure(4)
plt.plot(diff)
diff_fig.suptitle('Stationary Series from Differencing (lag=1)')
plt.ylim(-1.2, 1.2)
plt.xlim(0, 100)
plt.xlabel('Period')

acdiff_fig = plt.figure(5)
ac_plot(diff)
acdiff_fig.suptitle('Autocorrellation of Differenced Stationary Series')

plt.show()
