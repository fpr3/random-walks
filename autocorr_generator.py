import numpy as np
from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot as ac_plot

def sample_signal(n_samples, corr, mu=0, sigma=1):
    assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"

    # Find out the offset `c` and the std of the white noise `sigma_e`
    # that produce a signal with the desired mean and variance.
    # See https://en.wikipedia.org/wiki/Autoregressive_model
    #   Example:_An_AR.281.29_process
    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.normal(0, sigma_e))

    return np.array(signal)

def compute_corr_lag_1(signal):
    return np.corrcoef(signal[:-1], signal[1:])[0][1]

# Examples.
series = sample_signal(5000,.75, mu=2, sigma=3)
print(compute_corr_lag_1(series))
#print(compute_corr_lag_1(sample_signal(5000, 0.5)))
print(np.mean(series))
#print(np.mean(sample_signal(5000, 0.5, mu=2)))
print(np.std(series))
#print(np.std(sample_signal(5000, 0.5, sigma=3)))

series = sample_signal(5000,.75)
fig1=plt.figure(1)
plt.plot(series)
fig2=plt.figure(2)
ac_plot(series)
plt.xscale('log')
plt.show()
