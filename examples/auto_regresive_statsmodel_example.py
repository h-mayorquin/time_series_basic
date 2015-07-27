"""
A scrip where the Auto Regresive class and the statsmodels
function are tested.
"""

import matplotlib.pyplot as plt
import numpy as np
from signals.time_series_class import AR
import statsmodels.api as sm

plot = True

# Time parameters
dt = 1.0
Tmax = 100

# First we need the phi's vector
phi0 = 0.0
phi1 = -0.8
phi2 = 0.3

phi = np.array((phi0, phi1, phi2))


# Now we need the initial conditions
x0 = 1
x1 = 1
x2 = 0

initial_conditions = np.array((x0, x1, x2))

# Now we load the class and construct it
A = AR(phi, dt=dt, Tmax=Tmax)
A.initial_conditions(initial_conditions)
series = A.construct_series()

time = A.time

# Now we get the autocorrelation function
auto_correlation = sm.tsa.stattools.acf(series)
partial_correlation = sm.tsa.stattools.pacf(series)

# We will get the fitted model with an AR model now
sm_model = sm.tsa.ARMA(series, (2, 0))
# fit = sm_model.fit()

# print(fit.params)


if plot:
    plt.subplot(3, 1, 1)
    plt.plot(time, series, '*-')

    plt.subplot(3, 1, 2)
    plt.plot(auto_correlation, '*-')

    plt.subplot(3, 1, 3)
    plt.plot(partial_correlation, '*-')

    plt.show()
