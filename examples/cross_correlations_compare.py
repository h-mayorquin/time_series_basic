"""
This script can be used to compare cross-correlations. First
we have the cross-correlation as calculated by numpy and
then we also calculate them using statsmodels package.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

T = 30.0
dt = 0.1
Nt = int(T / dt)
t = np.arange(Nt) * dt

# Let's build the box
box_i = 1.0
Nbox_i = int(box_i / dt)
box = 3.0
Nbox = int(box / dt)
box_argument = np.arange(Nbox_i, Nbox_i + Nbox)
x = np.zeros(Nt)
x[box_argument] = 1

# The exponential
tau = 10
y = np.exp(- t / tau)

mode = 'full'
correlation = np.correlate(x, y, mode=mode)
correlation_inv = np.correlate(y, x, mode=mode)
correlation_sp = sm.tsa.stattools.ccf(x, y)
correlation_sp_inv = sm.tsa.stattools.ccf(y, x)

plt.subplot(4, 1, 1)
plt.plot(correlation)
plt.subplot(4, 1, 2)
plt.plot(correlation_inv)

plt.subplot(4, 1, 3)
plt.plot(correlation_sp)

plt.subplot(4, 1, 4)
plt.plot(correlation_sp_inv)

plt.show()

if False:
    plt.plot(t, x)
    plt.hold(True)
    plt.plot(t, y)
    plt.show()

