"""
Here we store an example of the class for the mix signals
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from signals.time_series_class import MixAr, AR
from signals.aux_functions import sidekick


plot = False
plot2 = True

# Time parameters
dt = 0.1
Tmax = 100

# Let's get the axuiliary class
amplitude = 1
w1 = 1
w2 = 5
beta = sidekick(w1, w2, dt, Tmax, amplitude)

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

# First we construct the series without the sidekick
B = AR(phi, dt=dt, Tmax=Tmax)
B.initial_conditions(initial_conditions)
normal_series = B.construct_series()

# Second we construct the series with the mix
A = MixAr(phi, dt=dt, Tmax=Tmax, beta=beta)
A.initial_conditions(initial_conditions)
mix_series = A.construct_series()

time = A.time

if plot:
    plt.subplot(3, 1, 1)
    plt.plot(time, beta)

    plt.subplot(3, 1, 2)
    plt.plot(time, normal_series)

    plt.subplot(3, 1, 3)
    plt.plot(time, mix_series)

    plt.show()
