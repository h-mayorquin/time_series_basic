import matplotlib.pyplot as plt
import numpy as np
from signals.signal_class import SpatioTemporalSignal

A = SpatioTemporalSignal(Tmax=100)

time = A.time

# Cross terms
a = np.cos(time)
b = np.sin(time)
c = np.cos(time)
d = np.sin(time)

interaction = np.array(((a, b), (d, c)))
initial_coditions = np.array((0, 1))

# Set the interaction and construct the series
A.set_interaction(interaction)
A.set_initial_conditions(initial_coditions)
A.construct_series()

signal_x = A.series[0, :]
signal_y = A.series[1, :]


# Plot
plot = True
if plot:
    plt.subplot(2, 1, 1)
    plt.plot(time, signal_x)
    plt.plot(time, signal_y)

    plt.subplot(2, 1, 2)
    plt.plot(time, a)
    plt.plot(time, b)

    plt.show()
