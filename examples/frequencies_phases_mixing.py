"""
A script that mixes the frequencies using the sublass of
the big main class.
"""

import matplotlib.pyplot as plt
import numpy as np
from signals.signal_class import TrigonometricMix

# Frequencies and phases
f1 = 1.0
f2 = 1.0
f3 = 1.0
f4 = 1.0

# Phases
phase_1 = 0.0
phase_2 = np.pi
phase_3 = np.pi
phase_4 = 0.0

f_matrix = np.array(((f1, f2), (f3, f4)))
phase_matrix = np.array(((phase_1, phase_2), (phase_3, phase_4)))

# Intialize the class
A = TrigonometricMix(Tmax=100, phase_m=phase_matrix, frequency_m=f_matrix)
time = A.time
initial_coditions = np.array((0, 1))
A.set_initial_conditions(initial_coditions)
# Construct the series
A.construct_series()

signal_x = A.series[0, :]
signal_y = A.series[1, :]


# Plot
plot = True
if plot:
    plt.plot(time, signal_x)
    plt.plot(time, signal_y)

    plt.show()
