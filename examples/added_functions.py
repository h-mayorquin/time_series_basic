"""
This is the script for the sidekick function that will play along
with our alpha.
"""

import matplotlib.pyplot as plt
import numpy as np
from signals.signal_class import TrigonometricMix

w1 = 1.0
w2 = 0.1

dt = 0.1
T = 100.0
Nt = int(T / dt)

time = np.arange(Nt) * dt

A1 = np.cos(w1 * time)
A2 = np.cos(w2 * time)
A3 = A1 + A2

plot = True

if plot:
    plt.subplot(3, 1, 1)
    plt.plot(time, A1)

    plt.subplot(3, 1, 2)
    plt.plot(time, A2)

    plt.subplot(3, 1, 3)
    plt.plot(time, A3)

    plt.show()
