"""
This is the script for the sidekick function that will play along
with our alpha.
"""

import matplotlib.pyplot as plt
import numpy as np


def sidekick(w1, w2, dt, T, A=1):
    """
    This function crates the sidekick time series provided
    the two mixing frequencies w1, w2, the time resolution dt
    and the total time T.

    returns the expresion A * (cos(w1 * t) + cos(w2 * t))
    where t goes from 0 to int (T / dt).
    """
    Nt = int(T / dt)
    time = np.arange(Nt) * dt

    A1 = np.cos(w1 * time)
    A2 = np.cos(w2 * time)
    A3 = A1 + A2

    return A3 * A


w1 = 1.0
w2 = 0.1

dt = 0.1
T = 100.0
Nt = int(T / dt)

time = np.arange(Nt) * dt

A1 = np.cos(w1 * time)
A2 = np.cos(w2 * time)
A3 = A1 + A2

A4 = sidekick(w1, w2, dt, T)

plot = True

if plot:
    plt.subplot(4, 1, 1)
    plt.plot(time, A1)

    plt.subplot(4, 1, 2)
    plt.plot(time, A2)

    plt.subplot(4, 1, 3)
    plt.plot(time, A3)

    plt.subplot(4, 1, 4)
    plt.plot(time, A4)

    plt.show()
