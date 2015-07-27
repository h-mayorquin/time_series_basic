"""
A couple of auxiliary functions
"""
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
