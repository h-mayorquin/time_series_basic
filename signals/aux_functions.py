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


def bump(t, offset, center, Max):
    """ 
    This function creates a bump with a quadratic 
    function. The center of the function is at the
    center. The zeros are at center +- offset and
    finally the value of the funtion at center is 
    Max. 

    Note that this function does not take caret that the 
    offsets are inside the vector t.
    """

    beta = offset ** 2
    D = Max / beta

    y = D * (beta - (t - center)**2)
    y[ y < 0] = 0

    return y
