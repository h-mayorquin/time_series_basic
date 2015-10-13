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


def gaussian_bump(x, mean=0, maximum=1.0,  baseline=0, HWHM_a=1.0, attenuation=2):
    """
    
    Returns a gaussian bump (something of the shape exp(-x^2)) with
    a given maximum and a given baseline.

    Parameters
    -----------
    x : the values of the argument

    mean : the value at which the function is centered

    maximum : the maximum value that the function attains at the mean

    baseline : the value far from the mean, the baseline (zero shited)

    HWHM_a : This come from the half width at half maximum terminology.
    In this case it denotes the value of the argument at which the
    function will have attenuated by the attenuation value (next arg)

    attenuation value: the attenuation value of the HWHM_a.

    In brief for the last two arguments:

    gaussian_bumps(HWHM_a) = maximum / attenuation
    
    """

    # First we will calculate sigma based on all the other values
    arg1 = attenuation * (maximum - baseline)
    arg2 = (maximum- attenuation * baseline)
    A = np.log(arg1 / arg2)

    sigma = (HWHM_a - mean) / np.sqrt(A)

    argument = (x - mean) / sigma
    gaussian = np.exp(-argument ** 2)
    
    return (maximum - baseline) * gaussian + baseline

