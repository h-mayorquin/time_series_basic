"""
This is the module for the class lag structure
"""
import numpy as np


class LagStructure(object):
    """
    A lag structure that consists on two numpy arrays one for the times that
    of the lag and other for the weights that the lag at those times should
    have:

    Times contains thet actual instants or points in time where that we want
    to peak into, for example:
    self.times = np.array((1, 2, 3, 4)) means that we should look at the signal
    at every second.
    on the other hand self.times = np.array((0.5, 1, 1.5, 2) means that we take
    a look on the signal every half a second.

    weights is how much weight should we give to a value at that point in time.
    It is equivalent of the signal at that point in time multiplied by
    the weight.

    Times has to be intialized to a numpy array, it gives an array with the
    first ten numbers by default. Weights can be unspecified and then a
    vector of equal weights as long as times is created for it
    """

    def __init__(self, times=np.arange(10), weights=None, window_size=10):

        # Test for input arrays
        if(not isinstance(times, np.ndarray)):
            raise ValueError("times should be numpy array")

        condition = isinstance(weights, np.ndarray) or weights is None
        if(not condition):
            raise ValueError("Weights should be numpy arrray or None")

        if weights is None:
            weights = np.ones(times.size)

        if weights.size != times.size:
            raise ValueError("times and weights should be same size")

        if window_size < 0:
            raise ValueError("Windows size should be positive")

        # Asign the values
        self.times = times
        self.weights = weights
        self.window_size = window_size
