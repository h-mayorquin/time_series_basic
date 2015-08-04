"""
Here we will put the class for the sensor. The sensor is the simplest object
in our framework it is just data with a sampling rate
"""


class Sensor:
    """
    This is the basic class for a sensor. It contains the data of a
    single instance (pixel, time series, any other object with
    multiple samples) and some methods
    to operate with them
    """

    def __init__(self, data, dt=1.0):
        """
        Initializes the sensor, data should be an array of the
        samples for the sensor and dt the sampling rate.
        """

        self.data = data
        self.dt = dt

        self.size = data.size

    def lag_ahead(self, lag, nlags):
        """
        This function lags the sensor ahead from its position

        To do:
        Describe parameters

        Add asertion for size
        """

        start = lag
        end = (self.size) - (nlags - lag)
        return self.data[start:end]

    def lag_back(self, lag, nlags):
        """
        This function lags the senor back fomr its position

        If the nlags is bigger than hialf of the data make a warning
        it if is bigger an exception.
        """

        start = (nlags - lag)
        end = self.size - lag
        return self.data[start:end]


class PerceptualSpace:
    """
    This con
