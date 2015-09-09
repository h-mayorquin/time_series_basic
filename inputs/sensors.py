"""
Here we will put the class for the sensor. The sensor is the simplest object
in our framework it is just data with a sampling rate
"""

import numpy as np


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
    This class contains a list with all the sensors that are
    relevant for the task.
    """

    def __init__(self, sensors, nlags):
        """
        Initializes the perceptual space

        sensors: a list of sensors with all the data
        nlag: the maximum number of lags

        # To do, check that the resoultions of the dt are the same
        # Check the both sensors and nlags are provided
        # Check that the sizes are the same
        # Check that the data are sensors otherwise convert them
        """

        # Get the data and make them sensors
        self.sensors = sensors
        self.nlags = nlags
        self.Nsensors = len(sensors)
        self.data_size = sensors[0].size

        # Create the lags
        self.lags = np.arange(self.nlags)

    def calculate_SLM(self):
        """
        This calculates the Sensor Lagged Matrix (SLM) of the
        set of sensors its dimensions should be:

        (nsensors * nlags, data_size - nlags)
        """

        # Initialize it
        self.SLM = np.zeros((self.Nsensors * self.nlags,
                             self.data_size - self.nlags))

        # Get all the possible lags and put it into a matrix
        for lag in self.lags:
            for sensor_index, sensor in enumerate(self.sensors):
                index = lag * self.Nsensors + sensor_index
                self.SLM[index, :] = sensor.lag_back(lag, self.nlags)

        return self.SLM

    def calculate_STDM(self):
        """
        From the SLM calculate the STDM (Spatio Temporal Distance Matrix)
        """

        # Check fist whether SLM is already calculated
        try:
            assert(self.SLM is not None)
        except:
            print("SLM was not calcluated")
            self.calculate_SLM()

        return np.corrcoef(self.SLM)
