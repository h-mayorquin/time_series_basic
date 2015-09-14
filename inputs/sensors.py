"""
Here we will put the class for the sensor. The sensor is the simplest object
in our framework it is just data with a sampling rate
"""

import numpy as np
from .lag_structure import LagStructure


class Sensor:
    """
    This is the basic class for a sensor. It contains the data of a
    single instance (pixel, time series, any other object with
    multiple samples) and some methods
    to operate with them
    """

    def __init__(self, data, dt=1.0,
                 lag_structure=LagStructure(window_size=10)):
        """
        Initializes the sensor, data should be an array of the
        samples for the sensor and dt the sampling rate,
        lag structure is an object from the LagStructure class
        """
        if(dt <= 0):
            raise ValueError("dt should be strictly positive")

        if(not isinstance(data, np.ndarray)):
            raise TypeError("Data has to be a numpy array")

        condition = isinstance(lag_structure, LagStructure)
        if(not condition and lag_structure is not None):
            raise TypeError("lag structure has to be a LagStructure instance")

        # Intialize the data
        self.data = data
        self.dt = dt
        self.lag_structure = lag_structure
        self.size = data.size

        # If no lag structure is not present create a simple one
        if(lag_structure is None):
            self.lag_structure = LagStructure(window_size=10)

        self.Nwindow_size = int(self.lag_structure.window_size / dt)

        # If the weights are not defined all of them to be equal
        if(self.lag_structure.weights is None):
            self.lag_structure.weights = np.ones(self.Nwindow_size)

        # Check the the last time windows falls
        last_index = int(lag_structure.lag_times[-1] / dt)
        max_delay_size = data.size - last_index - self.Nwindow_size

        if(max_delay_size <= 0):
            error_string = "Last window goes out of data"
            suggestion = ", Change the window size or the lag_times"
            information = ", The max delay is:" + str(max_delay_size)
            information += "\n For data size equal:" + str(data.size)
            raise IndexError(error_string + suggestion + information)

    def lag_back(self, lag):
        """
        From the perspective of the last data point this method
        moves back (towards the direction of the first data point)
        the whole array of data. The units here are given in terms
        of the times vector in the lag structure. So, if the first
        element of the times in the lag_structure is 3
        (times[0] = 3), the whole array will be moved three seconds
        back when lag.back(1) is called.

        This returns an array of size lag_structure.window_size weighted
        by the lag_structure.weights array (means multiplied)
        """
        if(self.lag_structure is None):
            raise ValueError("Need a lag structure to lag")

        if(lag < 1):
            raise IndexError("Lags need to be bigger than one")

        if(lag > self.lag_structure.lag_times.size):
            raise IndexError("Lag outside of lag_structure times")

        lag_index = int(self.lag_structure.lag_times[lag - 1] / self.dt)

        start = self.size - lag_index - self.Nwindow_size
        end = self.size - lag_index

        return self.data[start:end] * self.lag_structure.weights


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
