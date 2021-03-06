"""
Here we have the tests for everything in the inputs package
"""

import unittest
from unittest import TestCase
import numpy as np
import numpy.testing as nptest
import sys
sys.path.append('./')
from inputs.lag_structure import LagStructure
from inputs.sensors import Sensor, PerceptualSpace

######################
# Test Lag Structure
######################


class TestLagStructure(TestCase):
    """
    This is for testing the lag structure class in inputs
    """
    def test_default_init(self):
        """
        Test for the default conditions
        """
        lag_times = np.arange(10)

        test_lag = LagStructure()
        nptest.assert_almost_equal(lag_times, test_lag.lag_times)

    def test_input_not_numpyarray(self):
        """
        This test that only numpy array are taken as inputs
        """
        lag_times = [3, 2, 4]
        weights = {1, 2, 4}

        # All of those inputs have to raise an exception
        self.assertRaises(ValueError, LagStructure, lag_times=lag_times)
        self.assertRaises(ValueError, LagStructure, weights=weights)
        self.assertRaises(ValueError, LagStructure,
                          lag_times=lag_times, weights=weights)

    def test_window_size_positive(self):
        """
        Test that the window size is positive
        """
        lag_times = np.arange(100)
        self.assertRaises(ValueError, LagStructure,
                          lag_times=lag_times, window_size=-3.0)

    def test_times_come_sorted(self):
        """
        Test that the lag_times vector comes in sorted order.
        """

        lag_times1 = np.array((5, 3, 1))
        lag_times2 = np.random.randint(1, 15, 10)

        self.assertRaises(ValueError, LagStructure, lag_times=lag_times1)
        self.assertRaises(ValueError, LagStructure, lag_times=lag_times2)

######################
# Test Sensor
######################


class test_sensors(TestCase):
    """
    Testing the Sensor Class in inputs
    """

    def test_positive_sampling_rate(self):
        """
        Test that the sampling rate is positive
        """
        data = np.arange(100)
        dt = -0.1

        self.assertRaises(ValueError, Sensor, data, dt=dt)

    def test_data_is_numpy_array(self):
        """
        There should be an exception thrown if the data is not
        a numpy array
        """
        data1 = [0, 1, 3]
        data2 = {1, 2, 3}
        data3 = {1: 2, 3: 4, 5: 4}

        self.assertRaises(TypeError, Sensor, data1)
        self.assertRaises(TypeError, Sensor, data2)
        self.assertRaises(TypeError, Sensor, data3)

    def test_lag_structure_class(self):
        """
        There should be an exception if the lag structure is
        not of the lag structure class
        """

        data = np.arange(100)
        lag_structure1 = {100, 200, 300}
        lag_structure2 = [1, 2, 3]
        lag_structure3 = 100

        self.assertRaises(TypeError, Sensor, data,
                          lag_structure=lag_structure1)
        self.assertRaises(TypeError, Sensor, data,
                          lag_structure=lag_structure2)
        self.assertRaises(TypeError, Sensor, data,
                          lag_structure=lag_structure3)

    def test_window_size_vs_size_of_data(self):
        """
        This test that the window does not get out of the
        range of data for the last lag. 

        This method only works for push back
        """
        T = 10
        dt = 0.3
        Nt = T / dt
        epislon = 1.0
        last_lag = 3.0
        
        data = np.arange(Nt)
        lag_times = np.zeros(10)
        lag_times[-1] = T - last_lag  # Put the lag at where wanted

        # The windo size gets out of its place by epislon
        window_size = 3.0 + epislon
        lag_structure = LagStructure(lag_times, window_size=window_size)

        with self.assertRaises(IndexError):
            Sensor(data, dt, lag_structure)

    def test_lag_back_limits(self):
        """
        Test that the lab_back method throws an assertion when
        called with a lag smaller than 1 or bigger than the size
        of the whole data.
        """
        data = np.arange(100)
        dt = 0.5
        lag_times = np.arange(10)
        window_size = 10
        lag_structure = LagStructure(lag_times=lag_times,
                                     window_size=window_size)
        sensor = Sensor(data, dt, lag_structure)

        with self.assertRaises(IndexError):
            sensor.lag_back(0)

        with self.assertRaises(IndexError):
            sensor.lag_back(lag_times.size + 1)

    def test_lag_methods_without_lag_structure(self):
        """
        The lag methods should throw an exception if not lag
        method is defined when called
        """
        data = np.arange(1000)
        sensor = Sensor(data)
        self.assertRaises(TypeError, sensor.lag_back(1))

    def test_lag_back_method_return_size(self):
        """
        The lag bag method should return an array
        of size equal to the windows size / dt
        """

        min_dt = 0.001
        max_window_size = 20
        max_lag = 100
        # This is the maximum size that could be attained
        data_size = int(max_lag / min_dt + max_window_size / min_dt + 1)
        data = np.random.rand(data_size)
        lag_times = np.arange(max_lag)
        test_size = 1000
        window_sizes = np.random.uniform(0.001, max_window_size, test_size)

        test_size = 1000
        window_sizes = np.random.uniform(0.001, max_window_size, test_size)
        dts = np.random.uniform(min_dt, 10, test_size)

        for window_size, dt in zip(window_sizes, dts):

            Nwindow_size = int(window_size / dt)
            weights = np.ones(Nwindow_size)

            lag_structure = LagStructure(lag_times, weights, window_size)
            sensor = Sensor(data, dt=dt, lag_structure=lag_structure)
            self.assertAlmostEqual(Nwindow_size, sensor.lag_back(1).size)

    def test_lag_back_last_value(self):
        """
        This tests than lag back method gives numerically correct
        result for the last value.
        """
        data = np.random.rand(100)
        dt = 0.5
        lag_times = np.arange(1, 5.0)
        window_size = 5.0
        Nwindow_size = int(window_size / dt)
        weights = np.ones(Nwindow_size)

        lag_structure = LagStructure(lag_times, weights, window_size)
        sensor = Sensor(data, dt=dt, lag_structure=lag_structure)

        index = int(lag_times[0] / dt)
        weight = weights[0]
        lagged_value = data[-(index + 1)]

        # Get the last value of the lagged back sensor
        lagged_sensor = sensor.lag_back(1)

        self.assertAlmostEqual(lagged_value * weight, lagged_sensor[-1])

    def test_lag_back_values(self):
        """
        This tests than lag back method gives numerically correct
        for all the values with equal weights
        """

        data = np.random.rand(1000)
        dt = 0.5
        lag_times = np.arange(10.0)
        window_size = 10.0
        Nwindow_size = int(window_size / dt)
        lag = 4

        lag_structure = LagStructure(lag_times=lag_times,
                                     window_size=window_size)
        sensor = Sensor(data, dt=dt, lag_structure=lag_structure)

        first_lag_index = int(lag_times[lag - 1] / dt)
        start = data.size - first_lag_index - Nwindow_size

        result = np.zeros(Nwindow_size)

        for index in range(Nwindow_size):
            result[index] = data[start + index]

        nptest.assert_array_almost_equal(result, sensor.lag_back(lag))

    def test_lag_back_weights(self):
        """
        This tests than lag back method gives numerically correct
        for all the values with different weights.
        """

        data = np.random.rand(1000)
        dt = 0.1
        lag_times = np.arange(10.0)
        window_size = 10.0
        Nwindow_size = int(window_size / dt)
        weights = np.exp(-np.arange(window_size / dt))

        lag_structure = LagStructure(lag_times=lag_times, weights=weights,
                                     window_size=window_size)
        sensor = Sensor(data, dt=dt, lag_structure=lag_structure)

        first_lag_index = int(lag_times[0] / dt)
        start = data.size - first_lag_index - Nwindow_size

        result = np.zeros(Nwindow_size)

        weights = weights[::-1]
        for index in range(Nwindow_size):
            result[index] = weights[index] * data[start + index]

        nptest.assert_array_almost_equal(result, sensor.lag_back(1))

######################
#  Test PerceptualSpace
######################


class TestPerceptualSpace(TestCase):

    def test_SLM_first_value(self):
        """
        This tests that the SLM matrix returns the first row
        correctly.
        """
        # First we define the data
        dt = 0.3
        T = 100
        Tperiod = 20.0
        w = (2 * np.pi) / Tperiod
        t = np.arange(0, T, dt)
        data1 = np.sin(w * t)
        data2 = np.cos(w * t)

        # Now we define the lagged structure
        lag_times = np.arange(0, 10)
        window_size = 5.0
        lag_structure = LagStructure(lag_times=lag_times,
                                     window_size=window_size)

        # Build the sensor
        sensor1 = Sensor(data1, dt, lag_structure)
        sensor2 = Sensor(data2, dt, lag_structure)
        sensors = [sensor1, sensor2]

        # Build the perceptual space
        perceptual_space = PerceptualSpace(sensors, lag_first=False)
        SLM = perceptual_space.calculate_SLM()
        first_row = SLM[0]
        second_row = SLM[1]

        # Now we get the firs row manually
        result1 = sensor1.lag_back(1)
        result2 = sensor2.lag_back(1)
        nptest.assert_array_almost_equal(result1, first_row)
        nptest.assert_array_almost_equal(result2, second_row)

    def test_SLM_first_value_lag_first(self):
        """
        This tests that the SLM matrix returns the first row
        correctly.
        """
        # First we define the data
        dt = 0.3
        T = 100
        Tperiod = 20.0
        w = (2 * np.pi) / Tperiod
        t = np.arange(0, T, dt)
        data1 = np.sin(w * t)
        data2 = np.cos(w * t)

        # Now we define the lagged structure
        lag_times = np.arange(0, 10)
        window_size = 5.0
        lag_structure = LagStructure(lag_times=lag_times,
                                     window_size=window_size)

        # Build the sensor
        sensor1 = Sensor(data1, dt, lag_structure)
        sensor2 = Sensor(data2, dt, lag_structure)
        sensors = [sensor1, sensor2]

        # Build the perceptual space
        perceptual_space = PerceptualSpace(sensors, lag_first=True)
        SLM = perceptual_space.calculate_SLM()
        first_row = SLM[0]
        second_row = SLM[1]

        # Now we get the firs row manually
        result1 = sensor1.lag_back(1)
        result2 = sensor1.lag_back(2)
        nptest.assert_array_almost_equal(result1, first_row)
        nptest.assert_array_almost_equal(result2, second_row)


    def test_SLM_columns_to_time_map(self):
        """
        This tests that the map that converts the associates the columns
        of the SLM with times. This is done in such a way that each
        coulmn is mapped to the time that matches the most lagged element
        in lag_back
        """

        # Build signal
        signal1 = np.arange(1, 11, 1.0)
        signal2 = np.arange(-1, -11, -1.0)
        # Add noise
        signal1 += np.random.uniform(size=signal1.shape) * 0.01
        signal2 += np.random.uniform(size=signal1.shape) * 0.01
        # Pack them
        signals = [signal1, signal2]

        # PerceptualSpace
        dt = 1.0
        lag_times = np.arange(0, 3, 1)
        window_size = 8
        weights = None

        lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
        # sensors = [Sensor(signal1, dt, lag_structure,), Sensor(signal2, dt, lag_structure)]
        sensors = [Sensor(signal, dt, lag_structure) for signal in signals]
        perceptual_space = PerceptualSpace(sensors, lag_first=True)

        times = perceptual_space.map_SLM_columns_to_time()
        result = np.array((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        nptest.assert_array_almost_equal(result, times)

        
if __name__ == '__main__':
    unittest.main()
