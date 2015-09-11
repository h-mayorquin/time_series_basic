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
from inputs.sensors import Sensor


class test_lag_structure(TestCase):
    """
    This is for testing the lag structure class in inputs
    """
    def test_default_init(self):
        """
        Test for the default conditions
        """
        lag_times = np.arange(10)
        weights = np.ones(10)

        test_lag = LagStructure()
        nptest.assert_almost_equal(lag_times, test_lag.lag_times)
        nptest.assert_array_almost_equal(weights, test_lag.weights)

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
        range of data for the last lag
        """
        data = np.arange(1000)
        dt = 0.1
        lag_times = np.arange(90)
        window_size = 11.0
        lag_structure = LagStructure(lag_times, window_size=window_size)

        with self.assertRaises(IndexError):
            Sensor(data, dt, lag_structure)

    def test_lag_methods_without_lag_structure(self):
        """
        The lag methods should throw an exception if not lag
        method is defined when called
        """
        data = np.arange(1000)
        sensor = Sensor(data)
        self.assertRaises(TypeError, sensor.lag_back(1))


if __name__ == '__main__':
    unittest.main()
