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
        times = np.arange(10)
        weights = np.ones(10)

        test_lag = LagStructure()
        nptest.assert_almost_equal(times, test_lag.times)
        nptest.assert_array_almost_equal(weights, test_lag.weights)

    def test_equal_times_and_weights_size(self):
        """
        Test that the times vector and the weight vector are equal
        """
        times = np.arange(100)
        weights = np.arange(10)

        # This should raise an error
        self.assertRaises(LagStructure, times=times, weights=weights)

    def test_input_not_numpyarray(self):
        """
        This test that only numpy array are taken as inputs
        """
        times = [3, 2, 4]
        weights = {1, 2, 4}

        # All of those inputs have to raise an exception
        self.assertRaises(ValueError, LagStructure, times=times)
        self.assertRaises(ValueError, LagStructure, weights=weights)
        self.assertRaises(ValueError, LagStructure,
                          times=times, weights=weights)


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

        self.assertRaises(ValueError, Sensor, data1)
        self.assertRaises(ValueError, Sensor, data2)
        self.assertRaises(ValueError, Sensor, data3)

    def test_lag_structure_class(self):
        """
        There should be an exception if the lag structure is
        not of the lag structure class
        """

        data = np.arange(100)
        lag_structure1 = {100, 200, 300}
        lag_structure2 = [1, 2, 3]
        lag_structure3 = 100

        self.assertRaises(ValueError, Sensor, data,
                          lag_structure=lag_structure1)
        self.assertRaises(ValueError, Sensor, data,
                          lag_structure=lag_structure2)
        self.assertRaises(ValueError, Sensor, data,
                          lag_structure=lag_structure3)

    def test_lag_back(self):
        """
        Tests if the lag structure is proper
        """
        


def main():
    unittest.main()

if __name__ == '__main__':
    main()
