import unittest
from unittest import TestCase
import numpy as np
import numpy.testing as nptest
import sys
sys.path.append('./')
from inputs.lag_structure import LagStructure


class test_lag_structure(TestCase):
    """
    This is for testing the lag structure class.
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
        self.assertRaises(LagStructure, times=times)
        self.assertRaises(LagStructure, weights=weights)
        self.assertRaises(LagStructure, times=times, weights=weights)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
