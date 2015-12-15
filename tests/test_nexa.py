"""
Here we test things from the Nexa package
"""

import unittest
import numpy as np
import numpy.testing as nptest
from unittest import TestCase
import sys
sys.path.append('./')
import nexa.aux_functions as aux_functions

##################
# Test auxiliar functions
##################

class TestAuxiliaryFunctions(TestCase):
    """
    This is for the testing all the class in
    auxiliary functions
    """
    
    def test_normalization(self):
        """
        Test that value is normalized for 1
        """
        Number_of_tests = 1000
        low = -1000
        high = 1000
        for i in range(Number_of_tests):
            x = np.random.rand(100) * (high - low) + low
            y = aux_functions.softmax_base(x)
            result = np.sum(y)
            nptest.assert_almost_equal(result, 1.0)

    def test_finite(self):
        """
        This tests that the tests produces a non-inifite
        non nan value.
        """
        
        Number_of_tests = 1000
        low = -1000
        high = 1000
        for i in range(Number_of_tests):
            x = np.random.rand(100) * (high - low) + low
            y = aux_functions.softmax_base(x)

            # This should be True if all are finite
            all_finite = np.isfinite(y).all()
            self.assertTrue(all_finite)

if __name__ == '__main__':
    unittest.main()
