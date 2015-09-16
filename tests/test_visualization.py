import unittest
from unittest import TestCase
import numpy as np
import numpy.testing as nptest
import sys
sys.path.append('./')
from inputs.lag_structure import LagStructure
from inputs.sensors import Sensor, PerceptualSpace
from visualization.aux import linear_to_matrix


class TestAuxFunctions(TestCase):
    """
    This tests the auxiliary functions for visualization.
    """

    def test_aux_functions(self):
        """
        Test the auxiliary functions
        """

        indexes = np.arange(20)
        Nsensors = 2
        Nlags = 3

        matrix_indexes1 = linear_to_matrix(indexes, Nsensors, Nlags, True)
        matrix_indexes2 = linear_to_matrix(indexes, Nsensors, Nlags, False)

        result1 = np.arange(Nlags)
        result2 = np.arange(Nsensors)

        nptest.assert_almost_equal(result1, matrix_indexes1[:Nlags, 1])
        nptest.assert_almost_equal(result2, matrix_indexes2[:Nsensors, 0])

if __name__ == '__main__':
    unittest.main()
