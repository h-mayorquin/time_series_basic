"""
Some auxiliar functions
"""

import numpy as np


def linear_to_matrix(indexes, Nsensors, Nlags):
    """
    Transforms some linear indexes onto the index
    for a matrix
    """
    matrix_indexes = np.zeros((indexes.size, 2))
    for i, index in enumerate(indexes):
        matrix_indexes[i, 0] = index % Nsensors
        matrix_indexes[i, 1] = int(index / Nsensors)

    return matrix_indexes


def linear_to_matrix_with_values(linear_values, Nsensors, Nlags):
    """
    Transform a linear array on values dim = (Nsensors * Nlags)
    onto the values with matrix representation dim = (Nsensors, Nlags)
    """

    matrix_indexes = linear_to_matrix(np.arange(Nlags * Nsensors),
                                      Nsensors, Nlags)

    matrix = np.zeros((Nsensors, Nlags))

    for index, value in zip(matrix_indexes, linear_values):
        sensor_index = index[0]
        lag_index = index[1]
        matrix[sensor_index, lag_index] = value

    return matrix