"""
Here I will put the functions that help the multidimensional scaling
"""

import numpy as np


def calculate_temporal_distance(cross_correlation):
    """
    Once you have calculated the cross-correlation you can
    use this function in order to calculate the spatio
    temporal matrix that can be use with MDS.
    """
    Nseries = cross_correlation.shape[1]
    nlags = cross_correlation.shape[0] - 1
    A = np.zeros((Nseries * nlags, Nseries * nlags))

    for p in range(nlags):
        for l in range(nlags):
            for i in range(Nseries):
                for j in range(Nseries):
                    x_index = Nseries * p + i
                    y_index = Nseries * l + j
                    A[x_index, y_index] = cross_correlation[abs(p - l), i, j]

    return A
