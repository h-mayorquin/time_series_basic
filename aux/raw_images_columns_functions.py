"""
In this file there are auxiliary functions that help with the pre-processing of information
for the raw data of the columns and images
"""

import numpy as np

def extract_column_data(Nletters, Nside, max_lag, signals, policy='inclusive'):
    """
    Extracts the data from an array of matrices. That is,
    Extract chunks of size max_lag until it covers 
    Nletters
    """
    if policy == 'inclusive':

        # Get the total number of data minus the lag at the end
        Ndata = Nletters * Nside - max_lag
        letter_size = Nside

        data = np.zeros((Ndata, Nside, max_lag))

        for data_index in range(Ndata):
            x = np.zeros((Nside, max_lag))

            # Extract those five columns
            for lag_index in range(max_lag):
                image_index = (data_index + lag_index) // letter_size
                column_index = (data_index + lag_index) % letter_size
                x[:, lag_index] = signals[image_index, :, column_index]

            # Save it to data
            data[data_index, ...] = x

    if policy == 'exclusive':

        # Total number of data minus a lag for each letter (Nletters * max_lag)
        Ndata = Nletters * (Nside - max_lag + 1)
        letter_size = Nside - max_lag + 1

        data = np.zeros((Ndata, Nside, max_lag))

        for data_index in range(Ndata):
            x = np.zeros((Nside, max_lag))

            # Extract those five columns
            image_index = (data_index) // letter_size
            for lag_index in range(max_lag):
                column_index = (data_index % letter_size) + lag_index
                x[:, lag_index] = signals[image_index, :, column_index]

            # Save it to data
            data[data_index, ...] = x

    return data


def extract_letters_to_columns(Nletters, Nside, max_lag, letters_sequence, policy='inclusive', shift=0):
    """
    Exctact a data set of letters to use in the learning task for the
    columns of text
    """
    if policy == 'inclusive':

        # Get the total number of data minus the lag at the end
        Ndata = Nletters * Nside - max_lag
        letter_size = Nside

    if policy == 'exclusive':

        # Total number of data minus a lag for each letter (Nletters * max_lag)
        Ndata = Nletters * (Nside - max_lag + 1)
        letter_size = Nside - max_lag + 1


    letters_to_columns = np.zeros(Ndata, dtype='<U1')
    for index in range(Ndata):
        letter_index = index // letter_size
        letters_to_columns[index] = letters_sequence[letter_index + shift]

    return letters_to_columns

