import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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


# Load low resolution signal
signal_location_low = './data/wall_street_data_spaces.hdf5'
with h5py.File(signal_location_low, 'r') as f:
    dset = f['signal']
    signals_low = np.empty(dset.shape, np.float)
    dset.read_direct(signals_low)

# Load high resolution signal
signal_location_high = './data/wall_street_data_30.hdf5'
with h5py.File(signal_location_high, 'r') as f:
    dset = f['signal']
    signals_high = np.empty(dset.shape, np.float)
    dset.read_direct(signals_high)

# Load the letters
# Now we need to get the letters and align them
text_directory = './data/wall_street_letters_spaces.npy'
letters_sequence = np.load(text_directory)
Nletters = len(letters_sequence)
symbols = set(letters_sequence)

# Cover all policy
Nletters = 1000
shift = 0

# Extract the desired quantity
max_lag_low = 5
print(signals_low.shape)
Nside_low = signals_low.shape[1]

policy = 'exclusive'

# Transform the images into a contiguous representation
print('Loading data for low dimension')
data_low = extract_column_data(Nletters, Nside_low, max_lag_low, signals_low, policy=policy)

# Extract the letters
letters_low = extract_letters_to_columns(Nletters, Nside_low, max_lag_low, letters_sequence, policy=policy, shift=shift)

# Extract the desired quantity
max_lag_high = 15
Nside_high = signals_high.shape[1]

# Transform the images into a contiguous representation
print('Loading data for high dimension')
data_high = extract_column_data(Nletters, Nside_high, max_lag_high, signals_high, policy=policy)

# Extract the letters
letters_high = extract_letters_to_columns(Nletters, Nside_high, max_lag_high, letters_sequence, policy=policy, shift=shift)

# Now let's do classification for different number of data
print('Policy', policy)
print('Ndata for the low resolution', letters_low.size)
print('Ndata for the high resolution', letters_high.size)

Ndata_class = 1000
# First we get the classification for low resolution
X = data_low[:Ndata_class, ...].reshape(Ndata_class, Nside_low * max_lag_low)
y = letters_low[:Ndata_class, ...]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)
clf = LDA()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test) * 100
print('Score for low resolution', score)

# Now we get the classification for high resolution
X = data_high[:Ndata_class, ...].reshape(Ndata_class, Nside_high * max_lag_high)
y = letters_high[:Ndata_class, ...]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)
clf = LDA()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test) * 100
print('Score for high resolution', score)

