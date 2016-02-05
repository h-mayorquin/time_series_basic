"""
Just to play
"""

import numpy as np

from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure
from nexa.nexa import Nexa
from nexa.saving import NexaSaverHDF5
import h5py


signal_location = './data/wall_street_data.hdf5'

# Access the data and load it into signal
with h5py.File(signal_location, 'r') as f:
    dset = f['signal']
    signals = np.empty(dset.shape, np.float)
    dset.read_direct(signals)


# Get the data and copy it
Ndata = signals.shape[0]

signals_original = signals[:Ndata, ...]
signals_transposed = np.copy(signals_original)
signals_third_copy = np.copy(signals_original)

# Get the dimensions
Nside = signals_original.shape[1]

# Transpose the matrix
if True:
    for index, signal in enumerate(signals_original):
        signals_transposed[index, ...]= signal.T

# Let's get the first entry from the original matrix
first_original_entry = signals_original[0, ...]

# Reshape the both signals
reshaped_original = np.copy(signals_original).reshape(Ndata * Nside, Nside)
reshaped_transposed = np.copy(signals_transposed).reshape(Ndata * Nside, Nside)
signals_third_copy = signals_third_copy.swapaxes(1, 2).reshape(Ndata * Nside, Nside)

# Print to make comparisons
if True:
    print('first original entry \n', first_original_entry)
    print('original', reshaped_original[3])
    print('transpoed', reshaped_transposed[3])
    print('third copy', signals_third_copy[3])
