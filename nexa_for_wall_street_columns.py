"""
This preoduces the data for the example of Nexa for wall street columns
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
Nside = signals.shape[1]
Ndata = 50000
signals = signals[:Ndata, ...]
signals_columns = signals.swapaxes(1, 2).reshape(Ndata * Nside, Nside)
signals_columns += np.random.uniform(size=signals_columns.shape)
print('zeros', np.sum(signals_columns[0] == 0))
print('signals shape', signals_columns.shape)

# Now we need the nexa thing
dt = 1.0
lag_times = np.arange(0, 3, 1)
window_size = signals_columns.shape[0] - (lag_times[-1] + 1)
weights = None

lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
sensors = [Sensor(signal, dt, lag_structure) for signal in signals_columns.T]
perceptual_space = PerceptualSpace(sensors, lag_first=True)

Ntime_clusters = 3
Nspatial_clusters = 3
Nembedding = 3

nexa_object = Nexa(perceptual_space, Nspatial_clusters, Ntime_clusters, Nembedding)

# Calculate
# nexa_object.calculate_all()
# Now we calculate the distance matrix
nexa_object.calculate_distance_matrix()
print('STDM shape', nexa_object.STDM.shape)
print('Distance matrix calculated')
nexa_object.calculate_embedding()
print('Embedding calculated')
nexa_object.calculate_spatial_clustering()
print('Spatial clustering calculated')
nexa_object.calculate_cluster_to_indexes()
print('Cluster to index calculated')
nexa_object.calculate_time_clusters()
print('Time clusters calculated')

# Open the saver 
data_base_name = 'text_wall_street_columns'
saver = NexaSaverHDF5(data_base_name, 'a')
# Save 
run_name = 'test'
saver.save_complete_run(nexa_object, run_name)
print('Saved')

