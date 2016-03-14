"""
This preoduces the data for the example of Nexa for wall street with columns
"""

import numpy as np

from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure
from nexa.nexa import Nexa
from nexa.saving import NexaSaverHDF5
import h5py

signal_location = './data/wall_street_data.hdf5'
signal_location = './data/wall_street_data_spaces.hdf5'
signal_location = './data/wall_street_data_30.hdf5'

# Access the data and load it into signal
with h5py.File(signal_location, 'r') as f:
    dset = f['signal']
    signals = np.empty(dset.shape, np.float)
    dset.read_direct(signals)


# Get the data and copy it
Ndata = signals.shape[0]
Nside = signals.shape[1]
Ndata = 15000
signals = signals[:Ndata, ...]
signals_columns = signals.swapaxes(1, 2).reshape(Ndata * Nside, Nside)
signals_columns += np.random.uniform(size=signals_columns.shape)
print('zeros', np.sum(signals_columns[0] == 0))
print('signals shape', signals_columns.shape)

# Now we need the nexa thing
dt = 1.0
max_lag = 10.0
lag_times = np.arange(0, max_lag, 1)
window_size = signals_columns.shape[0] - (lag_times[-1] + 1)
weights = None

lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
sensors = [Sensor(signal, dt, lag_structure) for signal in signals_columns.T]
perceptual_space = PerceptualSpace(sensors, lag_first=True)

Nside_aux = 30  # The side of the image
index_to_cluster = np.zeros(lag_times.size * Nside_aux)
for index in range(index_to_cluster.size):
    index_to_cluster[index] = index % max_lag

Ntime_clusters = 3
Nspatial_clusters = max_lag
Nembedding = 3

Ntime_clusters_set = np.arange(10, 55, 5)

for Ntime_clusters in Ntime_clusters_set:
    print('------------------')
    print('Ntime clusters', Ntime_clusters, Ntime_clusters_set.size)
    # Get the normal nexa object
    nexa_object = Nexa(perceptual_space, Nspatial_clusters, Ntime_clusters, Nembedding)

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
    data_base_name = 'text_wall_street_columns_30'
    saver = NexaSaverHDF5(data_base_name, 'a')
    # Save 
    run_name = 'test'
    saver.save_complete_run(nexa_object, run_name)
    print('Saved Mix')

    # Get the independent nexa object
    nexa_object = Nexa(perceptual_space, Nspatial_clusters, Ntime_clusters, Nembedding)

    nexa_object.calculate_distance_matrix()
    print('STDM shape', nexa_object.STDM.shape)
    print('Distance matrix calculated')
    nexa_object.index_to_cluster = index_to_cluster
    print('Spatial clustering calculated')
    nexa_object.calculate_cluster_to_indexes()
    print('Cluster to index calculated')
    nexa_object.calculate_time_clusters_indp()
    print('Time clusters calculated')

    # Open the saver 
    data_base_name = 'text_wall_street_columns_30'
    saver = NexaSaverHDF5(data_base_name, 'a')
    # Save 
    run_name = 'indep'
    saver.save_complete_run(nexa_object, run_name)
    print('Saved Independent')
