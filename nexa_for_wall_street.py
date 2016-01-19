"""
This loads the data of the a text from an hdf5 file. Then
ir runs nexa and stores the results as a tradiditional nexa saver

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


# Reshape the data and limit it
Ndata = 100
signals = signals.reshape(signals.shape[0], signals.shape[1] * signals.shape[2])
# signals = signals[:Ndata, ...].astype('float')
signals += np.random.uniform(size=signals.shape) * 0.1
print('zeros', np.sum(signals[0] == 0))
print('signals shape', signals.shape)

dt = 1.0
lag_times = np.arange(0, 5, 1)
# lag_times = np.arange(0, 3, 1)  # For testing purposes
window_size = signals.shape[0] - (lag_times[-1] + 1)
weights = None

lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
sensors = [Sensor(signal, dt, lag_structure) for signal in signals.T]
perceptual_space = PerceptualSpace(sensors, lag_first=True)

# Get the nexa machinery right
Nspatial_clusters = 5
Ntime_clusters = 45
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
data_base_name = 'text_wall_street'
saver = NexaSaverHDF5(data_base_name, 'a')
# Save 
run_name = 'low-resolution'
saver.save_complete_run(nexa_object, run_name)
print('Saved')
