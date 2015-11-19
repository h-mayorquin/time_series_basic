import numpy as np
import matplotlib.pyplot as plt
import IPython

from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure
from nexa.nexa import Nexa

# Build signal
signal1 = np.arange(1, 11, 1.0)
signal2 = np.arange(-1, -11, -1.0)
# Add noise
signal1 += np.random.uniform(size=signal1.shape) * 0.01
signal2 += np.random.uniform(size=signal1.shape) * 0.01
# Pack them
signals = [signal1, signal2]
    
# PerceptualSpace
dt = 1.0
lag_times = np.arange(0, 4, 1)
window_size = signal1.size - lag_times[-1]
weights = None

lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
# sensors = [Sensor(signal1, dt, lag_structure,), Sensor(signal2, dt, lag_structure)]
sensors = [Sensor(signal, dt, lag_structure) for signal in signals]
perceptual_space = PerceptualSpace(sensors, lag_first=True)

# Get the nexa machinery right
Nspatial_clusters = 2
Ntime_clusters = 3
Nembedding = 2

x = perceptual_space.map_SLM_columns_to_time()
SLM = perceptual_space.calculate_SLM()

if True:
    nexa_object = Nexa(perceptual_space, Nspatial_clusters, Ntime_clusters, Nembedding)
    # Calculate distance matrix
    nexa_object.calculate_distance_matrix()
    # Calculate embedding
    nexa_object.calculate_embedding()
    # Calculate spatial clustering
    nexa_object.calculate_spatial_clustering()
    # Calculate cluster to index
    nexa_object.calculate_cluster_to_indexes()
    # Calculate time clusters
    nexa_object.calculate_time_clusters()
    # Ok, so let's get the signal index from the cluster to index (SLE index)
    code_vectors = nexa_object.build_code_vectors_pairs()
    print(signals)
    print('-------------')
    print(nexa_object.SLM)
    print(code_vectors)
