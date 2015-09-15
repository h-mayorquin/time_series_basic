"""
A script just to play
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure
from nexa.nexa import Nexa

# Time parameters
dt = 0.1
Tmax = 100
Tperiod = 20.0
w = (2 * np.pi) / Tperiod
# w = 4 * (2.0 * np.pi / Tmax)

# Let's get the axuiliary class
t = np.arange(0, Tmax, dt)
sine = np.sin(w * t)
sine_phase = np.sin(w * t + np.pi)

# Plot the things here
if False:
    plt.plot(t, sine)
    plt.hold(True)
    plt.plot(t, sine_phase)
    plt.show()

lag_times = np.arange(0, 3 * Tperiod)  # Go two times the period
# lag_times = np.arange(0, 3)

lag_structure = LagStructure(lag_times=lag_times, window_size=2 * Tperiod)
sensor1 = Sensor(sine, dt, lag_structure)
sensor2 = Sensor(sine, dt, lag_structure)
sensors = [sensor1, sensor2]
perceptual_space = PerceptualSpace(sensors, 3)

Nspatial_clusters = 2  # Number of spatial clusters
Ntime_clusters = 2  # Number of time clusters
Nembedding = 3  # Dimension of the embedding space

# Now the Nexa object
nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                   Ntime_clusters, Nembedding)


SLM = perceptual_space.calculate_SLM()

if False:
    # Calculate all the quantities
    nexa_object.calculate_all()

    from visualization.sensor_clustering import visualize_cluster_matrix
    from visualization.sensors import visualize_SLM
    from visualization.sensors import visualize_STDM_seaborn
    from visualization.time_cluster import visualize_time_cluster_matrix
    from visualization.code_vectors import visualize_code_vectors
    fig = visualize_SLM(nexa_object)
    plt.show(fig)
