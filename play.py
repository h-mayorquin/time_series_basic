import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the proper path
import sys
sys.path.append("../")

# Local libraries
from signals.aux_functions import bump
from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure
import matplotlib.cm as cm

Tmax = 1000
dt = 0.1
time = np.arange(0, Tmax, dt)

# ### Parameters of the bumbs

# In[29]:

center1 = 300.0
center2 = 700.0
offset = 100.0
Max = 100.0

signal1 = bump(time, offset, center1, Max)
signal2 = bump(time, offset, center2, Max)

signal1 += signal1 + np.random.rand(signal1.size)
signal2 += signal2 + np.random.rand(signal2.size)

lag_times = np.linspace(0, 600, 4) # Go two times the period
window_size = 2 * offset
Nwindowsize = int(window_size / dt) 
weights = None
lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
sensor1 = Sensor(signal1, dt, lag_structure)
sensor2 = Sensor(signal2, dt, lag_structure)
sensors = [sensor1, sensor2]
perceptual_space = PerceptualSpace(sensors, lag_first=True)


from nexa.nexa import Nexa

Nspatial_clusters = 2  # Number of spatial clusters
Ntime_clusters = 4  # Number of time clusters
Nembedding = 2  # Dimension of the embedding space

# Now the Nexa object
nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                   Ntime_clusters, Nembedding)

# Make all the calculations
nexa_object.calculate_distance_matrix()
SLM = perceptual_space.SLM
STDM = nexa_object.STDM

nexa_object.calculate_embedding()

if False:
    for i in range(SLM.shape[0]):
        plt.plot(SLM[i], label=str(i))
        plt.hold(True)

    plt.legend()
    plt.show()

embed = nexa_object.embedding
rainbow = cm.rainbow

plt.subplot(1, 2, 1)
for i in range(embed.shape[0]):
    plt.plot(embed[i, 0], embed[i, 1], label=str(i), marker='o',
             markersize=20)
    plt.hold(True)

plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(STDM)
plt.colorbar()

plt.show()

# Now let' s get the clustering
nexa_object.calculate_spatial_clustering()
nexa_object.calculate_cluster_to_indexes()

index_to_cluster = nexa_object.index_to_cluster
cluster_to_index = nexa_object.cluster_to_index
