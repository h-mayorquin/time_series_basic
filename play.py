import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the proper path
import sys
sys.path.append("../")

# Local libraries
from signals.aux_functions import gaussian_bump, combine_gaussian_bumps
from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure

Tmax = 1000
dt = 1.0
time = np.arange(0, Tmax, dt)

# First we define the parameters 
max_rate = 100
base = 10
value = 30
attenuation = 2

center1 = 250
distance = 300
center2 = center1 + distance


# **Now we define the gaussians bumps and combine them**

# In[4]:

# Create the gaussian bumpbs
gb1 = gaussian_bump(time, center1, max_rate, base, value, attenuation)
gb2 = gaussian_bump(time, center2, max_rate, base, value, attenuation)

# Add some noise
gb1 += np.random.rand(gb1.size)
gb2 += np.random.rand(gb2.size)


# #### Now we plot them
plt.plot(time, gb1)
plt.plot(time, gb2)
plt.ylim([0, max_rate + 20])


# ### Nexa Machinery
# Input structure
from signals.aux_functions import bump
from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure

# Nexa
from nexa.nexa import Nexa

lag_times = np.linspace(0, 600, 4) # Go two times the period
window_size = distance
Nwindowsize = int(window_size / dt) 
weights = None

lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
sensor1 = Sensor(gb1, dt, lag_structure)
sensor2 = Sensor(gb2, dt, lag_structure)
sensor3 = Sensor(gb2, dt, lag_structure)

sensors = [sensor1, sensor2, sensor3]
perceptual_space = PerceptualSpace(sensors, lag_first=True)


# #### Now we start Nexa
Nspatial_clusters = 2 # Number of spatial clusters
Ntime_clusters = 2  # Number of time clusters
Nembedding = 3  # Dimension of the embedding space

# Now the Nexa object
nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                   Ntime_clusters, Nembedding)


nexa_object.calculate_all()
code_vectors = nexa_object.build_code_vectors()
code_vectors_distance = nexa_object.build_code_vectors_distance()
code_vectors_softmax = nexa_object.build_code_vectors_softmax()
code_vectors_winner = nexa_object.build_code_vectors_winner()


