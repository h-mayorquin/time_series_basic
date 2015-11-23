"""
This file can be used to test the storage
"""

import numpy as np
import sys
# Somehow has to be run from ipython
sys.path.append('../')
print(sys.path)
from signals.aux_funipctions import gaussian_bump
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

# Create the gaussian bumpbs
gb1 = gaussian_bump(time, center1, max_rate, base, value, attenuation)
gb2 = gaussian_bump(time, center2, max_rate, base, value, attenuation)

# Add some noise
gb1 += np.random.rand(gb1.size)
gb2 += np.random.rand(gb2.size)


# Nexa
from nexa.nexa import Nexa

lag_times = np.linspace(0, 600, 4) # Go two times the period
window_size = distance
Nwindowsize = int(window_size / dt) 
weights = None
lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
sensor1 = Sensor(gb1, dt, lag_structure)
sensor2 = Sensor(gb2, dt, lag_structure)
sensors = [sensor1, sensor2]
perceptual_space = PerceptualSpace(sensors, lag_first=True)

Nspatial_clusters = 2  # Number of spatial clusters
Ntime_clusters = 4  # Number of time clusters
Nembedding = 2  # Dimension of the embedding space

# Now the Nexa object
# Now the Nexa object
nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                   Ntime_clusters, Nembedding)

nexa_object.calculate_all()
name = 'class_test'

from nexa.saving_nexa import NexaSaverHDF5

saver = NexaSaverHDF5(name, 'a')
saver.save_complete_run(nexa_object, 'type2')
