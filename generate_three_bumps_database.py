"""
This file should generate three bumpbs with varying distance
between them. 
"""
import numpy as np

# Local libraries for bump generation
from signals.aux_functions import gaussian_bump
from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure

# nexa
from nexa.nexa import Nexa
from nexa.saving import NexaSaverHDF5

# Time 
Tmax = 1000
dt = 1.0
time = np.arange(0, Tmax, dt)

# Parameters that the bumpbs share
max_rate = 100
base = 10
value = 50
attenuation = 2

# Perceptual Space Parameters
lag_times = np.linspace(0, 800, 5)
window_size = 200
Nwindowsize = int(window_size / dt)
weights = None
lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)

# nexa Parameters
Nspatial_clusters = 3
Ntime_clusters = 4
Nembedding = 3

# Saver parameters
centers1 = np.arange(100, 600, 100)
centers3 = np.arange(500, 1000, 100)

for center1 in centers1:
    for center3 in centers3:
        # Define three arangments for the values of the gaussian bumpbs
        center1 = center1
        center2 = 500
        center3 = center3

        # Now create the guassian bumps
        gb1 = gaussian_bump(time, center1, max_rate, base, value, attenuation)
        gb2 = gaussian_bump(time, center2, max_rate, base, value, attenuation)
        gb3 = gaussian_bump(time, center3, max_rate, base, value, attenuation)

        # Add some noise
        gb1 += np.random.rand(gb1.size)
        gb2 += np.random.rand(gb2.size)
        gb3 += np.random.rand(gb3.size)

        # Creat the perceptual space
        sensor1 = Sensor(gb1, dt, lag_structure)
        sensor2 = Sensor(gb2, dt, lag_structure)
        sensor3 = Sensor(gb3, dt, lag_structure)

        sensors = [sensor1, sensor2, sensor3]
        perceptual_space = PerceptualSpace(sensors, lag_first=True)

        # Create the nexa object
        nexa_object = Nexa(perceptual_space, Nspatial_clusters, Ntime_clusters, Nembedding)

        # Calculate all and save
        nexa_object.calculate_all()

        # Save everything
        run_name = str(center1) + '-'
        run_name += str(center2) + '-'
        run_name += str(center3)
        print('run name', run_name)

        name = 'three_bumps_distance'
        saver = NexaSaverHDF5(name, 'a')
        saver.save_complete_run(nexa_object, run_name)
