"""
A script just to play
"""

import numpy as np
import matplotlib.pyplot as plt
from signals.time_series_class import MixAr
from signals.aux_functions import sidekick
from input.sensors import PerceptualSpace, Sensor
from nexa.nexa import Nexa


plot = False
plot2 = False
plot3 = False

# Time parameters
dt = 0.1
Tmax = 100

# Let's get the axuiliary class
amplitude = 1
w1 = 1
w2 = 5
beta = sidekick(w1, w2, dt, Tmax, amplitude)

# First we need the phi's vector
phi0 = 0.0
phi1 = -0.8
phi2 = 0.3

phi = np.array((phi0, phi1, phi2))

# Now we need the initial conditions
x0 = 1
x1 = 1
x2 = 0

initial_conditions = np.array((x0, x1, x2))

# Second we construct the series with the mix
A = MixAr(phi, dt=dt, Tmax=Tmax, beta=beta)
A.initial_conditions(initial_conditions)
mix_series = A.construct_series()

time = A.time

# Here we will calculate correlations
Nlags = 100
unbiased = False
Nspatial_clusters = 2
Ntime_clusters = 2
Nembedding = 3  # Dimension of the embedding space

# We create the here perceptual space
aux = [Sensor(mix_series, dt), Sensor(beta, dt)]
perceptual_space = PerceptualSpace(aux, Nlags)

# Now the nexa object
nexa_object = Nexa(perceptual_space, Nlags, Nspatial_clusters,
                   Ntime_clusters)

# Now calculate the STDM
nexa_object.calculate_distance_matrix()

# Calculate the embedding
nexa_object.calculate_embedding(Nembedding)

# Calculate the clustering
nexa_object.calculate_spatial_clustering()

# Calculate cluster to index
nexa_object.calculate_cluster_to_indexes()

# Calculate time clusters
nexa_object.calculate_time_clusters()

# Visualize the spatial clusters
