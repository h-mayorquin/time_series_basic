"""
This script calculates the stress for a multidimensional scaling calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from signals.time_series_class import MixAr, AR
from signals.aux_functions import sidekick
from nexa.multidimensional_scaling import calculate_temporal_distance
from sklearn import manifold

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

# First we construct the series without the sidekick
B = AR(phi, dt=dt, Tmax=Tmax)
B.initial_conditions(initial_conditions)
normal_series = B.construct_series()

# Second we construct the series with the mix
A = MixAr(phi, dt=dt, Tmax=Tmax, beta=beta)
A.initial_conditions(initial_conditions)
mix_series = A.construct_series()

time = A.time

# Here we will calculate correlations
nlags = 100
unbiased = False

x_auto = sm.tsa.acf(mix_series, unbiased=unbiased, nlags=nlags)
y_auto = sm.tsa.acf(beta, unbiased=unbiased, nlags=nlags)
xy_cross = sm.tsa.ccf(mix_series, beta, unbiased=unbiased)[0:nlags + 1]

# Now the distance matrix
d = np.zeros((nlags + 1, 2, 2))
d[:, 0, 0] = x_auto
d[:, 1, 1] = y_auto
d[:, 1, 0] = xy_cross

d[:, 0, 1] = d[:, 1, 0]

####
# Let's transform to distance representation#
###

A = calculate_temporal_distance(d)

####
# Let's do MDS and plot the stress
####

Nmax = 100  # Maximum dimensions
dimensions = np.arange(2, Nmax)
stress = np.zeros_like(dimensions)

for index, i in enumerate(dimensions):
    print 'dimensions', i

    n_comp = i
    n_init = 10
    n_jobs = -1  # -1 To use all CPUs, 1 for only one
    disimi = 'precomputed'

    classifier = manifold.MDS(n_components=n_comp, n_init=n_init,
                              n_jobs=n_jobs, dissimilarity=disimi)
    embedd = classifier.fit_transform(A)
    stress[index] = classifier.stress_


plt.plot(dimensions, stress)
plt.xlabel('Dimensions')
plt.ylabel('Stress')
plt.show()
