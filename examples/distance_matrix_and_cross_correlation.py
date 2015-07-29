"""
A script used to display the cross-correlation matrix
and the the whole time seriesxX
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from signals.time_series_class import MixAr, AR
from signals.aux_functions import sidekick
from visualization import distance
import os

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

#########
# Here we will calculate correlations
#########

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

##############
#  Now plot the things
##############

#  fig = distance.linear(d, cmap='coolwarm', inter='none', origin='upper',
#                  fontsize=16, aspect='auto')

fig = distance.matrix(d, cmap='coolwarm', inter='none', origin='upper',
                      fontsize=16, aspect='auto')


# Save the figure

name = 'cross_correlation_transformed'

# Save the figure here
directory = './results/'
extension = '.pdf'
filename = directory + name + extension

plt.savefig(filename)
os.system("pdfcrop %s %s" % (filename, filename))

plt.show(fig)

if plot3:
    plt.subplot(3, 2, 1)
    plt.plot(x_auto)
    plt.ylim([-1, 1])
    plt.title('Autocorrelation of mix_series')

    plt.subplot(3, 2, 2)
    plt.plot(y_auto)
    plt.ylim([-1, 1])
    plt.title('Autocorrelation of sidekick')

    plt.subplot(3, 2, 3)
    plt.plot(xy_cross)
    plt.ylim([-1, 1])
    plt.title('Cross correlation')

    plt.subplot(3, 2, 4)
    plt.plot(time, beta)
    plt.title('Sidekick')

    plt.subplot(3, 2, 5)
    plt.plot(time, normal_series)
    plt.title('Normal series')

    plt.subplot(3, 2, 6)
    plt.plot(time, mix_series)
    plt.title('Mix series')

    plt.show()
