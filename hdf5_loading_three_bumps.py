import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from signals.aux_functions import gaussian_bump
import nexa.loading as load
from visualization.sensors import visualize_SLM_hdf5
from visualization.sensors import visualize_STDM_hdf5
from visualization.sensor_clustering import visualize_cluster_matrix_hdf5

# Load the database
location = './results_database/three_bumps_distance.hdf5'
database = h5py.File(location, 'r')

# Time 
Tmax = 1000
dt = 1.0
time = np.arange(0, Tmax, dt)

# Parameters that the bumpbs share
max_rate = 100
base = 10
value = 50
attenuation = 2

# Define three arangments for the values of the gaussian bumpbs
center1 = 100
center2 = 500
center3 = 700

# Now create the guassian bumps
gb1 = gaussian_bump(time, center1, max_rate, base, value, attenuation)
gb2 = gaussian_bump(time, center2, max_rate, base, value, attenuation)
gb3 = gaussian_bump(time, center3, max_rate, base, value, attenuation)

# Database extraction
run_name = str(center1) + '-'
run_name += str(center2) + '-'
run_name += str(center3)

nexa_arrangement = '3-4-3'
r = database[run_name]

# Load everything
SLM = load.get_SLM_hdf5(database, run_name)
STDM = load.get_STDM_hdf5(database, run_name, nexa_arrangement)
cluster_to_index = load.get_cluster_to_index_hdf5(database, run_name, nexa_arrangement)
index_to_cluster = load.get_index_to_cluster_hdf5(database, run_name, nexa_arrangement)
cluster_to_time_centers = load.get_cluster_to_time_centers_hdf5(database, run_name, nexa_arrangement)

# Now visualize the signals and the SLM
if False:
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gb1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time,gb2)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, gb3)

    ax4 = fig.add_subplot(gs[:, 1])
    visualize_SLM_hdf5(database, run_name, ax=ax4)

    plt.show()

# Now the signals and the STDM
if False:
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gb1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time,gb2)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, gb3)

    ax4 = fig.add_subplot(gs[:, 1])
    visualize_STDM_hdf5(database, run_name, nexa_arrangement, ax= ax4)

    plt.show()

    
# Now visualize the SLM and STDM
if False:
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    visualize_SLM_hdf5(database, run_name, ax=ax1)
    ax2 = fig.add_subplot(gs[:, 1])
    visualize_STDM_hdf5(database, run_name, nexa_arrangement, ax= ax2)
    fig.show()
    plt.close(fig)

# Now visualize the signals and the cluster matrix    
if True:
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gb1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, gb2)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, gb3)

    ax4 = fig.add_subplot(gs[:, 1])
    visualize_cluster_matrix_hdf5(database, run_name, nexa_arrangement, ax=ax4)

    plt.show()

