import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Input
from signals.aux_functions import gaussian_bump
from inputs.sensors import Sensor, PerceptualSpace
from inputs.lag_structure import LagStructure
# nexa
from nexa.nexa import Nexa
# Visualization libraries
from visualization.sensor_clustering import visualize_cluster_matrix
from visualization.sensors import visualize_SLM_axis
from visualization.sensors import visualize_STDM_seaborn

def set_gaussian_bumps(base, distance, value, time, max_rate):
    """
    A convenience function for setting the bumps
    """

    base = base
    value = value
    attenuation = 2

    center1 = 200
    distance = distance
    center2 = center1 + distance

    # Create the gaussian bumpbs
    gb1 = gaussian_bump(time, center1, max_rate, base, value, attenuation)
    gb2 = gaussian_bump(time, center2, max_rate, base, value, attenuation)

    # Add some noise
    gb1 += np.random.rand(gb1.size)
    gb2 += np.random.rand(gb2.size)

    return gb1, gb2

    
def set_perceptual_space(gb1, gb2, dt):
    """
    A convenience function to set the percpetual space
    """
    
    # lag_times = np.linspace(0, 800, 5) # Go two times the period
    lag_times = np.arange(0, 1000, 200)
    window_size = 200
    weights = None
    lag_structure = LagStructure(lag_times=lag_times, weights=weights, window_size=window_size)
    sensor1 = Sensor(gb1, dt, lag_structure)
    sensor2 = Sensor(gb2, dt, lag_structure)
    sensors = [sensor1, sensor2]
    perceptual_space = PerceptualSpace(sensors, lag_first=True)

    return perceptual_space


def parameter_swep_SLM(base, distance, value):
    """
    Sweps parameter looking for the SLM
    """
    Tmax = 1100
    dt = 1.0
    time = np.arange(0, Tmax, dt)

    # First we define the parameters 
    max_rate = 450

    Nspatial_clusters = 2  # Number of spatial clusters
    Ntime_clusters = 4  # Number of time clusters
    Nembedding = 2  # Dimension of the embedding space
    
    # Set the gaussian bumpbs
    gb1, gb2 = set_gaussian_bumps(base, distance, value, time, max_rate)
    # Get the perceptual_space
    perceptual_space = set_perceptual_space(gb1, gb2, dt)

    # Let's do the plotin here
    gs = mpl.gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[:, 1])


    # Now the nexa object
    nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                       Ntime_clusters, Nembedding)

    # Visualize object

    visualize_SLM_axis(nexa_object, ax=ax1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gb1)
    ax1.set_ylim((0, max_rate  + 20))
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, gb2)
    ax2.set_ylim((0, max_rate + 20))

    return fig

def parameter_swep_STDM(base, distance, value):
    """
    Sweps parameter looking for the STDM
    """
    Tmax = 1100
    dt = 1.0
    time = np.arange(0, Tmax, dt)

    # First we define the parameters 
    max_rate = 450

    Nspatial_clusters = 2  # Number of spatial clusters
    Ntime_clusters = 4  # Number of time clusters
    Nembedding = 2  # Dimension of the embedding space
    
    # Set the gaussian bumpbs
    gb1, gb2 = set_gaussian_bumps(base, distance, value, time, max_rate)
    # Get the perceptual_space
    perceptual_space = set_perceptual_space(gb1, gb2, dt)


    # Let's do the plotin here

    gs = mpl.gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[:, 1])


    # Now the nexa object
    nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                       Ntime_clusters, Nembedding)

    # Visualize object
    nexa_object.calculate_distance_matrix()
    visualize_STDM_seaborn(nexa_object, ax=ax1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gb1)
    ax1.set_ylim((0, max_rate  + 20))
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, gb2)
    ax2.set_ylim((0, max_rate + 20))

    return fig

def parameter_swep_cluster(base, distance, value):
    """
    Sweps parameter looking for the STDM
    """
    Tmax = 1100
    dt = 1.0
    time = np.arange(0, Tmax, dt)

    # First we define the parameters 
    max_rate = 450

    Nspatial_clusters = 2  # Number of spatial clusters
    Ntime_clusters = 4  # Number of time clusters
    Nembedding = 2  # Dimension of the embedding space
    
    # Set the gaussian bumpbs
    gb1, gb2 = set_gaussian_bumps(base, distance, value, time, max_rate)
    # Get the perceptual_space
    perceptual_space = set_perceptual_space(gb1, gb2, dt)


    # Let's do the plotin here

    gs = mpl.gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[:, 1])


    # Now the nexa object
    nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                       Ntime_clusters, Nembedding)

    nexa_object.calculate_distance_matrix()
    nexa_object.calculate_embedding
    nexa_object.calculate_spatial_clustering()

    # Visualize object
    visualize_cluster_matrix(nexa_object, ax=ax1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gb1)
    ax1.set_ylim((0, max_rate  + 20))
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, gb2)
    ax2.set_ylim((0, max_rate + 20))

    return fig

def parameter_swep_cluster_SLM(base, distance, value):
    """
    Sweps parameter looking for the STDM
    """
    Tmax = 1100
    dt = 1.0
    time = np.arange(0, Tmax, dt)

    # First we define the parameters 
    max_rate = 450

    Nspatial_clusters = 2  # Number of spatial clusters
    Ntime_clusters = 4  # Number of time clusters
    Nembedding = 2  # Dimension of the embedding space
    
    # Set the gaussian bumpbs
    gb1, gb2 = set_gaussian_bumps(base, distance, value, time, max_rate)
    # Get the perceptual_space
    perceptual_space = set_perceptual_space(gb1, gb2, dt)


    # Let's do the plotin here

    gs = mpl.gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(16, 12))



    # Now the nexa object
    nexa_object = Nexa(perceptual_space, Nspatial_clusters,
                       Ntime_clusters, Nembedding)

    nexa_object.calculate_distance_matrix()
    nexa_object.calculate_embedding
    nexa_object.calculate_spatial_clustering()

    # Visualize the cluster on the right side
    ax1 = fig.add_subplot(gs[:, 1])
    visualize_cluster_matrix(nexa_object, ax=ax1)
    # Visualize the SLE on the left side
    ax2 = fig.add_subplot(gs[:, 0])
    visualize_SLM_axis(nexa_object, ax=ax2)


    return fig



def create_filename(directory, name, figure_format, base, distance, value):
    """
    Creation of a unique filename
    """

    filename = directory + name
    filename += '-'
    filename += '{:.2f}'.format(base)
    filename += '-'
    filename += '{:.2f}'.format(distance)
    filename += '-'
    filename += '{:.2f}'.format(value)

    filename += figure_format


    return filename

