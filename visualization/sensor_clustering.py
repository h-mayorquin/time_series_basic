"""
Here will go the routines that are used to visualize the
clustering of the sensors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .aux import linear_to_matrix, linear_to_matrix_with_values
from .aux import linear_to_matrix_pairs
import nexa.loading as load


def visualize_cluster_matrix(nexa_object, cmap='coolwarm', inter='none',
                             origin='upper', fontsize=16, aspect='auto',
                             colorbar=False, ax=None):
    """
    Documentation
    """

    Nlags = nexa_object.sensors.nlags
    Nsensors = nexa_object.sensors.Nsensors
    values = nexa_object.index_to_cluster

    lags_first = nexa_object.lags_first

    to_plot = linear_to_matrix_with_values(values, Nsensors,
                                           Nlags, lags_first)

    # First the parameters
    to_plot_title = 'Clustering asigned to sensors'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Lags'
    ylabel = 'Sensor'

    # If the axis is none it creates the figure, the axis and the image.
    # Otherwise just add the figure to the axis and get the figure from it

    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes(axes_position)
        im = ax.imshow(to_plot, interpolation=inter, cmap=cmap,
                       origin=origin, aspect=aspect)

    else:
        im = ax.imshow(to_plot, interpolation=inter, cmap=cmap,
                       origin=origin, aspect=aspect)
        fig = im.get_figure()

    # Se the labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(to_plot_title)

    # Change the font sizes
    axes = fig.get_axes()
    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    # Colorbar (This makes the axes to display proper)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.solids.set_edgecolor('face')

    if ax is None:
        return fig
    else:
        return ax

    
def visualize_cluster_matrix_hdf5(database, run_name, nexa_arrangement,
                                  cmap='coolwarm', inter='none',
                                  origin='upper', fontsize=16, aspect='auto',
                                  colorbar=False, ax=None):
    """
    This plots the clustering matrix extracting it from hdf5
    """
    attrs = database[run_name].attrs
    Nlags = attrs['Nlags']
    Nsensors = attrs['Nsensors']
    values = load.get_index_to_cluster_hdf5(database, run_name, nexa_arrangement)
    lags_first = attrs['lags_first']

    to_plot = linear_to_matrix_with_values(values, Nsensors,
                                           Nlags, lags_first)

    # First the parameters
    to_plot_title = 'Clustering asigned to sensors'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Lags'
    ylabel = 'Sensor'

    # If the axis is none it creates the figure, the axis and the image.
    # Otherwise just add the figure to the axis and get the figure from it

    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes(axes_position)
        im = ax.imshow(to_plot, interpolation=inter, cmap=cmap,
                       origin=origin, aspect=aspect)

    else:
        im = ax.imshow(to_plot, interpolation=inter, cmap=cmap,
                       origin=origin, aspect=aspect)
        fig = im.get_figure()

    # Se the labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(to_plot_title)

    # Change the font sizes
    axes = fig.get_axes()
    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    # Colorbar (This makes the axes to display proper)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.solids.set_edgecolor('face')

    if ax is None:
        return fig
    else:
        return ax


def visualize_clusters_text_to_image(nexa, f, run_name):
    """
    Takes the nexa file in hdf5 format and plots the
    cluster with a matrix for each lag in the tex to
    image format.
    """

    interpolation = 'none'
    origin = 'lower'
    cmap = 'jet'
    cmap = 'brg'
    cmap = 'gnuplot'

    cluster_to_index = nexa['cluster_to_index']
    lags = f[run_name + '/lags']

    Nlags = f[run_name].attrs['Nlags']
    Nsensors = f[run_name].attrs['Nsensors']
    Nside = int(np.sqrt(Nsensors))

    matrix = np.zeros((Nside, Nside, Nlags))

    Nspatial_clusters = nexa.attrs['Nspatial_clusters']
    
    for cluster in range(Nspatial_clusters):
        cluster_indexes = np.array(cluster_to_index[str(cluster)])
        for index in cluster_indexes:
            sensor_number = index // Nlags  # This needs to be mapped to matrix
            sensor_number_x = sensor_number // Nside
            sensor_number_y = sensor_number % Nside
            lag_number = index % Nlags
            matrix[sensor_number_x, sensor_number_y, lag_number] = cluster

    # Now plot it
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2)

    for (row_index, column_index), lag in  zip(linear_to_matrix_pairs(lags), lags):
        ax = fig.add_subplot(gs[row_index, column_index])
        ax.imshow(matrix[..., lag], cmap=cmap, interpolation=interpolation, origin=origin)
        ax.set_title('Lag ' + str(lag))

    return fig
