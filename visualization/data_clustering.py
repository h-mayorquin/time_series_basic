"""
Here will go the routines that are used to visualize the
clustering in time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .aux import linear_to_matrix_pairs



def visualize_time_cluster_matrix(nexa_object, cluster, time_center,
                                  cmap='coolwarm', inter='none',
                                  origin='upper', fontsize=16, aspect='auto',
                                  colorbar=True):
    """
    Documentation

    time center: is the time center that we want to plot
    """

    Nsensors = nexa_object.sensors.Nsensors
    Nlags = nexa_object.sensors.nlags

    cluster_to_index = nexa_object.cluster_to_index
    # This contains all the time centers for a given cluster
    time_centers = nexa_object.cluster_to_time_centers

    # It is not obvious that the positional order fo these two should coincide
    values = time_centers[cluster][time_center]
    indexes = cluster_to_index[cluster]

    # Transform the values to plot
    to_plot = np.zeros((Nsensors, Nlags))

    # Transform the values to plot
    to_plot = np.zeros((Nsensors, Nlags))

    if nexa_object.lags_first:
        for index, value in zip(indexes, values):
            # Transform the indexes to matrix representation
            sensor_index = int(index / Nlags)
            lag_index = index % Nlags
            to_plot[sensor_index, lag_index] = value

    else:
        for index, value in zip(indexes, values):
            # Transform the indexes to matrix representation
            sensor_index = index % Nsensors
            lag_index = int(index / Nsensors)
            to_plot[sensor_index, lag_index] = value

    # Calculate min and max
    aux1 = np.min(to_plot)
    aux2 = np.max(to_plot)

    vmax = np.max(np.abs(aux1), np.abs(aux2))
    vmin = -vmax

    # Now the parameters
    to_plot_title = 'Time cluster'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Time lags'
    ylabel = 'Sensor'

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(axes_position)
    im = plt.imshow(to_plot, interpolation=inter, cmap=cmap,
                    origin=origin, aspect=aspect, vmin=vmin, vmax=vmax)

    # Se the labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(to_plot_title)

    # Se the ticks names for x
    # x_labels = np.arange(Nseries * Nseries + 1)
    # ax.xaxis.set_major_formatter(plt.FixedFormatter(x_labels))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))

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


def visualize_data_cluster_text_to_image(nexa, f, run_name,
                                         cluster, data_center, colorbar=True):
    """
    Returns a figure of of the time center for a particular time center
    """
    # Get the indexes
    cluster_to_index = nexa['cluster_to_index']
    cluster_to_data_centers = nexa['cluster_to_time_centers']
    
    cluster_indexes = cluster_to_index[str(cluster)]
    data_centers = cluster_to_data_centers[str(cluster)]

    # Get the size parameters
    Nsensors = f[run_name].attrs['Nsensors']
    Nlags = f[run_name].attrs['Nlags']
    Nside = int(np.sqrt(Nsensors))
    lags = f[run_name + '/lags']

    # Matrix to save and fill
    matrix = np.zeros((Nside, Nside, Nlags))

    for i, index in enumerate(cluster_indexes):
        sensor_number = index // Nlags
        sensor_number_x = sensor_number // Nside
        sensor_number_y = sensor_number % Nside
        lag_number = index % Nlags
        matrix[sensor_number_x, sensor_number_y, lag_number] = data_centers[data_center, i]

    # Extract minimum and maximum for the color limits
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    # Plot it
    interpolation = 'none'
    origin = 'lower'
    cmap = 'inferno_r'

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2)

    for (row_index, column_index), lag in  zip(linear_to_matrix_pairs(lags), lags):
        ax = fig.add_subplot(gs[row_index, column_index])
        im = ax.imshow(matrix[..., lag], cmap=cmap, interpolation=interpolation,
                       origin=origin, vmin=min_value, vmax=max_value)
        ax.set_title('Lag ' + str(lag) + ' || data center=' + str(data_center))

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.solids.set_edgecolor('face')

    return fig
