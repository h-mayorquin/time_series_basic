"""
Here will go the routines that are used to visualize the
clustering in time
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_time_cluster_matrix(nexa_object, cluster, time_center,
                                  cmap='coolwarm', inter='none',
                                  origin='upper', fontsize=16, aspect='auto',
                                  colorbar=True):
    """
    Documentation
    """

    Nsensors = nexa_object.sensors.Nsensors
    Nlags = nexa_object.Nlags

    cluster_to_index = nexa_object.cluster_to_index
    time_centers = nexa_object.cluster_to_time_centers

    values = time_centers[cluster][time_center]
    indexes = cluster_to_index[cluster]

    # Transform the values to plot
    to_plot = np.zeros((Nsensors, Nlags))

    for index, value in zip(indexes, values):
        sensor_index = index % Nsensors
        lag_index = int(index / Nsensors)
        to_plot[sensor_index, lag_index] = value

    # Now the parameters
    to_plot_title = 'Time cluster'

    cmap = cmap
    inter = inter
    origin = origin
    
    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Time lags'
    ylabel = 'Sensors'

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(axes_position)
    im = plt.imshow(to_plot, interpolation=inter, cmap=cmap,
                    origin=origin, aspect=aspect)

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
