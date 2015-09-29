"""
Here will go the routines that are used to visualize the
clustering of the sensors
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .aux import linear_to_matrix, linear_to_matrix_with_values


def visualize_cluster_matrix(nexa_object, cmap='coolwarm', inter='none',
                             origin='upper', fontsize=16, aspect='auto',
                             colorbar=False):
    """
    Documentation
    """

    Nlags = nexa_object.sensors.nlags
    Nsensors = nexa_object.sensors.Nsensors
    values = nexa_object.index_to_cluster

    to_plot = linear_to_matrix_with_values(values, Nsensors,
                                           Nlags, nexa_object.lags_first)

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

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(axes_position)
    im = plt.imshow(to_plot, interpolation=inter, cmap=cmap,
                    origin=origin, aspect=aspect)

    # Se the labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(to_plot_title)

    # Se the ticks names for x as the lags
    x_labels = nexa_object.sensors.lags
    # ax.xaxis.set_major_formatter(plt.FixedFormatter(x_labels))
    # Set the axis every then lags
    # ax.xaxis.set_major_locator(plt.MultipleLocator(int(x_labels.size / 5)))

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

    return fig
