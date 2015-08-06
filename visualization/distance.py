"""
Here I will write the functions for visualizing the matrices
needed by nexa. That is, the distance matrix after flattening
in its two representations (matrix and linear) and the functions
to convert the clusters in the reduced space to the original
space. Furthermore routines for clustering will also be added.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def linear(d, cmap='coolwarm', inter='none', origin='upper',
           fontsize=16, aspect='auto'):
    """
    Plots the flattened matrix in a linear representations that is the
    3 x 3 matrix in time look like:

    time 0 : 11, 12, 13, 21, 22, 23, 31, 32, 33
    time 1 : 11, 12, 13, 21, 22, 23, 31, 32, 33
    .
    .
    .
    time klags : 11, 12, 13, 21, 22, 23, 31, 32, 33

    Parameters:
    distance: a matrix of distance with dimensions (klags, Nseries, Nseries)
    where Nseries is the number of time series and klags is the number
    of lags in the cross correlation matrix.

    cmap: the imshow colormap

    inter: the imshow interpolation

    origin: the imshow origin

    fontisze: the fontsize of the axis, ticks, etc

    aspect: imshow apsect ratio

    Returns:
    fig: A matplotlib figure object
    """

    klags = d.shape[0]
    Nseries = d.shape[1]

    to_plot = d.reshape((Nseries * klags, Nseries))
    to_plot = d.reshape((klags, Nseries * Nseries))
    # to_plot = d.reshape((nlags + 1, 4))

    # First the parameters
    to_plot_title = 'Cross Correlation Matrix [Linear Representation]'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Correlation #'
    ylabel = 'Time Lags'

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(axes_position)
    im = plt.imshow(to_plot, interpolation=inter, cmap=cmap,
                    origin=origin, aspect=aspect)

    # Se the labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(to_plot_title)

    # Se the ticks names for x
    x_labels = np.arange(Nseries * Nseries + 1)
    ax.xaxis.set_major_formatter(plt.FixedFormatter(x_labels))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    # Change the font sizes
    axes = fig.get_axes()
    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    # Colorbar (This makes the axes to display proper)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.solids.set_edgecolor('face')

    return fig


def matrix(d, cmap='coolwarm', inter='none', origin='upper',
           fontsize=16, aspect='auto'):
    """
    Plots the flattened matrix in a linear representations that is the
    3 x 3 matrix in time look like:

    time 0 : 11, 12, 13
             21, 22, 23
             31, 32, 33
    
    time 1 : 11, 12, 13
             21, 22, 23
             31, 32, 33
    .
    .
    .
    time klags : 11, 12, 13
                 21, 22, 23
                 31, 32, 33

    Parameters:
    distance: a matrix of distance with dimensions (klags, Nseries, Nseries)
    where Nseries is the number of time series and klags is the number
    of lags in the cross correlation matrix.

    cmap: the imshow colormap

    inter: the imshow interpolation

    origin: the imshow origin

    fontisze: the fontsize of the axis, ticks, etc

    aspect: imshow apsect ratio

    Returns:
    fig: A matplotlib figure object
    """

    klags = d.shape[0]
    Nseries = d.shape[1]

    to_plot = d.reshape((Nseries * klags, Nseries))

    # First the parameters
    to_plot_title = 'Cross Correlation Matrix [Matrix Representation]'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Time series #'
    ylabel = 'Time Series * Time Lag '

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(axes_position)
    im = plt.imshow(to_plot, interpolation=inter, cmap=cmap,
                    origin=origin, aspect=aspect)

    # Se the labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(to_plot_title)

    # Se the ticks names for x
    x_labels = np.arange(Nseries * Nseries + 1)
    ax.xaxis.set_major_formatter(plt.FixedFormatter(x_labels))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    # Change the font sizes
    axes = fig.get_axes()
    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    # Colorbar (This makes the axes to display proper)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.solids.set_edgecolor('face')

    return fig
