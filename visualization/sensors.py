"""
Visualize function realted to the sensors
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import nexa.loading as load

sns.set(style='white')

def visualize_STDM(nexa_object, ax=None):
    """
    Routine which plots using seaborn
    """

    to_plot = nexa_object.STDM
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(to_plot, mask=None,  cmap=cmap,
                vmax=1.0, vmin=-1.0,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    plt.title('Spatio Temporal Distance Matrix (Distances)')

    if ax is None:
        return fig
    else:
        return ax

def visualize_SLM(nexa_object, cmap='coolwarm', inter='none',
                  origin='upper', fontsize=16, aspect='auto',
                  colorbar=True, ax=None, symmetry=True):
    """
    Document
    """

    SLM = nexa_object.SLM
    to_plot = SLM

    # First the parameters
    to_plot_title = 'Sensor Lagged Matrix'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize

    xlabel = 'Time Windows'
    ylabel = 'Lagged Sensors'

    if ax is None:
        fig_size = (16, 12)
        axes_position = [0.1, 0.1, 0.8, 0.8]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes(axes_position)

    if symmetry:
        # We create symmetric vmin and vmax
        max_value = np.abs(np.max(to_plot))
        min_value = np.abs(np.min(to_plot))
        vmax = np.max((max_value, min_value))
        vmin = -vmax
                    
        im = ax.imshow(to_plot, interpolation=inter, vmin=vmin,
                       vmax=vmax, cmap=cmap, origin=origin,
                       aspect=aspect)
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

    return im


def visualize_STDM(nexa_object, cmap='coolwarm', inter='none',
                   origin='upper', fontsize=16, aspect='auto',
                   colorbar=True):
    """
    Document
    """

    Nlags = nexa_object.Nlags
    Nsensors = nexa_object.sensors.Nsensors
    STDM = nexa_object.STDM

    to_plot = STDM

    # First the parameters
    to_plot_title = 'Spatio Temporal Distance Matrix'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Time lags * Sensors'
    ylabel = xlabel

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

    return fig


def visualize_SLM_hdf5(database, run_name, cmap='coolwarm', inter='none',
                       origin='upper', fontsize=16, aspect='auto',
                       colorbar=True, ax=None, symmetry=True):
    """
    This visualizes the SLM for a particular database
    of a hdf5 storage and a particular run.
    """
    SLM = load.get_SLM_hdf5(database, run_name)
    to_plot = SLM

    # First the parameters
    to_plot_title = 'Sensor Lagged Matrix'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize

    xlabel = 'Time Windows'
    ylabel = 'Lagged Sensors'

    if ax is None:
        fig_size = (16, 12)
        axes_position = [0.1, 0.1, 0.8, 0.8]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes(axes_position)

    if symmetry:
        # We create symmetric vmin and vmax
        max_value = np.abs(np.max(to_plot))
        min_value = np.abs(np.min(to_plot))
        vmax = np.max((max_value, min_value))
        vmin = -vmax
                    
        im = ax.imshow(to_plot, interpolation=inter, vmin=vmin,
                       vmax=vmax, cmap=cmap, origin=origin,
                       aspect=aspect)
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

    return im


def visualize_STDM_hdf5(database, run_name, nexa_arrangement,
                        ax=None):
    """
    Routine which plots the STDM using seaborn
    and extracting this from a hdf5 representation
    """

    to_plot = load.get_STDM_hdf5(database, run_name, nexa_arrangement)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(to_plot, mask=None,  cmap=cmap,
                vmax=1.0, vmin=-1.0,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    plt.title('Spatio Temporal Distance Matrix (Distances)')

    if ax is None:
        return fig
    else:
        return ax

