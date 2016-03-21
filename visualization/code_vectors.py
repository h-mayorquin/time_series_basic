"""
Visualizations for the code vectors
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def seaborn_code_vectors(code_vectors):
    """
    Seaborn version to visualize code vectors
    """

    to_plot = np.array(code_vectors)

    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(to_plot, mask=None,  cmap=cmap,
                vmax=1.0, vmin=-1.0,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=0, cbar_kws={"shrink": .5}, ax=ax)

    plt.title('Spatio Temporal Distance Matrix (Distances)')

    return fig


def visualize_code_vectors(code_vectors, cmap='Paired', inter='none',
                           origin='upper', fontsize=16, aspect='auto',
                           colorbar=True):
    """
    Document
    """

    to_plot = np.array(code_vectors)

    # First the parameters
    to_plot_title = 'Code Vectors in Time'

    cmap = cmap
    inter = inter
    origin = origin

    fontsize = fontsize  # The fontsize
    fig_size = (16, 12)
    axes_position = [0.1, 0.1, 0.8, 0.8]

    xlabel = 'Sensor Clusters'
    ylabel = 'Time'

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


def visualize_representation_winners(code_vectors, Nsensor_clusters, Ndata_clusters, ax=None):
    """
    Here we plot an histogram with the frequencies of winning for each of 
    the data clusters in each receptive field (Nsensors clusters)

    Parameters:
    --------------
    code_vectors: the code vectors in winner takes all format.
    Nsensor_clusters: The number of sensor cluster or receptive fields.
    Ndata_clusters: The number of data clusters per receptiv field.
    ax: an axes instance of matplotlib.

    If the axes instance is passed the axes instance is returned. Otherwise 
    just run plt.show() after the function to plot it.
    """

    # Create an axes instance if it is not passed
    if ax is None:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)

    # Get the frequencies and the indexes
    frequencies = code_vectors.mean(axis=0) 
    left_coordinates = np.arange(frequencies.size)

    # Get the color map
    Nfeatures = Nsensor_clusters * Ndata_clusters
    color_aux = np.arange(Nfeatures) // Ndata_clusters
    color_aux = color_aux / np.max(color_aux)
    color = plt.cm.Paired(color_aux)

    # Plot the bar and configure it
    ax.bar(left_coordinates, frequencies, color=color, align='center', alpha=0.8)
    ax.set_xlim(-1, Nfeatures + 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Receptive Fields')
    ax.set_ylabel('Level of activation')

    # Set the format for the thicks
    formatting = []
    for cluster in range(Nsensor_clusters + 1):
        formatting.append(str(cluster))

    ax.xaxis.set_major_formatter(plt.FixedFormatter(formatting))
    ax.xaxis.set_major_locator(plt.MultipleLocator(Ndata_clusters))
    
    
    return ax



