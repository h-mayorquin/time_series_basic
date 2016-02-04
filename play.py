"""
Just to play
"""

import numpy as np
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
# Now plot


# First we load the file 
file_location = './results_database/text_wall_street_big.hdf5'
run_name = '/low-resolution'
f = h5py.File(file_location, 'r')

# Now we need to get the letters and align them
text_directory = './data/wall_street_letters.npy'
letters_sequence = np.load(text_directory)
Nletters = len(letters_sequence)
symbols = set(letters_sequence)

# Load the particular example
Nspatial_clusters = 8
Ntime_clusters = 40
Nembedding = 3

parameters_string = '/' + str(Nspatial_clusters)
parameters_string += '-' + str(Ntime_clusters)
parameters_string += '-' + str(Nembedding)

nexa = f[run_name +parameters_string]

cluster = 2
time_center = 1

from visualization.data_cluster import visualize_data_cluster_text_to_image

fig = visualize_data_cluster_text_to_image(nexa, f, run_name,
                                           cluster, time_center)
plt.show(fig)


