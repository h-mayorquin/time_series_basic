import numpy as np
import h5py
import matplotlib.pyplot as plt
from visualization.data_clustering import visualize_data_cluster_text_to_image_columns

# First we load the file 
file_location = './results_database/text_wall_street_columns_spaces.hdf5'
run_name = '/test'
f = h5py.File(file_location, 'r')


# Now we need to get the letters and align them
text_directory = './data/wall_street_letters_spaces.npy'
letters_sequence = np.load(text_directory)
Nletters = len(letters_sequence)
symbols = set(letters_sequence)

# Nexa parameters
Nspatial_clusters = 3
Ntime_clusters = 3
Nembedding = 3

parameters_string = '/' + str(Nspatial_clusters)
parameters_string += '-' + str(Ntime_clusters)
parameters_string += '-' + str(Nembedding)

nexa = f[run_name + parameters_string]
cluster_to_index = nexa['cluster_to_index']

cluster = 2
data_center = 1

if True:
    fig = visualize_data_cluster_text_to_image_columns(nexa, f, run_name,
                                                       cluster, data_center, colorbar=True)
    plt.show(fig)


    
