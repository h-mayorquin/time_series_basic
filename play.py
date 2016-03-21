import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.code_vectors import visualize_representation_winners

# First we load the file 
file_location = './results_database/text_wall_street_columns_30_semi_constantNdata.hdf5'
f = h5py.File(file_location, 'r')

Nembedding = 3
max_lag = 12
Nspatial_clusters = max_lag
Ntime_clusters = 60 // max_lag
    
# Here calculate the scores for the mixes
run_name = '/test' + str(max_lag)


parameters_string = '/' + str(Nspatial_clusters)
parameters_string += '-' + str(Ntime_clusters)
parameters_string += '-' + str(Nembedding)

nexa = f[run_name + parameters_string]
cluster_to_index = nexa['cluster_to_index']
code_vectors_softmax = np.array(nexa['code-vectors-softmax'])
code_vectors_winner = np.array(nexa['code-vectors-winner'])

visualize_representation_winners(code_vectors_winner, Nspatial_clusters, Ntime_clusters, ax=None)









