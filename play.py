import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Parmeter search
from parameter_exploration.parameter_swep import parameter_swep_SLM
from parameter_exploration.parameter_swep import parameter_swep_STDM
from parameter_exploration.parameter_swep import parameter_swep_cluster
from parameter_exploration.parameter_swep import parameter_swep_cluster_SLM
from parameter_exploration.parameter_swep import create_filename

# Directory to save
directory = './results/'
name = 'parameter_swep_cluster'
figure_format = '.png'
    
base = 10
distance = 300
value = 50

distances = np.linspace(0, 600, 601)
bases = np.linspace(0, 200, 201)
values = np.linspace(10, 200, 191)

swep_distances = True
swep_bases = True
swep_values = True
verbose = True

########
# Swep distances
########

for base in bases:
    for distance in distances:
        for value in values:
            if verbose:
                print(base, distance, value)
            # SLE
            fig = parameter_swep_SLM(base, distance, value)
            # Get the filename righ
            name = 'parameter_swep'
            name += '_SLM'
            filename = create_filename(directory, name, figure_format, base, distance, value)
            # Save the figure
            plt.savefig(filename)
            # Clear the figure
            plt.close(fig)

            # STDM
            fig = parameter_swep_STDM(base, distance, value)
            # Get the filename right
            name = 'parameter_swep'
            name += '_STDM'
            filename = create_filename(directory, name, figure_format, base, distance, value)
            # Save the figure
            plt.savefig(filename)
            # Clear the figure
            plt.close(fig)

            # Cluster
            fig = parameter_swep_cluster(base, distance, value)
            # Get the filename right
            name = 'parameter_swep'
            name += '_cluster'
            filename = create_filename(directory, name, figure_format, base, distance, value)
            # Save the figure
            plt.savefig(filename)
            # Clear the figure
            plt.close(fig)

            # Cluster-SLE
            fig = parameter_swep_cluster_SLM(base, distance, value)
            # Get the filename right
            name = 'parameter_swep'
            name += '_cluster_SLM'
            filename = create_filename(directory, name, figure_format, base, distance, value)
            # Save the figure
            plt.savefig(filename)
            # Clear the figure
            plt.close(fig)
