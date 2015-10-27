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

distances = np.linspace(0, 600, 100)
bases = np.linspace(0, 200, 500)
values = np.linspace(0, 200, 50)

swep_distances = True
swep_bases = True
swep_values = True

########
# Swep distances
########

if swep_distances:
    for distance in distances:
        # SLE
        fig = parameter_swep_SLM(base, distance, value)
        # Get the filename righ
        name = 'parameter_swep'
        name += '_SLM'
        filename = create_filename(directory, name, figure_format, base, distance, value)
        # Save the figure
        plt.savefig(filename)
        print(filename)
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
        print(filename)
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
        print(filename)
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
        print(filename)
        # Clear the figure
        plt.close(fig)

########
# Swep bases
########
        
if swep_bases:
    for base in bases:
        # SLE
        fig = parameter_swep_SLM(base, distance, value)
        # Get the filename righ
        name = 'parameter_swep'
        name += '_SLM'
        filename = create_filename(directory, name, figure_format, base, distance, value)
        # Save the figure
        plt.savefig(filename)
        print(filename)
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
        print(filename)
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
        print(filename)
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
        print(filename)
        # Clear the figure
        plt.close(fig)

########
# Swep values
########
if swep_values:        
    for value in values:
        # SLE
        fig = parameter_swep_SLM(base, distance, value)
        # Get the filename righ
        name = 'parameter_swep'
        name += '_SLM'
        filename = create_filename(directory, name, figure_format, base, distance, value)
        # Save the figure
        plt.savefig(filename)
        print(filename)
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
        print(filename)
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
        print(filename)
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
        print(filename)
        # Clear the figure
        plt.close(fig)



