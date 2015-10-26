import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Parmeter search
from parameter_exploration.parameter_swep import parameter_swep_SLE
from parameter_exploration.parameter_swep import parameter_swep_STDM
from parameter_exploration.parameter_swep import parameter_swep_cluster
from parameter_exploration.parameter_swep import create_filename

# Directory to save
directory = './results/'
name = 'parameter_swep_STDM'
# name = 'parameter_swep_SLE'
name = 'parameter_swep_cluster'
figure_format = '.png'
    
base = 10
distance = 300
value = 50

distances = np.arange(0, 600, 100)
bases = np.linspace(0, 200, 10)

for base in bases:
    filename = create_filename(directory, name, figure_format, base, distance, value)
    # fig = parameter_swep_SLE(base, distance, value)
    # fig = parameter_swep_STDM(base, distance, value)
    fig = parameter_swep_cluster(base, distance, value)
    plt.savefig(filename)
    # Clear the figure
    plt.close(fig)




