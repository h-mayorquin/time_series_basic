import numpy as np
import matplotlib.pyplot as plt
from signals.signal_class import SpatioTemporalSignal

# First the paramters
dt = 0.1  # Resoultion
delay = 10   # In s
Tmax = 20  # In s

# Plotting
plot = True
verbose = False

# Let's define the object
A = SpatioTemporalSignal(dt=dt, delay=delay, Tmax=Tmax, Nseries=2)
time = A.time

####
#  Filter contrusction
####

# First the time
filter_time = np.arange(A.Ndelay) * dt
# Now the filter
b
tau = 1
Amp = -100.0

decay = np.exp(-filter_time / tau)
periodic = Amp * np.cos(filter_time)
alpha = decay * periodic

alpha_to_plot = alpha
alpha_to_plot = periodic
# alpha_to_plot = decay * A.Ndelay

decay = np.exp(-time / tau)
periodic = Amp * np.cos(time)
alpha = decay * periodic
#  alpha = decay * A.Ndelay
# alpha = periodic

# The rest of the interaction terms
b = np.zeros(A.NTmax)
c = np.zeros(A.NTmax)
d = np.zeros(A.NTmax)

interaction = np.array(((alpha, b), (c, d)))

###
#  Complete the series and extract it
###

initial_coditions = np.array((1, 0))
A.set_initial_conditions(initial_coditions)
A.set_interaction(interaction)
if verbose:
    A.construct_series_verbose()
else:
    A.construct_series()

result = A.series[0, :]
correlation = np.correlate(result, result, mode='same')

correlation_to_plot = correlation[correlation.size/2:]


if plot:
    plt.subplot(3, 1, 1)
    plt.plot(time, result)

    plt.subplot(3, 1, 2)
    plt.plot(filter_time, alpha_to_plot)

    plt.subplot(3, 1, 3)
    plt.plot(correlation_to_plot)

    plt.show()
