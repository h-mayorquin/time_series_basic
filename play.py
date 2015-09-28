"""
A script just to play
"""

import numpy as np
import matplotlib.pyplot as plt
from signals.spikes import inh_poisson_generator
from signals.spikes import sparse_to_dense
from scipy.signal import hamming, fftconvolve
from scipy.signal import boxcar


rate = [100., 100., 100.]
rate = np.asarray(rate)
t = [0., 1000., 2000.]
t = np.asarray(t)
t_stop = 3000.0

spike_train = inh_poisson_generator(rate, t, t_stop)

# Build a dense spike
dt = 1.0  # in ms
dense_spikes = sparse_to_dense(spike_train, dt, t_stop)

mean_rate = (1000.0 / t_stop) * spike_train.size
print mean_rate

kernel_duration = 100.0
kernel_size = int(kernel_duration / dt)  # ms

kernel = hamming(kernel_size)
kernel2 = boxcar(kernel_size)

norm = 1000.0 / kernel_duration

convolution = norm * fftconvolve(dense_spikes, kernel2, 'valid')
convolution = norm * fftconvolve(dense_spikes, kernel, 'valid')
plt.plot(convolution)
plt.show()
