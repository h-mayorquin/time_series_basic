"""
A script just to play
"""

import numpy as np
from inputs.sensors import Sensor
from inputs.lag_structure import LagStructure

data = np.random.rand(1000)
dt = 0.5
lag_times = np.arange(10.0)
window_size = 10.0
Nwindow_size = int(window_size / dt)
lag = 4

lag_structure = LagStructure(lag_times=lag_times,
                             window_size=window_size)
sensor = Sensor(data, dt=dt, lag_structure=lag_structure)

first_lag_index = int(lag_times[lag - 1] / dt)
start = data.size - first_lag_index - Nwindow_size

result = np.zeros(Nwindow_size)

for index in range(Nwindow_size):
    result[index] = data[start + index]

lagged_sensor = sensor.lag_back(lag)
print(result, sensor.lag_back(lag))
