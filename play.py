"""
Just to play
"""

import numpy as np
import h5py
from sklearn import svm, cross_validation
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns

# First we load the file 
file_location = './results_database/text_wall_street_big.hdf5'
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

# Set the parameters for the simulation
maximal_lags = np.arange(8, 21, 3)
# Run the delay analysis
N = 1000
delays = np.arange(3, 21, 2)
accuracy_matrix = np.zeros((maximal_lags.size, delays.size))


for maximal_lag_index, maximal_lag in enumerate(maximal_lags):
    # Extract the appropriate database
    run_name = '/low-resolution' + str(maximal_lag)
    nexa = f[run_name + parameters_string]

    # Now we load the time and the code vectors
    time = nexa['time']
    code_vectors = nexa['code-vectors']
    code_vectors_distance = nexa['code-vectors-distance']
    code_vectors_softmax = nexa['code-vectors-softmax']
    code_vectors_winner = nexa['code-vectors-winner']

    for delay_index, delay in enumerate(delays):
        X = code_vectors_softmax[:(N - delay)]
        y = letters_sequence[delay:N]
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)

        clf = svm.SVC(C=1.0, cache_size=200, kernel='linear')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) * 100.0
        accuracy_matrix[maximal_lag_index, delay_index] = score
        print('delay_index', delay_index)
        print('maximal_lag_index', maximal_lag_index)
        print('maximal_lag', maximal_lag)
        print('delay', delay)
        print('score', score)


# Plot it

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
for maximal_lag_index in range(maximal_lags.size):
    ax.plot(delays, accuracy_matrix[maximal_lag_index, :], 'o-', lw=2, markersize=10,
            label=str(maximal_lags[maximal_lag_index]))

ax.set_xlabel('Delays')
ax.set_ylim([0, 105])
ax.set_title('Latency analysis for different lags')
ax.legend()

plt.show()
