"""
Column prediction
"""

import numpy as np
from sklearn import svm, cross_validation
import h5py


# First we load the file 
file_location = './results_database/text_wall_street_columns.hdf5'
run_name = '/test'
f = h5py.File(file_location, 'r')


# Now we need to get the letters and align them
text_directory = './data/wall_street_letters.npy'
letters_sequence = np.load(text_directory)
Nletters = len(letters_sequence)
symbols = set(letters_sequence)

# Transform to letters
if False:
    symbol_to_number = {}
    for number, symbol in enumerate(symbols):
        symbol_to_number[symbol] = number

    letters_sequence = [symbol_to_number[letter] for letter in letters_sequence]

# Nexa parameters
Nspatial_clusters = 3
Ntime_clusters = 48
Nembedding = 3

parameters_string = '/' + str(Nspatial_clusters)
parameters_string += '-' + str(Ntime_clusters)
parameters_string += '-' + str(Nembedding)

nexa = f[run_name + parameters_string]
cluster_to_index = nexa['cluster_to_index']
code_vectors_softmax = np.array(nexa['code-vectors-softmax'])

Ndata = 10000
targets = []

for index in range(Ndata):
    letter_index = index // 10
    targets.append(letters_sequence[letter_index])

# Transform to array
targets = np.array(targets)

# Now we need to classify
X  = code_vectors_softmax[:Ndata]
y = targets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)

clf_linear = svm.SVC(C=1.0, kernel='linear')
clf_linear.fit(X_train, y_train)
score = clf_linear.score(X_test, y_test) * 100.0
print('score', score)
prediction = clf_linear.predict(X_train)

run_name = '/indep'
f = h5py.File(file_location, 'r')


# Now we need to get the letters and align them
text_directory = './data/wall_street_letters.npy'
letters_sequence = np.load(text_directory)
Nletters = len(letters_sequence)
symbols = set(letters_sequence)

# Transform to letters
if False:
    symbol_to_number = {}
    for number, symbol in enumerate(symbols):
        symbol_to_number[symbol] = number

    letters_sequence = [symbol_to_number[letter] for letter in letters_sequence]

# Nexa parameters
Nspatial_clusters = 3
Ntime_clusters = 3
Nembedding = 3

parameters_string = '/' + str(Nspatial_clusters)
parameters_string += '-' + str(Ntime_clusters)
parameters_string += '-' + str(Nembedding)

nexa = f[run_name + parameters_string]
cluster_to_index = nexa['cluster_to_index']
code_vectors_softmax = np.array(nexa['code-vectors-softmax'])

targets = []

for index in range(Ndata):
    letter_index = index // 10
    targets.append(letters_sequence[letter_index])

# Transform to array
targets = np.array(targets)

# Now we need to classify
X  = code_vectors_softmax[:Ndata]
y = targets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)

clf_linear = svm.SVC(C=1.0, kernel='linear')
clf_linear.fit(X_train, y_train)
score = clf_linear.score(X_test, y_test) * 100.0
print('score ind', score)
prediction_ind = clf_linear.predict(X_train)
