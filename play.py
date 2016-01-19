"""
Just to play
"""

import numpy as np
import h5py
from sklearn import svm, cross_validation
from sklearn.naive_bayes import MultinomialNB

# First we load the file 
file_location = './results_database/text_wall_street.hdf5'
run_name = '/low-resolution'

Nspatial_clusters = 5
Ntime_clusters = 15
Nembedding = 3

parameters_string = '/' + str(Nspatial_clusters)
parameters_string += '-' + str(Ntime_clusters)
parameters_string += '-' + str(Nembedding)

f = h5py.File(file_location, 'r')
nexa = f[run_name + parameters_string]

# Now we load the time and the code vectors
time = nexa['time']
code_vectors = nexa['code-vectors']
code_vectors_distance = nexa['code-vectors-distance']
code_vectors_softmax = nexa['code-vectors-softmax']
code_vectors_winner = nexa['code-vectors-winner']

# Now we need to get the letters and align them
text_directory = './data/wall_street_letters.npy'
letters_sequence = np.load(text_directory)
Nletters = len(letters_sequence)
symbols = set(letters_sequence)

# Make prediction with scikit-learn
N = 1000 

X = code_vectors_softmax[:(N-1)]
X = code_vectors_winner[:(N -1)]
y = letters_sequence[1:N]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)

calculate_SVM = True
if calculate_SVM:
    clf = svm.SVC(C=1.0, cache_size=200, kernel='linear')
    clf.fit(X_train, y_train)
    print('SVM score', clf.score(X_test, y_test))

    clf_b = MultinomialNB()
    clf_b.fit(X_train, y_train)
    print('Multionmial score', clf_b.score(X_test, y_test))

