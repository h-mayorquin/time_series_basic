"""
The main class for Nexa will be here.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, cluster

from input.sensors import PerceptualSpace


class Nexa():
    """
    This is the class for Nexa.
    """

    n_init = 4
    n_jobs = -1  # -1 To use all CPUs, 1 for only one

    def __init__(self, sensors, Nlags, Nspatial_clusters,
                 Ntime_clusters, SLM=None):
        """
        Describe the parameters
        """

        self.Nlags = Nlags
        self.Nspatial_clusters = Nspatial_clusters
        self.Ntime_clusters = Ntime_clusters

        # Check that sensors are a PerceptualSpace instance
        # To do: Make this a try statmeent
        if isinstance(sensors, PerceptualSpace):
            self.sensors = sensors
        else:
            self.sensors = PerceptualSpace(sensors, self.NLags)

        if SLM is None:
            self.SLM = self.sensors.calculate_SLM()

        # Initiate values for the methods
        self.STDM = None
        self.embedding = None
        self.index_to_cluster = None
        self.cluster_to_index = None
        self.cluster_to_time_centers = None
        self.dimensions_of_embedding_space = None

    def calculate_distance_matrix(self):
        self.STDM = self.sensors.calculate_STDM()

    def calculate_embedding(self, n_comp):
        """
        This calculates the euclidian embedding of our
        distance matrix using MDS.

        To Do: assertion for STDM not previously calculated
        """
        disimi = 'precomputed'
        n_init = Nexa.n_init
        n_jobs = Nexa.n_jobs

        classifier = manifold.MDS(n_components=n_comp, n_init=n_init,
                                  n_jobs=n_jobs, dissimilarity=disimi)

        self.embedding = classifier.fit_transform(self.STDM)

        return classifier.stress_

    def calculate_spatial_clustering(self):
        """
        This class calculates the spatial clustering
        """

        n_jobs = Nexa.n_jobs
        n_clusters = self.Nspatial_clusters

        classifier = cluster.KMeans(n_clusters=n_clusters, n_jobs=n_jobs)
        self.index_to_cluster = classifier.fit_predict(self.embedding)

    def calculate_cluster_to_indexes(self):
        """
        Calculates the dictionary
        """
        self.cluster_to_index = {}

        for cluster_n in range(self.Nspatial_clusters):
            indexes = np.where(self.index_to_cluster == cluster_n)[0]
            self.cluster_to_index[cluster_n] = indexes

    def calculate_time_clusters(self):
        """
        This calculates a dictionary where the keys are sensor
        cluster indexes and the accessed elements are an array
        of the cluster centers.

        TO DO: Return the signal indexes when asigned to cluster
        """

        n_jobs = Nexa.n_jobs
        t_clusters = self.Ntime_clusters

        self.cluster_to_time_centers = {}

        for cluster_n, cluster_indexes in self.cluster_to_index.items():

            data_in_the_cluster = self.SLM[cluster_indexes, :]
            classifier = cluster.KMeans(n_clusters=t_clusters, n_jobs=n_jobs)
            classifier.fit_predict(data_in_the_cluster.T)
            centers = classifier.cluster_centers_
            self.cluster_to_time_centers[cluster_n] = centers
