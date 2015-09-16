"""
The main class for Nexa will be here.
"""
import numpy as np
from sklearn import manifold, cluster

from inputs.sensors import PerceptualSpace


class Nexa():
    """
    This is the class for Nexa.
    """

    n_init = 10
    n_jobs = -1  # -1 To use all CPUs, 1 for only one

    def __init__(self, sensors, Nspatial_clusters,
                 Ntime_clusters, Nembedding, SLM=None, lag_first=True):
        """
        Describe the parameters
        """

        self.Nspatial_clusters = Nspatial_clusters
        self.Ntime_clusters = Ntime_clusters
        self.Nembedding = Nembedding
        self.lag_first = lag_first

        # Check that sensors are a PerceptualSpace instance
        # To do: Make this a try statmeent
        if isinstance(sensors, PerceptualSpace):
            self.sensors = sensors
        else:
            self.sensors = PerceptualSpace(sensors, self.NLags, lag_first)

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

    def calculate_embedding(self):
        """
        This calculates the euclidian embedding of our
        distance matrix using MDS.

        Calculates the embedding, that is, it embedds every
        sensor on an Euclidian space with dimensions equal to
        self.Nembedding. Threfore it should return an array
        with a shape = (self.Nsensors, self.Nembedding).

        To Do: assertion for STDM not previously calculated
        """
        disimi = 'precomputed'
        n_init = Nexa.n_init
        n_jobs = Nexa.n_jobs
        n_comp = self.Nembedding

        classifier = manifold.MDS(n_components=n_comp, n_init=n_init,
                                  n_jobs=n_jobs, dissimilarity=disimi)

        self.embedding = classifier.fit_transform(self.STDM)

        return classifier.stress_

    def calculate_spatial_clustering(self):
        """
        This class calculates the spatial clustering. Once there is
        an embedding this function performs a clustering in the
        embedded space with as many clusters as self.Nspatial_clusters

        It returns an index to cluster which maps every sensor to the
        cluster (via a number) that it belongs to.
        """

        n_jobs = Nexa.n_jobs
        n_clusters = self.Nspatial_clusters

        classifier = cluster.KMeans(n_clusters=n_clusters, n_jobs=n_jobs)
        self.index_to_cluster = classifier.fit_predict(self.embedding)

    def calculate_cluster_to_indexes(self):
        """
        Calculates the dictionary where each cluster maps
        to the set of all sensors that belong to it. It should
        therefore return a dictionary with as many elements as
        there a clusters each mapping to a subset of the sensor
        set.
        """
        self.cluster_to_index = {}

        for cluster_n in range(self.Nspatial_clusters):
            indexes = np.where(self.index_to_cluster == cluster_n)[0]
            self.cluster_to_index[cluster_n] = indexes

    def calculate_time_clusters(self):
        """
        This calculates a dictionary where the keys are sensor
        cluster indexes and the values are an array
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

    def calculate_all(self):
        """
        Calculates all the quantities of the object in one go
        """
        self.calculate_distance_matrix()
        self.calculate_embedding()
        self.calculate_spatial_clustering()
        self.calculate_cluster_to_indexes()
        self.calculate_time_clusters()

    def build_code_vectors(self):
        """
        A function that builds the code vectors for the current
        SLM

        You will have a list with as many code vectors as time
        points you have.

        Each code vector will have size equal to the number of
        spatial cluster. Each spatial cluster will b given a
        value according to which time cluster the piece of data
        is closer too.
        """

        code_vectors = []
        cluster_to_index = self.cluster_to_index
        cluster_to_time_centers = self.cluster_to_time_centers

        Nt = self.SLM.shape[1]
        for t in range(Nt):
            vector = np.zeros(self.Nspatial_clusters)
            for Ncluster, cluster_indexes in cluster_to_index.items():
                cluster_data = self.SLM[cluster_indexes, t]
                time_centers = cluster_to_time_centers[Ncluster]
                dot = np.dot(time_centers, cluster_data)
                vector[Ncluster] = np.argmax(dot)

            code_vectors.append(vector)

        return code_vectors
