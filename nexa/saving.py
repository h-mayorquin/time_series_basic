"""
This contains the classes for saving and storing nexa files and their output
"""
import h5py


class NexaSaverHDF5():
    """
    This class is responsible for creating an hdf5 database
    and coordinating the interactio of a nexa object with it.
    """

    # This should have a try statement to see if the file exists
    folder = './results_database/'
    extension = '.hdf5'

    def __init__(self, name, mode='a'):
        """
        Initializes the data base

        name: is an identifier that should be related to the input
        mode: the identifier for the hdf5 file: w+, a, r
        """

        self.filename = self.folder + name + self.extension
        print('Creating a data base at:')
        print(self.filename + ' with mode ' + mode)
        self.f = h5py.File(self.filename, mode)

    def save_complete_run(self, nexa_object, run_name='lag_structure'):
        """
        This file saves a complete run of the nexa framework.
        It should be used after the simulation was run and it serves
        as a handy way of saving everything. 

        This takes an object where all the quantities have been calculated
        before
        """

        # Things for shorthand access
        f = self.f
        a_sensor = nexa_object.sensors.sensors[0]
        
        # Time structure
        root = '/'
        lag_structure_type = run_name
        file_level = root + lag_structure_type + '/'
        # import IPython
        # IPython.embed()

        # If that directory is not already add it and all the quantities
        # Otherwise skip to the level bellow
        if file_level not in f.keys():
            print('Accesing database for the first time')
            # SLM
            name_to_store = 'SLM'
            f[file_level + name_to_store] = nexa_object.SLM
            # STDM
            name_to_store = 'STDM'
            f[file_level + name_to_store] = nexa_object.STDM
            # Lags
            name_to_store = 'lags'
            f[file_level + name_to_store] = a_sensor.lag_structure.lag_times
            # Weights or filters
            name_to_store = 'weights'
            f[file_level + name_to_store] = a_sensor.lag_structure.weights

            # Attributes
            f[file_level].attrs['dt'] = a_sensor.dt
            f[file_level].attrs['window_size'] = a_sensor.lag_structure.window_size
            f[file_level].attrs['Nwindow_size'] = nexa_object.sensors.Nwindow_size
            f[file_level].attrs['Nlags'] = nexa_object.Nlags
            f[file_level].attrs['Nsensors'] = nexa_object.Nsensors
            f[file_level].attrs['format of spatial'] = 'Nspatial_clusters-Ntime_clusters-Nembedding'
            f[file_level].attrs['lags_first'] = nexa_object.lags_first

        # nexa structure
        spatial_name = str(nexa_object.Nspatial_clusters) + '-'
        spatial_name += str(nexa_object.Ntime_clusters) + '-'
        spatial_name += str(nexa_object.Nembedding)

        file_level += spatial_name + '/'

        # Code vectors
        name_to_store = '/code-vectors'
        f[file_level + name_to_store] = nexa_object.build_code_vectors()

        name_to_store = '/code-vectors-distance'
        f[file_level + name_to_store] = nexa_object.build_code_vectors_distance()

        name_to_store = '/code-vectors-winner'
        f[file_level + name_to_store] = nexa_object.build_code_vectors_winner()

        name_to_store = '/code-vectors-softmax'
        f[file_level + name_to_store] = nexa_object.build_code_vectors_softmax()

        # Time
        name_to_store = '/time'
        f[file_level + name_to_store] = nexa_object.sensors.map_SLM_columns_to_time()

        # Index to cluster
        name_to_store = '/index_to_cluster'
        f[file_level + name_to_store] = nexa_object.index_to_cluster

        # Save attributes
        f[file_level].attrs['Nspatial_clusters'] = nexa_object.Nspatial_clusters
        f[file_level].attrs['Ndata_clusters'] = nexa_object.Ntime_clusters
        f[file_level].attrs['Nembeeding'] = nexa_object.Nembedding
        f[file_level].attrs['Git'] = 'Not implemented yet'

        # Create cluster to index map
        grp = f.create_group(file_level + 'cluster_to_index')

        # Cluster to index
        for cluster, indexes in nexa_object.cluster_to_index.items():
            grp[str(cluster)] = indexes

        # Cluster to time centers
        grp = f.create_group(file_level + 'cluster_to_time_centers')
        for cluster, time_centers in nexa_object.cluster_to_time_centers.items():
            grp[str(cluster)] = time_centers

        # Close the file
        f.close()

