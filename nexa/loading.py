"""
Here will be the functions that load from the hdf5 storage
representation of the hdf5 format.
"""

def get_SLM_hdf5(database, run_name):
    """"
    This gets the SLM matrix from a particular data set
    and a particular run. 
    """

    r = database[run_name]
    SLM = r['SLM']
    return SLM[...]


def get_STDM_hdf5(database, run_name, nexa_arrangement):
    """
    This gets the STDM matrix from a particular dataset and a
    a particular run. The particular nexa_arrangement can be provided
    as well.
    """
    r = database[run_name]
    n = r[nexa_arrangement]
    STDM = n['STDM']

    return STDM[...]


def get_cluster_to_index_hdf5(database, run_name, nexa_arrangement):
    """
    This should return the spatial clustering from an hdf5 database
    It should return the hdf5 group as an object.

    TO-DO maybe return it as a dictionary.
    """
    r = database[run_name]
    n = r[nexa_arrangement]

    return n['cluster_to_index']

def get_index_to_cluster_hdf5(database, run_name, nexa_arrangement):
    """
    Returns the mapping between the index of SLM and the spatial 
    cluster to which something belongs directly from the hdf5 
    storage representation.
    """
    r = database[run_name]
    n = r[nexa_arrangement]
    index_to_cluster = n['index_to_cluster']
    return index_to_cluster[...]
    
def get_cluster_to_time_centers_hdf5(database, run_name, nexa_arrangement):
    """
    This should return the cluster to time centers mapping
    from an hdf5 database. It should return the hdf5 
    group as an object.

    TO-DO maybe return it as a dictionary.
    """
    r = database[run_name]
    n = r[nexa_arrangement]

    return n['cluster_to_time_centers']
