# -*- coding: utf-8 -*-
"""
Some benchmark utils
"""


import os
import math
import sys
import numpy as np
import csv


def walklevel(some_dir, level=1):
    """
    Modified function os.walk to handle the level of walk depth through the folders.
    
    Parameters
    ----------
    some_dir : string
        path to the directory.
    level : int, optional
        DESCRIPTION. The default is 1.

    Yields
    ------
    root : list
        list woth subdirectories.
    dirs : list
        list with directories.
    files : list
        file list.

    """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
            
def estimate_similarity(a, b, similarity_metric = 'dot_product'):
    
    similarity = 0.0
    if (similarity_metric == 'euclidean_distance'):
        #euclidean distance
        similarity = numpy.linalg.norm(a-b)/(np.linalg.norm(a)*np.linalg.norm(b))
        pass
    elif (similarity_metric == 'dot_product'):
        #dot product
        similarity = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        pass
    else:
        sys.exit("metric is not chosen")
    
    
    return (similarity)


def check_protocol_data(protocols_path, protocol_name, dataset_path):
    """
    Function checks the generated protocol with the dataset in the dataset_path. 
    Checks the existense of images from the protocol.
    Parameters
    ----------
    protocol_path : string
        path to the protocol folder
    protocol_name : string
        name of the protocol folder
    dataset_path :string
        path to the benchmark datasets.
    Returns
    -------
    boolean 
    Signifies if the protocol is ok (True)
    """
    path = os.path.join(protocols_path, protocol_name)
    
    with open('%s/%s/%s.csv' %(protocols_path, protocol_name, "dataset"), newline='') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    
    not_found = 0
    for path in dataset:
        full_path = dataset_path + path[0]
        if not os.path.exists(full_path):
            not_found += 1
        
    if not_found == 0:
        print("Protocol is okay")
        return(True)
    else:
        print("Protocol is not okay. {} files not found".format(not_found))
        return(False)



def find_nearest_value(array, value):
    """
    Simple function to find nearest value in np array

    Parameters
    ----------
    array : numpy array
        array to search the value
    value : float
        value to search

    Returns
    -------
    closest_element : numpy input array element type
        Closest element in the array
    idx : int
        index of the closest elemets

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    closest_element = array[idx]

    return  closest_element, idx


def test_find_nearest_value():
    
    array_size = 10
    array = np.random.random(10)
    print("random array - ", array)
    
    value = 0.5
    print("value to search", value)
    
    print("Resulting nearest value with index  - ", find_nearest_value(array, value))


if __name__ == '__main__':
    test_find_nearest_value()

