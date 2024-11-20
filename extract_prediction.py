# -*- coding: utf-8 -*-
"""
Exctracting prediction for a model by name.
Update this file for each new model by implementing a function that 
extract the prediction feature array for a model.

See examples test_model or random_model
"""
import numpy as np
import argparse


def test_model(filename, args):
    """
    Reads pregenerated features for the respective filename 

    Parameters
    ----------
    filename : str filename of an image
    args : args with the defined pathes

    Returns
    -------
    prediction : numpy array 

    """    
    
    #reading from file
    
    file_embedding_name = filename.replace(args.test_dataset_folder, args.models_path+args.model_name+"/predictions/")
    file_embedding_name = file_embedding_name.replace('.jpg', '.npy')
    
    #print(file_embedding_name)
    prediction = np.load(file_embedding_name)
    return prediction



def random_model(filename, args):
    """
    Function generates a random embedding for any input

    Parameters
    ----------
    filename : any filename
    args : any args

    Returns
    -------
    prediction : numpy array  random prediction

    """
    
    prediction = np.random.uniform(low=-1.0, high=1.0, size=(args.size_array,))
    prediction =  prediction / np.linalg.norm(prediction)
    
    return prediction



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', default = "./test_images/1_aligned.jpg", help='test image path')
    parser.add_argument('--size_array', default = 64 , help='Size of feature array')
    
    args = parser.parse_args()
    
    return args


def test_random_model(args):
    
    prediction = random_model(args.filename, args)
    print(prediction)
    
if __name__ == '__main__':
    
    args = parse_args()
    
    test_random_model(args)    
    