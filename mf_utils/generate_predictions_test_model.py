# -*- coding: utf-8 -*-
"""
This script generates random test embeddings for estimating MMPMR rates 
Namelly it crates similar embeddings for the intra class group and interpolate embeddings for morphs. 
Level of similarity is controlled by intra_class_variance_coefficient variables

@author: iurii.medvedev
"""

import os
import numpy as np
import sys
import random   
import tqdm
import glob
import argparse
 


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_data', default = "../data/data_aligned/alignment_default/", help='Path to the benchmark data')
    parser.add_argument('--path_to_predictions', default = "../models/test_model/predictions", help='Path to the predictions to save')
    parser.add_argument('--dataset_enrollment', default = "/FACING2/Originals/Enrollment", help='Relative path to the enrollment dataset')
    parser.add_argument('--dataset_reference', default = "/FACING2/Originals/Reference", help='Relative path to the reference dataset')
    parser.add_argument('--datasets_morph', default = "/FACING2/Morphs/", help='Relative path to the morph dataset')

    parser.add_argument('--feature_size', default=128, help='Size of a feature vector')
    parser.add_argument('--extention', default=".jpg", help='Image extention')

    parser.add_argument('--similarity_enrollemnt', default=0.8, help='Similarity coefficient for enrollment data')
    parser.add_argument('--similarity_reference', default=0.8, help='Similarity coefficient for reference data')
    parser.add_argument('--similarity_morph', default=0.9, help='Similarity coefficient for morph data')

    args = parser.parse_args()
    
    return args




def generate_enrollment_predictions(data_enrollment_path, 
                                    dst_predictions_path,
                                    size_array = 64,
                                    intra_class_variance_coefficient = 0.8,
                                    extention = ".jpg"):
  
    """
    Generates enrollment predictions by creating synthetic embeddings for files within a specified directory.
    This function simulates enrollment embeddings for a set of files, applying random perturbations 
    to represent intra-class variability.

    Parameters:
    -----------
    data_enrollment_path : str
        Path to the root directory containing subdirectories of images for each class or identity.
    dst_predictions_path : str
        Destination path for storing generated prediction embeddings as .npy files.
    size_array : int, optional
        Size of the embedding vector for each file. Default is 64.
    intra_class_variance_coefficient : float, optional
        Coefficient controlling the level of intra-class variance in embeddings. Default is 0.1.
    extention : str, optional
        File extension of the image files to process (e.g., ".jpg"). Default is ".jpg".

    """
    
    #create subdirs in aligned folder
    for subdir, dirs, files in os.walk(data_enrollment_path):
        
        if len(dirs)<1:
            continue
        
        for idc in tqdm.tqdm(range(len(dirs)), desc = "Generating enrollment embeddings"):
            
            dirct = dirs[idc]
            os.makedirs(dst_predictions_path+'/'+dirct, exist_ok=True)
        
            #run within each subfolder
            files = [f for f in glob.glob(data_enrollment_path+"/"+dirct+"/" + "*"  + extention)]
                        
            #generate class embedding
            baseline = np.random.uniform(low=-1.0, high=1.0, size=(size_array,))
            baseline = baseline / np.linalg.norm(baseline)
            
            for i, file in enumerate(files):  
                
                jitter = np.random.uniform(low=-1.0, high=1.0, size=(size_array,))
                jitter = intra_class_variance_coefficient * jitter / np.linalg.norm(jitter)
                
                embedding  = baseline + jitter
                embedding = embedding / np.linalg.norm(embedding)
                
                #print(embedding)
                
                file = file.replace(data_enrollment_path, dst_predictions_path)
                file = file.replace(os.sep, '/')
                file = file.replace(extention.upper(), extention)
                file = file.replace(extention, '.npy')
                
                #print(file, np.linalg.norm(embedding), extention.upper())
                 
                np.save(file, embedding )
              


def generate_reference_predictions(data_reference_path, 
                                   data_enrollment_predictions_path,
                                   dst_predictions_path,
                                   size_array = 64,
                                   intra_class_variance_coefficient = 0.9,
                                   extention = ".jpg"):
    
    """
    Generates reference predictions by creating embeddings for files in the reference directory 
    based on enrollment embeddings. It adds random perturbations to simulate intra-class variability.
    
    Parameters:
    -----------
    data_reference_path : str
        Path to the directory containing reference images organized in class-based subdirectories.
    data_enrollment_predictions_path : str
        Path to the directory containing enrollment embeddings (.npy files) for each class.
    dst_predictions_path : str
        Path where the generated reference embeddings (.npy files) will be saved.
    size_array : int, optional
        Dimensionality of the embedding vectors. Default is 64.
    intra_class_variance_coefficient : float, optional
        Factor to control the extent of intra-class variance applied to each reference embedding. Default is 0.15.
    extention : str, optional
        File extension of reference image files to process (e.g., ".jpg"). Default is ".jpg".

    """
    
    
    
    #create subdirs in aligned folder
    for subdir, dirs, files in os.walk(data_reference_path):
        
        if len(dirs)<1:
            continue
        
        for idc in tqdm.tqdm(range(len(dirs)), desc = "Generating reference embeddings"):
            
            dirct = dirs[idc]
            
            #print(dst_predictions_path+'/'+dirct, dirct)
            
            os.makedirs(dst_predictions_path+'/'+dirct, exist_ok=True)
        
            #run within each subfolder
            files = [f for f in glob.glob(data_reference_path+"/"+dirct+"/" + "*"  + extention)]
               
            #print(files)
            
            #run within each subfolder
            files_enrollment = [f for f in glob.glob(data_enrollment_predictions_path+"/"+dirct+"/" + "*"  + ".npy")]
                
            #generate class embedding
            baseline = np.load(files_enrollment[0])
            
            for i, file in enumerate(files):  
                
                jitter = np.random.uniform(low=-1.0, high=1.0, size=(size_array,))
                jitter = intra_class_variance_coefficient * jitter / np.linalg.norm(jitter)
                
                embedding  = baseline + jitter
                embedding = embedding / np.linalg.norm(embedding)
                
                #print(embedding)
                
                file = file.replace(data_reference_path, dst_predictions_path)
                file = file.replace(os.sep, '/')
                file = file.replace(extention.upper(), extention)
                file = file.replace(extention, '.npy')
                
                #print(file, np.linalg.norm(embedding), extention.upper())
                 
                np.save(file, embedding )
        

def generate_morph_predictions(data_morph_path, 
                               data_enrollment_predictions_path, 
                               dst_predictions_path,
                               size_array = 64,
                               intra_class_variance_coefficient = 0.9,
                               extention = ".jpg"):

    """
    Generates morph predictions by creating synthetic embeddings for morph images, 
    which combine features from two different enrolled classes. 
    Each morph embedding is created by averaging the embeddings of two classes 
    and adding random variance.
    
    Parameters:
    -----------
    data_morph_path : str
        Path to the directory containing morph images organized in class-based subdirectories.
        Each morph image filename should follow the convention 'class1_class2.ext'.
    data_enrollment_predictions_path : str
        Path to the directory containing enrollment embeddings (.npy files) for each class.
    dst_predictions_path : str
        Path to save the generated morph embeddings (.npy files).
    size_array : int, optional
        Dimensionality of the embedding vectors. Default is 64.
    intra_class_variance_coefficient : float, optional
        Factor to control the level of added variance in morph embeddings. Default is 0.3.
    extention : str, optional
        File extension of morph image files (e.g., ".jpg"). Default is ".jpg".
    
    """

    #create subdirs in aligned folder
    for subdir, dirs, files in os.walk(data_morph_path):
        
        if len(dirs)<1:
            continue
        
        for idc in tqdm.tqdm(range(len(dirs)), desc = "Generating morph embeddings"):
            
            dirct = dirs[idc]
            
            #print(dst_predictions_path+'/'+dirct, dirct)
            
            os.makedirs(dst_predictions_path+'/'+dirct, exist_ok=True)
        
            #run within each subfolder
            files = [f for f in glob.glob(data_morph_path+"/"+dirct+"/" + "*"  + extention)]
               
            
            for i, file in enumerate(files):  
                
                filename = file.replace(data_morph_path +"/"+dirct, "")
                filename = filename.replace(os.sep, '')
                filename = filename.replace(extention, "")
                dirct_1, dirct_2 = filename.split("_")
                
                
                #load enrollment predictions
                files_enrollment_1 = [f for f in glob.glob(data_enrollment_predictions_path+"/"+dirct_1+"/" + "*"  + ".npy")]
                files_enrollment_2 = [f for f in glob.glob(data_enrollment_predictions_path+"/"+dirct_2+"/" + "*"  + ".npy")]
                 
                
                #generate class embedding
                baseline_1 = np.load(files_enrollment_1[0])
                baseline_2 = np.load(files_enrollment_2[0])
                
                
                jitter = np.random.uniform(low=-1.0, high=1.0, size=(size_array,))
                jitter = intra_class_variance_coefficient * jitter / np.linalg.norm(jitter)
                
                embedding  = (baseline_1 + baseline_2)/2 + jitter
                embedding = embedding / np.linalg.norm(embedding)
                
                #print(embedding)
                
                file = file.replace(data_morph_path, dst_predictions_path)
                file = file.replace(os.sep, '/')
                file = file.replace(extention.upper(), extention)
                file = file.replace(extention, '.npy')
                
                #print(file, np.linalg.norm(embedding), extention.upper())
                 
                np.save(file, embedding )

    
    
    

if __name__ == '__main__':
    
    args = parse_args()
        
       
    path_to_data = args.path_to_data
    path_to_predictions = args.path_to_predictions
    
    dataset_enrollment = args.dataset_enrollment
    dataset_reference = args.dataset_reference

    datasets_morph = args.datasets_morph
    
    extention = args.extention
    feature_array_size = args.feature_size
    
    intra_class_variance_coefficient_enrollment = args.similarity_enrollemnt
    intra_class_variance_coefficient_reference = args.similarity_reference
    intra_class_variance_coefficient_morph = args.similarity_morph
    
    
    #running gerenation of the data
    generate_enrollment_predictions(data_enrollment_path = path_to_data + dataset_enrollment,
                                    dst_predictions_path = path_to_predictions + dataset_enrollment,
                                    size_array = feature_array_size,
                                    intra_class_variance_coefficient = intra_class_variance_coefficient_enrollment,
                                    extention = extention)
    
       
    generate_reference_predictions(data_reference_path = path_to_data + dataset_reference, 
                                   data_enrollment_predictions_path = path_to_predictions + dataset_enrollment, 
                                   dst_predictions_path = path_to_predictions + dataset_reference,
                                   size_array = feature_array_size,
                                   intra_class_variance_coefficient = intra_class_variance_coefficient_reference,
                                   extention = extention)
    
    generate_morph_predictions(data_morph_path = path_to_data + datasets_morph, 
                               data_enrollment_predictions_path = path_to_predictions + dataset_enrollment, 
                               dst_predictions_path = path_to_predictions + datasets_morph,
                               size_array = feature_array_size,
                               intra_class_variance_coefficient = intra_class_variance_coefficient_morph,
                               extention = extention)
    

    