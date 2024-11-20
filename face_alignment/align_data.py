# -*- coding: utf-8 -*-
"""
Example alignment.
Aligned the labled dataset - folder with folders.
Detect and align faces. 
Then copy resized files to the dst folder

"""

from alignment import _read_image_2_cvMat
from alignment import _align_faces, _align_face_insightface
import cv2
import glob
import os, sys, inspect
import numpy as np
from mtcnn.mtcnn import MTCNN
import argparse
import tqdm
    


def parse_args():
    parser = argparse.ArgumentParser()
  
    parser.add_argument('-s', '--src_folder', default="../data/data_raw/FACING2/Morphs", help='Source path to the directory with folders of images')
    parser.add_argument('-d', '--dst_folder', default="../data/data_aligned_1/FACING2/Morphs", help='Destination directory for aligned images')
    parser.add_argument('-f', '--face_size', default=224 , help='Size of the aligned face')
   
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()
   
    src_folder = args.src_folder  
    dst_folder = args.dst_folder
    
    face_size=(int(args.face_size), int(args.face_size))
    
    print("start alignment")
    
    number_of_aligned = 0
    not_aligned_list = []

    
    detector = MTCNN()  
    
    #create subdirs in aligned folder
    for subdir, dirs, files in os.walk(src_folder):
        #print(files)
        for dirct in dirs:
            #create directory for saving the model

            os.makedirs(dst_folder+'/'+dirct, exist_ok=True)
        
            
            #run within each subfolder
            files = [f for f in glob.glob(src_folder+"/"+dirct+"/" + "*"  + ".jpg")]
    	
            for i, file in enumerate(files):  

                #print("file", file)
                file = file.replace(os.sep, '/')
                
                #defining filename and brisque filename
                new_filename = file.replace(src_folder, dst_folder)
                
                #print("new_filename", new_filename)
                
                image = cv2.imread(file)
                
                #run alignment    
                result, number_of_detected, main_idx , landmarks, _ = _align_faces(cv_image = image,
                                                                                detector = detector, 
                                                                                face_size = face_size, 
                                                                                main_face_criteria = "size")
                
                
              
                if (number_of_detected > 0):
                
                    cv2.imwrite(new_filename, result[main_idx])
                    number_of_aligned += 1
                    
                
                    
                else:
                    print("file", file)
                    not_aligned_list.append(file)
                

    print("not aligned images, align them manually:"  )
    print(not_aligned_list)
       
    
 
if __name__ == '__main__':
    main()
