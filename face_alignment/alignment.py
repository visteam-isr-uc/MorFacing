# -*- coding: utf-8 -*-

"""
Face alignment which follows the common academic alignment

@author: iurii
"""

import os
import sys
import random
import cv2
import numpy as np
from skimage import transform as trans

import math


def crop_face(face_image, faces_rect, face_center, modify_rect=True, d_mult = 1.0):
    """

    :param face_image:
    :param faces_rect: Tuple with the following values [x, y, width, height]
    :param face_center: Tuple with the following values [center_x, center_y]
    :param modify_rect:
    :return:
    """
    # Check if the face is empty
    if face_image is None or faces_rect is None:
        raise Exception("Empty Source")

    # Crop face using the rectangle region along the face center
    # If modify_rect is True, the function returns an image area cropped by a square with size of max(width, height)
    # multiplied by the scaling parameter
    if modify_rect:
        half_side_value = int(d_mult * max(faces_rect[2], faces_rect[3]) / 2)
        x = face_center[0] - half_side_value
        y = face_center[1] - half_side_value
        width = 2 * half_side_value
        height = 2 * half_side_value
    # If the modify_rect is False, simply crop the image using the input rectangle
    else:
        x = face_center[0] - (faces_rect[2] / 2)
        y = face_center[1] - (faces_rect[3] / 2)
        width = faces_rect[2]
        height = faces_rect[3]

    # Cast every thing to int because all the dimensions need to be integers
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    # Crop the image
    cropped_face = crop_any_rect(face_image, (x, y, width, height))

    return cropped_face

def crop_any_rect(src, roi):
    """
    Simple function that crops and image using the desired rectangle. If the rectangle lays outside of the image
    dimensions, the cropped image is filled with pixels similar to the border of the original image.
    :param src: An image loaded using opencv
    :param roi: Desired rectangle around the face. Needs to be in the following order: [x, y, width, height]
    :return: Returns the image cropped with the desired rectangle
    """
    bt_mrg = 0
    tp_mrg = 0
    lft_mrg = 0
    rgt_mrg = 0

    if roi[0] < 0:
        lft_mrg = abs(roi[0])

    if roi[1] < 0:
        tp_mrg = abs(roi[1])

    # Attention to this, ndarray.shape return (height, width) so the indexes in the src need to be switched
    gp_cols = src.shape[1] - roi[0] - roi[2]
    gp_rows = src.shape[0] - roi[1] - roi[3]

    if gp_cols < 0:
        rgt_mrg = abs(gp_cols)

    if gp_rows < 0:
        bt_mrg = abs(gp_rows)

    src = cv2.copyMakeBorder(src, tp_mrg, bt_mrg, lft_mrg, rgt_mrg, cv2.BORDER_CONSTANT, value=0)

    new_x = roi[0] + lft_mrg
    new_y = roi[1] + tp_mrg

    crop_img = src[new_y:new_y + roi[3], new_x:new_x + roi[2]]

    return crop_img

#Default FR Alignment parameters
def _get_gt_insightface_params():
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32 ) #center = [48.02616, 71.90078]
    dst[:,0] += 8.0                            #center = [56.02616, 71.90078]
    
    target_insightface_size = (112, 112)
    return(dst, target_insightface_size)


def _align_face_insightface(src_image, src_ldms, face_size):
    
    #assuring the float format of landmarks
    src_ldms = (np.array(src_ldms)).astype(np.float32)
    
    #getting ground truth landmarks
    dst_ldms, target_insightface_size = _get_gt_insightface_params()
    
    #rescaling the landmarks
    dst_ldms[:,0] *= face_size[0]/target_insightface_size[0]
    dst_ldms[:,1] *= face_size[1]/target_insightface_size[1]
    

    
    #estimating similarity with opencv
    M = cv2.estimateAffinePartial2D(src_ldms.astype(np.float32), dst_ldms.astype(np.float32), method=cv2.LMEDS, refineIters=10)
    #print(M)
    M = M[0]


    aligned_image = cv2.warpAffine(src_image, M, face_size, borderValue=0.0, flags=cv2.INTER_LINEAR)

    return(aligned_image)
 

def distance_2_pts(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def _find_center_pt(keypoints):
    #find central point of the face landmarks
    x = 0
    y = 0
    num = len(keypoints)
    for pt in keypoints:        
        x += keypoints[pt][0]
        y += keypoints[pt][1]
        
    x //= num
    y //= num
    return (x,y)

def _find_face_dim(bounding_box):
    #find dimention of face in pixels as average of width and height
    dim = (bounding_box[2]+ bounding_box[3])/2
    return dim

def _find_face_dim_keypoints(keypoints):
    #find dimention of face in pixels as average of width and height
    left_eye = (keypoints['left_eye'])
    right_eye = (keypoints['right_eye'])
    nose = (keypoints['nose'])
    mouth_left = (keypoints['mouth_left'])
    mouth_right = (keypoints['mouth_right'])
    
    center_of_face = _find_center_pt(keypoints) 
    
    left_eye_2_center = distance_2_pts(left_eye,center_of_face)
    right_eye_2_center = distance_2_pts(right_eye,center_of_face)
    nose_2_center = distance_2_pts(nose,center_of_face)
    mouth_left_2_center = distance_2_pts(mouth_left,center_of_face)
    mouth_right_2_center = distance_2_pts(mouth_right,center_of_face)
    
    distances = [left_eye_2_center,
                  right_eye_2_center,
                  nose_2_center,
                  mouth_left_2_center,
                  mouth_right_2_center]
    
    
    return max(distances)


#simply cropp the central part of the image
def _no_detected_face(cv_image, crop_ratio = 0.7, face_size=(224,224)):
    
    h, w, c = cv_image.shape
    new_bounding_box=[h/2-h*crop_ratio/2, w/2-w*crop_ratio/2, h*crop_ratio, w*crop_ratio]
    
    cropped_image = crop_face(cv_image, new_bounding_box, (w/2,h/2)) 
    
    return cv2.resize(cropped_image, face_size, interpolation=cv2.INTER_CUBIC)   

# return align faces, numberof detected faces,index on the main face
def _align_faces(cv_image,
                 detector, 
                 face_size=(224,224), 
                 main_face_criteria = "size"):
    """
    

    Parameters
    ----------
    cv_image : np.array,  Input raw cv mat image.
    detector : face_detecror, Face detector with method detect_face.
    face_size : tuple of (int,int), optional. Face image size. The default is (224,224).
    main_face_criteria : string, optional,'size'|'center'        DESCRIPTION. The default is "size".

    Returns
    -------
    output_imgs : list of np.array, output images.
    face_len : int, number of deteected faces.
    main_idx : int, index of the main face.
    output_ldms : list of lists of points, list of landmarks o detected faces.
    frontality_scores : list of floats, frontality scores estimated by landmarks.

    """
    
    
    # #check if image is too small, and if so then increase the size
    # if(cv_image.shape[0]+cv_image.shape[1]<200):
    #     print('enlarged')
    #     cv_image = cv2.resize(cv_image, (cv_image.shape[1]*3,cv_image.shape[0]*3), interpolation=cv2.INTER_CUBIC)
    
    h, w, c = cv_image.shape 
        
    #detect faces
    detection = detector.detect_faces(cv_image) 
    del detector
    
    #print("face detected ", len(detection))
    output_imgs = list()
    
    #print("face detected ", len(detection))
    output_ldms = list()
    
    #print("face detected ", len(detection))
    frontality_scores = list()
    
    #index of the main face (closest to the center)
    main_idx = 0 
    
    #current distance from the center. search for minimal
    l_dist = h*w
    
    #current face dimention
    face_dim_max = 0
    
    face_len = 0 
    
    #cycle idx
    idx = 0
    for face in detection:
        #face location
        bounding_box = face['box']
        keypoints = face['keypoints']
        
        left_eye = (keypoints['left_eye'])
        right_eye = (keypoints['right_eye'])
        nose = (keypoints['nose'])
        mouth_left = (keypoints['mouth_left'])
        mouth_right = (keypoints['mouth_right'])
        
        # estimating frontality of the face
        rel1 = distance_2_pts(left_eye, nose)/max(0.0001,distance_2_pts(right_eye, nose))
        rel2 = distance_2_pts(mouth_left, nose)/max(0.0001,distance_2_pts(mouth_right, nose))
        
        #result frontality score
        rel = abs(1-rel1)+abs(1-rel2)
        frontality_scores.append(rel)
        
        # thresh = 0.2
        # if(abs(1-rel1)>thresh)or(abs(1-rel2)>thresh):
        #     print("non frontal mtcnn skip")
        #     continue
            
        #find center of the face
        center_of_face = _find_center_pt(keypoints)   
        #chech distance to the center. if true assign new index  
        center_distance = distance_2_pts((w/2,h/2),center_of_face)
        #print("points ", w,h,center_of_face, center_distance,l_dist)
            
        # #find face dimention
        # face_dim =_find_face_dim(bounding_box)*face_scale
        face_dim =_find_face_dim_keypoints(keypoints) 
            
        #defining the main face :
        if main_face_criteria == "center" :
            if(l_dist > center_distance):
                main_idx = idx
                l_dist = center_distance            
            

        elif (main_face_criteria == "size" ):
            if(face_dim > face_dim_max):
                main_idx = idx
                face_dim_max = face_dim  
        else:
            sys.exit('incorrect type of main_face_criteria')
        
       
        src_ldms = np.array([
               [left_eye[0], left_eye[1]],
               [right_eye[0], right_eye[1]],
               [nose[0], nose[1]],
               [mouth_left[0], mouth_left[1]],
               [mouth_right[0], mouth_right[1]]], dtype=np.float32 )
        
        
        dst,trgt = _get_gt_insightface_params()

        
        aligned_image  = _align_face_insightface(cv_image, src_ldms, face_size)
       
        #appending aligned image
        output_imgs.append(aligned_image)
        
        
        #write to the output
        output_ldms.append([left_eye,right_eye,nose,mouth_left, mouth_right])   
        #print("left_eye",left_eye, left_eye[0],left_eye[1], len(left_eye))
        idx = idx+1
        face_len = face_len +1
        
    #if no faces detected ruth the spacial handler_method
   
    if face_len<1:
          if_detected = False
          print("empty image ")
          output_imgs.append(_no_detected_face(cv_image, crop_ratio = 0.7, face_size=face_size))
          output_ldms.append([[0,0],[0,0],[0,0],[0,0],[0,0]])
         
    return output_imgs, face_len ,main_idx, output_ldms, frontality_scores


def _read_image_2_cvMat(imagepath):
    image = cv2.cvtColor(cv2.imread(imagepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image

def test_alignment():
    src_ldms = np.array([
       [312.0, 354.0],
       [479.0, 348.0],
       [390.0, 456.0],
       [315.0, 545.0],
       [473.0, 536.0]], dtype=np.float32 )
    
    imagepath = "../test_images/1.jpg"
    imagesavepath = "../test_images/1_aligned.jpg"
    src_image = _read_image_2_cvMat(imagepath)
    face_size = (224,224)
    
    dst,trgt = _get_gt_insightface_params()
    print("dst landmarks", dst)
    print("target size", trgt)
    
    aligned_image  = _align_face_insightface(src_image, src_ldms, face_size)
    cv_save_aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(imagesavepath, cv_save_aligned_image)
        
if __name__ == '__main__':
    test_alignment()