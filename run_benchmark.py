# -*- coding: utf-8 -*-
"""
This script is used for benchmarking.


This code is designed to benchmark face recognition models  and
evaluate various metrics, including MMPMR (Mated Morph Presentation Match Rate),  
ROC (Receiver Operating Characteristic) curves 
and DET (Detection Error Tradeoff) . The benchmarking is done using different
face recognition protocols.

Before running the test on your model - implement the extract_prediction function of 
your model(keep thesame name as the model) in extract_prediction.py 

Example:
python run_benchmark.py --models_path "./models/" --model_name "name_of_your_model" 

  
 
"""



import argparse
import os, sys
import csv
from glob import glob
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, det_curve, accuracy_score
import json
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import extract_prediction
from mf_utils.mf_utils import find_nearest_value


def parse_args():
    parser = argparse.ArgumentParser()
         
    parser.add_argument('-p','--protocol_labels', nargs='+', default=['opencv','opencv_sm','stylegan','stylegan_r','diffae'], help='Labels for protocols to be tested')
    parser.add_argument('-m', '--models_path', default = "./models/", help='Path to models')
    parser.add_argument('-n', '--model_name', default = "test_model" , help='Model name')
    parser.add_argument('-t', '--test_dataset_folder', default = "./data/data_aligned/alignment_default/", help='Path to the data which will be used in testing')
    parser.add_argument('-s', '--protocols_path', default = "./benchmark_protocols/", help='Path of the protocols files')
    parser.add_argument('-l', '--load_11', default = False, help='Binary if loading of 11 results is needed.')
    parser.add_argument('-r', '--load_predictions', default = False, help='Binary if loading of predictions is needed')

    args = parser.parse_args()
    
    return args



#settings for the filenames

dataset_file_name = "dataset.txt"
protocol_11_file_name = "protocol_11.txt"
predictions_11_file_name = "predictions_11.txt"
labels_11_file_name = "labels_11.txt"

predictions_file_name = "predictions.txt"

#protocol properties
protocol_max_select_number = 5

#list of fnmrs for comparison 
FNMR_compare = [0.01,0.001,0.0001,0.00001]

   
def full_test_model(args, 
                    load_11 = False, 
                    load_predictions = False,
                    FNMR_compare = [0.01,0.001,0.0001,0.00001]):
    """
    Core function to perform the full model evaluation. It computes predictions based on the specified protocol 
    and generates metrics like ROC and DET curves. It also computes additional metrics like MMPMR, 
    MinMax MMPMR, and Prod MMPMR.
    
    **Parameters**:
    - `args`: The parsed arguments/configuration.
    - `load_11`: Boolean to indicate whether to load pre-computed 1-1 protocol predictions.
    - `load_predictions`: Boolean to indicate whether to load pre-computed feature vectors.
    - `FNMR_compare`: Values of FNMR to compute and save MMPMR values.
    
    """ 
    

    
    #protocol vars
    dataset_path = args.test_dataset_folder
    protocol_name = args.protocol_name
    protocols_path = args.protocols_path 
    modelname = args.model_name
   
    extract_model_prediction = getattr(extract_prediction, modelname)
 
       
    print("Evaluating %s on %s" %(modelname, protocol_name))
    

    #create protocol folder in the model folder
    if not os.path.exists('%s/%s/%s' %(args.models_path, args.model_name, protocol_name)):
        os.makedirs('%s/%s/%s' %(args.models_path, args.model_name, protocol_name), exist_ok=True)
    
    
    
    #processing 1-1 protocol
    predictions_11 = []
    labels_11 = []
    
    if not load_11:
    
        #load protocol
        protocol_11 = []
        counter_p = 0
        with open("%s/%s/%s" %(protocols_path, protocol_name, protocol_11_file_name)) as f:
            for line in f:
                filepath = str(line)
                filepath = filepath.replace('\n', '')
                elems = filepath.split(",")
                protocol_11.append((elems[0],elems[1],int(elems[2])))
                #print(counter_p, line)
                counter_p += 1   
        f.close()
        #run through the protocol collect predictions #TODO rewrite extracting predictions not to extract them for repeating images

        for i in tqdm(range(len(protocol_11)), desc="Processing 1-1 Protocol"): 
            
            p_11  = protocol_11[i]
            file, file_2 = p_11[0], p_11[1]       

            #extract prediction from the image 1
            file_path = file.replace(os.sep, '/')
            emb_np_norm  = extract_model_prediction('%s/%s' %(dataset_path, file_path), args) 
           

            #extract prediction from the image 2
            file_path_2 = file_2.replace(os.sep, '/')
            emb_np_norm_2 = extract_model_prediction('%s/%s' %(dataset_path, file_path_2), args) 

            #print(file_path, file_path_2)
            similarity_11 = np.dot(emb_np_norm, emb_np_norm_2)
                      
            #add prediction and label to the list
            predictions_11.append(similarity_11)
            labels_11.append(p_11[2])
            
            #print(similarity_11,"gt_label", p_11[2],"stage",len(predictions_11), "of", len(protocol_11))
            
              
        np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, predictions_11_file_name),  np.array(predictions_11),fmt='%10.7f')
        np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, labels_11_file_name),  np.array(labels_11), fmt='%i')

    
    else:
        
        with open("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, predictions_11_file_name)) as f:
            for line in f:
                prediction = str(line)
                prediction = prediction.replace('\n', '')
                prediction = prediction.replace(' ', '')
                predictions_11.append(float(prediction))
        f.close()
        
        with open("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, labels_11_file_name)) as f:
            for line in f:
                filepath = str(line)
                filepath = filepath.replace('\n', '')
                labels_11.append(int(filepath))
        f.close()
            
            
    
    
    
    #compute metrics and get the threshold list
    fpr, tpr, thresholds_list = roc_curve(labels_11, predictions_11, pos_label=1)
    
    
    FMR = fpr
    FNMR = 1-tpr
    
    
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("%s/%s/%s/%s" %(args.models_path,  args.model_name, protocol_name, "ROC_protocol_11.png"))
   

    plt.figure()
    lw = 1
    plt.plot(FMR, FNMR, color='black',lw=lw, label='DET curve ')
    plt.xlim([0.000001, 0.01])
    plt.ylim([0.000001, 0.01])
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non Match Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('DET_' + protocol_name) #+ model_name +"_" 
    plt.legend(loc="lower right")
    #plt.show()
    #save ROC curve with appropriate name
    plt.savefig("%s/%s/%s/%s" %(args.models_path,  args.model_name, protocol_name, "DET_protocol_11.png"))



    """PROCESSING THE MMPMR METRICS """   
    
    #init lists of  labels and predictions
    predictions = []
   
     
    dataset = []
    
    with open("%s/%s/%s" %(protocols_path,protocol_name,dataset_file_name)) as f:
        for line in f:
            filepath = str(line)
            filepath = filepath.replace('\n', '')
            dataset.append(filepath)

              
     
    #Make predictions and write them with the labels into a file 

    if load_predictions:
        try:
            
            predictions = np.loadtxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, predictions_file_name))
            print('predictions are loaded - ', predictions)
            
        except:
            raise Exception("Cant load predictions")
    else:
        
        print("Start making predictions")
        
        for i in tqdm(range(len(dataset)), desc="Extracting predictions for MMPMR metrics"): 

            file  = dataset[i]
                
            file_path = file.replace(os.sep, '/')
            emb_np_norm  = extract_model_prediction('%s/%s' %(dataset_path, file_path), args) 
            
            #add prediction and label to the list
            predictions.append(emb_np_norm)
                               
            #print(emb_np_norm)


       
        predictions = np.array(predictions)
        
        #saving dataset and labels
        np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, dataset_file_name),  np.array(dataset),fmt="%s",)
        np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, predictions_file_name),  np.array(predictions), fmt='%10.7f')
           
        #Final results
        print('Predictions are extracted')
      


    #defining the thresholds list  basing on the 1-1 verification.
    thresholds_array = thresholds_list
    print("threshold array shape",thresholds_array.shape)             
    
    # Opening JSON file
    with open(args.protocols_path+"/"+args.protocol_name+"/mmpmr_protocol_N"+str(protocol_max_select_number)+"_id.json", 'r') as json_file:
        mmpmr_protocol_id = json.load(json_file)
     
    print("lenght of the protocol", len(mmpmr_protocol_id))
    
    
    json_file.close()
   
   
    
    #combine protocol by id
    MMPMR = [] #this is not exactly correct formulation. 
               #Here MMPMR is computed by taking the first element from the each ref_lists.
    MinMax_MMPMR = []
    Prod_MMPMR = []
    
    current_id = 0
    
    print("Start extracting MMPMR metrics")
    
    for i in tqdm(range(len(thresholds_array)), desc="Evaluating MMPMR metricts for threshold values"): 

        thresh = thresholds_array[i]
        MMPMR_current = 0
        MinMax_MMPMR_current = 0
        Prod_MMPMR_current = 0
        
        for mp in mmpmr_protocol_id:
            
            morph =  mp["morph"]
            ref_list_1 = mp["org_1"]
            ref_list_2 = mp["org_2"]

            #print(morph, ref_list_1, ref_list_2) 
           
            #compute MMPMR
            first_sim = np.dot(predictions[morph], predictions[ref_list_1[0]])
            second_sim = np.dot(predictions[morph], predictions[ref_list_2[0]])
            res_S = min(first_sim, second_sim)
            if res_S>thresh:
                MMPMR_current+=1
                
            #Compute MinMax_MMPMR
            sims_1_MinMax = []
            sims_2_MinMax = []
            for r_1 in ref_list_1:
                sims_1_MinMax.append(np.dot(predictions[morph], predictions[r_1]))
            
            for r_2 in ref_list_2:
                sims_2_MinMax.append(np.dot(predictions[morph], predictions[r_2]))
            
            
            first_sim_MinMax = max(sims_1_MinMax)
            second_sim_MinMax = max(sims_2_MinMax)

            res_S_MinMax = min(first_sim_MinMax, second_sim_MinMax)
            if res_S_MinMax>thresh:
                MinMax_MMPMR_current+=1
            
            
            #Compute Prod_MMPMR 
            sims_1_Prod = []
            sims_2_Prod = []
            
            for r_1 in ref_list_1:
                sims_1_Prod.append(np.dot(predictions[morph], predictions[r_1]))
            
            for r_2 in ref_list_2:
                sims_2_Prod.append(np.dot(predictions[morph], predictions[r_2]))
                
             
            first_sim_Prod = (sum(1.0 for x in sims_1_Prod if float(x) > thresh))/len(sims_1_Prod)
            second_sim_Prod = (sum(1.0 for x in sims_2_Prod if float(x) > thresh))/len(sims_2_Prod)
            #print(first_sim_Prod, second_sim_Prod)
            Prod_MMPMR_current += first_sim_Prod*second_sim_Prod
            
  
    
        MMPMR_current = float(MMPMR_current)/len(mmpmr_protocol_id)
        MinMax_MMPMR_current = float(MinMax_MMPMR_current)/len(mmpmr_protocol_id)
        Prod_MMPMR_current = float(Prod_MMPMR_current)/len(mmpmr_protocol_id)   
    
        MMPMR.append(MMPMR_current)
        MinMax_MMPMR.append(MinMax_MMPMR_current)
        Prod_MMPMR.append(Prod_MMPMR_current)
    
    #saving the result
    
    np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, "Thresholds.txt"),  np.array(thresholds_array),fmt='%10.7f')
    np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, "FNMRs.txt"),  np.array(FNMR),fmt='%10.7f')
    np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, "FMRs.txt"),  np.array(FMR),fmt='%10.7f')
    np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, "MMPMR.txt"),  np.array(MMPMR),fmt='%10.7f')
    np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, "MinMax_MMPMR.txt"),  np.array(MinMax_MMPMR),fmt='%10.7f')
    np.savetxt("%s/%s/%s/%s" %(args.models_path, args.model_name, protocol_name, "Prod_MMPMR.txt"),  np.array(Prod_MMPMR),fmt='%10.7f')


    
    # plot MMPMR rates 
    plt.figure()
    lw = 1
    plt.plot(FNMR, MMPMR, color='yellow',lw=lw)
    plt.plot(FNMR, MinMax_MMPMR, color='blue', lw=lw)
    plt.plot(FNMR, Prod_MMPMR, color='red', lw=lw)
    plt.xlim([0.00001, 0.1])
    #plt.ylim([0.0, 0.01])
    plt.xscale('log')
    plt.xlabel('thresholds')
    plt.ylabel('Rate')
    plt.title('MMPMR Rates')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("%s/%s/%s/%s" %(args.models_path,  args.model_name, protocol_name, "Rates.png"))
    
                
    
    # extracting metrics MMPMR @ FNMR
    if len(FNMR_compare)>0 :

        FMR_results = []
        MMPMR_results = []
        MinMax_MMPMR_results = []
        Prod_MMPMR_results = []
        
        for i in range(len(FNMR_compare)):
            
            FNMR_closest, FNMR_closest_index = find_nearest_value(FNMR, FNMR_compare[i])
            
            #print("FMR closest", FMR_closest, FMR_closest_index)
            FMR_results.append(FMR[FNMR_closest_index])
            MMPMR_results.append(MMPMR[FNMR_closest_index])
            MinMax_MMPMR_results.append(MinMax_MMPMR[FNMR_closest_index])
            Prod_MMPMR_results.append(Prod_MMPMR[FNMR_closest_index])

        print("",FNMR_compare)
        print(FMR_results)     
        print(MMPMR_results)  
        print(MinMax_MMPMR_results)  
        print(Prod_MMPMR_results) 
         
        with open('%s/%s/%s/FMR@FNMR.csv' %(args.models_path, args.model_name, protocol_name), mode='w') as FMR_FNMR_file:
                csv_writer = csv.writer(FMR_FNMR_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(FNMR_compare)
                csv_writer.writerow(FMR_results)       
        FMR_FNMR_file.close()
    
        with open('%s/%s/%s/MMPMR@FNMR.csv' %(args.models_path, args.model_name, protocol_name), mode='w') as MMPMR_FNMR_file:
                csv_writer = csv.writer(MMPMR_FNMR_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(FNMR_compare)
                csv_writer.writerow(MMPMR_results)       
        MMPMR_FNMR_file.close()
        
        with open('%s/%s/%s/MinMax_MMPMR@FNMR.csv' %(args.models_path, args.model_name, protocol_name), mode='w') as MinMax_MMPMR_FNMR_file:
                csv_writer = csv.writer(MinMax_MMPMR_FNMR_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(FNMR_compare)
                csv_writer.writerow(MinMax_MMPMR_results)       
        MinMax_MMPMR_FNMR_file.close()
        
        with open('%s/%s/%s/Prod_MMPMR@FNMR.csv' %(args.models_path, args.model_name, protocol_name), mode='w') as Prod_MMPMR_FNMR_file:
                csv_writer = csv.writer(Prod_MMPMR_FNMR_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(FNMR_compare)
                csv_writer.writerow(Prod_MMPMR_results)       
        Prod_MMPMR_FNMR_file.close()
    
    

if __name__ == '__main__':
    
    
    args = parse_args()
    
    
    #combining list of protocols for running experiment
    args.protocol_name_list = []
    for pl in args.protocol_labels:
        args.protocol_name_list.append("protocol_mmpmr_Facing2_"+pl)

    
    for iny, p_name in enumerate(args.protocol_name_list):
        
        #assigning the protocol name into args
        args.protocol_name = args.protocol_name_list[iny]
        
        #creating folder for trhe results of the protocol evaluation
        os.makedirs('%s/%s/%s' %(args.models_path, args.model_name, p_name), exist_ok=True)
        
        #full benchmark
        full_test_model(args, 
                        load_11 = args.load_11, 
                        load_predictions = args.load_predictions,
                        FNMR_compare = FNMR_compare)
        
        

    
   
    
    