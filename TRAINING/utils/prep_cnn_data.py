# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:51:34 2020

@author: baili, Khmaissia
"""

import os
import pandas as pd
import numpy as np
from config import config
#from utils.utils import *
import pickle
from utils.extract_features import *
import pandas as pd
from utils.test import test
from collections import Counter
##############################################################################
    
def prepare_data(txt_file_path, OptRange, extraction_mode):
    
    set_id = txt_file_path.split('.')[0]
    ClassMapping = config['ClassMapping']

    # _FEAT_DIR = './features/'+'/Feature_Vectors/'+set_id
    # if not os.path.exists('./features/'+'/Feature_Vectors/' ):
    #     os.mkdir('./features/'+'/Feature_Vectors/')

    # if not os.path.exists(_FEAT_DIR ):
    #     os.mkdir(_FEAT_DIR)
    
    training_files = []
    training_sampling = []
    with open(txt_file_path, 'r') as f:
        for line in f:
            training_files.append(line.rstrip('\n').split(",")[0])
            training_sampling.append(int(line.rstrip('\n').split(",")[1]))    

    train_set = dict()
    ul_set = dict()
    for i in range(len(training_files)):
        if extraction_mode == "YOLO":
            print("*** Extracting data from Yolo detections from video " +str(i+1) + "...")
            data_set, unlabeled, _, _ = extract_udata_yolo([training_files[i]], [training_sampling[i]], OptRange, shifted_data=False, expanded_data=False)  
        elif extraction_mode == "GT":
            print("*** Extracting data from ground truth boxes from video " +str(i+1) + "...")
            data_set, unlabeled, _, _ = extract_udata_GT([training_files[i]], [training_sampling[i]], OptRange, shifted_data=False, expanded_data=False)  
        elif extraction_mode == "BOTH":
            data_set, unlabeled, _, _ = extract_udata_both([training_files[i]], [training_sampling[i]], OptRange, shifted_data=False, expanded_data=False)    
        else:
            print("*** Make sure to set up extraction mode: YOLO, GT or BOTH ...")
      
        if train_set == {}:
            train_set = data_set
            ul_set = unlabeled
        else:
            train_set["images"] = np.concatenate((train_set["images"], data_set["images"]),axis=0)
            train_set["labels"] = np.concatenate((train_set["labels"], data_set["labels"]),axis=0)

            ul_set["images"] = np.concatenate((ul_set["images"], unlabeled["images"]),axis=0)
            ul_set["labels"] = np.concatenate((ul_set["labels"], unlabeled["labels"]),axis=0)



    ul_set['labels'] = np.zeros(len(ul_set['labels']), dtype=int) -1
         
    
    # _DATA_DIR = './features/'+'/CNN_data'
    # if not os.path.exists(_DATA_DIR ):
    #     os.mkdir(_DATA_DIR)
   
    return train_set, ul_set


