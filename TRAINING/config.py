# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:51:34 2020

@author: Khmaissia
"""

import numpy as np
import os
from utils import prep_data

############################ DEFINE CONTEXT ###############################
#define_ranges = {1000 : ['cegr01923*.arf','cegr02003*.arf'],
#                 1500 : ['cegr01925*.arf','cegr02005*.arf'],
#                 2000 : ['cegr01927*.arf','cegr02007*.arf'],
#                 2500 : ['cegr01929*.arf','cegr02009*.arf'],
#                 3000 : ['cegr01931*.arf','cegr02011*.arf'],
#                 3500 : ['cegr01933*.arf','cegr02013*.arf'],
#                 4000 : ['cegr01935*.arf','cegr02015*.arf'],
#                 4500 : ['cegr01937*.arf','cegr02017*.arf'],
#                 5000 : ['cegr01939*.arf','cegr02019*.arf']
#}  

define_ranges = {'R1' : 650,
                 'R2' : 750,
                 'R3' : 950,
                 'R4' : 1325,
                 'R5' : 1875} 
 
RangesList = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

ClassMapping = {'PICKUP':0,'SUV':0,'BTR70':1,'BRDM2':1,'BMP2':1,
                     'T72':1,'ZSU23':1,'2S3':1,'MTLB':1,'D20':1, 'M60':1, 'HMMWV':1, '2S1':1, 'BTR80':1, 'M1114':1}


###################### FEATURE EXTRACTION PARAMETERS #######################

data_config = {
    "root" :  r'D:\ATR Database',
    #"extraction_method" : "GT", 
    "yolo_thresh": 0.3,
    "projection_model_path" : r'.\models\TSNE_Proj_Model' ,
    "input_size" : 32,
    "expand_BB" : 0.1,
}
        
gen_unlabeled_data = {
        "shift_v_ratio" : 0.5, # shift using a random ratio between -0.5 and 0.5
        "shift_h_ratio" : 0.5,
        "blurr_filter" : 0.9,
        "scale_ratio" : 0.2,
        "Selector" : [1,1,1,0],
        "percentage" : 0.5, # percentage per target
        }
### pre-processing algorithms ###
invert_config = {
    "select" : 0,
}

clip1_config = {
    "select" : 0, 
    "lower_thresh" : 0.1,
    "upper_thresh" : 99.9,
}

normalize_config = {
    "select" : 0,
    "technique" : 3,
    #opts=1 ==> mean, std per frame 
    #opts=2 ==> mean, std per pixel (using mean and std images computed across all training set) 
    #opts=3 ==> med, mad normalization per frame 
    #opts=4 ==> med, mad normalization per pixe (using med and mad frames computed across all training set)
    "averaged_frames" : dict(),
}

clip2_config = {
    "select" : 0, 
    "lower_lim" : -5,
    "upper_lim" : 10,
}

mm_config = {
    # mixmatch
    "lr" : 3e-3,
    "consis_coef" : 100,
    "alpha" : 0.75,
    "T" : 0.5,
    "K" : 2,
}
supervised_config = {
    "lr" : 4e-5
}
### master ###

train_config = {
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : prep_data.data,
    #"num_classes" : 3,
    "iteration" : 5000,
    "warmup" : 2000,
    "lr_decay_iter" : 4000,
    "lr_decay_factor" : 0.02,
    "batch_size_tr" : 100,
    "batch_size" : 128,
    "lr" : 4e-4,
    "expand_BB" : 0.1,
    "MM": mm_config,
    "supervised": supervised_config,
}

config = {
    "HasRange" : True,
    "data_config" : data_config,
    "train_config" : train_config,
    "invert" : invert_config ,
    "clip1" : {"select" : clip1_config['select'], "threshold" : np.array([clip1_config['lower_thresh'], clip1_config['upper_thresh']])},
    "normalize" : normalize_config,
    "clip2" : {"select" : clip2_config['select'], "lims" : np.array([clip2_config['lower_lim'], clip2_config['upper_lim']])},
    "exp_id" : 'SSL_MixMatch',
    "def_ranges" : define_ranges,
    #"def_day_night" : define_day_night,
    "save_images" : False, # or False save target patches under ./features/target_images/
    "gen_unlabeled" : gen_unlabeled_data,
    "model_path" : "./models/SSL_MixMatch",
    "data_path" : "./exp_res/exp/Data",
    "ClassMapping" : ClassMapping,
    "marker_sizes" : [ 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70 , 80, 90, 100, 110, 120],
    "markers_list" : 'o+v*><38phX|_DPsxo+v*><',
    "Ranges" : RangesList
}


if not os.path.exists(r'./models/'+config['exp_id']):
    os.mkdir(r'./models/'+config['exp_id'])
    
if not os.path.exists("./features/target_images"):
        os.mkdir("./features/target_images")
        
