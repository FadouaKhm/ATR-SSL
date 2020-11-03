# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:51:34 2020

@author: Khmaissia
"""

import numpy as np

ClassMapping = {'PICKUP':0,'SUV':0,'BTR70':1,'BRDM2':1,'BMP2':1,
                     'T72':1,'ZSU23':1,'2S3':1,'MTLB':1,'D20':1, 'M60':1, 'HMMWV':1, '2S1':1, 'BTR80':1, 'M1114':1}


###################### FEATURE EXTRACTION PARAMETERS #######################

data_config = {
    "root" :  r'D:\ATR Database', 
    "yolo_thresh": 0.3,
    "input_size" : 32,
    "range_limit": 3500,
    "transform" : [True, False, False],
    "batch_size": 128,
}
        
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


config = {

    "data_config" : data_config,
    "invert" : invert_config ,
    "clip1" : {"select" : clip1_config['select'], "threshold" : np.array([clip1_config['lower_thresh'], clip1_config['upper_thresh']])},
    "normalize" : normalize_config,
    "clip2" : {"select" : clip2_config['select'], "lims" : np.array([clip2_config['lower_lim'], clip2_config['upper_lim']])},
    "ClassMapping" : ClassMapping,
}

        
