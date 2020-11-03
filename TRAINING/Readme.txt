##########################################################################################
###############################       OVERVIEW        ####################################
##########################################################################################
This code performs the data extraction and training for MixMAtch semi-supervised model.

##########################################################################################
###############################     Requirements       ###################################
##########################################################################################
torch
torchvision
numpy
pandas
matplotlib
PIL
cv2
sklearn

##########################################################################################
###############################     Quick Start        ###################################
##########################################################################################
You can run the main training script from the command line by simply typing:

train.py  --mode GT --num_cls 2 --train training_files.txt --val validation_files.txt  --alg MM --output Exp0
or
train.py  -m GT -c 2 -t training_files.txt -v validation_files.txt  -a MM -o Exp0

USE CASES:
    1- Train MixMatch, 2 classes with GT, save outputs under exp_1:
        RUN: python train.py  -a MM -m GT -c 2 -t training_files.txt -v validation_files.txt  -o exp_1
    2- Train supervised, 3 classes with BOTH yolo and GT, save outputs under exp_2:
        RUN: python train.py  -a supervised -m BOTH -c 3 -t training_files.txt -v validation_files.txt  -o exp_2
        
        
##########################################################################################
###############################     Description       ####################################
##########################################################################################
The script takes as input:

1. Data extraction mode: This can be selected by setting the option (--mode or -m) to either: 'GT', 'YOLO' or 'BOTH'

2. Number of classes: Set the option --num_cls (or simply -c) to 2 or 3. Remember to adjust the class mapping (ClassMapping) in the config file.

3. List of training files: This should be the path to the txt file containing the training videos. Select it by setting the option --train (or simply -t)

4. List of validation files: This should be the path to the txt file containing the validation videos. Select it by setting the option --val (or simply -v)

5. Training algorithm: Set it up by configuring the option --alg (simply -a). It could be MixMatch -a 'MM' or wResNet28 -a 'supervised'.

6. Output folder: Location where you want your trained model and training logs/plots to be saved. Set it up by setting the option --output (or -o) to 'Experiment_id'

##########################################################################################
########################     Implementation details   ####################################
##########################################################################################

NUMBER OF CLASSES :
------------------
The provided script can train a MixMatch model for a variation of settings depending on the data extraction mode (GT, YOLO or BOTH) and the number of classes.
If you chose 2 classes, it doesn't include the non-target boxes.
If you chose 3 classes, it will include the non target classes.

UNLABELED DATA GENERATION :
--------------------------
All input training videos are used as labeled data. 
To generate the unlabeled data, four augmented sets are generated  from the original tarining set:

The four used augmentations are:
 - Horizontal shifting: ("shift_h_ratio" )
 - Vertical shifting: ("shift_v_ratio" )
 - Scaling: ("scale_ratio")
 - Blurring: ("blurr_filter")

The amount of the included unlabeled data in the training process is selected based on: 
    - A Selector:  [1,2,3,4] where each index refers to one of the augmented sets. If it is set to zero, the set will not be included, and included if not.
    - A percentage: between 0 and 1, Determines the ratio of samples per class to be selected from each of the augmented sets. 
 
 To control the parameters of the unlabeled data generation, set the corresponding parameters in the config file:
 gen_unlabeled_data = {
        "shift_v_ratio" : 0.5, # shift using a random ratio between -0.5 and 0.5
        "shift_h_ratio" : 0.5,
        "blurr_filter" : 0.9,
        "scale_ratio" : 0.2,
        "Selector" : [1,1,1,0],
        "percentage" : 0.5, # percentage per target
        }