##########################################################################################
###############################     Overview       ####################################
##########################################################################################

This code performs testing on a trained model saved at a givel model_path (default setting: model_path = './models/SSL_model_0.pth')

#######################################################################################################################
####################################### Requirements ##################################################################
#######################################################################################################################
torch
torchvision
numpy
pandas
matplotlib
PIL
cv2
sklearn

##############################################################################################
###########################      Quick Start guide     #######################################
##############################################################################################

To test the model, simply execute the code using a command line. For example: 
		>Python test.py -p model_path = './models/MODEL_NAME.pth' -t "./data/testing_files.txt" -m "YOLO" -NT "True"

The code loads the saved model, and extracts the BBs from the list of input videos. It generates the predictions and their 
confdences and saves the output in json declaration files which will be save under './results/MODEL_NAME'

##############################################################################################
################################      Description     ########################################
##############################################################################################

In the config file:
1- Set up your class mapping: ClassMapping
2- Update the root folder path (path to the videos database)
3- Set up the testing range limit : "range_limit": 3500: default
4- Select preprocessing parameters if needed.