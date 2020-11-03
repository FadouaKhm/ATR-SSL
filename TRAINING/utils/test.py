# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:38:58 2020

@author: Fadoua Khmaissia
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse, math, time, json, os
from utils.lib import wrn,wrn64, transform
from utils.train_config import tr_config
from utils.lib import wrn, transform



from config import config

#from prepare_datasets import gen_test_data, gen_shifted_test_data
####
def test(test_set,model_path):
   
    TsAcc = np.array(0)

    
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    dataset_cfg = tr_config['data']
    transform_fn = transform.transform(*dataset_cfg["transform"])

    model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    #model = svhnet.STNSVHNet((32,32), 3, 3, 5, train_config["num_classes"]).cuda()
    #print(dataset_cfg["num_classes"])
    

    model.load_state_dict(torch.load(model_path)) 
    dataset_cfg = tr_config['data']
    test_dataset = dataset_cfg["dataset"](test_set)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        model.eval()
        #print("### test ###")
        sum_acc = 0.

        confusion_matrix = torch.zeros(dataset_cfg["num_classes"],dataset_cfg["num_classes"])
        pr=torch.tensor((), dtype=torch.int64,device=device)
        lab=torch.tensor((), dtype=torch.int64,device=device)
        Fts_test=np.empty([1,128,1,1])
        #conf=np.empty([1,10])
        conf = torch.tensor((), dtype=torch.float32,device=device)
        for j, data in enumerate(test_loader):
            input, target = data
            input, target = input.to(device).float(), target.to(device).long()
            [output, ft] = model(input)
            #print(output.shape, ft.shape)
            Fts_test=np.concatenate((Fts_test,ft.cpu().detach().numpy()),0)
            #conf = np.concatenate((conf,output.cpu().detach().numpy()),0)
            output = torch.reshape(output, (input.shape[0], dataset_cfg["num_classes"]))
            conf=torch.cat((conf, output), 0)
            pred_label = output.max(1)[1]
            pr=torch.cat((pr, pred_label), 0)
            lab=torch.cat((lab, target), 0)
            sum_acc += (pred_label == target).float().sum()
#            for tt, pp in zip(target.view(-1), pred_label.view(-1)):
#                confusion_matrix[tt.long(), pp.long()] += 1
            #if ((j+1) % 10) == 0:
            #   d_p_s = 100/(time.time()-s)
            #    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(
            #           j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s
            #   ), "\r", end="")
            #    s = time.time()
            
        test_acc = sum_acc / float(len(test_dataset))
        TsAcc = np.append(TsAcc,test_acc.item()*100)
        #print("test accuracy : {}".format(test_acc))  

    
    outpts = dict()
    outpts ['test_accuracy'] = test_acc
    outpts ['conf_mat'] = confusion_matrix
    outpts ['targets'] = lab
    outpts ['preds'] = pr
    outpts['TsAccs'] = TsAcc
    outpts['Fts_test']=Fts_test
    #outpts['conf']=np.delete(conf,0,0)
    outpts['conf']=conf
    return outpts

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    