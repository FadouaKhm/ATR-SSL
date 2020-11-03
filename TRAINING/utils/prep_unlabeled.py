# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:02:39 2020

@author: user
"""
import cv2, pickle, json
import numpy as np
from utils import arf, preprocess
from utils.read_json_files import *
from collections import namedtuple



def split_l_u(data1,data2,data3,data4, select_data, percentage):
    pct_labels_per_cls = (1/np.sum(select_data))*percentage
    # NOTE: this function assume that train_set is shuffled.
    source =[]
    if select_data[0] == 1:
        u0 = get_ul_per_class(data1,pct_labels_per_cls)
        source.append(u0)
    if select_data[1] == 1:
        u1 = get_ul_per_class(data2,pct_labels_per_cls)
        source.append(u1)
    if select_data[2] == 1:
        u2 = get_ul_per_class(data3,pct_labels_per_cls)
        source.append(u2)
    if select_data[3] == 1:
        u3 = get_ul_per_class(data4,pct_labels_per_cls)
        source.append(u3)
        
    #source = [u0,u1,u2,u3]
    result = {}
    for key in source[0]:
        result[key] = np.concatenate([d[key] for d in source])

    return result

def get_ul_per_class(train_set,pct_labels_per_cls):
    
    rng = np.random.RandomState(1)
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices] 
    
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    
    u_images = []
    u_labels = []
    lbls = []
    for c in classes:
        
        n_labels_per_cls = int(pct_labels_per_cls*np.count_nonzero(labels == c))
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]

        u_images += [c_images[:n_labels_per_cls]]
        lbls += [c_labels[:n_labels_per_cls]]
        #u_labels += [np.zeros_like(c_labels[:n_labels_per_cls]) - 1] # dammy label
        u_labels = lbls
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}
    #lbls =  np.concatenate(lbls, 0)
    return u_train_set

#def split_train_u_val_test_2c(val_rng, selector, pct):
#    val_set = np.load(config["data_path"] + "/GT/{}.npy".format(val_rng), allow_pickle=True).item()
#    train_set_sampled = dict()
#    unlabeled_set = dict()
#    for rng in range(1000, 4000, 500):
#        if rng != val_rng:
#            
#            with open(config["data_path"] + "/GT/features_{}.pickle".format(rng), "rb") as f:
#                features = pickle.load(f)
#            orig_labels = np.array(features["orig_labels"])
#            if train_set_sampled == {}:
#
#                u_train = np.load(config["data_path"] + "/GT/u{}.npy".format(rng), allow_pickle=True).item()
#                unlabeled_set = split_l_u(u_train['Shifted_v'], u_train['Shifted_h'], u_train['Scaled'], u_train['Blurred'], selector, pct)
#                 
#                 
#            
#                train_set = np.load(config["data_path"] + "/GT/{}.npy".format(rng), allow_pickle=True).item()
#                train_set_sampled = {"images": np.concatenate(
#                    [train_set["images"][orig_labels == 0], train_set["images"][orig_labels == 1]], axis=0),
#                                     "labels": np.concatenate(
#                                         [train_set["labels"][orig_labels == 0], train_set["labels"][orig_labels == 1]],
#                                         axis=0)}
#                for i in range(2, 10):
#                    train_set_sampled["images"] = np.concatenate(
#                        [train_set_sampled["images"], train_set["images"][orig_labels == i][::4]], axis=0)
#                    train_set_sampled["labels"] = np.concatenate(
#                        [train_set_sampled["labels"], train_set["labels"][orig_labels == i][::4]], axis=0)
#            else:
#                train_set = np.load(config["data_path"] + "/GT/{}.npy".format(rng), allow_pickle=True).item()
#                train_set_sampled["images"] = np.concatenate([train_set_sampled["images"], np.concatenate(
#                    [train_set["images"][orig_labels == 0], train_set["images"][orig_labels == 1]], axis=0)], axis=0)
#                train_set_sampled["labels"] = np.concatenate([train_set_sampled["labels"], np.concatenate(
#                    [train_set["labels"][orig_labels == 0], train_set["labels"][orig_labels == 1]], axis=0)], axis=0)
#                for i in range(2, 10):
#                    train_set_sampled["images"] = np.concatenate(
#                        [train_set_sampled["images"], train_set["images"][orig_labels == i][::4]], axis=0)
#                    train_set_sampled["labels"] = np.concatenate(
#                        [train_set_sampled["labels"], train_set["labels"][orig_labels == i][::4]], axis=0)
#
#                u_train = np.load(config["data_path"] + "/GT/u{}.npy".format(rng), allow_pickle=True).item()
#                u_train_set = split_l_u(u_train['Shifted_v'], u_train['Shifted_h'], u_train['Scaled'], u_train['Blurred'], selector, pct)
#                unlabeled_set["images"] = np.concatenate((unlabeled_set["images"], u_train_set["images"]),axis=0)
#                unlabeled_set["labels"] = np.concatenate((unlabeled_set["labels"], u_train_set["labels"]),axis=0)
#                    
#
#    rng = np.random.RandomState(1)
#    indices = rng.permutation(len(train_set_sampled["images"]))
#    train_set_sampled["images"] = train_set_sampled["images"][indices]
#    train_set_sampled["labels"] = train_set_sampled["labels"][indices]
#
#    with open(config["data_path"]+"/GT/features_{}.pickle".format(val_rng), "rb") as f:
#        features = pickle.load(f)
#    masks = [np.array(features["aspects"]) == 1, np.array(features["aspects"]) == 3, np.array(features["aspects"]) == 5, np.array(features["aspects"]) == 7]
#    total_mask = masks[0] + masks[1] + masks[2] + masks[3]
#    validation_set = {"images": val_set["images"][total_mask], "labels": val_set["labels"][total_mask]}
#    test_set = {"images": val_set["images"][~total_mask], "labels": val_set["labels"][~total_mask]}
#    
#
#    
##    Udata = dict()
##    Udata['images'] = np.concatenate((unlabeled_set['images'][mask0,:,:,:],unlabeled_set['images'][mask1,:,:,:][0::4,:,:,:]),axis=0)
##    Udata['labels'] = np.concatenate((unlabeled_set['labels'][mask0],unlabeled_set['labels'][mask1][0::4]),axis=0)
#    Udata = sample_set(unlabeled_set, 4)
#    v_set = sample_set(validation_set, 4)
#
#    return train_set_sampled, v_set, test_set, Udata

def split_train_u_val_test_2c(val_rng, selector, pct, GT):
    val_set = np.load(config["data_path"] + "/" +  GT + "/{}.npy".format(val_rng), allow_pickle=True).item()
    train_set_sampled = dict()
    unlabeled_set = dict()
    for rng in range(1000, 4000, 500):
        if rng != val_rng:
            
            with open(config["data_path"] + "/" +  GT + "/features_{}.pickle".format(rng), "rb") as f:
                features = pickle.load(f)
            orig_labels = np.array(features["orig_labels"])
            if train_set_sampled == {}:

                u_train = np.load(config["data_path"] + "/" +  GT + "/uI{}.npy".format(rng), allow_pickle=True).item()

#                u_train['Inverted']['labels'] = u_train['Scaled']['labels']
#                u_train['Inverted']['images'] = np.array(u_train['Inverted']['images']).astype('float32')
#                u_train['Inverted']['images'] = np.transpose(u_train['Inverted']['images'],(0,3,1,2))
    
                unlabeled_set = split_l_u(u_train['Shifted_v'], u_train['Shifted_h'], u_train['Inverted'], u_train['Blurred'], selector, pct)
                 
                 
            
                train_set = np.load(config["data_path"] + "/" +  GT + "/{}.npy".format(rng), allow_pickle=True).item()
                train_set_sampled = {"images": np.concatenate(
                    [train_set["images"][orig_labels == 0], train_set["images"][orig_labels == 1]], axis=0),
                                     "labels": np.concatenate(
                                         [train_set["labels"][orig_labels == 0], train_set["labels"][orig_labels == 1]],
                                         axis=0)}
                for i in range(2, 10):
                    train_set_sampled["images"] = np.concatenate(
                        [train_set_sampled["images"], train_set["images"][orig_labels == i][::4]], axis=0)
                    train_set_sampled["labels"] = np.concatenate(
                        [train_set_sampled["labels"], train_set["labels"][orig_labels == i][::4]], axis=0)
            else:
                train_set = np.load(config["data_path"] + "/" +  GT + "/{}.npy".format(rng), allow_pickle=True).item()
                train_set_sampled["images"] = np.concatenate([train_set_sampled["images"], np.concatenate(
                    [train_set["images"][orig_labels == 0], train_set["images"][orig_labels == 1]], axis=0)], axis=0)
                train_set_sampled["labels"] = np.concatenate([train_set_sampled["labels"], np.concatenate(
                    [train_set["labels"][orig_labels == 0], train_set["labels"][orig_labels == 1]], axis=0)], axis=0)
                for i in range(2, 10):
                    train_set_sampled["images"] = np.concatenate(
                        [train_set_sampled["images"], train_set["images"][orig_labels == i][::4]], axis=0)
                    train_set_sampled["labels"] = np.concatenate(
                        [train_set_sampled["labels"], train_set["labels"][orig_labels == i][::4]], axis=0)

                u_train = np.load(config["data_path"] + "/" +  GT + "/u{}.npy".format(rng), allow_pickle=True).item()
                
                u_train['Inverted']['labels'] = u_train['Scaled']['labels']
                u_train['Inverted']['images'] = np.array(u_train['Inverted']['images']).astype('float32')
                u_train['Inverted']['images'] = np.transpose(u_train['Inverted']['images'],(0,3,1,2))
                
                u_train_set = split_l_u(u_train['Shifted_v'], u_train['Shifted_h'], u_train['Scaled'], u_train['Blurred'], selector, pct)
                unlabeled_set["images"] = np.concatenate((unlabeled_set["images"], u_train_set["images"]),axis=0)
                unlabeled_set["labels"] = np.concatenate((unlabeled_set["labels"], u_train_set["labels"]),axis=0)
                    

    rng = np.random.RandomState(1)
    indices = rng.permutation(len(train_set_sampled["images"]))
    train_set_sampled["images"] = train_set_sampled["images"][indices]
    train_set_sampled["labels"] = train_set_sampled["labels"][indices]

    with open(config["data_path"]+"/" +  GT + "/features_{}.pickle".format(val_rng), "rb") as f:
        features = pickle.load(f)
    masks = [np.array(features["aspects"]) == 1, np.array(features["aspects"]) == 3, np.array(features["aspects"]) == 5, np.array(features["aspects"]) == 7]
    total_mask = masks[0] + masks[1] + masks[2] + masks[3]
    validation_set = {"images": val_set["images"][total_mask], "labels": val_set["labels"][total_mask]}
    test_set = {"images": val_set["images"][~total_mask], "labels": val_set["labels"][~total_mask]}
    

    
#    Udata = dict()
#    Udata['images'] = np.concatenate((unlabeled_set['images'][mask0,:,:,:],unlabeled_set['images'][mask1,:,:,:][0::4,:,:,:]),axis=0)
#    Udata['labels'] = np.concatenate((unlabeled_set['labels'][mask0],unlabeled_set['labels'][mask1][0::4]),axis=0)
    Udata = sample_set(unlabeled_set, 4)
    v_set = sample_set(validation_set, 4)

    return train_set_sampled, v_set, test_set, Udata

def sample_set(unlabeled_set, sample_rate):
    
    mask0 = unlabeled_set["labels"] == 0
    mask1 = unlabeled_set["labels"] == 1
    
    Udata = dict()
    Udata['images'] = np.concatenate((unlabeled_set['images'][mask0,:,:,:],unlabeled_set['images'][mask1,:,:,:][0::sample_rate,:,:,:]),axis=0)
    Udata['labels'] = np.concatenate((unlabeled_set['labels'][mask0],unlabeled_set['labels'][mask1][0::sample_rate]),axis=0)
    return Udata

        