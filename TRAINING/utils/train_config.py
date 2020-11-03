"""
Created on Sun Nov 17 15:19:44 2019

@author: Fadoua Khmaissia
"""

from utils.lib.datasets import prep_data
import numpy as np

shared_config = {
    "iteration" : 10000,
    "warmup" : 9000,
    "lr_decay_iter" : 4000,
    "lr_decay_factor" : 0.02,
    "batch_size" : 100,
}
### dataset ###
data_config = {
    "transform" : [True, False, False], # flip, rnd crop, gaussian noise
    "dataset" : prep_data.data,
    "num_classes" : 3,
    "FA_class_id" : 10,
}

### algorithm ###
vat_config = {
    # virtual adversarial training
    "xi" : 1e-6,
    "eps" : {"cifar10":6, "svhn":1},
    "consis_coef" : 0.3,
    "lr" : 3e-3
}
pl_config = {
    # pseudo label
    "threashold" : 0.95,
    "lr" : 3e-4,
    "consis_coef" : 1,
}
mt_config = {
    # mean teacher
    "ema_factor" : 0.95,
    "lr" : 4e-4,
    "consis_coef" : 8,
}
pi_config = {
    # Pi Model
    "lr" : 3e-4,
    "consis_coef" : 20.0,
}
ict_config = {
    # interpolation consistency training
    "ema_factor" : 0.999,
    "lr" : 4e-4,
    "consis_coef" : 100,
    "alpha" : 0.1,
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
tr_config = {
    "shared" : shared_config,
    "data" : data_config,
    "VAT" : vat_config,
    "PL" : pl_config,
    "MT" : mt_config,
    "PI" : pi_config,
    "ICT" : ict_config,
    "MM" : mm_config,
    "supervised" : supervised_config
}
