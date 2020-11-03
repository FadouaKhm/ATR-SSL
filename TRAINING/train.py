# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:27:41 2020

@author: Khmaissia
"""
from utils.training_code import training_code
import argparse
import os
import pickle
from utils.prep_cnn_data import prepare_data
import numpy as np
from config import config

"""
Main training script. you can run from the command line by:
train.py  --mode GT --num_cls 2 --train training_files.txt --val validation_files.txt  --alg MM --output Exp0
or
train.py  -m GT -c 2 -t training_files.txt -v validation_files.txt  -a MM -o Exp0

USE CASES:
    1- Train MixMatch, 2 classes with GT, save outputs under exp_1:
        RUN: python train.py  -a MM -m GT -c 2 -t training_files.txt -v validation_files.txt  -o exp_1
    2- Train supervised, 3 classes with BOTH yolo and GT, save outputs under exp_2:
        RUN: python train.py  -a supervised -m BOTH -c 3 -t training_files.txt -v validation_files.txt  -o exp_2
        
        

"""

def main(args):
    print('EXCTRAING TRAINING FRAMES .. \n')
    l_train_set, u_train_set = prepare_data(args.train, config['HasRange'], args.mode)
    print('TRAINING FRAMES EXTRACTED \n')

    print('EXCTRAING VALIDATION FRAMES .. \n')
    val_set, _ = prepare_data(args.val, config['HasRange'], 'YOLO')
    print('VALIDATION FRAMES EXTRACTED \n')
    
    print('TRAINING .. \n')
    training_code(args.alg, l_train_set, val_set, u_train_set, args.NonTarget, args.output)
    


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", default='training_files.txt' , type=str, help="List of training videos .txt file path")
    parser.add_argument("--val", "-v", default='validation_files.txt' , type=str, help="List of validation videos .txt file path")
    parser.add_argument("--alg", "-a", default="MM", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
    parser.add_argument("--NonTarget", "-NT", default=False, type=int, help="Whether or not to include a Non Target class")
    parser.add_argument("--mode", "-m", default='GT', type=str, help="Extraction mode: GT, YOLO or BOTH")
    parser.add_argument("--output", "-o", default="Exp0", type=str, help="Output folder name")
    
    
    args = parser.parse_args()
    
    main(args)
