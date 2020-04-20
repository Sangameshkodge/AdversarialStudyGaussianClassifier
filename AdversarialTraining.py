#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:07:08 2020

@author: skodge
"""

import numpy as np
import matplotlib.pyplot as plt
from attack import CW_attack_fast
from utils import display_image, mean_cov
from defense import Adv_training_data
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='Defense for CW attack on Gaussian classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_bits',              default=8,      type=int,     help='Number of bits of quantization')
parser.add_argument('--quantize',            default=True,    type=str2bool,   help='Source Model')
parser.add_argument('--stride',              default=1,    type=int,   help='1 for overlapping case 8 for non overlapping case')
global args
args = parser.parse_args()
print(args)


#loading the training dataset
train_cat=np.matrix(np.loadtxt('./dataset/train_cat.txt',delimiter=','))
train_grass=np.matrix(np.loadtxt('./dataset/train_grass.txt',delimiter=','))

#loading the test dataset
Y = plt.imread ('./dataset/cat_grass.jpg')/255
truth = plt.imread ('./dataset/truth.png')/255

#computing the parameters for gaussian classifier (Training)
mean_cat,cov_cat, pi_cat = mean_cov(train_cat,train_grass)
mean_grass,cov_grass, pi_grass = mean_cov(train_grass,train_cat)

Lamda = [0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
Alpha = [0.0001, 0.0002, 0.0003]

augmented_cat =train_cat
augmented_grass = train_grass
for l in Lamda:
    for a in Alpha:
        adv_train_cat   = Adv_training_data(training_data=train_cat,
                                            mean_cat=mean_cat, 
                                            cov_cat=cov_cat,
                                            pi_cat=pi_cat, 
                                            mean_grass=mean_grass,
                                            cov_grass=cov_grass, 
                                            pi_grass=pi_grass, 
                                            l=l, 
                                            target_index=1, 
                                            stride=args.stride, 
                                            alpha=a)
        
        adv_train_grass = Adv_training_data(training_data=train_grass,
                                            mean_cat=mean_cat, 
                                            cov_cat=cov_cat,
                                            pi_cat=pi_cat, 
                                            mean_grass=mean_grass,
                                            cov_grass=cov_grass, 
                                            pi_grass=pi_grass, 
                                            l=l, 
                                            target_index=1, 
                                            stride=args.stride, 
                                            alpha=a)
        augmented_cat = np.concatenate((augmented_cat, adv_train_cat), axis=1)
        augmented_grass = np.concatenate((augmented_grass, adv_train_cat), axis=1)
        mean_cat,cov_cat, pi_cat = mean_cov(augmented_cat,augmented_grass)
        mean_grass,cov_grass, pi_grass = mean_cov(augmented_grass,augmented_cat)
        



#Inference
display_image(img_perturbed = Y, 
              mean_cat=mean_cat, 
              cov_cat=cov_cat, 
              pi_cat=pi_cat, 
              mean_grass=mean_grass,
              cov_grass=cov_grass, 
              pi_grass=pi_grass,
              original_img = Y,
              truth = truth,
              title="NonAttackNonOverlap", 
              stride=args.stride,
              save=False, 
              infer=True)  


# non overlaping
stride = args.stride 

### Analysis for Lamda variation
display = [5]
lam=[1]
alpha=[0.0001]
for i in range(len(display)): 
    l = lam[i]
    disp = display[i] 
    a = alpha[i]
    img_perturbed = CW_attack_fast(   img_0=Y, 
                                      mean_cat=mean_cat, 
                                      cov_cat=cov_cat, 
                                      pi_cat=pi_cat, 
                                      mean_grass=mean_grass,
                                      cov_grass=cov_grass, 
                                      pi_grass=pi_grass,
                                      original_img = Y,
                                      truth = truth,
                                      l=l, 
                                      alpha=a,
                                      display_iter=disp, 
                                      stride=stride, 
                                      title="lamda_{}_stride_{}_".format(l,stride))
    
    display_image(img_perturbed = img_perturbed, 
                  mean_cat=mean_cat, 
                  cov_cat=cov_cat, 
                  pi_cat=pi_cat, 
                  mean_grass=mean_grass,
                  cov_grass=cov_grass, 
                  pi_grass=pi_grass,
                  original_img = Y,
                  truth = truth,
                  title="lamda_{}_stride_{}_final".format(l,stride), 
                  stride=stride, 
                  save=False)     

