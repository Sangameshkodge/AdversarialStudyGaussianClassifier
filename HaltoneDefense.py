#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:07:27 2020

@author: skodge
"""

import numpy as np
import matplotlib.pyplot as plt
from attack import CW_attack_fast
from utils import display_image, mean_cov
from defense import Halftone_image, Halftone_patch
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
parser.add_argument('--quantize',            default=True,    type=str2bool,   help='Source Model')
parser.add_argument('--stride',              default=1,    type=int,   help='1 for overlapping case 8 for non overlapping case')
parser.add_argument('--attack_type',         default='blackbox',    type=str,   help='blackbox and whitebox attacks')
global args
args = parser.parse_args()
print(args)


#loading the training dataset
train_cat=np.matrix(np.loadtxt('./dataset/train_cat.txt',delimiter=','))
train_grass=np.matrix(np.loadtxt('./dataset/train_grass.txt',delimiter=','))
                                        
#loading the test dataset
Y = plt.imread ('./dataset/cat_grass.jpg')/255
truth = plt.imread ('./dataset/truth.png')

#computing the parameters for gaussian classifier (Training)
ht = Halftone_patch(quantize=args.quantize)
ht_img = Halftone_image(quantize=args.quantize) 
mean_cat_defense,cov_cat_defense, pi_cat_defense = mean_cov(ht.forward(train_cat),ht.forward(train_grass))
mean_grass_defense,cov_grass_defense, pi_grass_defense = mean_cov(ht.forward(train_grass),ht.forward(train_cat))

if args.attack_type.lower() == 'whitebox':
    mean_cat_attack = mean_cat_defense
    cov_cat_attack = cov_cat_defense
    pi_cat_attack = pi_cat_defense
    mean_grass_attack = mean_grass_defense
    cov_grass_attack = cov_grass_defense
    pi_grass_attack = pi_grass_defense
elif args.attack_type.lower() == 'blackbox':        
    mean_cat_attack,cov_cat_attack, pi_cat_attack = mean_cov(train_cat,train_grass)
    mean_grass_attack,cov_grass_attack, pi_grass_attack = mean_cov(train_grass,train_cat)
    print("Inference without Halftone")
    #Inference
    display_image(img_perturbed = Y, 
                  mean_cat=mean_cat_attack, 
                  cov_cat=cov_cat_attack, 
                  pi_cat=pi_cat_attack, 
                  mean_grass=mean_grass_attack,
                  cov_grass=cov_grass_attack, 
                  pi_grass=pi_grass_attack,
                  original_img = Y,
                  truth = truth,
                  title="NonAttackNonDefense", 
                  stride=args.stride,
                  save=False, 
                  infer=True) 
else:
    raise ValueError

#Inference
print("Inference with Halftone")
display_image(img_perturbed = Y, 
              mean_cat=mean_cat_defense, 
              cov_cat=cov_cat_defense, 
              pi_cat=pi_cat_defense, 
              mean_grass=mean_grass_defense,
              cov_grass=cov_grass_defense, 
              pi_grass=pi_grass_defense,
              original_img = Y,
              truth = truth,
              title="NonAttackDefense", 
              stride=args.stride,
              save=False, 
              infer=True, 
              preprocessing = ht_img)  


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
                                      mean_cat_attack=mean_cat_attack, 
                                      cov_cat_attack=cov_cat_attack, 
                                      pi_cat_attack=pi_cat_attack, 
                                      mean_grass_attack=mean_grass_attack,
                                      cov_grass_attack=cov_grass_attack, 
                                      pi_grass_attack=pi_grass_attack,
                                      mean_cat_defense = mean_cat_defense,
                                      cov_cat_defense = cov_cat_defense,
                                      pi_cat_defense = pi_cat_defense,
                                      mean_grass_defense = mean_grass_defense,
                                      cov_grass_defense = cov_grass_defense,
                                      pi_grass_defense = pi_grass_defense,
                                      original_img = Y,
                                      truth = truth,
                                      l=l, 
                                      alpha=a,
                                      display_iter=disp, 
                                      stride=stride, 
                                      title="lamda_{}_stride_{}_".format(l,stride), 
                                      preprocessing=[ht,ht_img],
                                      attack_type=args.attack_type)
    
    display_image(img_perturbed = img_perturbed,  
                  mean_cat=mean_cat_defense, 
                  cov_cat=cov_cat_defense, 
                  pi_cat=pi_cat_defense, 
                  mean_grass=mean_grass_defense,
                  cov_grass=cov_grass_defense, 
                  pi_grass=pi_grass_defense,
                  original_img = Y,
                  truth = truth,
                  title="lamda_{}_stride_{}_final".format(l,stride), 
                  stride=stride, 
                  save=False,
                  preprocessing=ht_img)     




