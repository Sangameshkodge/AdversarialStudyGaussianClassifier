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
from defense import Adv_training_data_CW, Adv_training_data_PGD
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
parser.add_argument('--train',              default='PGD',    type=str,   help='PGD for training with PGD CW for CW')
parser.add_argument('--stride',              default=1,    type=int,   help='1 for overlapping case 8 for non overlapping case')
parser.add_argument('--attack_type',         default='blackbox',    type=str,   help='blackbox and whitebox attacks')
global args
args = parser.parse_args()
print(args)

if args.train=='CW':
     Adv_training_data = Adv_training_data_CW
elif args.train=='PGD':
    Adv_training_data = Adv_training_data_PGD
else:
    raise ValueError
#loading the training dataset
train_cat=np.matrix(np.loadtxt('./dataset/train_cat.txt',delimiter=','))
train_grass=np.matrix(np.loadtxt('./dataset/train_grass.txt',delimiter=','))

#loading the test dataset
Y = plt.imread ('./dataset/cat_grass.jpg')/255
truth = plt.imread ('./dataset/truth.png')

#computing the parameters for gaussian classifier (Training)
mean_cat,cov_cat, pi_cat = mean_cov(train_cat,train_grass)
mean_grass,cov_grass, pi_grass = mean_cov(train_grass,train_cat)

Lamda =  np.linspace (0.5,5.5,10)
Alpha =  np.linspace (0.00002, 0.0003,10)

augmented_cat =train_cat
augmented_grass = train_grass
#parameters for blackbox attack
mean_cat_defense = mean_cat
cov_cat_defense = cov_cat
pi_cat_defense = pi_cat
mean_grass_defense = mean_grass
cov_grass_defense = cov_grass
pi_grass_defense = pi_grass

for l in Lamda:
    for a in Alpha:
        adv_train_cat   = Adv_training_data(training_data=train_cat,
                                            mean_cat=mean_cat_defense, 
                                            cov_cat=cov_cat_defense,
                                            pi_cat=pi_cat_defense, 
                                            mean_grass=mean_grass_defense,
                                            cov_grass=cov_grass_defense, 
                                            pi_grass=pi_grass_defense, 
                                            l=l, 
                                            target_index=1, 
                                            stride=args.stride, 
                                            alpha=a)
        
        adv_train_grass = Adv_training_data(training_data=train_grass,
                                            mean_cat=mean_cat_defense, 
                                            cov_cat=cov_cat_defense,
                                            pi_cat=pi_cat_defense, 
                                            mean_grass=mean_grass_defense,
                                            cov_grass=cov_grass_defense, 
                                            pi_grass=pi_grass_defense, 
                                            l=l, 
                                            target_index=0, 
                                            stride=args.stride, 
                                            alpha=a)
        augmented_cat = np.concatenate((augmented_cat, adv_train_cat), axis=1)
        augmented_grass = np.concatenate((augmented_grass, adv_train_cat), axis=1)
        mean_cat_defense,cov_cat_defense, pi_cat_defense = mean_cov(augmented_cat,augmented_grass)
        mean_grass_defense,cov_grass_defense, pi_grass_defense = mean_cov(augmented_grass,augmented_cat)
        

if args.attack_type.lower() == 'whitebox':
    mean_cat_attack = mean_cat_defense
    cov_cat_attack = cov_cat_defense
    pi_cat_attack = pi_cat_defense
    mean_grass_attack = mean_grass_defense
    cov_grass_attack = cov_grass_defense
    pi_grass_attack = pi_grass_defense
elif args.attack_type.lower() == 'blackbox':
    print("Inference without Adversarial Training")
    mean_cat_attack = mean_cat
    cov_cat_attack = cov_cat
    pi_cat_attack = pi_cat
    mean_grass_attack = mean_grass
    cov_grass_attack = cov_grass
    pi_grass_attack = pi_grass
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
                  path="./Outputs/Defense/AdversarialTraining/"+args.train+'/'+args.attack_type+'/',
                  save=True, 
                  infer=True) 
else:
    raise ValueError

#Inference
print("Inference with Adversarial Training")
display_image(img_perturbed = Y, 
              mean_cat=mean_cat_defense, 
              cov_cat=cov_cat_defense, 
              pi_cat=pi_cat_defense, 
              mean_grass=mean_grass_defense,
              cov_grass=cov_grass_defense, 
              pi_grass=pi_grass_defense,
              original_img = Y,
              truth = truth,
              title="NonAttackDefenseNonOverlap", 
              stride=args.stride,
              save=True, 
              path="./Outputs/Defense/AdversarialTraining/"+args.train+'/'+args.attack_type+"/",
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
                                      path="./Outputs/Defense/AdversarialTraining/"+args.train+'/'+args.attack_type+"/"
                                    )
    
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
                  path="./Outputs/Defense/AdversarialTraining/"+args.train+'/'+args.attack_type+"/"
              )     

