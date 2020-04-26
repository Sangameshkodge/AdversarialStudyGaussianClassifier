#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:11:46 2020

@author: skodge
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import  display_image, mean_cov
from attack import PGD_attack_fast

#loading the training dataset
train_cat=np.matrix(np.loadtxt('./dataset/train_cat.txt',delimiter=','))
train_grass=np.matrix(np.loadtxt('./dataset/train_grass.txt',delimiter=','))

#loading the test dataset
Y = plt.imread ('./dataset/cat_grass.jpg')/255
truth = plt.imread ('./dataset/truth.png')

#computing the parameters for gaussian classifier (Training)
mean_cat,cov_cat, pi_cat = mean_cov(train_cat,train_grass)
mean_grass,cov_grass, pi_grass = mean_cov(train_grass,train_cat)

#overlapping
stride = 1

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
              stride= stride,
              save=False, 
              infer=True)  


### Analysis for Alpha variation
display = [5, 5, 2]
alpha=[0.0001, 0.0002, 0.0004] 
for i in range(len(display)): 
    a = alpha[i]
    disp = display[i]
    
    img_perturbed = PGD_attack_fast(img_0=Y, 
                                  mean_cat_attack=mean_cat, 
                                  cov_cat_attack=cov_cat, 
                                  pi_cat_attack=pi_cat, 
                                  mean_grass_attack=mean_grass,
                                  cov_grass_attack=cov_grass, 
                                  pi_grass_attack=pi_grass,
                                  mean_cat_defense = mean_cat,
                                  cov_cat_defense = cov_cat,
                                  pi_cat_defense = pi_cat,
                                  mean_grass_defense = mean_grass,
                                  cov_grass_defense = cov_grass,
                                  pi_grass_defense = pi_grass,
                                  path = './Outputs/rough/',
                                  original_img = Y,
                                  truth = truth,
                                  alpha=a, 
                                  display_iter=disp, 
                                  stride=stride, 
                                  title="alpha_{}_stride_{}_".format(a,stride))
    
    display_image(img_perturbed = img_perturbed, 
                  mean_cat=mean_cat, 
                  cov_cat=cov_cat, 
                  pi_cat=pi_cat, 
                  mean_grass=mean_grass,
                  cov_grass=cov_grass, 
                  pi_grass=pi_grass,  
                  original_img = Y,
                  truth = truth,
                  title="alpha_{}_stride_{}_final".format(a,stride), 
                  stride=stride)   
