#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:25:12 2020

@author: skodge
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import display_image, mean_cov
from attack import CW_attack_fast

#loading the training dataset
train_cat=np.matrix(np.loadtxt('./dataset/train_cat.txt',delimiter=','))
train_grass=np.matrix(np.loadtxt('./dataset/train_grass.txt',delimiter=','))

#loading the test dataset
Y = plt.imread ('./dataset/cat_grass.jpg')/255
truth = plt.imread ('./dataset/truth.png')

#computing the parameters for gaussian classifier (Training)
mean_cat,cov_cat, pi_cat = mean_cov(train_cat,train_grass)
mean_grass,cov_grass, pi_grass = mean_cov(train_grass,train_cat)


# non overlaping
stride = 8 


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


### Analysis for Lamda variation
display = [50, 10, 5]
lam=[1, 5, 10]
for i in range(len(display)): 
    l = lam[i]
    disp = display[i] 
    img_perturbed = CW_attack_fast(img_0=Y, 
                                  mean_cat=mean_cat, 
                                  cov_cat=cov_cat, 
                                  pi_cat=pi_cat, 
                                  mean_grass=mean_grass,
                                  cov_grass=cov_grass, 
                                  pi_grass=pi_grass,
                                  mean_cat_infer = mean_cat,
                                  cov_cat_infer = cov_cat,
                                  pi_cat_infer = pi_cat,
                                  mean_grass_infer = mean_grass,
                                  cov_grass_infer = cov_grass,
                                  pi_grass_infer = pi_grass,
                                  original_img = Y,
                                  truth = truth,
                                  l=l, 
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
                  stride=stride)     
# analyze for Alpha variation
display = [50, 10, 5, 5]
alpha=[0.0001, 0.0002, 0.0004, 0.001]
for i in range(len(display)): 
    a = alpha[i]
    disp = display[i]
    
    img_perturbed = CW_attack_fast(img_0=Y, 
                                  mean_cat=mean_cat, 
                                  cov_cat=cov_cat, 
                                  pi_cat=pi_cat, 
                                  mean_grass=mean_grass,
                                  cov_grass=cov_grass, 
                                  pi_grass=pi_grass,
                                  mean_cat_infer = mean_cat,
                                  cov_cat_infer = cov_cat,
                                  pi_cat_infer = pi_cat,
                                  mean_grass_infer = mean_grass,
                                  cov_grass_infer = cov_grass,
                                  pi_grass_infer = pi_grass, 
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
