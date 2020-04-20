#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:17:41 2020

@author: skodge
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import mean_cov, display_image

#loading the training dataset
train_cat=np.matrix(np.loadtxt('./dataset/train_cat.txt',delimiter=','))
train_grass=np.matrix(np.loadtxt('./dataset/train_grass.txt',delimiter=','))

#loading the test dataset
Y = plt.imread ('./dataset/cat_grass.jpg')/255
truth = plt.imread ('./dataset/truth.png')/255

#computing the parameters for gaussian classifier (Training)
mean_cat,cov_cat, pi_cat = mean_cov(train_cat,train_grass)
mean_grass,cov_grass, pi_grass = mean_cov(train_grass,train_cat)


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
              stride=8,
              infer=True)  

display_image(img_perturbed = Y, 
              mean_cat=mean_cat, 
              cov_cat=cov_cat, 
              pi_cat=pi_cat, 
              mean_grass=mean_grass,
              cov_grass=cov_grass, 
              pi_grass=pi_grass,
              original_img = Y,
              truth = truth,
              title="NonAttackOverlap", 
              stride=1,
              infer=True)  
