#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:08:13 2020

@author: skodge
"""
import numpy as np 
from utils import gradient, get_parameters

class Quantize():
    def __init__(self, n_bits=1, quantize=True):
        assert n_bits <= 8.0
        self.n_bits = n_bits
        self.quantize=quantize
    
    def forward(self, data):
        if self.quantize:
            data = np.float_(np.int_(data*255.0))
            bin_width = 255.0/2**(self.n_bits)
            data=np.where(data==255.0,data-0.5 * bin_width * np.ones_like(data), data)
            data = np.floor( data/bin_width ) * bin_width + 0.5 * bin_width * np.ones_like(data)
            data = data / 255.0
        return np.matrix(data)


    
def Adv_training_data(training_data, mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, l=5, target_index=1, stride=8, alpha=0.0001):
    perturbed_data_k = training_data 
    W_cat, w_cat, w_0_cat = get_parameters(mean_cat,cov_cat, pi_cat)
    W_grass, w_grass, w_0_grass = get_parameters(mean_cat,cov_cat, pi_cat)
    for i in range(300):
        current_grad = gradient(patch_vec_k= perturbed_data_k,
                                patch_vec_0=training_data,
                                mean_cat=mean_cat, 
                                cov_cat=cov_cat, 
                                pi_cat=pi_cat, 
                                mean_grass=mean_grass,
                                cov_grass=cov_grass, 
                                pi_grass=pi_grass,
                                W_cat=W_cat,
                                w_cat=w_cat,
                                w_0_cat=w_0_cat,
                                W_grass=W_grass,
                                w_grass=w_grass,
                                w_0_grass=w_0_grass,
                                l=l, 
                                target_index=target_index)
        perturbed_data_k_1 = np.clip(perturbed_data_k - alpha * current_grad,0,1)
        change = np.linalg.norm((perturbed_data_k_1-perturbed_data_k))  
        perturbed_data_k = perturbed_data_k_1
        if  change < 0.001/(2850) and stride == 8:
            break
        elif change <0.01/(180564) and stride == 1:
            break
    return perturbed_data_k_1