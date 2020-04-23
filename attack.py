#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:23:17 2020

@author: skodge
"""

import numpy as np 
from utils import gradient, parallel, unparallel_grad, display_image, get_parameters


def CW_attack_fast (img_0, 
                    mean_cat_attack, cov_cat_attack, pi_cat_attack, mean_grass_attack,cov_grass_attack, pi_grass_attack,
                    mean_cat_defense, cov_cat_defense, pi_cat_defense, mean_grass_defense,cov_grass_defense, pi_grass_defense,
                    original_img, truth, 
                    l=5, target_index=1, stride=8, alpha=0.0001, display_iter=300, title='',
                    preprocessing = [None,None], attack_type = 'blackbox'):
    iter_num=0
    parallel_img_0 =parallel(img_0, stride=stride)
    img_k = img_0
    W_cat, w_cat, w_0_cat = get_parameters(mean_cat_attack,cov_cat_attack, pi_cat_attack)
    W_grass, w_grass, w_0_grass = get_parameters(mean_grass_attack,cov_grass_attack, pi_grass_attack)
        
    while iter_num<300:
        iter_num+=1
        parallel_img_k = parallel(img_k, stride=stride)
        if attack_type =='whitebox' and preprocessing[0]!=None:
            parallel_img_k = preprocessing[0].forward(parallel_img_k)
            parallel_img_0 = preprocessing[0].forward(parallel_img_0)
            
        current_grad = gradient(patch_vec_k = parallel_img_k , 
                                patch_vec_0 = parallel_img_0, 
                                mean_cat=mean_cat_attack, 
                                cov_cat=cov_cat_attack, 
                                pi_cat=pi_cat_attack, 
                                mean_grass=mean_grass_attack,
                                cov_grass=cov_grass_attack, 
                                pi_grass=pi_grass_attack,
                                W_cat=W_cat,
                                w_cat=w_cat,
                                w_0_cat=w_0_cat,
                                W_grass=W_grass,
                                w_grass=w_grass,
                                w_0_grass=w_0_grass,
                                l=l, 
                                target_index=target_index)
        grad = unparallel_grad(current_grad, img_0, stride = stride)
        img_k_1 = np.clip (img_k - alpha * grad, 0, 1) 
        change = np.linalg.norm((img_k_1-img_k)) 
        img_k = img_k_1
        
        if (iter_num)%display_iter==0:
            print("\n")
            display_image(img_perturbed=img_k_1, 
                          mean_cat=mean_cat_defense, 
                          cov_cat=cov_cat_defense, 
                          pi_cat=pi_cat_defense, 
                          mean_grass=mean_grass_defense,
                          cov_grass=cov_grass_defense, 
                          pi_grass=pi_grass_defense,
                          original_img = original_img,
                          truth = truth,
                          title=title+'iter_'+str(iter_num), 
                          stride = stride, 
                          preprocessing=preprocessing[1])
            
            print(' Change:{}'.format(change))
        if  change < 0.001 and stride == 8:
            print("\n\nMax Iteration:" + str(iter_num))
            break
        elif change <0.01 and stride == 1:
            print("\n\nMax Iteration:" + str(iter_num))
            break
    
    return img_k_1

#CW attack function
def CW_attack (img_0, mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, original_img, truth, l=5, target_index=1, stride=8, alpha=0.0001, display_iter=300, title='',
                    preprocessing = None):
    iter_num=0
    img_perturbed_k = np.copy(img_0)
    img_perturbed_k_1 = np.copy(img_0)
    W_cat, w_cat, w_0_cat = get_parameters(mean_cat,cov_cat, pi_cat)
    W_grass, w_grass, w_0_grass = get_parameters(mean_grass,cov_grass, pi_grass)
    while iter_num<300:
        iter_num+=1
        grad = np.zeros_like(img_0)
        for i in range(4,img_0.shape[0]-4,stride): #loop starting form zero to center the output 
            for j in range (4,img_0.shape[1]-4,stride): #loop starting form zero to center the output 
                patch_vec_0=img_0[i-4:i+4,j-4:j+4].reshape((64,1))
                patch_vec_k=img_perturbed_k[i-4:i+4,j-4:j+4].reshape((64,1))
                grad[i-4:i+4,j-4:j+4] += gradient(  patch_vec_k=patch_vec_k, 
                                                    patch_vec_0=patch_vec_0, 
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
                                                    target_index=target_index).reshape((8,8))
        
        img_perturbed_k_1 = np.clip( img_perturbed_k - alpha * grad ,0,1)
        change = np.linalg.norm((img_perturbed_k_1-img_perturbed_k))  
        img_perturbed_k = img_perturbed_k_1
        if (iter_num)%display_iter==0:
            print("\n")
            display_image(img_perturbed=img_perturbed_k, 
                          mean_cat=mean_cat, 
                          cov_cat=cov_cat, 
                          pi_cat=pi_cat, 
                          mean_grass=mean_grass,
                          cov_grass=cov_grass, 
                          pi_grass=pi_grass,
                          original_img = original_img,
                          truth = truth,
                          title=title+'iter_'+str(iter_num), 
                          stride = stride, 
                          preprocessing=preprocessing)
            print(' Change:{}'.format(change))
        if  change < 0.001 and stride == 8:
            break
        elif change <0.01 and stride == 1:
            break
    return img_perturbed_k
