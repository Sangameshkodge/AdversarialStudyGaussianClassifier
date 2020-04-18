#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:23:17 2020

@author: skodge
"""

import numpy as np 
import math
import matplotlib.pyplot as plt


#function computing mean and covariance for gaussian classifier
def mean_cov(A,B):
    return np.mean(A,axis=1), np.cov(A), A.shape[1]/(A.shape[1]+B.shape[1])


#discriminant function fot gaussian classifier given mean, covariance and prior
def discriminant(x,m,c,pi=1,d=64):
    return -0.5*(x-m).T*np.linalg.pinv(c) *(x-m)- np.log(np.linalg.det(c))/2 +np.log(pi)  

        

#Testing the image Y using the gaussian classifier
def inference(img, mean_cat, cov_cat, pi_cat, mean_grass, cov_grass, pi_grass, stride=8):
    output=np.zeros_like(img) # leaving the boundary pixels 0
    for i in range(4,img.shape[0]-4,stride): #loop starting form zero to center the output 
        for j in range (4,img.shape[1]-4,stride): #loop starting form zero to center the output 
            z=img[i-4:i+4,j-4:j+4].reshape((64,1))
            if (discriminant(z,mean_cat,cov_cat,pi_cat) >  discriminant(z,mean_grass,cov_grass,pi_grass) ):
                if stride==1:
                    output[i,j]=1 
                elif stride==8:
                    output[i-4:i+4 , j-4:j+4]=1
                else:
                    raise ValueError
    return output

#computing loss for gaussian classifier
def MAE(pred, target):
    return    np.linalg.norm(pred-target)/(pred.shape[0]*pred.shape[1])

#parameter of gaussian classifier 
def get_parameters(m,c,pi=1):
    W = -np.linalg.pinv(c)
    w = np.matmul(np.linalg.pinv(c),m)
    w_0 = -(0.5*np.matmul(np.matmul((m).T, np.linalg.inv(c) ),(m)) + 0.5*math.log(np.linalg.det(c)) - math.log(pi))
    return W, w, w_0


# Carlini Wagner attack
def gradient(patch_vec_k, patch_vec_0, mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, l=5, target_index=1):
    g_cat = discriminant(patch_vec_k,mean_cat,cov_cat,pi_cat)
    g_grass = discriminant(patch_vec_k,mean_grass,cov_grass,pi_grass)
    if target_index==0:
        g_target = g_cat
        g_not_target = g_grass
        W_target, w_target, w_0_target = get_parameters(mean_cat,cov_cat, pi_cat)
        W_not_target, w_not_target, w_0_not_target = get_parameters(mean_grass,cov_grass, pi_grass)
    elif target_index==1: 
        g_target = g_grass
        g_not_target = g_cat
        W_target, w_target, w_0_target = get_parameters(mean_grass,cov_grass, pi_grass)
        W_not_target, w_not_target, w_0_not_target = get_parameters( mean_cat,cov_cat, pi_cat)
    else:
        raise ValueError ("Unsupported Target Index {}. Expect it to be 0 or 1 ".format(target_index))
    grad = np.zeros_like(patch_vec_0)
    if g_target>g_not_target:
        return grad
    else:
        grad = 2*(patch_vec_k-patch_vec_0)+ l*(np.matmul(W_not_target - W_target,patch_vec_k) + w_not_target-w_target)
        return grad
    
#function that analysis the data and displays the images
def display_image(img_perturbed,  mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, original_img, truth, title='', stride = 8):
    img_infer = inference ( img=img_perturbed, 
                            mean_cat=mean_cat, 
                            cov_cat=cov_cat, 
                            pi_cat=pi_cat, 
                            mean_grass=mean_grass,
                            cov_grass=cov_grass, 
                            pi_grass=pi_grass,
                            stride= stride)
    noise_perturbed = (original_img-img_perturbed)+0.5
    plt.figure(figsize=(5,5))
    plt.title("Perturbed Image")
    plt.imsave('./Outputs/Perturbed_'+ title + '.png', img_perturbed*255,cmap='gray')
    plt.close()
    plt.figure(figsize=(5,5))
    plt.title("Perturbation")
    plt.imsave('./Outputs/noise_'+ title + '.png', noise_perturbed*255,cmap='gray')
    plt.close()
    plt.figure(figsize=(5,5))
    plt.title("Classifier Output")
    plt.imsave('./Outputs/inference_'+ title + '.png', img_infer*255,cmap='gray')
    plt.close()
    norm = np.linalg.norm(noise_perturbed-0.5)
    cat_count = (img_infer==1).sum()/(stride*stride)
    grass_count = (img_infer==0).sum()/(stride*stride)
    print("\n\nIter:{}\n norm:{} \n # of cat pixels:{} \n # of grass pixels:{}\n MAE Loss:{}".format(title, norm, cat_count, grass_count, MAE(img_infer,truth)) ) 
    return

#CW attack function
def CW_attack (img_0, mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, original_img, truth, l=5, target_index=1, stride=8, alpha=0.0001, display_iter=300, title=''):
    iter_num=0
    img_perturbed_k = np.copy(img_0)
    img_perturbed_k_1 = np.copy(img_0)
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
                                                    l=l, 
                                                    target_index=target_index).reshape((8,8))
        img_perturbed_k_1 = np.clip( img_perturbed_k - alpha * grad ,0,1)
        change = np.linalg.norm((img_perturbed_k_1-img_perturbed_k))  
        img_perturbed_k = img_perturbed_k_1
        if (iter_num)%display_iter==0:
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
                          stride = stride)
            print(' Change:{}'.format(change))
        if  change < 0.001 and stride == 8:
            break
        elif change <0.01 and stride == 1:
            break
    return img_perturbed_k