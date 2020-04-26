#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:09:27 2020

@author: skodge

"""

import numpy as np 
import math
import matplotlib.pyplot as plt

## Important functions to increase the speed of computation. 
def parallel (img, stride=8):
    #retruns the patch matrix for faster computation 
    dim1= len(range(4,img.shape[0]-4,stride))
    dim2= len(range (4,img.shape[1]-4,stride))
    rangei = range(4,img.shape[0]-4,stride)
    rangej = range (4,img.shape[1]-4,stride)
    img_parallel = np.zeros((dim1*dim2, 64))
    
    for idx_i in range(dim1): #loop starting form zero to center the output 
            for idx_j in range (dim2): #loop starting form zero to center the output 
                i = rangei[idx_i]
                j = rangej[idx_j]
                img_parallel[dim2*idx_i+idx_j] =img[i-4:i+4,j-4:j+4].reshape((64,))
    return img_parallel.T

def unparallel_infer(img_infer, img, stride=8):
    #from the infered parallel patch returns the 2D image
    rangei = range(4,img.shape[0]-4,stride)
    rangej = range (4,img.shape[1]-4,stride)
    dim1=len(rangei)
    dim2=len(rangej)
    img_unparallel = np.zeros_like(img)
    for idx_i in range(dim1): #loop starting form zero to center the output 
            for idx_j in range (dim2): #loop starting form zero to center the output 
                i = rangei[idx_i]
                j = rangej[idx_j]
                if stride ==1:
                    img_unparallel[idx_i, idx_j] += img_infer.T[dim2*idx_i+idx_j] 
                else:
                    img_unparallel[i-4:i+4,j-4:j+4]=img_infer.T[dim2*idx_i+idx_j] 
    return img_unparallel

def unparallel_grad(grad_parallel, img, stride=8):
    #From patch returns stitched gradient
    rangei = range(4,img.shape[0]-4,stride)
    rangej = range (4,img.shape[1]-4,stride)
    dim1=len(rangei)
    dim2=len(rangej)
    grad_unparallel = np.zeros_like(img)
    for idx_i in range(dim1): #loop starting form zero to center the output 
            for idx_j in range (dim2): #loop starting form zero to center the output 
                i = rangei[idx_i]
                j = rangej[idx_j]
                grad_unparallel[i-4:i+4,j-4:j+4] += grad_parallel.T[dim2*idx_i+idx_j].reshape((8,8))
    return grad_unparallel

#Testing the image Y using the gaussian classifier
def inference_fast(img, mean_cat, cov_cat, pi_cat, mean_grass, cov_grass, pi_grass, stride=8):
    output=np.zeros_like(img) # leaving the boundary pixels 0
    parallel_img = parallel(img, stride=stride)
    parallel_infer = np.where(discriminant(parallel_img,mean_cat,cov_cat,pi_cat) >  discriminant(parallel_img,mean_grass,cov_grass,pi_grass), np.ones((parallel_img.shape[1],1)), np.zeros((parallel_img.shape[1],1)))
    output = unparallel_infer(parallel_infer.T, img, stride=stride)
    return output

# Carlini Wagner attack
def gradient_CW(patch_vec_k, patch_vec_0, mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, W_cat, w_cat, w_0_cat, W_grass, w_grass, w_0_grass, l=5, target_index=1):
    g_cat = discriminant(patch_vec_k,mean_cat,cov_cat,pi_cat)
    g_grass = discriminant(patch_vec_k,mean_grass,cov_grass,pi_grass)
    if target_index==0:
        g_target = g_cat
        g_not_target = g_grass
        
        W_target = W_cat
        w_target = w_cat
        W_not_target = W_grass
        w_not_target = w_grass
    
    elif target_index==1: 
        g_target = g_grass
        g_not_target = g_cat
       
        W_not_target = W_cat
        w_not_target = w_cat
        W_target = W_grass
        w_target = w_grass
    else:
        raise ValueError ("Unsupported Target Index {}. Expect it to be 0 or 1 ".format(target_index))
    grad = np.zeros_like(patch_vec_0)
    grad = np.where( (g_target>g_not_target).T, grad, 2*(patch_vec_k-patch_vec_0)+ l*((np.matmul(W_not_target - W_target,patch_vec_k) + w_not_target-w_target)) )
    return grad

def gradient_FGSM(patch_vec_k, patch_vec_0, mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, W_cat, w_cat, w_0_cat, W_grass, w_grass, w_0_grass, target_index=1):
    if target_index==0:
        W_target = W_cat
        w_target = w_cat
        W_not_target = W_grass
        w_not_target = w_grass
    
    elif target_index==1: 
        W_not_target = W_cat
        w_not_target = w_cat
        W_target = W_grass
        w_target = w_grass
    else:
        raise ValueError ("Unsupported Target Index {}. Expect it to be 0 or 1 ".format(target_index))
    grad = np.sign((np.matmul(W_not_target - W_target,patch_vec_k) + w_not_target-w_target))
    return grad

#function that analysis the data and displays the images
def display_image(img_perturbed,  mean_cat, cov_cat, pi_cat, mean_grass,cov_grass, pi_grass, original_img, truth, path="./Outputs/rough/",title='', stride = 8, save=True, infer=False, preprocessing=None):
    
    if preprocessing !=None:
        img_perprocessed = preprocessing.forward(img_perturbed)
    else : 
        img_perprocessed = img_perturbed
        
    img_infer = inference_fast ( img=img_perprocessed, 
                            mean_cat=mean_cat, 
                            cov_cat=cov_cat, 
                            pi_cat=pi_cat, 
                            mean_grass=mean_grass,
                            cov_grass=cov_grass, 
                            pi_grass=pi_grass,
                            stride= stride)
    noise_perturbed = (original_img-img_perturbed)+0.5
    norm = np.linalg.norm(noise_perturbed-0.5)
    cat_count = (img_infer==1).sum()/(stride*stride)
    grass_count = (img_infer==0).sum()/(stride*stride)
    print("Iter:{}\n norm:{} \n # of cat pixels:{} \n # of grass pixels:{}\n Misprediction Count:{}".format(title, norm, cat_count, grass_count, misprediction_count(img_infer,truth)) ) 
    plt.figure(figsize=(5,5))
    plt.title("Perturbed Image")
    if not(infer):
        if save:
            plt.imsave(path+'Perturbed_'+ title + '.png', img_perturbed*255,cmap='gray')
        else:
        
            plt.imshow(img_perturbed*255,cmap='gray', vmin=0, vmax=255)
            plt.show()
    plt.close()
    plt.figure(figsize=(5,5))
    plt.title("perturbation")
    if not(infer):
        if save:
            plt.imsave(path+'noise_'+ title + '.png', noise_perturbed*255,cmap='gray')
        else:
            plt.imshow(noise_perturbed*255,cmap='gray', vmin=0, vmax=255)
            plt.show()
    plt.close()
    plt.figure(figsize=(5,5))
    plt.title("Classifier Output")
    if save:
        plt.imsave(path+'inference_'+ title + '.png', img_infer*255,cmap='gray')
    else:
        plt.imshow(img_infer*255,cmap='gray', vmin=0, vmax=255)
        plt.show()
    plt.close()
    return

#discriminant function fot gaussian classifier given mean, covariance and prior
def discriminant(x,m,c,pi=1,d=64):
    return -0.5*np.sum(np.multiply((x-m).T*np.linalg.pinv(c),(x-m).T), axis=1)- np.log(np.linalg.det(c))/2 +np.log(pi)  
    
#parameter of gaussian classifier 
def get_parameters(m,c,pi=1):
    W = -np.linalg.pinv(c)
    w = np.matmul(np.linalg.pinv(c),m)
    w_0 = -(0.5*np.matmul(np.matmul((m).T, np.linalg.inv(c) ),(m)) + 0.5*math.log(np.linalg.det(c)) - math.log(pi))
    return W, w, w_0

#function computing mean and covariance for gaussian classifier
def mean_cov(A,B):
    return np.mean(A, axis=1), np.cov(A), A.shape[1]/(A.shape[1]+B.shape[1])

#computing loss for gaussian classifier
def MAE(pred, target):
    return    np.linalg.norm(pred-target)/(pred.shape[0]*pred.shape[1])


def misprediction_count(pred, target):
    return    (pred!=target).sum()

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







