#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:16:55 2022

@author: cmccurley
"""

###############################################################################
########################### Import Packages ###################################
###############################################################################

from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import slic, mark_boundaries
from copy import deepcopy

## General packages
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

# PyTorch packages
import torch
import torchvision.transforms as transforms

## Custom packages
import initialize_network
from util import convert_gt_img_to_mask, cam_img_to_seg_img
from dataloaders import loaderPASCAL
from cam_functions import ActivationCAM,OutputScoreCAM, LayerCAM, GradCAM
from cam_functions.utils.image import show_cam_on_image

## General packages
import os
import json
import ctypes
import dask.array as da

# PyTorch packages
import torch
import torchvision
from torch.utils.data import random_split
from torchvision import datasets

torch.manual_seed(24)
torch.set_num_threads(1)


def gen_data_train_two_class(Parameters):
    
    Bags = []
    Labels = []
    
    vgg_idx = [[0,63],[64,191],[192,447],[448,959],[960,1471]]
    
    ## Import parameters 
    parameters = Parameters
            
    transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize(256), 
                                transforms.CenterCrop(224)])           

    if (parameters.DATASET == 'pascal'):
        
        ## Define data transforms
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize(256), 
                                    transforms.CenterCrop(224)])         
    
        target_transform = transforms.Compose([transforms.ToTensor()])
        
#        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
#        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/JPEGImages'
#        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
        'motorbike':14, 'person':15, 'pottedplant':16,
        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
        
        classes = parameters.bg_classes + parameters.target_classes
        
        parameters.NUM_CLASSES = len(classes)
        
        parameters.pascal_im_size = (224,224)
        
        trainset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'train', parameters, transform, target_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        
    nBagsTrain = parameters.nBagsTrain
            
    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    
    model = initialize_network.get_torchray_model(parameters)
 
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
                    
    ######################################################################
    ########################### Compute CAMs #############################
    ######################################################################
    
    norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ## Turn on gradients for CAM computation 
    for param in model.parameters():
        param.requires_grad = True

    model.eval()             
    
    activation_models = dict()
    
    if (parameters.model == 'vgg16'):
        for layer_idx in parameters.layers:
            activation_models[str(layer_idx)] = OutputScoreCAM(model=model, target_layers=[model.features[layer_idx]], use_cuda=parameters.cuda)

    target_category = None
        
    #######################################################################
    ########################### Load results ##############################
    #######################################################################
    
    means_savepath = './importance_means.npy'
    stds_savepath = './importance_stds.npy'
    means = np.load(means_savepath, allow_pickle=True)
    stds = np.load(stds_savepath, allow_pickle=True)
    
    #######################################################################
    ########################### Get Rankings ##############################
    #######################################################################

    
    current_class_vals = means[0,:]
    rank_7_idx_most = (-current_class_vals).argsort()
    rank_7_most = current_class_vals[rank_7_idx_most]
    
    
#    ## Least Important for 1
#    current_class_vals = means[1,:]
#    rank_1_idx_most = (-current_class_vals).argsort()
#    rank_1_most = current_class_vals[rank_1_idx_most]
    
    ## Get ranking in each stage class 7
    stage_idx_list_7 = []
    stage_idx_list_actual_7 = []
    for idk in range(len(vgg_idx)):
        current_class_vals = means[7,:][vgg_idx[idk][0]:vgg_idx[idk][1]]
        current_idx = (-current_class_vals).argsort()
        stage_idx_list_7.append(current_idx)
        if (idk>0):
            current_idx_actual = current_idx + vgg_idx[idk-1][1]+1
            stage_idx_list_actual_7.append(current_idx_actual)
        else:
            stage_idx_list_actual_7.append(current_idx)
    
    
    ###########################################################################
    ############### Extract activation maps and importance weights ############
    ###########################################################################
    
    with tqdm(total=nBagsTrain) as progress_bar:
        
        for tClass in [0,1]:
            
            parameters.mnist_target_class = tClass
            
            print(f'\n!!!! CLASS {str(tClass)} !!!!\n')
            
            most_idx = rank_7_idx_most
            stage_idx_list = stage_idx_list_7
            stage_idx_list_actual = stage_idx_list_actual_7
        
            sample_idx = 0
            for data in train_loader:
                
                images, labels = data[0].to(device), data[1].to(device)
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                ########### Get Activation Maps and Model Outputs per Map #########
                scores_by_stage = dict()
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                    
                    ## Get activation maps
                    activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=0)
                    importance_weights = importance_weights[0,:].numpy()
                    
                    if not(stage_idx):
                        all_activations = activations
                        all_importance_weights = importance_weights
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
                        all_importance_weights = np.append(all_importance_weights, importance_weights, axis=0)
                
                ########### Exctract out the stage we're interested in and invert if desired #########
                
                ## Evaluate IoU of feature maps
                if sample_idx < nBagsTrain:
                    
                    ## Get features from first stage
                    X = all_activations[stage_idx_list_actual[0][0:Parameters.nActivations],:,:]
                    
                    # Flip activations
                    flipped_activations = X
               
                    ## Downsample and add inverted activations
                    for act_idx in range(flipped_activations.shape[0]):
                        
                        flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                        flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
        
                    X = flipped_activations
                    
                    
                    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
                    X = X.T
                    
                    Bags.append(X)
                    if tClass == 1:
                        Labels.append(1)
                    else:
                        Labels.append(0)
                        
                        
                    sample_idx += 1
                    progress_bar.update()
                else:

                    progress_bar.reset()
                    
                    break
    
    allBags = np.zeros((len(Bags),),dtype=np.object)
    allLabels = np.zeros((len(Labels),),dtype=np.uint8)
    
    for idx in range(len(Bags)):
        allBags[idx] = Bags[idx]
        allLabels[idx] = Labels[idx]
    
    return allBags, allLabels


def gen_data_test(Parameters):
    
    Bags = []
    Labels = []
    Images = []
    
    vgg_idx = [[0,63],[64,191],[192,447],[448,959],[960,1471]]
    
    ## Import parameters 
    parameters = Parameters
            
    transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize(256), 
                                transforms.CenterCrop(224)])           

    if (parameters.DATASET == 'pascal'):
        
        ## Define data transforms
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize(256), 
                                    transforms.CenterCrop(224)])         
    
        target_transform = transforms.Compose([transforms.ToTensor()])
        
#        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
#        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/JPEGImages'
#        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
        'motorbike':14, 'person':15, 'pottedplant':16,
        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
        
        classes = parameters.bg_classes + parameters.target_classes
        
        parameters.NUM_CLASSES = len(classes)
        
        parameters.pascal_im_size = (224,224)
        
        trainset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'test', parameters, transform, target_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        
    nBagsTest = parameters.nBagsTest
            
    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    
    model = initialize_network.get_torchray_model(parameters)
 
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
                    
    ######################################################################
    ########################### Compute CAMs #############################
    ######################################################################
    
    norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ## Turn on gradients for CAM computation 
    for param in model.parameters():
        param.requires_grad = True

    model.eval()             
    
    activation_models = dict()
    
    if (parameters.model == 'vgg16'):
        for layer_idx in parameters.layers:
            activation_models[str(layer_idx)] = OutputScoreCAM(model=model, target_layers=[model.features[layer_idx]], use_cuda=parameters.cuda)

    target_category = None
        
    #######################################################################
    ########################### Load results ##############################
    #######################################################################
    
    means_savepath = './importance_means.npy'
    stds_savepath = './importance_stds.npy'
    means = np.load(means_savepath, allow_pickle=True)
    stds = np.load(stds_savepath, allow_pickle=True)
    
    #######################################################################
    ########################### Get Rankings ##############################
    #######################################################################

    
    current_class_vals = means[0,:]
    rank_7_idx_most = (-current_class_vals).argsort()
    rank_7_most = current_class_vals[rank_7_idx_most]
    
    
#    ## Least Important for 1
#    current_class_vals = means[1,:]
#    rank_1_idx_most = (-current_class_vals).argsort()
#    rank_1_most = current_class_vals[rank_1_idx_most]
    
    ## Get ranking in each stage class 7
    stage_idx_list_7 = []
    stage_idx_list_actual_7 = []
    for idk in range(len(vgg_idx)):
        current_class_vals = means[7,:][vgg_idx[idk][0]:vgg_idx[idk][1]]
        current_idx = (-current_class_vals).argsort()
        stage_idx_list_7.append(current_idx)
        if (idk>0):
            current_idx_actual = current_idx + vgg_idx[idk-1][1]+1
            stage_idx_list_actual_7.append(current_idx_actual)
        else:
            stage_idx_list_actual_7.append(current_idx)
    
    
    ###########################################################################
    ############### Extract activation maps and importance weights ############
    ###########################################################################
    
    target_class = 0
    
    with tqdm(total=nBagsTest) as progress_bar:
        
        for tClass in [1]:
            
            parameters.mnist_target_class = tClass
            
            print(f'\n!!!! CLASS {str(tClass)} !!!!\n')
            
            most_idx = rank_7_idx_most
            stage_idx_list = stage_idx_list_7
            stage_idx_list_actual = stage_idx_list_actual_7
        
            sample_idx = 0
            for data in train_loader:
                
                images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
                gt_image_input = images.permute(2,3,1,0)
                gt_image_input = gt_image_input.detach().cpu().numpy()
                gt_image_input = gt_image_input[:,:,:,0]
                
                ########### Get Activation Maps and Model Outputs per Map #########
                scores_by_stage = dict()
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                    
                    ## Get activation maps
                    activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                    importance_weights = importance_weights[0,:].numpy()
                    
                    if not(stage_idx):
                        all_activations = activations
                        all_importance_weights = importance_weights
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
                        all_importance_weights = np.append(all_importance_weights, importance_weights, axis=0)
                
                ########### Exctract out the stage we're interested in and invert if desired #########
                
                ## Evaluate IoU of feature maps
                if sample_idx < nBagsTest:
                    
                    ## Get features from first stage
                    X = all_activations[stage_idx_list_actual[0][0:Parameters.nActivations],:,:]
                    
                    if (Parameters.INCLUDE_INVERTED == 'y'):
                    
                        # Flip activations
                        flipped_activations = X
                   
                        ## Downsample and add inverted activations
                        for act_idx in range(flipped_activations.shape[0]):
                            
                            flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                            flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
            
                        X = flipped_activations
                    
                    
                    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
                    X = X.T
                        
                    Bags.append(X)
                    Labels.append(gt_img)
                    Images.append(gt_image_input)
                       
                    sample_idx += 1
                    progress_bar.update()
                else:

                    progress_bar.reset()
                    
                    break
    
    allBags = np.zeros((len(Bags),),dtype=np.object)
    allLabels = np.zeros((len(Labels),),dtype=np.object)
    allImages = np.zeros((len(Images),),dtype=np.object)
    
    for idx in range(len(Bags)):
        allBags[idx] = Bags[idx]
        allLabels[idx] = Labels[idx]
        allImages[idx] = Images[idx]
    
    return allBags, allLabels, allImages

def gen_data_train_single_class(Parameters):
    
    Bags = []
    Labels = []
    
    vgg_idx = [[0,63],[64,191],[192,447],[448,959],[960,1471]]
    
    ## Import parameters 
    parameters = Parameters
            
    transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize(256), 
                                transforms.CenterCrop(224)])           

    if (parameters.DATASET == 'pascal'):
        
        ## Define data transforms
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize(256), 
                                    transforms.CenterCrop(224)])         
    
        target_transform = transforms.Compose([transforms.ToTensor()])
        
#        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
#        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/JPEGImages'
#        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
        'motorbike':14, 'person':15, 'pottedplant':16,
        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
        
        classes = parameters.bg_classes + parameters.target_classes
        
        parameters.NUM_CLASSES = len(classes)
        
        parameters.pascal_im_size = (224,224)
        
        trainset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'train', parameters, transform, target_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        
    nBagsTrain = parameters.nBagsTrain
            
    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    
    model = initialize_network.get_torchray_model(parameters)
 
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
                    
    ######################################################################
    ########################### Compute CAMs #############################
    ######################################################################
    
    norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ## Turn on gradients for CAM computation 
    for param in model.parameters():
        param.requires_grad = True

    model.eval()             
    
    activation_models = dict()
    
    if (parameters.model == 'vgg16'):
        for layer_idx in parameters.layers:
            activation_models[str(layer_idx)] = OutputScoreCAM(model=model, target_layers=[model.features[layer_idx]], use_cuda=parameters.cuda)

    target_category = None
        
    #######################################################################
    ########################### Load results ##############################
    #######################################################################
    
    means_savepath = './importance_means.npy'
    stds_savepath = './importance_stds.npy'
    means = np.load(means_savepath, allow_pickle=True)
    stds = np.load(stds_savepath, allow_pickle=True)
    
    #######################################################################
    ########################### Get Rankings ##############################
    #######################################################################

    
    current_class_vals = means[0,:]
    rank_7_idx_most = (-current_class_vals).argsort()
    rank_7_most = current_class_vals[rank_7_idx_most]
    
    ## Get ranking in each stage class 7
    stage_idx_list_7 = []
    stage_idx_list_actual_7 = []
    for idk in range(len(vgg_idx)):
        current_class_vals = means[7,:][vgg_idx[idk][0]:vgg_idx[idk][1]]
        current_idx = (-current_class_vals).argsort()
        stage_idx_list_7.append(current_idx)
        if (idk>0):
            current_idx_actual = current_idx + vgg_idx[idk-1][1]+1
            stage_idx_list_actual_7.append(current_idx_actual)
        else:
            stage_idx_list_actual_7.append(current_idx)
    
    
    ###########################################################################
    ############### Extract activation maps and importance weights ############
    ###########################################################################
    
    target_class = 0
    
    with tqdm(total=nBagsTrain) as progress_bar:
        
        for tClass in [1]:
            
            parameters.mnist_target_class = tClass
            
            print(f'\n!!!! CLASS {str(tClass)} !!!!\n')
            
            most_idx = rank_7_idx_most
            stage_idx_list = stage_idx_list_7
            stage_idx_list_actual = stage_idx_list_actual_7
        
            sample_idx = 0
            for data in train_loader:
                
                images, labels = data[0].to(device), data[1].to(device)
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                ########### Get Activation Maps and Model Outputs per Map #########
                scores_by_stage = dict()
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                    
                    ## Get activation maps
                    activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                    importance_weights = importance_weights[0,:].numpy()
                    
                    if not(stage_idx):
                        all_activations = activations
                        all_importance_weights = importance_weights
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
                        all_importance_weights = np.append(all_importance_weights, importance_weights, axis=0)
                
                ########### Exctract out the stage we're interested in and invert if desired #########
                
                ## Evaluate IoU of feature maps
                if sample_idx < nBagsTrain:
                    
                    
#                    path = './results_single_class/sample_' + str(sample_idx)
#                    os.mkdir(path)
#                    
#                    
#                    for idx in range(all_activations.shape[0]):
#                        plt.figure()
#                        plt.imshow(all_activations[idx,:,:])
#                        savepath = path + '/act_' + str(idx) + '.png'
#                        plt.savefig(savepath)
#                        plt.close()
                    
                    ## Get features from first stage
                    X = all_activations[stage_idx_list_actual[0][0:Parameters.nActivations],:,:]
                    
                    path = './results_single_class/sample_' + str(sample_idx)
#                    os.mkdir(path)
#                    
#                    
#                    for idx in range(X.shape[0]):
#                        plt.figure()
#                        plt.imshow(X[idx,:,:])
#                        savepath = path + '/act_' + str(idx) + '.png'
#                        plt.savefig(savepath)
#                        plt.close()
                    
                    if (Parameters.INCLUDE_INVERTED == 'y'):
                    
                        # Flip activations
                        flipped_activations = X
                   
                        ## Downsample and add inverted activations
                        for act_idx in range(flipped_activations.shape[0]):
                            
                            flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                            flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
            
                        X = flipped_activations
                    
                    
                    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
                    X = X.T
                        
                    tBags, tLabels = gen_bags_gradcam(X, images, cam_input, target_class, model, path, parameters)
                    
                    for idx in range(len(tLabels)):
                        Bags.append(tBags[idx])
                        Labels.append(tLabels[idx])
                    
                    sample_idx += 1
                    progress_bar.update()
                else:

                    progress_bar.reset()
                    
                    break
    
    allBags = np.zeros((len(Bags),),dtype=np.object)
    allLabels = np.zeros((len(Labels),),dtype=np.uint8)
    
    for idx in range(len(Bags)):
        allBags[idx] = Bags[idx]
        allLabels[idx] = Labels[idx]
    
    return allBags, allLabels


def gen_bags_gradcam(X, images, cam_input, target_class, model, path, parameters):
    
    Bags = np.zeros((10,),dtype=np.object)
    Labels = np.zeros((10,),dtype=np.uint8)
     
    random_seed = 12
    random.seed(random_seed)
    
    cam_model = GradCAM(model=model, target_layers=[model.features[30]], use_cuda=parameters.cuda)
    grayscale_cam = cam_model(input_tensor=cam_input,target_category=target_class)
    grayscale_cam = grayscale_cam[0, :]
        
    ## Binarize CAM for segmentation
    image = deepcopy(images.permute(2,3,1,0))
    image = image.detach().cpu().numpy()
    image = image[:,:,:,0]
    img = image
    
#    cam_image = show_cam_on_image(img, grayscale_cam, True)
#    plt.figure()
#    plt.imshow(cam_image)
#    plt.axis('off')
#    savepath = path + '/gradcam.png'
#    plt.savefig(savepath)
#    plt.close()
    
    
    sp_img = slic(img,n_segments=300)
    sp_seg_img = np.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
    
    for val in np.unique(sp_img):
        sp_seg_img[np.where(sp_img==val)] = np.mean(grayscale_cam[np.where(sp_img==val)])
    
    ## Binarize CAM 
    thresh = threshold_otsu(sp_seg_img)
    binary_img = cam_img_to_seg_img(sp_seg_img, thresh)
    binary_img = binary_img.reshape((binary_img.shape[0]*binary_img.shape[1]))
          
    neg_coords = np.where(binary_img==0)[0].tolist()
    pos_coords = np.where(binary_img==1)[0].tolist()
     
    ## Make negative bags
    for idk in range(int(len(Bags)/2)):
        bag = np.zeros((parameters.nPntsBags,X.shape[1]))
        
        neg_ind = random.sample(neg_coords,parameters.nPntsBags)
        
        bag = X[neg_ind,:]
        
        Bags[idk] = bag
        Labels[idk] = 0
        
    ## Make positive bags
    for idk in range(int(len(Bags)/2),len(Bags)):
        bag = np.zeros((parameters.nPntsBags,X.shape[1]))
        
        nNeg = int(0.75*parameters.nPntsBags)
        nPos = int(0.25*parameters.nPntsBags)
        
        neg_ind = random.sample(neg_coords,nNeg)
        pos_ind = random.sample(pos_coords,nPos)
      
        bag[0:nNeg,:] = X[neg_ind,:]
        bag[nNeg:,:] = X[pos_ind,:]
      
        Bags[idk] = bag
        Labels[idk] = 1
            
    return Bags, Labels



