#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:57:04 2021

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  initialize_network.py
    *
    *  Desc: 
    *
    *  Written by:  
    *
    *  Latest Revision:  
**********************************************************************
"""
######################################################################
######################### Import Packages ############################
######################################################################

## Torchray models
from torchray.benchmark import models

######################################################################
##################### Function Definitions ###########################
######################################################################

def get_torchray_model(parameters):
    
    ## Get pre-trained TorchRay model
    if (parameters.model == 'vgg16') and (parameters.DATASET == 'pascal'):
        model = models.get_model(arch='vgg16', dataset='voc', convert_to_fully_convolutional=False)
        print('Initialized VGG16 Model on PASCAL!')
    elif (parameters.model == 'resnet50') and (parameters.DATASET == 'pascal'):
        model = models.get_model(arch='resnet50', dataset='voc', convert_to_fully_convolutional=False)
        print('Initialized ResNet50 Model on PASCAL!')
    elif (parameters.model == 'vgg16') and (parameters.DATASET == 'coco'):
        model = models.get_model(arch='vgg16', dataset='coco', convert_to_fully_convolutional=False)
        print('Initialized VGG16 Model on COCO!')
    elif (parameters.model == 'resnet50') and (parameters.DATASET == 'coco'):
        model = models.get_model(arch='resnet50', dataset='coco', convert_to_fully_convolutional=False)
        print('Initialized ResNet50 Model on COCO!')
    elif (parameters.model == 'vgg16') and (parameters.DATASET == 'imagenet'):
        model = models.get_model(arch='vgg16', dataset='imagenet', convert_to_fully_convolutional=False)
        print('Initialized VGG16 Model on ImageNet!')    
    elif (parameters.model == 'resnet50') and (parameters.DATASET == 'imagenet'):
        model = models.get_model(arch='resnet50', dataset='imagenet', convert_to_fully_convolutional=False)
        print('Initialized ResNet50 Model on ImageNet!')
    else:
        print('!!!Invalid model parameters!!!')
    
    
    return model

