#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:27:06 2021

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  dataLoaders.py
    *
    *
    *  Desc:  This file provides class definitions for data objects.
    *         Specifically, this handles loading the samples and 
    *         corresponding image-level labels. 
    *
    *
    *  Written by:  Connor H. McCurley
    *
    *  Latest Revision:  2021-03-16
    *
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## Pytorch packages
import torch
from torch.utils.data import Dataset

## Packages for super-pixel segmentation
from skimage.segmentation import slic as slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
from skimage.util import random_noise

######################################################################
######################### Class Definitions ##########################
######################################################################
    
#################################################################################
################################# PASCAL Data ###################################
#################################################################################
class loaderPASCAL(Dataset):
    """
    ******************************************************************
        *  Func:    tripletPASCAL()
        *  Desc:    Provides the class definition for PASCALVOC data loader.  
        *           This handles the actual data loading for the file paths defined 
        *           in the training and validation .txt files.
        *
        *  Inputs:
        *           Dataset - 
        *               .txt file of full image paths for data subset including
        *               bag label.
        *
    ******************************************************************
    """ 
    
    def __init__(self, target_classes, bg_classes, data_set_flag, config, img_transform = None, label_transform = None):  #root_dir is a list
        """
        ******************************************************************
            *  Func:    __init__()
            *  Desc:    Initializes class object upon instantiation.
            *  Inputs:    
            *           self - 
            *
            *           target_classes - 
            *               List of classes to be considered as target
            *
            *           bg_classes - 
            *               List of classes to be considered as background
            *
            *           data_set_flag -
            *               str indicating the type of dataset (train, val, test)
            *
            *
            *           config - 
            *               Object of global script parameters.
            *
            *           img_transform - 
            *               Transformation to apply to images (typically conversion
            *               to Tensor).
            *
            *           label_transform - 
            *               Transformation to apply to labels (typically conversion
            *               to Tensor).
            *  Outputs:   
            *           self - 
            *               Set inherent object parameters.
            *
        ******************************************************************
        """   
        
        ## Read file names and labels in data list
        
        ## Only load files with semantic segmentation map
        if(data_set_flag == 'test'):
            target_files, bg_files, all_files = read_files_pascal(config.pascal_data_path, target_classes, bg_classes, 'val')
            
            data_path = config.pascal_data_path.replace('Main','Segmentation')
            target_files, bg_files, all_files = read_test_files_pascal(data_path, target_files, bg_files, 'val')
            
        else:
            target_files, bg_files, all_files = read_files_pascal(config.pascal_data_path, target_classes, bg_classes, data_set_flag)
        
        self.filelist = all_files
        self.target_filelist = target_files

        ## Define object parameters
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.im_size = config.pascal_im_size
        self.parameters = config
    
        
        ## Get path of sample from data list along with corresponding label
        self.train_data_paths = []
        self.gt_data_paths = []
        self.train_bag_labels = []
        self.train_bag_label_set = set()
        
        for file in self.filelist:
            
            ## Load sample path
            datafile = config.pascal_image_path + '/'+ str(file[0]) + '.jpg'
            self.train_data_paths.append(datafile)
            self.train_bag_labels.append(int(file[1]))
            
            ## Add pixel-level groundtruth file
            gt_datafile = config.pascal_gt_path + '/'+ str(file[0]) + '.png'
            self.gt_data_paths.append(gt_datafile)
            
            ## Get bag-level label
            self.train_bag_label_set.add(int(file[1]))
            
        self.label_to_indices = {label: np.where(np.asarray(self.train_bag_labels) == label) for label in self.train_bag_label_set}
    
        
    def __len__(self):  
        """
        ******************************************************************
            *  Func:    __len__()
            *
            *  Desc:    Returns the nummber of samples in the data subset.
            *
            *  Inputs:    
            *           self - 
            *
            *  Outputs: 
            *           Returns number of samples in the data subset.
            *
        ******************************************************************
        """ 
        
        return len(self.filelist)
    
    def __getitem__(self, index): 
        """
        ******************************************************************
            *  Func:    __getitem__()
            *
            *  Desc:    Returns structure of actual image, image label, and
            *           initial confidence map for a sample at a given path.
            *
            *  Inputs:
            *           self -
            *
            *           index - 
            *               Index of desired sample in the full data list.
            *
            *  Outputs:   
            *           triplet set of images (a,+,-), bag-level label list [a,+,-]
            *
        ******************************************************************
        """ 
        ## Get image, bag-level label, and pixel-level groundtruth
        img, label, img_gt = self.train_data_paths[index], self.train_bag_labels[index], self.gt_data_paths[index]
   
        ## Load images and resize
        img = Image.open(img).convert('RGB').resize(self.im_size)
        
        try:
            img_gt = np.array(Image.open(img_gt).resize(self.im_size))
#            img_gt = np.arrImage.open(img_gt).resize(self.im_size))
        except:
            img_gt = np.zeros((self.im_size))
    
        ## Apply image and label transformations (if applicable)
        if self.img_transform is not None:
            img = self.img_transform(img)
#            img_gt = self.img_transform(img_gt)
            
        if self.label_transform is not None:
            label = torch.tensor(label)
      

        return [img,label,img_gt]
    
######################### PASCAL File Reader ##############################
def read_files_pascal(data_path, target_categories, bg_categories, data_set_flag):
    """
    ******************************************************************
        *  Func:    read_files_pascal()
        *
        *  Desc:    Returns list of file aliases according to selected 
        *           category parameters.
        *
        *  Inputs:
        *
        *           data_path -
        *               Full path to PASCAL images
        *
        *           target_categories -
        *               List of categories to be considered as target
        *
        *           bg_categories -
        *               List of categories to be considered as background
        *
        *           data_set_flag - 
        *              str indicating dataset type ('train', 'val' or 'test')
        *           
        *
        *  Outputs:   
        *
        *           target_all - 
        *               List of all file aliases used as targets
        *
        *           bg_all - 
        *               List of all file aliases used as background
        *
        *           all_files - 
        *               List of all file aliases (targets + background)
        *
    ******************************************************************
    """ 
    target_all = []
    bg_all = []
    all_files = []
    
    ## Get target files
    files = os.listdir(data_path)
    for fi in files:
        if('trainval' not in fi):
            num = 1
            for i,str in enumerate(target_categories,1):
                if ((str in fi) and (data_set_flag in fi)):
                
                    f = open(data_path+"/"+fi)
                    iter_f = iter(f)
                    for line in iter_f:
                        if not(line[12] == '-'):
                            line = line[0:11]
                            target_all.append([line,num])
                        
                    break
                    
    ## Remove any target files from background set
    bg_categories = list(set(bg_categories)-set(target_categories))
    
    ## Get background files
    for fi in files:
        if('trainval' not in fi):
            num = 0
            for i,str in enumerate(bg_categories,1):
                if ((str in fi) and (data_set_flag in fi)):
                
                    f = open(data_path+"/"+fi)
                    iter_f = iter(f)
                    for line in iter_f:
                        if not(line[12] == '-'):
                            line = line[0:11]
                            bg_all.append([line,num])
                        
                    break
    
    ## Combine file lists          
    all_files = target_all + bg_all
                    
    return target_all, bg_all, all_files

def read_test_files_pascal(data_path, target_all, bg_all, data_set_flag):
    """
    ******************************************************************
        *  Func:    read_test_files_pascal()
        *
        *  Desc:    Returns list of file aliases according to selected 
        *           category parameters.
        *
        *  Inputs:
        *
        *           gt_data_path -
        *               Full path to PASCAL images
        *
        *           target_all -
        *               L
        *
        *           bg_all -
        *               
        *
        *           data_set_flag - 
        *              str indicating dataset type ('train', 'val' or 'test')
        *           
        *
        *  Outputs:   
        *
        *           target_all - 
        *               List of all file aliases used as targets
        *
        *           bg_all - 
        *               List of all file aliases used as background
        *
        *           all_files - 
        *               List of all file aliases (targets + background)
        *
    ******************************************************************
    """ 
    
    gt_files = []
    target_gt_all = []
    bg_gt_all = []
    all_gt_files = []
    
    ## Get target files
    files = os.listdir(data_path)
    for fi in files:
        if(data_set_flag in fi):
            
            f = open(data_path+"/"+fi)
            iter_f = iter(f)
            for line in iter_f:
                gt_files.append(line[:-1])
                
            break
                    
  
    ## Get target files
    for entry in target_all:
        current_file = entry[0]

        if current_file in gt_files:
            target_gt_all.append(entry)
    
    ## Get bg files
    for entry in bg_all:
        current_file = entry[0]

        if current_file in gt_files:
            bg_gt_all.append(entry)
    
    
    ## Combine file lists          
    all_gt_files = target_gt_all + bg_gt_all
                    
    return target_gt_all, bg_gt_all, all_gt_files   




#################################################################################
################################# DSIAC Data ####################################
#################################################################################
class loaderDSIAC(Dataset):
    """
    ******************************************************************
        *  Func:    tripletPASCAL()
        *  Desc:    Provides the class definition for PASCALVOC data loader.  
        *           This handles the actual data loading for the file paths defined 
        *           in the training and validation .txt files.
        *
        *  Inputs:
        *           Dataset - 
        *               .txt file of full image paths for data subset including
        *               bag label.
        *
    ******************************************************************
    """   
    
    def __init__(self, filelistName, parameters, dataset_flag = 'train', img_transform = None, label_transform = None):  #root_dir is a list
        """
        ******************************************************************
            *  Func:    __init__()
            *  Desc:    Initializes class object upon instantiation.
            *  Inputs:    
            *           self - 
            *
            *           filelistName - 
            *               .txt of paths to individual training files along with label.
            *
            *           config - 
            *               Object of global script parameters.
            *
            *           img_transform - 
            *               Transformation to apply to images (typically conversion
            *               to Tensor).
            *
            *           label_transform - 
            *               Transformation to apply to labels (typically conversion
            *               to Tensor).
            *  Outputs:   
            *           self - 
            *               Set inherent object parameters.
            *
        ******************************************************************
        """         
        self.filelist = []
        
        if (dataset_flag == 'test'):
        
            ## Only test CAMs for target.  (Targets are the only files with GT)
            with open(filelistName) as f:
                for line in f:
                    if int(line[-2]):
                        self.filelist.append(line[:-1])
        
            self.gtfilelist = []
            for line in self.filelist:
                
                    new_filename = line.replace(parameters.dsiac_data_path, parameters.dsiac_gt_path)
                    filename = new_filename.split('patches/')[1][:-6]
                    replace_patches = filename + '_json/label.png'
                    new_filename = new_filename.split('patches/')[0] + replace_patches
                    self.gtfilelist.append(new_filename)
            
        else:
        
            ## Consider all files for train and validate
            with open(filelistName) as f:
                for line in f:
#                    ###########!!!!!!!!! Remove !!!!!!!!###########
#                    if int(line[-2]):
                    self.filelist.append(line[:-1])

        #Down-sample
        ##########!!!!!!!!! Remove !!!!!!!!###########
        self.filelist = self.filelist[0:10]

        self.img_transform = img_transform
        self.label_transform = label_transform
        self.dataset_flag = dataset_flag
        
    def __len__(self):  
        """
        ******************************************************************
            *  Func:    __len__()
            *
            *  Desc:    Returns the nummber of samples in the data subset.
            *
            *  Inputs:    
            *           self - 
            *
            *  Outputs: 
            *           Returns number of samples in the data subset.
            *
        ******************************************************************
        """ 
        
        return len(self.filelist)
    
    
    def __getitem__(self, index): 
        """
        ******************************************************************
            *  Func:    __getitem__()
            *  Desc:    Returns structure of actual image, image label, and
            *           initial confidence map for a sample at a given path.
            *  Inputs:
            *           self -
            *
            *           index - 
            *               Index of desired sample in the full data list.
            *
            *  Outputs:   
            *           sample - 
            *               Dictionary containing actual sample image, 
            *               corresponding image label and initial 
            *               confidence map.
            *
        ******************************************************************
        """ 
 
        
        img = self.filelist[index][:-2]
        label = self.filelist[index][-1]
        
        if (self.dataset_flag == 'test'):
            img_gt = self.gtfilelist[index]
        
        try:
            label = np.array(int(label))
        except ValueError:
            print(f'Label Error -- file:{self.filelist[index]}, datafile:{img}, label:{label}, index:{index}')

        ########################  Normal Way ################################
        try:
            ## Load sample image and corresponding confidence map
            img = Image.open(img).convert('RGB')
        except IOError:
             print(f'Image Error -- file:{self.filelist[index]}, datafile:{img}, label:{label}, index:{index}')
        
        if (self.dataset_flag == 'test'):
            try:
                ## Load sample image and corresponding confidence map
                img_gt = np.array(Image.open(img_gt))
                img_gt[np.where(img_gt != 0)] = 1
            except IOError:
                 print(f'Image Error -- file:{self.filelist[index]}, datafile:{img}, label:{label}, index:{index}')
        else:
            img_gt = np.zeros((510,720))
            

        ## Apply image and label transformations (if applicable)
        if self.img_transform is not None:
            img = self.img_transform(img)
#            img_gt = self.img_transform(img_gt)
            
        if self.label_transform is not None:
            label = torch.tensor(label)

        return [img,label,img_gt]
    





    
 
    
    
    
    