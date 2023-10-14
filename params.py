#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:59:22 2022

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  params.py
    *  Name:  Connor H. McCurley
    *  Date:  04/29/2022
    *  Desc:  Implementation of the Fusion-CAM algorithm by C. McCurley, 
    *         "Discriminatve Feature Learning with Imprecise, Uncertain, 
    *         and Ambiguous Data," Ph.D Thesis, Gainesville, FL, 2022.
    *
    *         This code defines the parameters for the Fusion-CAM algorithm.
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: 04/29/2022
    *  This product is Copyright (c) 2022 University of Florida
    *  All rights reserved
**********************************************************************
"""
    

import argparse

#######################################################################
########################## Define Parameters ##########################
#######################################################################
def set_parameters(args):
    
    ######################################################################
    ############################ Define Parser ###########################
    ######################################################################
    
    parser = argparse.ArgumentParser(description='Parameters for the Multiple Instance Choquet Integral.')
    
    parser.add_argument('--experiment_name', help='Where to save results.', default='test', type=str)
    parser.add_argument('--measure_type', help='Parameter for full or binary fuzzy measure.', default='binary', type=str) ##full_measure_random_init, full_measure_bfm_init
    parser.add_argument('--num_sources', help='Number of sources in ChI.', default=4, type=int)
    parser.add_argument('--source_indices', help='Sources', nargs='+', default=[1,2,3,4,5], type=int)
    
    ## N Choose K
    parser.add_argument('--n_choose_k', help='Flag for n choose k selection.  n computes a single measure.', default='y', type=str)
    parser.add_argument('--initial_num_sources', help='N in  N choose K selection', default=5, type=int)
    
    ## Init with BFM then learn FFM
    parser.add_argument('--bfm_and_ffm', help='Parameter to run bfm, init search with bfm, then learn ffm', default='y', type=str)
    
    ######################################################################
    ######################### Input Parameters ###########################
    ######################################################################
    
    parser.add_argument('--num_runs', help='How many times to run the optimization process for computing mean and std.', default=20, type=int)
    
    parser.add_argument('--nPop', help='Size of population, preferably even numbers.', default=100, type=int)
    parser.add_argument('--sigma', help='Sigma of Gaussians in fitness function.', default=0.1, type=float)
    parser.add_argument('--maxIterations', help='Max number of iteration.', default=50, type=int)
    parser.add_argument('--fitnessThresh', help='Stopping criteria: when fitness change is less than the threshold.', default=0.0001, type=float)
    parser.add_argument('--eta', help='Percentage of time to make small-scale mutation.', default=0.5, type=float)
    parser.add_argument('--sampleVar', help='Variance around sample mean.', default=0.1, type=float)
    parser.add_argument('--mean', help='Mean of ci in fitness function. Is always set to 1 if the positive label is "1".', default=1.0, type=float)
    parser.add_argument('--analysis', action='store_true', help='If ="1", record all intermediate results.', default=False)
    parser.add_argument('--p', help='p power value for the softmax. p(1) should->Inf, p(2) should->-Inf.', nargs='+', default=[10,-10], type=float)
    parser.add_argument('--use_parallel', action='store_true', help='If ="1", use parallel processing for population sampling.', default=False)
    
    ## Parameters for binary fuzzy measure sampling
    parser.add_argument('--U', help='How many times we are willing to sample a new measure before deeming the search exhaustive.', default=500, type=int)
    parser.add_argument('--Q', help='How many times the best fitness value can remain unchanged before stopping.', default=100, type=int)

    parser.add_argument('--DATASET', help='Dataset selection.', default='pascal', type=str)
    parser.add_argument('--DATABASENAME', help='Relative path to the training data list.', default='', type=str)
 
    parser.add_argument('--target_classes', help='List of target classes.', nargs='+', default=['aeroplane'], type=str)
    parser.add_argument('--bg_classes', help='List of background classes.', nargs='+', default=[], type=str)
    
    ######################################################################
    ##################### Training Hyper-parameters ######################
    ######################################################################
    
    parser.add_argument('--starting_parameters', help='Initial parameters when training from scratch. (0 uses pre-training on ImageNet.)', default='./model_eoe_80.pth', type=str)
    parser.add_argument('--model', help='Neural network model', default='vgg16', type=str)
    parser.add_argument('--BATCH_SIZE', help='Input batch size for training.', default=1, type=int)
    parser.add_argument('--parameter_path', help='Where to save/load weights after each epoch of training', default='/trainTemp.pth', type=str)
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training.', default=False)
    parser.add_argument('--cuda', help='Enable CUDA training.', default=True)
    
    parser.add_argument('--fitness', help='Fitness function to use in genetic algorithm optimization.', default='genmean', type=str)
    parser.add_argument('--CAM_SEG_THRESH', help='Value for binary threshold.', default=0.3, type=float)
    
    parser.add_argument('--layers', help='Layers to compute CAM at.', nargs='+', default=[4], type=int)
    parser.add_argument('--nBagsTrain', help='Input batch size for training.', default=4, type=int)
    parser.add_argument('--nBagsTest', help='Input batch size for testing.', default=4, type=int)
    parser.add_argument('--onlyTarget', help='', default='y', type=str)
    parser.add_argument('--nPntsBags', help='Number of points per bag.', default=1000, type=int)
    parser.add_argument('--nActivations', help='Number of activations to pull from Grad-CAM ranking', default=20, type=int)
    parser.add_argument('--INCLUDE_INVERTED', help='Parameter for including inverted activation features.', default='n', type=str)
    parser.add_argument('--SELECTION_APPROACH', help='Approach for feature selection.', default='iou', type=str)
    

    return parser.parse_args(args)



                  

