#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:10:53 2022

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  mil_source_selection.py
    *  Name:  Connor H. McCurley
    *  Date:  
    *  Desc:  Implementation of the Fusion-CAM algorithm by C. McCurley, 
    *         "Discriminatve Feature Learning with Imprecise, Uncertain, 
    *         and Ambiguous Data," Ph.D Thesis, Gainesville, FL, 2022.
    *
    *         This code defines the source selection procedures.
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: 
    *  This product is Copyright (c) 2022 University of Florida
    *  All rights reserved
**********************************************************************
"""


"""
%=====================================================================
%================= Import Settings and Packages ======================
%=====================================================================
"""

######################################################################
######################### Import Packages ############################
######################################################################

## General packages
import os
import json
import copy
from copy import deepcopy
import numpy as np
import itertools
from tqdm import tqdm
import dask.array as da
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
from skimage.filters import threshold_otsu
from sklearn.metrics import precision_recall_fscore_support as prfs

## Custom packages
from cm_MICI.util.cm_mi_choquet_integral_binary import BinaryMIChoquetIntegral
#from cm_MICI.util.cm_mi_choquet_integral_dask import MIChoquetIntegral

"""
%=====================================================================
%======================= Function Definitions ========================
%=====================================================================
"""

###############################################################################
############################# Dask MIL Selection ##############################
###############################################################################

def compute_fitness_binary_measure(indices, chi, Bags, Labels, NUM_SOURCES, Parameters):
    
    tempBags = np.zeros((len(Bags)),dtype=np.object)
    for idb, bag in enumerate(Bags):
        tempbag = bag[:,indices]
        tempBags[idb] = tempbag
    
    ## Train ChI with current set of sources
    chi.train_chi(tempBags, Labels, Parameters) 

    ## Return indices of NUM_SOURCES sources and corresponding fitness for the set of sources
    return indices, chi.fitness

def select_mici_binary_measure(Bags, Labels, NUM_SOURCES, Parameters):
    
#    print('\n Running selection for MICI Min-Max with Binary Measure... \n')

    #######################################################################
    ########################### Select Sources ############################
    #######################################################################
    
    nAllSources = Bags[0].shape[1]
    
    ind_set = []
    fitness_set = []
    
#    all_ind = np.array(list(itertools.combinations(range(nAllSources), NUM_SOURCES)))
    all_ind = np.array(list(itertools.permutations(range(nAllSources), NUM_SOURCES)))
    
#    all_measures = np.zeros((all_ind.shape[0],))
    
    chi = BinaryMIChoquetIntegral()
#    
#    ## In parallel
#    num_cores = mp.cpu_count()-2
#    pool = mp.Pool(num_cores)
#    
##    res = [pool.apply_async(func=compute_fitness_binary_measure, args=(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)) for k in tqdm(range(all_ind.shape[0]))]
#
#    res = [pool.apply(func=compute_fitness_binary_measure, args=(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)) for k in tqdm(range(all_ind.shape[0]))]
##    res = [pool.apply(compute_fitness_binary_measure, (all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters) for k in tqdm(range(all_ind.shape[0])))]
#
#    print('Selected Sources!')
#    print('Aggregating results...')
#
#    with tqdm(total=all_ind.shape[0]) as progress_bar:
#        for k in range(all_ind.shape[0]):
##            ind_set.append(res[k].get()[0].tolist())
##            fitness_set.append(res[k].get()[1])
#            ind_set.append(res[k][0].tolist())
#            fitness_set.append(res[k][1])
#            
#            if not(k % 20):
#                progress_bar.update(20)
#    
#    pool.close()
#    pool.join()
    
#    ## Initialize selection with each potential source
#    with tqdm(total=all_ind.shape[0]) as progress_bar:
#        ## In series
#        for k in range(all_ind.shape[0]):
#
#            ind, fitness = compute_fitness_binary_measure(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)
#        
#            ind_set.append(ind.tolist())
#            fitness_set.append(fitness)
#            
#            progress_bar.update()
#    search = True
    
    ## Initial search to remove sources
    sources_to_remove = []
    sources_to_keep = []
    with tqdm(total=Parameters.initial_num_sources) as progress_bar:
        for idx in range(Parameters.initial_num_sources):
            
            if idx == 7:
                print('here')
            
            ind = np.where(all_ind[:,0] == idx)
            
            tempBags = np.zeros((len(Bags)),dtype=np.object)
            for idb, bag in enumerate(Bags):
                tempbag = bag[:,all_ind[ind[0][0],:]]
                tempBags[idb] = tempbag
            
            ## Train ChI with current set of sources
            chi.train_chi(tempBags, Labels, Parameters)
            
            if (chi.measure[0] == 0):
                all_ind = np.delete(all_ind, ind, 0)
                sources_to_remove.append(idx)
            else:
                sources_to_keep.append(idx)
    
            progress_bar.update()
   
    ## Genearte new sources to search
    all_ind = np.array(list(itertools.combinations(range(nAllSources), NUM_SOURCES)))
    
    sources = []
    for idx in sources_to_keep:
        ind = np.where(all_ind == idx)[0]
        sources.append(all_ind[ind,:])
        
    if len(sources_to_keep) > 1:
        all_ind = np.vstack(sources)
        
    ## In parallel
    num_cores = mp.cpu_count()-2
    pool = mp.Pool(num_cores)

    res = [pool.apply(func=compute_fitness_binary_measure, args=(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)) for k in tqdm(range(all_ind.shape[0]))]

    print('Selected Sources!')
    print('Aggregating results...')

    with tqdm(total=all_ind.shape[0]) as progress_bar:
        for k in range(all_ind.shape[0]):
            ind_set.append(res[k][0].tolist())
            fitness_set.append(res[k][1])
            
            if not(k % 20):
                progress_bar.update(20)
    
    pool.close()
    pool.join()    
    
    print('Done! Returning best sources.')
    
    #######################################################################
    ########################### Save Results ##############################
    #######################################################################
    
    ## Order fitness values from greatest to least
    top_order = (-np.array(fitness_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    ## Values to return - indices of selected sources
    return_indices = ind_set[top_order_ind]    

    return return_indices

###############################################################################
############################# Dask MIL Selection ##############################
###############################################################################

def compute_iou_binary_measure(indices, chi, Bags, Labels, NUM_SOURCES, Bags_test, Labels_test, Parameters):
    
    tempBags = np.zeros((len(Bags)),dtype=np.object)
    for idb, bag in enumerate(Bags):
        tempbag = bag[:,indices]
        tempBags[idb] = tempbag
    
    ## Train ChI with current set of sources
    chi.train_chi(tempBags, Labels, Parameters) 
    
    ## Compute output with mean measure elements
    iou = np.zeros((len(Bags_test)))
    mse = np.zeros((len(Bags_test)))
    metric_precision = np.zeros((len(Bags_test)))
    metric_recall = np.zeros((len(Bags_test)))
    metric_f1_score = np.zeros((len(Bags_test)))
    
    if (Parameters.n_choose_k == 'y'):
        ## Train ChI with selected sources
        tempBags = np.zeros((len(Bags_test),),dtype=np.object)
        for idb, bag in enumerate(Bags_test):
            temp_bag = bag[:,indices]
            tempBags[idb] = temp_bag
        currentBagsTest = deepcopy(tempBags)
    else:
        currentBagsTest = deepcopy(Bags_test)
    
    for idx in range(len(currentBagsTest)):
        y_true = Labels_test[idx].reshape((Labels_test[idx].shape[0]*Labels_test[idx].shape[1])).astype('float64') 
        tempTestBag = np.zeros((1,),dtype=np.object)
        tempTestBag[0] = currentBagsTest[idx]
        y_est = chi.compute_chi(tempTestBag,len(tempTestBag),chi.measure)
       
        error = y_true - y_est
        mse[idx] = np.dot(error,error)/(2*len(y_true))
    
        ## Compute iou
        gt_img = y_true > 0
        
        try:
            if(Parameters.CAM_SEG_THRESH == 0): 
                img_thresh = threshold_otsu(y_est)
            else:
                img_thresh = Parameters.CAM_SEG_THRESH
            binary_feature_map = y_est > img_thresh
        except:
            binary_feature_map = y_est > 0.1
    
        ## Compute fitness as IoU to pseudo-groundtruth
        intersection = np.logical_and(binary_feature_map, gt_img)
        union = np.logical_or(binary_feature_map, gt_img)
        
        try:
            iou[idx] = np.sum(intersection) / np.sum(union)
        except:
            iou[idx] = 0
            
        prec, rec, f1, _ = prfs(gt_img, binary_feature_map, pos_label=1, average='binary') 
                        
        metric_precision[idx], metric_recall[idx], metric_f1_score[idx] = round(prec,5), round(rec,5), round(f1,5)
            
    results_miou_mean = np.mean(iou)
    results_miou_std = np.std(iou)
    results_mprec_mean = np.mean(metric_precision)
    results_mprec_std = np.std(metric_precision)
    results_mrec_mean = np.mean(metric_recall)
    results_mrec_std = np.std(metric_recall)
    results_mf1_mean = np.mean(metric_f1_score)
    results_mf1_std = np.std(metric_f1_score)

    ## Return indices of NUM_SOURCES sources and corresponding fitness for the set of sources
    return indices, chi.fitness, results_miou_mean, results_mprec_mean, results_mrec_mean, results_mf1_mean, results_miou_std, results_mprec_std, results_mrec_std, results_mf1_std

def select_mici_binary_measure_validation_iou(Bags, Labels, NUM_SOURCES, Bags_test, Labels_test, Parameters):
    
#    print('\n Running selection for MICI Min-Max with Binary Measure... \n')

    #######################################################################
    ########################### Select Sources ############################
    #######################################################################
    
    nAllSources = Bags[0].shape[1]
    
    ind_set = []
    fitness_set = []
    iou_set = []
    precision_set = []
    recall_set = []
    f1_set = []
    iou_set_std = []
    precision_set_std = []
    recall_set_std = []
    f1_set_std = []
    
#    all_ind = np.array(list(itertools.combinations(range(nAllSources), NUM_SOURCES)))
    all_ind = np.array(list(itertools.permutations(range(nAllSources), NUM_SOURCES)))
    
#    all_measures = np.zeros((all_ind.shape[0],))
    
    chi = BinaryMIChoquetIntegral()
    
#    ## In parallel
#    num_cores = mp.cpu_count()-2
#    pool = mp.Pool(num_cores)
#    
##    res = [pool.apply_async(func=compute_fitness_binary_measure, args=(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)) for k in tqdm(range(all_ind.shape[0]))]
#
#    res = [pool.apply(func=compute_iou_binary_measure, args=(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Bags_test, Labels_test, Parameters)) for k in tqdm(range(all_ind.shape[0]))]
##    res = [pool.apply(compute_fitness_binary_measure, (all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters) for k in tqdm(range(all_ind.shape[0])))]
#
#    print('Selected Sources!')
#    print('Aggregating results...')
#
#    with tqdm(total=all_ind.shape[0]) as progress_bar:
#        for k in range(all_ind.shape[0]):
##            ind_set.append(res[k].get()[0].tolist())
##            fitness_set.append(res[k].get()[1])
#            ind_set.append(res[k][0].tolist())
#            fitness_set.append(res[k][1])
#            iou_set.append(res[k][2])
#            precision_set.append(res[k][3])
#            recall_set.append(res[k][4])
#            f1_set.append(res[k][5])
#            
#            if not(k % 20):
#                progress_bar.update(20)
#    
#    pool.close()
#    pool.join()
    
#    ## Initialize selection with each potential source
#    with tqdm(total=all_ind.shape[0]) as progress_bar:
#        ## In series
#        for k in range(all_ind.shape[0]):
#
#            ind, fitness = compute_fitness_binary_measure(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)
#        
#            ind_set.append(ind.tolist())
#            fitness_set.append(fitness)
#            
#            progress_bar.update()
#            
#    progress_bar.reset()
    
    ## Initial search to remove sources
    sources_to_remove = []
    sources_to_keep = []
    with tqdm(total=Parameters.initial_num_sources) as progress_bar:
        for idx in range(Parameters.initial_num_sources):
            
            if idx == 7:
                print('here')
            
            ind = np.where(all_ind[:,0] == idx)
            
            tempBags = np.zeros((len(Bags)),dtype=np.object)
            for idb, bag in enumerate(Bags):
                tempbag = bag[:,all_ind[ind[0][0],:]]
                tempBags[idb] = tempbag
            
            ## Train ChI with current set of sources
            chi.train_chi(tempBags, Labels, Parameters)
            
            if (chi.measure[0] == 0):
                all_ind = np.delete(all_ind, ind, 0)
                sources_to_remove.append(idx)
            else:
                sources_to_keep.append(idx)
    
            progress_bar.update()
   
    ## Genearte new sources to search
    all_ind = np.array(list(itertools.combinations(range(nAllSources), NUM_SOURCES)))
    
    sources = []
    for idx in sources_to_keep:
        ind = np.where(all_ind == idx)[0]
        sources.append(all_ind[ind,:])
        
    if len(sources_to_keep) > 1:
        all_ind = np.vstack(sources)
        
    ## In parallel
    num_cores = mp.cpu_count()-2
    pool = mp.Pool(num_cores)

    res = [pool.apply(func=compute_iou_binary_measure, args=(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Bags_test, Labels_test, Parameters)) for k in tqdm(range(all_ind.shape[0]))]
   
    print('Selected Sources!')
    print('Aggregating results...')

    with tqdm(total=all_ind.shape[0]) as progress_bar:
        for k in range(all_ind.shape[0]):
            ind_set.append(res[k][0].tolist())
            fitness_set.append(res[k][1])
            iou_set.append(res[k][2])
            precision_set.append(res[k][3])
            recall_set.append(res[k][4])
            f1_set.append(res[k][5])
            iou_set_std.append(res[k][6])
            precision_set_std.append(res[k][7])
            recall_set_std.append(res[k][8])
            f1_set_std.append(res[k][9])
            
            
            if not(k % 20):
                progress_bar.update(20)
    
    pool.close()
    pool.join()
    
    print('Done! Returning best sources.')
    
    #######################################################################
    ########################### Save Results ##############################
    #######################################################################

    return ind_set, fitness_set, iou_set, precision_set, recall_set, f1_set, iou_set_std, precision_set_std, recall_set_std, f1_set_std


    



