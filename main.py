#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:05:14 2022

@author: cmccurley
."""


"""
***********************************************************************
    *  File:  main.py
    *  Name:  Connor H. McCurley
    *  Date:  04/29/2022
    *  Desc:  Implementation of the Fusion-CAM algorithm by C. McCurley, 
    *         "Discriminatve Feature Learning with Imprecise, Uncertain, 
    *         and Ambiguous Data," Ph.D Thesis, Gainesville, FL, 2022.
    *
    *         This code defines the parameters for the Fusion-CAM algorithm.
    *
    *         This script is a demo for training and testing the MICI using 
    *         the generalized mean (softmax) model.
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


"""
%=====================================================================
%================= Import Settings and Packages ======================
%=====================================================================
"""

######################################################################
######################### Import Packages ############################
######################################################################
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## General packages
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from skimage.filters import threshold_otsu
from sklearn.metrics import precision_recall_fscore_support as prfs

## Custom packages
from params import set_parameters
import mil_source_selection
from cm_MICI.util.cm_mi_choquet_integral import MIChoquetIntegral
from cm_MICI.util.cm_mi_choquet_integral_binary import BinaryMIChoquetIntegral
from cam_functions.utils.image import show_cam_on_image
import gen_data

"""
%=====================================================================
%============================ Main ===================================
%=====================================================================
"""

if __name__== "__main__":
    
#    print('================= Running Main =================\n')
    
    ######################################################################
    ######################### Set Parameters  ############################
    ######################################################################
    args = None
    parameters = set_parameters(args)
    
    print(f'Running Experiment: {parameters.measure_type} measure; {parameters.experiment_name}')
    
    if(parameters.measure_type == 'binary'):
        base_path = './results/binary_measure'
    elif(parameters.measure_type == 'full_measure_random_init'):
        base_path = './results/full_measure_random_init'
    elif(parameters.measure_type == 'full_measure_bfm_init'):
        base_path = './results/full_measure_bfm_init'
    elif(parameters.measure_type == 'full_measure_init_percent_from_true'):
        base_path = './results/full_measure_init_' + str(parameters.percent_diff) + '_percent_from_true' 
            
    try:
        os.mkdir(base_path)
    except:
        pass
        
    savepath = base_path + '/' + parameters.experiment_name
    
    ######################################################################
    ############################ Load Data ###############################
    ######################################################################
    
    ##################### Load training data ######################
    if (parameters.onlyTarget == 'y'):
        Bags, Labels = gen_data.gen_data_train_single_class(parameters)
    else:
        Bags, Labels = gen_data.gen_data_train_two_class(parameters)
    
    ind = parameters.source_indices
    for elem in range(len(ind)):
        ind[elem] = ind[elem] - 1
    
    ## Train ChI with selected sources
    tempBags = np.zeros((len(Bags),),dtype=np.object)
    for idb, bag in enumerate(Bags):
        temp_bag = bag[:,ind]
        tempBags[idb] = temp_bag
    Bags = deepcopy(tempBags)
    
    ##################### Load testing data #######################
    Bags_test, Labels_test, Images_test = gen_data.gen_data_test(parameters)
    
    ## Train ChI with selected sources
    tempBags = np.zeros((len(Bags_test),),dtype=np.object)
    for idb, bag in enumerate(Bags_test):
        temp_bag = bag[:,ind]
        tempBags[idb] = temp_bag
    Bags_test = deepcopy(tempBags)
    
    nElements = (2**(parameters.num_sources))-1
    
    ######################################################################
    ############################ Define ChI ##############################
    ######################################################################
    
    ## Initialize Choquet Integral instance
    if(parameters.measure_type == 'binary') or (parameters.measure_type == 'full_measure_bfm_init'):
        chi = BinaryMIChoquetIntegral() ## Min-max objective with binary fuzzy measure
    elif(parameters.measure_type == 'full_measure_random_init') or (parameters.measure_type == 'full_measure_init_percent_from_true'):
        chi = MIChoquetIntegral() ## Generalized mean objective with full fuzzy measure
#        parameters.use_parallel = False

    ######################################################################
    ######################### Define results #############################
    ######################################################################
    
    temp_num_runs = parameters.num_runs
    if(parameters.measure_type == 'binary') or (parameters.measure_type == 'full_measure_bfm_init'):
        parameters.num_runs = 1
    
    results_measure = np.zeros((parameters.num_runs, nElements)) ##[Runs x lenMeasure]
    results_mse = np.zeros((parameters.num_runs))
    results_iou = np.zeros((parameters.num_runs))
    results_prec = np.zeros((parameters.num_runs))
    results_rec = np.zeros((parameters.num_runs))
    results_f1 = np.zeros((parameters.num_runs))
    results_fitness = np.zeros((parameters.num_runs))
    results_all_selected_sources = []
    results_n_choose_k_hist = np.zeros((parameters.initial_num_sources,))
    results_fitness_true = np.zeros((parameters.num_runs))
    results_indx_true = []

    ######################################################################
    ########################## Train/Test ChI ############################
    ######################################################################
    for idk in tqdm(range(parameters.num_runs)):
        
        ## If N choose K, select sources and train, 
        ## Otherwise, learn a single measure with all sources
        if (parameters.n_choose_k == 'y'):
            
            ## Select indices
            if (parameters.SELECTION_APPROACH == 'fitness'):
                indices = mil_source_selection.select_mici_binary_measure(Bags, Labels, parameters.num_sources, parameters)
                results_all_selected_sources.append(indices)
            
            elif (parameters.SELECTION_APPROACH == 'iou'):
                ind_set, fitness_set, iou_set, precision_set, recall_set, f1_set, iou_set_std, precision_set_std, recall_set_std, f1_set_std = mil_source_selection.select_mici_binary_measure_validation_iou(Bags, Labels, parameters.num_sources, Bags_test, Labels_test, parameters)
                results = dict()
                results['experiment_name'] = parameters.experiment_name
                results['ind_set'] = ind_set
                results['fitness_set'] = fitness_set
                results['iou_set'] = iou_set
                results['precision_set'] = precision_set
                results['recall_set'] = recall_set
                results['f1_set'] = f1_set
                results['iou_set_std'] = iou_set_std
                results['precision_set_std'] = precision_set_std
                results['recall_set_std'] = recall_set_std
                results['f1_set_std'] = f1_set_std
                
                try:
                    os.mkdir(savepath)
                except:
                    pass
                
                saveto = savepath + '/results.npy'
                np.save(saveto, results, allow_pickle=True)
                
                results_to_print = dict()
                results_to_print['experiment_name'] = parameters.experiment_name
                results_to_print['ind_set'] = ind_set
                results_to_print['fitness_set'] = fitness_set
                results_to_print['iou_set'] = iou_set
                results_to_print['precision_set'] = precision_set
                results_to_print['recall_set'] = recall_set
                results_to_print['f1_set'] = f1_set
                
                saveto = savepath + '/results.txt'
                with open(saveto, 'w') as f:
                    for key, value in results_to_print.items():
                        f.write('%s:%s\n' % (key,value))
                
                quit()
            
            ## Track which sources were selected for histogram
            for num in indices:
                results_n_choose_k_hist[num] = results_n_choose_k_hist[num] + 1
            
            ## Train ChI with selected sources
            tempBags = np.zeros((len(Bags),),dtype=np.object)
            for idb, bag in enumerate(Bags):
                temp_bag = bag[:,indices]
                tempBags[idb] = temp_bag
                
            chi.train_chi(tempBags, Labels, parameters)
            
        else:
            chi.train_chi(Bags, Labels, parameters) 
        
        ######################################################################
        ########################## Testing Stage #############################
        ######################################################################
        
        ## Compute output with mean measure elements
        iou = np.zeros((len(Bags_test)))
        mse = np.zeros((len(Bags_test)))
        metric_precision = np.zeros((len(Bags_test)))
        metric_recall = np.zeros((len(Bags_test)))
        metric_f1_score = np.zeros((len(Bags_test)))
        
        if (parameters.n_choose_k == 'y'):
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
                if(parameters.CAM_SEG_THRESH == 0): 
                    img_thresh = threshold_otsu(y_est)
                else:
                    img_thresh = parameters.CAM_SEG_THRESH
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
                
        results_mmse_mean = np.mean(mse)
        results_mmse_std = np.std(mse)
        results_miou_mean = np.mean(iou)
        results_miou_std = np.std(iou)
        results_mprec_mean = np.mean(metric_precision)
        results_mprec_std = np.std(metric_precision)
        results_mrec_mean = np.mean(metric_recall)
        results_mrec_std = np.std(metric_recall)
        results_mf1_mean = np.mean(metric_f1_score)
        results_mf1_std = np.std(metric_f1_score)
        
        ######################################################################
        ####################### Add Results to Dictionary ####################
        ######################################################################
        
        results_measure[idk,:] = chi.measure
        results_mse[idk] = np.mean(results_mmse_mean)
        results_iou[idk] = np.mean(results_miou_mean)
        results_prec[idk] = np.mean(results_mprec_mean)
        results_rec[idk] = np.mean(results_mrec_mean)
        results_f1[idk] = np.mean(results_mf1_mean)
        results_fitness[idk] = chi.fitness
      
    ######################################################################
    ######################## Accumulate Results ##########################
    ######################################################################
    
    results_measure_mean = np.round(np.mean(results_measure, axis=0),5) 
    results_measure_std = np.round(np.std(results_measure, axis=0),5) 
    results_mse_mean = np.round(np.mean(results_mse),5)
    results_fitness_mean = np.round(np.mean(results_fitness),5)
    results_fitness_std = np.round(np.std(results_fitness),5)
    best_fitness_idx = (-results_fitness).argsort()[::-1][0]
    results_fitness_best = results_fitness[best_fitness_idx]
    results_iou_mean = np.round(np.mean(results_iou),5)
    results_prec_mean = np.round(np.mean(results_prec),5)
    results_rec_mean = np.round(np.mean(results_rec),5)
    results_f1_mean = np.round(np.mean(results_f1),5)
    
    if(parameters.measure_type == 'binary') or (parameters.measure_type == 'full_measure_bfm_init'):
        results_mse_std = results_mmse_std
        results_iou_std = np.round(results_miou_std,5)
        results_prec_std = np.round(results_mprec_std,5)
        results_rec_std = np.round(results_mrec_std,5)
        results_f1_std = np.round(results_mf1_std,5)
    else:
        results_mse_std = np.round(np.std(results_mse),5)
        results_iou_std = np.round(np.std(results_iou),5)
        results_prec_std = np.round(np.std(results_prec),5)
        results_rec_std = np.round(np.std(results_rec),5)
        results_f1_std = np.round(np.std(results_f1),5)
    
    results = dict()
    results['experiment_name'] = parameters.experiment_name
    results['measure_mean'] = results_measure_mean
    results['measure_std'] = results_measure_std
    results['mse_mean'] = results_mse_mean
    results['mse_std'] = results_mse_std
    results['fitness_mean'] = results_fitness_mean
    results['fitness_std'] = results_fitness_std
    results['best_fitness'] = results_fitness_best
    results['parameters'] = parameters
    results['results_iou_mean'] = results_iou_mean
    results['results_iou_std'] = results_iou_std
    results['results_prec_mean'] = results_prec_mean
    results['results_prec_std'] = results_prec_std
    results['results_rec_mean'] = results_rec_mean
    results['results_rec_std'] = results_rec_std
    results['results_f1_mean'] = results_f1_mean
    results['results_f1_std'] = results_f1_std
    
    try:
        os.mkdir(savepath)
    except:
        pass
    
    saveto = savepath + '/results.npy'
    np.save(saveto, results, allow_pickle=True)
    
    results_to_print = dict()
    results_to_print['experiment_name'] = parameters.experiment_name
    results_to_print['measure_mean'] = results_measure_mean
    results_to_print['measure_std'] = results_measure_std
    results_to_print['mse_mean'] = results_mse_mean
    results_to_print['mse_std'] = results_mse_std
    results_to_print['fitness_mean'] = results_fitness_mean
    results_to_print['fitness_std'] = results_fitness_std
    results_to_print['best_fitness'] = results_fitness_best
    results_to_print['results_iou_mean'] = results_iou_mean
    results_to_print['results_iou_std'] = results_iou_std
    results_to_print['results_prec_mean'] = results_prec_mean
    results_to_print['results_prec_std'] = results_prec_std
    results_to_print['results_rec_mean'] = results_rec_mean
    results_to_print['results_rec_std'] = results_rec_std
    results_to_print['results_f1_mean'] = results_f1_mean
    results_to_print['results_f1_std'] = results_f1_std
    
    saveto = savepath + '/results.txt'
    with open(saveto, 'w') as f:
        for key, value in results_to_print.items():
            f.write('%s:%s\n' % (key,value))
    
    ######################################################################
    ############################## Plots #################################
    ######################################################################
    if(parameters.measure_type == 'binary') or (parameters.measure_type == 'full_measure_random_init') or (parameters.measure_type == 'full_measure_init_percent_from_true'):

        if(parameters.n_choose_k == 'y'):
            fittest_ind = (-results_fitness).argsort()[0]
            indices = results_all_selected_sources[fittest_ind]
            
        if (parameters.n_choose_k == 'y'):
            ## Train ChI with selected sources
            tempBags = np.zeros((len(Bags_test),),dtype=np.object)
            for idb, bag in enumerate(Bags_test):
                temp_bag = bag[:,indices]
                tempBags[idb] = temp_bag
            currentBagsTest = deepcopy(tempBags)
        else:
            currentBagsTest = deepcopy(Bags_test)    
        
        ## Compute output with mean measure elements
        for idx in range(len(currentBagsTest)):
            y_true = Labels_test[idx].astype('float64') 
            tempTestBag = np.zeros((1,),dtype=np.object)
            tempTestBag[0] = currentBagsTest[idx]
            y_est = chi.compute_chi(tempTestBag,len(tempTestBag),results_measure[best_fitness_idx,:])
            cam_image_true = show_cam_on_image(Images_test[idx], y_true, True)
            cam_image_est = show_cam_on_image(Images_test[idx], y_est.reshape((224,224)), True)
    
            ## Plot true and estimated labels
            if(parameters.measure_type == 'binary'):
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(cam_image_true,cmap='jet')
                ax[0].set_title('True Labels')
                ax[1].imshow(cam_image_est,cmap='jet')
                ax[1].set_title('Fusion result: MICI Min-Max + BFM')
            else:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(cam_image_true,cmap='jet')
                ax[0].set_title('True Labels')
                ax[1].imshow(cam_image_est,cmap='jet')
                ax[1].set_title('Fusion result: MICI Gen-Mean + FFM')
        
            saveto = savepath + '/chi_output_' + str(idx) + '.png'
            plt.savefig(saveto)
            plt.close()
        
        ## Compute histogram of selected sources
        if(parameters.n_choose_k == 'y'):
            results_n_choose_k_hist = results_n_choose_k_hist/50
            hist_bins = list(np.arange(1,len(results_n_choose_k_hist)+1))
            plt.figure()
            ax = plt.axes()
            plt.xlabel('Index of Source')
            plt.ylabel('Normalized Selection Frequency')
            plt.bar(hist_bins,results_n_choose_k_hist,align='center')
            saveto = savepath + '/selection_frequency.png'
            plt.savefig(saveto)
            plt.close()
    
        ######################################################################
        ############################# Learned ################################
        ######################################################################
        
        ## Get equations
        eqts = chi.get_linear_eqts()
        saveto = savepath + '/equations_learned.txt'
        
        with open(saveto, 'w') as f:
            for key, value in eqts.items():
                f.write('%s:%s\n' % (key,value))
                
    ############################################################################################################################
    ############################################################################################################################
    ############################################################################################################################
    
    ################# Initialize FFM with learned BFM #########################
    elif(parameters.measure_type == 'full_measure_bfm_init'):
        print('Training FULL measure with BFM initialization')
        
        gbfm = results_measure[best_fitness_idx,:]
        
        chi = MIChoquetIntegral()
#        parameters.use_parallel = False
        
        ######################################################################
        ######################### Define results #############################
        ######################################################################
        
        parameters.num_runs = temp_num_runs
        results_measure = np.zeros((parameters.num_runs, nElements)) ##[Runs x lenMeasure]
        results_mse = np.zeros((parameters.num_runs))
        results_iou = np.zeros((parameters.num_runs))
        results_prec = np.zeros((parameters.num_runs))
        results_rec = np.zeros((parameters.num_runs))
        results_f1 = np.zeros((parameters.num_runs))
        results_fitness = np.zeros((parameters.num_runs))
        results_all_selected_sources = []
        results_n_choose_k_hist = np.zeros((parameters.initial_num_sources,))
        results_fitness_true = np.zeros((parameters.num_runs))
        results_indx_true = []
    
        ######################################################################
        ########################## Train/Test ChI ############################
        ######################################################################
        
        for idk in tqdm(range(parameters.num_runs)):
            
            ## If N choose K, select sources and train, 
            ## Otherwise, learn a single measure with all sources
            if (parameters.n_choose_k == 'y'):
                
#                ## Select indices
#                indices = mil_source_selection.select_mici_binary_measure(Bags, Labels, parameters.num_sources, parameters)
#                results_all_selected_sources.append(indices)
#                
#                ## Track which sources were selected for histogram
#                for num in indices:
#                    results_n_choose_k_hist[num] = results_n_choose_k_hist[num] + 1
                
                ## Train ChI with selected sources
                tempBags = np.zeros((len(Bags),),dtype=np.object)
                for idb, bag in enumerate(Bags):
                    temp_bag = bag[:,indices]
                    tempBags[idb] = temp_bag
                
                chi.train_chi(tempBags, Labels, parameters, ginit=gbfm)      
                
            else:
                chi.train_chi(Bags, Labels, parameters, ginit=gbfm) 
        
            ######################################################################
            ########################## Testing Stage #############################
            ######################################################################
        
            ## Compute output with mean measure elements
            iou = np.zeros((len(Bags_test)))
            mse = np.zeros((len(Bags_test)))
            metric_precision = np.zeros((len(Bags_test)))
            metric_recall = np.zeros((len(Bags_test)))
            metric_f1_score = np.zeros((len(Bags_test)))
            
            if (parameters.n_choose_k == 'y'):
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
                    if(parameters.CAM_SEG_THRESH == 0): 
                        img_thresh = threshold_otsu(y_est)
                    else:
                        img_thresh = parameters.CAM_SEG_THRESH
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
                    
            results_mmse_mean = np.mean(mse)
            results_mmse_std = np.std(mse)
            results_miou_mean = np.mean(iou)
            results_miou_std = np.std(iou)
            results_mprec_mean = np.mean(metric_precision)
            results_mprec_std = np.std(metric_precision)
            results_mrec_mean = np.mean(metric_recall)
            results_mrec_std = np.std(metric_recall)
            results_mf1_mean = np.mean(metric_f1_score)
            results_mf1_std = np.std(metric_f1_score)
            
            ######################################################################
            ####################### Add Results to Dictionary ####################
            ######################################################################
            
            results_measure[idk,:] = chi.measure
            results_mse[idk] = np.mean(results_mmse_mean)
            results_iou[idk] = np.mean(results_miou_mean)
            results_prec[idk] = np.mean(results_mprec_mean)
            results_rec[idk] = np.mean(results_mrec_mean)
            results_f1[idk] = np.mean(results_mf1_mean)
            results_fitness[idk] = chi.fitness
        
        #######################################################################
        ######################## Save Results #################################
        #######################################################################
        results_measure_mean = np.round(np.mean(results_measure, axis=0),5) 
        results_measure_std = np.round(np.std(results_measure, axis=0),5) 
        results_mse_mean = np.round(np.mean(results_mse),5)
        results_fitness_mean = np.round(np.mean(results_fitness),5)
        results_fitness_std = np.round(np.std(results_fitness),5)
        best_fitness_idx = (-results_fitness).argsort()[::-1][0]
        results_fitness_best = results_fitness[best_fitness_idx]
        results_iou_mean = np.round(np.mean(results_iou),5)
        results_prec_mean = np.round(np.mean(results_prec),5)
        results_rec_mean = np.round(np.mean(results_rec),5)
        results_f1_mean = np.round(np.mean(results_f1),5)
        
        if(parameters.measure_type == 'binary') or (parameters.measure_type == 'full_measure_bfm_init'):
            results_mse_std = results_mmse_std
            results_iou_std = np.round(results_miou_std,5)
            results_prec_std = np.round(results_mprec_std,5)
            results_rec_std = np.round(results_mrec_std,5)
            results_f1_std = np.round(results_mf1_std,5)
        else:
            results_mse_std = np.round(np.std(results_mse),5)
            results_iou_std = np.round(np.std(results_iou),5)
            results_prec_std = np.round(np.std(results_prec),5)
            results_rec_std = np.round(np.std(results_rec),5)
            results_f1_std = np.round(np.std(results_f1),5)
        
        results = dict()
        results['experiment_name'] = parameters.experiment_name
        results['measure_mean'] = results_measure_mean
        results['measure_std'] = results_measure_std
        results['mse_mean'] = results_mse_mean
        results['mse_std'] = results_mse_std
        results['fitness_mean'] = results_fitness_mean
        results['fitness_std'] = results_fitness_std
        results['best_fitness'] = results_fitness_best
        results['parameters'] = parameters
        results['init_bfm_measure'] = gbfm
        results['results_iou_mean'] = results_iou_mean
        results['results_iou_std'] = results_iou_std
        results['results_prec_mean'] = results_prec_mean
        results['results_prec_std'] = results_prec_std
        results['results_rec_mean'] = results_rec_mean
        results['results_rec_std'] = results_rec_std
        results['results_f1_mean'] = results_f1_mean
        results['results_f1_std'] = results_f1_std
        
        saveto = savepath + '/results.npy'
        np.save(saveto, results, allow_pickle=True)
        
        results_to_print = dict()
        results_to_print['experiment_name'] = parameters.experiment_name
        results_to_print['measure_mean'] = results_measure_mean
        results_to_print['measure_std'] = results_measure_std
        results_to_print['mse_mean'] = results_mse_mean
        results_to_print['mse_std'] = results_mse_std
        results_to_print['fitness_mean'] = results_fitness_mean
        results_to_print['fitness_std'] = results_fitness_std
        results_to_print['best_fitness'] = results_fitness_best
        results_to_print['init_bfm_measure'] = gbfm
        results_to_print['results_iou_mean'] = results_iou_mean
        results_to_print['results_iou_std'] = results_iou_std
        results_to_print['results_prec_mean'] = results_prec_mean
        results_to_print['results_prec_std'] = results_prec_std
        results_to_print['results_rec_mean'] = results_rec_mean
        results_to_print['results_rec_std'] = results_rec_std
        results_to_print['results_f1_mean'] = results_f1_mean
        results_to_print['results_f1_std'] = results_f1_std
        
        saveto = savepath + '/results.txt'
        with open(saveto, 'w') as f:
            for key, value in results_to_print.items():
                f.write('%s:%s\n' % (key,value))
        
        ######################################################################
        ############################## Plots #################################
        ######################################################################
        if(parameters.n_choose_k == 'y'):
            fittest_ind = (-results_fitness).argsort()[0]
            indices = results_all_selected_sources[fittest_ind]
            
        if (parameters.n_choose_k == 'y'):
            ## Train ChI with selected sources
            tempBags = np.zeros((len(Bags_test),),dtype=np.object)
            for idb, bag in enumerate(Bags_test):
                temp_bag = bag[:,indices]
                tempBags[idb] = temp_bag
            currentBagsTest = deepcopy(tempBags)
        else:
            currentBagsTest = deepcopy(Bags_test)
            
        ## Compute output with mean measure elements
        for idx in range(len(currentBagsTest)):
            y_true = Labels_test[idx].reshape((Labels_test[idx].shape[0]*Labels_test[idx].shape[1])).astype('float64')  
            tempTestBag = np.zeros((1,),dtype=np.object)
            tempTestBag[0] = currentBagsTest[idx]
            y_est = chi.compute_chi(tempTestBag,len(tempTestBag),results_measure[best_fitness_idx,:])
            cam_image_true = show_cam_on_image(Images_test[idx], y_true, True)
            cam_image_est = show_cam_on_image(Images_test[idx], y_est.reshape((224,224)), True)
    
            ## Plot true and estimated labels
            if(parameters.measure_type == 'binary'):
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(cam_image_true,cmap='jet')
                ax[0].set_title('True Labels')
                ax[1].imshow(cam_image_est,cmap='jet')
                ax[1].set_title('Fusion result: MICI Min-Max + BFM')
            else:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(cam_image_true,cmap='jet')
                ax[0].set_title('True Labels')
                ax[1].imshow(cam_image_est,cmap='jet')
                ax[1].set_title('Fusion result: MICI Gen-Mean + FFM')
        
            saveto = savepath + '/chi_output_' + str(idx) + '.png'
            plt.savefig(saveto)
            plt.close()
        
        ## Compute histogram of selected sources
        if(parameters.n_choose_k == 'y'):
            results_n_choose_k_hist = results_n_choose_k_hist/50
            hist_bins = list(np.arange(1,len(results_n_choose_k_hist)+1))
            plt.figure()
            ax = plt.axes()
            plt.xlabel('Index of Source')
            plt.ylabel('Normalized Selection Frequency')
            plt.bar(hist_bins,results_n_choose_k_hist,align='center')
            saveto = savepath + '/selection_frequency.png'
            plt.savefig(saveto)
            plt.close()
    
        ######################################################################
        ############################# Learned ################################
        ######################################################################
        
        ## Get equations
        eqts = chi.get_linear_eqts()
        saveto = savepath + '/equations_learned.txt'
        
        with open(saveto, 'w') as f:
            for key, value in eqts.items():
                f.write('%s:%s\n' % (key,value))

        