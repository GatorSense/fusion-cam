#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:53:57 2022

@author: cmccurley
"""

import torch
import tqdm
from cam_functions.base_cam import BaseCAM
from sklearn.preprocessing import StandardScaler

import cv2
import numpy as np
import ttach as tta
from cam_functions.activations_and_gradients import ActivationsAndGradients
from cam_functions.utils.svd_on_activations import get_2d_projection

class OutputScoreCAMZNorm:
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True,
                 norm_by_layer=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.norm_by_layer = norm_by_layer
        self.scalar = StandardScaler()


    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)
        
        ## scale and upsample activations
        upsampled_activations = self.return_activations(input_tensor,
                                                   target_category,
                                                   eigen_smooth)
        
        feature_scores = self.get_feature_scores(input_tensor,target_category,upsampled_activations)
    
    
        return upsampled_activations, feature_scores

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def return_activations(
            self,
            input_tensor,
            target_category,
            eigen_smooth):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
#        grads_list = [g.cpu().data.numpy()
#                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

#        for target_layer, layer_activations, layer_grads in \
#                zip(self.target_layers, activations_list, grads_list):
           
        upsampled_activations = self.upsample_activations(activations_list[0][0,:,:,:], target_size)
        
        all_activations = np.zeros((upsampled_activations.shape[0], upsampled_activations.shape[1]*upsampled_activations.shape[2]))
        for k in range(upsampled_activations.shape[0]):
            all_activations[k,:] = upsampled_activations[k,:,:].reshape((upsampled_activations.shape[1]*upsampled_activations.shape[2]))
        
        if not(self.norm_by_layer):
            for k in range(all_activations.shape[0]):
                if not(np.std(all_activations[k,:]) == 0):
                    all_activations[k,:] = (all_activations[k,:] - np.mean(all_activations[k,:])/np.std(all_activations[k,:]))
                else: 
                    all_activations[k,:] = np.zeros((all_activations.shape[1]))
        else:
            all_activations = self.scalar.fit_transform(all_activations)
            
        for k in range(all_activations.shape[0]):
            upsampled_activations[k,:,:] = all_activations[k,:].reshape((upsampled_activations.shape[1],upsampled_activations.shape[2]))
        
        
        
        return upsampled_activations
    
    def get_feature_scores(self,
                        input_tensor,
                        target_category,
                        activations):
        with torch.no_grad():
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = torch.unsqueeze(activation_tensor, dim=0)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for batch_index, tensor in enumerate(input_tensors):
                category = target_category[batch_index]
#                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = self.model(batch).cpu().numpy()[:, category]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(upsampled.shape[0], upsampled.shape[1])

#            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return scores
    
    
    def return_importance_weights(
            self,
            input_tensor,
            target_category,
            eigen_smooth):
        
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
#        target_size = self.get_target_width_height(input_tensor)
#        
        target_layer = self.target_layers[0]

        importance_weights = self.get_cam_weights(input_tensor, target_layer,
                                       target_category, activations_list[0], grads_list[0])
        
        return importance_weights
   
        
    ## Returns CAM weights by Grad-CAM
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))

#    
#    ## Perform basic fusion of activation maps
#    def aggregate_multi_layers(self, cam_per_target_layer):
#        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
#        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
#        result = np.mean(cam_per_target_layer, axis=1)
#        return self.scale_cam_image(result)

    ## Scale each activation with min/max and upsample to input size
    def upsample_activations(self, activations, target_size=None):
        result = []
        for img in activations:
            
            if not(self.norm_by_layer):
                img = img - np.min(img)
                img = img / (1e-7 + np.max(img))
                
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result


    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):

        return self.forward(input_tensor,
                            target_category, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

