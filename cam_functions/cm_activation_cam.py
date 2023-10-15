#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:20:13 2022

@author: cmccurley
"""

import cv2
import numpy as np
import torch
import ttach as tta
from cam_functions.activations_and_gradients import ActivationsAndGradients
from cam_functions.utils.svd_on_activations import get_2d_projection

class ActivationCAM:
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True):
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
        
        importance_weights = self.return_importance_weights(input_tensor,
                                                            target_category,
                                                            eigen_smooth)
        
#        return self.aggregate_multi_layers(cam_per_layer)
    
        return upsampled_activations, importance_weights

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
        
        return upsampled_activations
    
    
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
