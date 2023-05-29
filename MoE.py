# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:54:23 2023

@author: oowoyele
"""

import torch

class MoE(): # fully connected neural network class
    def __init__(self, fcn_list, kappa = 5):
        # fcn_list = list of fully connected neural network objects, created using the MLP() class.
        # kappa is a parameter that controls how strongly we want to separate the experts
        #self.num_experts = num_experts
        self.num_experts = len(fcn_list)
        self.kappa = kappa
        self.fcn_list = fcn_list
        self.alpha = None
        
    def compute_SE(self, fcn):
        return (fcn.output - fcn.y)**2

    def compute_MSE(self, fcn):
        return torch.mean((fcn.output - fcn.y)**2)
    
    def compute_weighted_MSE(self, fcn, alpha, update_y = True):
        # computes and returns the weighted (each sample is weighted using alpha)
        # if update_y is true, it updates the model predictions using latest weights before computing the MSE
        if update_y:
            fcn.pred()
        return torch.mean(self.alpha*(fcn.output - fcn.y)**2)
    
    def get_num_winning_points(self):
        # computes and returns the number of points "won" by each model

        num_wp = [len(torch.where(torch.argmax(self.alpha, axis=1) == iexp)[0].detach().numpy()) for iexp in torch.arange(self.num_experts)]

        return num_wp
    
    def get_winning_points_inds(self):
        # computes and returns the indices of points "won" by each model
        
        inds_exp = [torch.where(torch.argmax(self.alpha, axis=1) == iexp)[0] for iexp in torch.arange(self.num_experts)]
        return inds_exp
    
    def compute_alpha(self):
        
        # computes the weights for the MSE (stored as alpha)
        with torch.no_grad():
            errors = [self.compute_SE(fcn) for fcn in self.fcn_list]
            
            errors_mat = torch.concatenate(errors, axis = 1)
            errors_mat_norm = errors_mat/torch.amax(errors_mat, axis = 1)[:,None]
            self.alpha = torch.exp(-self.kappa*errors_mat_norm)/torch.sum(torch.exp(-self.kappa*errors_mat_norm), axis = 1)[:,None]
        
        return self.alpha
    
    def compute_weighted_mse(self, update_y = True):
        self.compute_alpha()
        self.wmse = [self.compute_weighted_MSE(fcn, self.alpha[:,iexp:iexp+1], update_y) for iexp, fcn in enumerate(self.fcn_list)]
        
        
        return self.wmse