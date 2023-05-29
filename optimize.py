# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:52:56 2023

@author: oowoyele
"""

import torch


class optimizer(): # fully connected neural network class
    def __init__(self, fcn = None, parameters = None, learning_rate = 0.01):
        
        if parameters == None:
            self.parameters = fcn.parameters
        else:
            self.parameters = parameters
            
        self.optim = torch.optim.Adam(parameters, lr=learning_rate)
        #self.fcn = fcn
        
        
    def step(self, loss):
        #self.fcn.pred()
        #mse = self.fcn.mse()
        #torch.autograd.set_detect_anomaly(True)
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
        
    
class optimizerMoE(): # fully connected neural network class
    def __init__(self, fcn_list = None, parameters = None, learning_rate = 0.01):

        if fcn_list is not  None:
            fcn_given = True
            self.num_experts = len(fcn_list)
        elif parameters is not None:
            params_given = True
            self.num_experts = len(parameters)
            
        self.optim = []
        #self.fcn_list = fcn_list
        self.mse_list = [[]]*len(fcn_list)
        
        for iexp in range(self.num_experts):
            if fcn_given:
                self.parameters = fcn_list[iexp].parameters
            elif params_given:
                self.parameters = parameters[iexp]
            
            self.optim += [torch.optim.Adam(self.parameters, lr=learning_rate)]
            if fcn_given:
                self.mse_list[iexp] = fcn_list[iexp].mse()
            
        
    def step(self, loss_list):
        for iexp in range(self.num_experts):
            self.optim[iexp].zero_grad()
            loss_list[iexp].backward(retain_graph=True)
            self.optim[iexp].step()
            #fcn.pred()
            #loss_list[iexp] = fcn.mse()