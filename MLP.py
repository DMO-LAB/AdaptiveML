# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:56:02 2023

@author: oowoyele
"""

import torch
import sys

def initialize(shape = None, dtype=torch.float64):
    # initialize a matrix with random numbers    
    # function initializes and returns a 2D array with specified shape
    inp_shape = shape[0]
    out_shape = shape[1]
    scale = 1/(inp_shape + out_shape)
    limit = torch.sqrt(scale)
    
    var = torch.rand((inp_shape, out_shape), dtype=dtype) * limit
    var.requires_grad = True
    return var

def initialize1D(shape = None, dtype = torch.float64):
    # initialize a vector with random numbers
    # function initializes and returns a 1D array with specified shape
    inp_shape = shape[0]
    scale = 1/(inp_shape)
    limit = torch.sqrt(scale)
    
    var = torch.rand((inp_shape, ), dtype=dtype) * limit

    var.requires_grad = True
    return var

class MLP(): # fully connected neural network class
    # input_tensor: numpy array containing input to neural network
    # output_tensor: numpy array containing desired output neural network
    # annstruct: structure of neural network as list
    # lin_output: whether or not we want the output layer of the neural network to be activated
    def __init__(self, input_tensor = None, output_tensor = None, annstruct = None, activation = 'sigmoid', lin_output = False, dtype = torch.float64, weights = [], biases = []):
        self.annstruct = [torch.tensor(annstr) for annstr in annstruct]  # structure of ann as a list
        self.nTargets = self.annstruct[-1] # number of targets
        self.nLayers = int(len(self.annstruct) - 2) # number of hidden layer
        self.actLayers = [[]]*(self.nLayers + 1) # empty list to hold activated layers
        self.Layers = [[]]*(self.nLayers + 1) # empty list to hold layers before activation
        self.lin_output = lin_output # whether we have a linear activation at the last layer
        self.dtype = dtype
        self.weights = weights # weights
        self.biases = biases # biases
        self.activation =  activation # type of activation to use
        self.x = torch.from_numpy(input_tensor)
        self.y = torch.from_numpy(output_tensor)
        
        if self.activation != 'sigmoid':#  and self.activation != 'leaky_relu':
        # only implemented for sigmoid
            sys.exit('Error. Only implemented for sigmoid')
        
        # initialize weights. store as a list
        if self.weights == []:
            self.weights = [initialize(shape = (self.annstruct[0], self.annstruct[1]))]
            for ii in range(self.nLayers-1):
                self.weights += [initialize(shape = (self.annstruct[ii+1], self.annstruct[ii+2]))]
            self.weights += [initialize(shape = (self.annstruct[-2], self.nTargets))]
        
        # initialize biases. store as a list
        if self.biases == []:
            self.biases = [initialize1D(shape = (self.annstruct[1],))]
            for ii in range(self.nLayers-1):
                self.biases += [initialize1D(shape = (self.annstruct[ii+2],))]
            self.biases += [initialize1D(shape = (self.nTargets,))]
            
        self.pred()
    
    def sigma(self, x): # apply activation
        #return torch.where(self.activation == 'leaky_relu', torch.where(x > 0, x, x*0.05), 1/(1 + torch.exp(-x)))
    
        return 1/(1 + torch.exp(-x))
    
    def d_sigma(self, x): # apply gradient of activation
        sigmoid_x = 1/(1 + torch.exp(-x))
        #return torch.where(self.activation == 'leaky_relu', torch.where(x > 0, 1, 0.05), sigmoid_x * (1 - sigmoid_x))
        return sigmoid_x * (1 - sigmoid_x)

    def pred(self): # create the neural network connections
        # compute first layer and activate it
        self.Layers[0] = torch.matmul(self.x, self.weights[0]) +  self.biases[0]
        self.actLayers[0] = self.sigma(self.Layers[0])
        
        # loop over all remaining hidden units and activate them
        for ii in range(1, self.nLayers):
            self.Layers[ii] = torch.matmul(self.actLayers[ii-1], self.weights[ii]) +  self.biases[ii]
            self.actLayers[ii] = self.sigma(self.Layers[ii])
        
        # compute last layer 
        ii = self.nLayers
        self.Layers[ii] = torch.matmul(self.actLayers[ii-1], self.weights[ii]) +  self.biases[ii]
        
        # we have an activated hidden unit, apply activation, and assign result to list containing activated layers
        if self.lin_output == False:
            self.actLayers[ii] = self.sigma(self.Layers[ii])
        else:
            self.actLayers[ii] = self.Layers[ii]
        
        self.output = self.actLayers[ii] # assign to output variable
        
        self.parameters = self.weights + self.biases
        
        return self.output
    
    def mse(self, update_y = True):
        if update_y:
            self.pred()
        return torch.mean((self.y - self.output)**2)
    
    def get_grads(self, x):
        # function to analytically compute gradients of error wrt weights and biases. assumes you have one target, the network is fully connected.
        
        self.num_samples = x.shape[0] # number of samples
        actLayers_ = [x] + self.actLayers[:-1] # list of activated layers
        actLayers_ = [nl[:,None,:] for nl in actLayers_] # expand arrays by adding another dimension
        
        # we start from last layer and back propagate to first
        
        # last layer
        if self.lin_output == False:
            sig = [-self.d_sigma(self.Layers[self.nLayers])[:,None,:]]
        else:
            qwe = -1*torch.eye(self.nTargets, dtype=self.dtype)[None, :]
            sig = [torch.tile(qwe, [self.num_samples, 1, 1])]
        
        gradW = [torch.matmul(torch.permute(actLayers_[self.nLayers], [0,2,1]), sig[-1])]

        # loop over all layers
        for i in range(1, self.nLayers+1):
            a = sig[-1]
            b = self.weights[self.nLayers - i + 1].T
            c = self.d_sigma(self.Layers[self.nLayers-i])
            d = torch.tensordot(a, b, dims=[[2],[0]])
            sig += [d * c[:,None,:]]
            gradW += [torch.matmul(torch.permute(actLayers_[self.nLayers-i], [0,2,1]), sig[-1])]

        # reverse lists so they go from first to last (dW1 ... dW_last)
        sig.reverse()
        gradb = sig
        gradW.reverse()
        grad_vars = []
        
        # combine gradients from weights and biases in a single list
        for j in range(len(gradW)):
            grad_vars += [gradW[j][:,None, :, :]]
        for j in range(len(gradb)):
            grad_vars += [gradb[j][:,None, :, :]]
    
        return grad_vars # return gradients