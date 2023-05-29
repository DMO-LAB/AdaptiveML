# -*- coding: utf-8 -*-
"""
Created on Thu May 25 23:19:05 2023

@author: oowoyele
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from MLP import MLP
from optimize import optimizerMoE
from MoE import MoE
import matplotlib.pyplot as plt

species=['H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'CO', 'HCO', 'CO2', 'CH3', 'CH4', 'HO2', 'H2O2', 
         'C2H2', 'C2H4', 'NC12H26', 'CH2(S)', 'C2H6', 'HCCO', 'C12H25-2', 'C9H19-1', 'C12H25O-5', 'CH2O']

isp = species.index("OH")
species_array = np.array(species)

inp_list = ['chi', 'PV', 'Zvar', 'Z']
num_inputs = len(inp_list)
num_targets = len(species)

#data = np.load('../MOE/self_organizing_NNs/st_500.npy')
data = np.load('./st_500.npy')

data_len  = data.shape[0]
kk = np.random.permutation(data_len)[:50000]
data = data[kk,:]
inputs_unscaled = data[:,:4]
targets_unscaled = data[:,num_inputs : num_inputs + num_targets]
inp_scaler = MinMaxScaler()
inp_scaler.fit(inputs_unscaled)
inp = inp_scaler.transform(inputs_unscaled)

out_scaler = MinMaxScaler()
out_scaler.fit(targets_unscaled)
targets = out_scaler.transform(targets_unscaled)
out = targets[:,isp:isp+1]

inp_torch = torch.from_numpy(inp)
out_torch = torch.from_numpy(out)

dtype = torch.float64

fcn1 = MLP(inp, out, annstruct = [4, 10, 10, 1], activation = 'sigmoid', lin_output = True, dtype = torch.float64)
fcn2 = MLP(inp, out, annstruct = [4, 10, 10, 1], activation = 'sigmoid', lin_output = True, dtype = torch.float64)

opt = optimizerMoE(fcn_list = [fcn1, fcn2], learning_rate=0.005)

moe = MoE([fcn1, fcn2], kappa = 10)

for it in range(10000):
    loss_list = moe.compute_weighted_mse()
    loss_ = [loss.detach().numpy() for loss in loss_list]
    wp_np = [nwp for nwp in moe.get_num_winning_points()]
    opt.step(loss_list)
    
    if it%50 == 0:
        print(it, loss_, wp_np)