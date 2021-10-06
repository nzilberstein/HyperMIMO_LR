import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt
from numpy import linalg as LA
from MMNet_base import *
from HyperNetwork_base import *

class MMNet(nn.Module):

    def __init__(self, num_layers, NT, NR, constellation, device='cuda'):
        super(MMNet, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.MMNetbase = MMNet_base(NT, NR, constellation, self.device)
        self.NT = NT
        self.NR = NR
        self.HyperNet = HyperNetwork(self.NT,self.NR, device)

    
    def forwardHyperNet(self, H):
        batch_size = H.shape[0]
        
        Theta = self.HyperNet.forward(H)
        Theta = torch.reshape(Theta, (batch_size, self.num_layers,  2 * self.NT + (2 * self.NT ) * (2 * self.NR)) )
        W_theta = Theta[:,:,:-(2 * self.NT)]
        theta_vec = Theta[:,:,-(2 * self.NT):]

        return W_theta, theta_vec

    def forward(self, H, y, noise_sigma):
        batch_size = H.shape[0]
        x_size = H.shape[-1]

        x_prev = torch.zeros(batch_size, x_size, dtype=torch.float64).to(device=self.device)
        x_list=[x_prev]
        Theta = self.HyperNet.forward(H)
        Theta = torch.reshape(Theta, (batch_size, self.num_layers,  2 * self.NT + (2 * self.NT ) * (2 * self.NR)) )
        W_theta = Theta[:,:,:-(2 * self.NT)]
        theta_vec = Theta[:,:,-(2 * self.NT):]

        for index in range(self.num_layers):
            xout = self.MMNetbase.forward(H, W_theta[:,index,:], theta_vec[:,index,:], y, x_list[-1].float(), noise_sigma, batch_size)
            x_list.append(xout.double())

            
        
        return (x_list[1:])