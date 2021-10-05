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
from utils import *

class MMNet_base():

    def __init__(self, NT, NR, constellation, device):
        super(MMNet_base, self).__init__()
        self.device = device
        self.NT = 2*NT
        self.NR = 2*NR
        self.lgst_constel = constellation.double()
        self.M = int(self.lgst_constel.shape[0])

    def gaussian(self, zt, tau2_t):
        zt = zt
        #zt - symbols
        arg = torch.reshape(zt,[-1,1]).to(device=self.device) - self.lgst_constel.to(device=self.device)
        arg = torch.reshape(arg, [-1, self.NT, self.M]) 
        #-|| z - symbols||^2 / 2sigma^2
        arg = -torch.square(arg)/ 2. /  tau2_t
        arg = torch.reshape(arg, [-1, self.M]) 
        softMax = nn.Softmax(dim=1) 
        x_out = softMax(arg) 
        del arg
        # sum {xi exp()/Z}
        x_out = torch.matmul(x_out, torch.reshape(self.lgst_constel, [self.M,1]).to(device=self.device)) 
        x_out = torch.reshape(x_out, [-1, self.NT])  
        return x_out
    
    

    def MMNet_denoiser(self, H, W_theta, theta_vec, zt, xhat, rt, noise_sigma, batch_size):    
        HTH = torch.bmm(H.permute(0, 2, 1), H) 
        v2_t = torch.divide(torch.sum(torch.square(rt), dim=1, keepdim=True) - self.NR * torch.square(noise_sigma.unsqueeze(dim=1).to(device=self.device)) / 2, torch.unsqueeze(batch_trace(HTH), dim=1))
        v2_t = torch.maximum(v2_t , 1e-9 * torch.ones(v2_t.shape).to(device=self.device))
        v2_t = torch.unsqueeze(v2_t, dim=2)
        
        C_t = batch_identity_matrix(self.NT, H.shape[-1], batch_size).to(device=self.device) - torch.matmul(W_theta, H)
        tau2_t = 1./self.NT * torch.reshape(batch_trace(torch.bmm(C_t, C_t.permute(0, 2, 1)) ), [-1,1,1]) * v2_t + torch.square(torch.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * torch.reshape(batch_trace(torch.bmm(W_theta, W_theta.permute(0, 2, 1))), [-1,1,1])

        xhat = self.gaussian(zt, tau2_t/theta_vec.to(device=self.device))
        
        return xhat

    
    def MMNet_linear(self, H, W_theta, y, xhat, batch_size):

        rt = y - batch_matvec_mul(H.double(), xhat.double())
        zt = xhat + batch_matvec_mul(W_theta, rt.double())

        return zt, rt


    def process_forward(self, H, W_theta, theta_vec, y, x_out, noise_sigma, batch_size, HyperNet):
        
        if HyperNet == "MLP":
            W_theta = torch.reshape(W_theta, shape=(batch_size, self.NT, self.NR))
            theta_vec = torch.unsqueeze(torch.abs(theta_vec), dim=2)
        else:
            theta_vec = torch.abs(theta_vec)          
            
        zt, rt = self.MMNet_linear(H, W_theta, y, x_out, batch_size)
        x_out = self.MMNet_denoiser(H, W_theta, theta_vec, zt, x_out, rt, noise_sigma, batch_size)

        return x_out

    def forward(self, H, W_theta, theta_vec, y, x_out, noise_sigma, batch_size, HyperNet):
        return self.process_forward(H,  W_theta, theta_vec, y, x_out, noise_sigma, batch_size, HyperNet)