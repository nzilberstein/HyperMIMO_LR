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
from sample_generator import *

from MMNet_base import MMNet_base
from HyperNetwork_base import *
from MMNet import *
from MMNet_singleH import *

#parameters
NT = 2
NR = 4

snrdb_list = {16:np.arange(11.0, 22.0), 6:np.arange(10.0, 21.0), 2:np.arange(5.0, 15.0)}

num_layers = 6 #Cambiar en HyperNet module
train_iter = 50000
pre_train_iter = 1 * 2000
# preTrain_batch_size = 500
train_batch_size = 100
test_batch_size = 5000
learning_rate = 1e-3
mod_n = 4

corr_flag = True
batch_corr = True

test_set_flag = True
batch_size = 100
MMNet_batch_size = 700
time_seq = 5
PATH = '/home/nicolas/MIMO_detection_project/HyperMIMO_final/rho_model_kron/H_param_50seq'
model_filename = '/home/nicolas/MIMO_detection_project/HyperMIMO/HyperMIMO_seq5hop.pth'


def train3(Hr, Hi, H_test, model, generator, device='cpu'):

    learning_rate = 1e-3
    model.train()
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08, verbose = True)
    
    alpha = torch.tensor(1).to(device=device)
    beta = torch.tensor(1).to(device=device)

    H_MMNet, Thetas_MMNet, ThetaReal_MMNet, ThetaImag_MMNet, Theta_Vec = getBatchHMMnet(MMNet_batch_size, num_layers, PATH)
    H_MMNet = H_MMNet.to(device=device).double()
    Thetas_MMNet, Theta_Vec = processThetaMMNet(Thetas_MMNet, Theta_Vec, MMNet_batch_size, num_layers)

    for i in range(train_iter):

        rho = 0.6
        H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=snrdb_list[NT][0], snr_db_max=snrdb_list[NT][-1], batch_size=batch_size, correlated_flag=corr_flag, rho=rho)
        H = H.to(device=device).double()
        y = y.to(device=device).double()
        noise_sigma = noise_sigma.to(device=device).double()

        list_batch_x_predicted = model.forward(H, y, noise_sigma)
  
        Thetas_HyperNet, theta_vec = model.module.forwardHyperNet(H_MMNet)
        Thetas_HyperNet = torch.reshape(Thetas_HyperNet, shape=(MMNet_batch_size * num_layers, 2*NT, 2*NR))
        Thetas_HyperNet = torch.reshape(Thetas_HyperNet, shape=(-1, 2*NT * 2*NR))
        theta_vec = torch.reshape(theta_vec, shape=(MMNet_batch_size * num_layers, 2*NT))
        
        x = x.to(device=device)
        j_indices = j_indices.to(device=device)

        loss, SER = loss_fn_eval(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, generator)
        loss_theta = loss_fn_theta_fast(Thetas_HyperNet, theta_vec, Thetas_MMNet, Theta_Vec, MMNet_batch_size, num_layers, device)
        
        optimizer.zero_grad()
        (alpha * loss +  beta * loss_theta).backward(retain_graph = True)
        optimizer.step()
        del  H, y, x, j_indices, noise_sigma, list_batch_x_predicted
        
        if (i%500==0):
            model.eval()
            accs_NN_general = model_eval(NT, model, snrdb_list[NT][0], snrdb_list[NT][-1], batch_size * time_seq, generator, 'cuda', iterations=500, test_set_flag = test_set_flag, test_set =  H_test)
            print('iteration number : ', i, 'SER : ',  accs_NN_general[-1], 'Loss:', loss, 'Loss-Theta:', loss_theta,  'Lossglobal:', loss_theta + loss)
            
            lr_scheduler.step(alpha * loss +  beta * loss_theta) 
            model.train()


def main():
    device = 'cuda'
    generator = sample_generator(1 * train_batch_size, mod_n, NR)
    model = MMNet(num_layers, NT, NR, generator.constellation, device=device)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device=device)
    
    if (load_pretrained_model):
        model.load_state_dict(torch.load(model_filename))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', 0.91, 0, True, 0.0001, 'rel', 0, 0, 1e-08)
        print('*******Successfully loaded pre-trained model***********')
    else:
        
    with open('/home/nicolas/MIMO_detection_project/HyperMIMO_final/rho_model_kron/H_0', 'rb') as fp:
        H = pkl.load(fp)

    Hr = H[0:1,0:NR,0:NT]
    Hi = H[0:1,NR:,0:NT]

    with open('/home/nicolas/MIMO_detection_project/HyperMIMO/rho_model_kron/H_test', 'rb') as fp:
        H_test = pkl.load(fp)

    train3(Hr, Hi, H_test.double(), model, generator, device)        
    print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
    main()