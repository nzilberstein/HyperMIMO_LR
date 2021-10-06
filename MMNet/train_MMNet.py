import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt
# from numpy import linalg as LA
import scipy.linalg as LA
import pandas as pd
import seaborn as sns
from settings import *
from utils import *
from sample_generator import *
from MMNet import *
import io
import os

#Training parameters
train_iter = 5000
train_batch_size = 500
learning_rate = 1e-3

batch_size = 100
time_seq = 5

PATH = os.getcwd()

test_set_flag = True

corr_flag = True


def train(H, model, optimizer, generator, device='cuda'):
    criterion = nn.MSELoss().to(device=device)
    model.train()
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)

    for i in range(train_iter):

        y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, NT, snr_db_min=snrdb_classical_list[NT][0], snr_db_max=snrdb_classical_list[NT][-1], batch_size=train_batch_size)
        H = H.to(device=device)
        y = y.to(device=device)
        
        list_batch_x_predicted = model.forward(H, y, noise_sigma.to(device=device))

        x = x.to(device=device)
        j_indices = j_indices.to(device=device)

        loss, SER = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion)                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del y, x, j_indices, noise_sigma, list_batch_x_predicted

        if (i%500==0):
            model.eval()
            y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, NT, snr_db_min=snrdb_classical_list[NT][-1], snr_db_max=snrdb_classical_list[NT][-1], batch_size=train_batch_size)
            H = H.to(device=device)
            y = y.to(device=device)
            noise_sigma = noise_sigma.to(device=device)
            with torch.no_grad():
                list_batch_x_predicted = model.forward(H, y, noise_sigma)

                x = x.to(device=device)
                j_indices = j_indices.to(device=device)
                loss_last, SER_final = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion)                
                results = [loss_last.detach().item(), 1 - SER_final]
                print_string = [i]+results
                print(' '.join('%s' % np.round(x,6) for x in print_string))
            del y, x, j_indices, noise_sigma, list_batch_x_predicted
            model.train()

            
def main():
    generator = sample_generator(train_batch_size, mod_n, NR)
    rho = 0.6

    SER = []
    accs_NN = []
    SER_ACUM = np.zeros(11)
    total_sum = 1
    with open(PATH + '/H_test', 'rb') as fp:
        H = pkl.load(fp)
    for i in range(0, 1):
        SER = []

        device = 'cuda:0'
        model = MMNet(num_layers, NT, NR, train_batch_size, generator.constellation, device=device)
        model = model.to(device=device)
        R = H[i:i+1,:,:].repeat_interleave(train_batch_size, dim=0)
        print('******************************** Starting training **********************************************')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train(R, model, optimizer, generator, device)
        print('******************************** Finish training **********************************************')
        print('******************************** Starting testing *********************************************')

        H0 = torch.empty((batch_size, 2 * NR, 2 * NT))
        H1 = torch.empty((batch_size, 2 * NR, 2 * NT))
        H2 = torch.empty((batch_size, 2 * NR, 2 * NT))
        H3 = torch.empty((batch_size, 2 * NR, 2 * NT))
        H4 = torch.empty((batch_size, 2 * NR, 2 * NT))

        with open(PATH + '/H_test', 'rb') as fp:
            H = pkl.load(fp)
        for ii in range(0, batch_size):
            H0[ii] = H[0 + ii * time_seq:1 + ii*time_seq,:,:]
            H1[ii] = H[1 + ii * time_seq:2 + ii*time_seq,:,:]
            H2[ii] = H[2 + ii * time_seq:3 + ii*time_seq,:,:]
            H3[ii] = H[3 + ii * time_seq:4 + ii*time_seq,:,:]
            H4[ii] = H[4 + ii * time_seq:5 + ii*time_seq,:,:]


        accs_NN_old = []
        accs_NN_old.append(model_eval(NT, model, snrdb_classical_list[NT][0], snrdb_classical_list[NT][-1], train_batch_size , generator, 'cuda', iterations=500,  test_set_flag = test_set_flag, test_set = H.double()))
        print(accs_NN_old)

         #This is in if the user wants to generate the parameters for the regularizer
#        with open('/home/nicolas/MIMO_detection_project/HyperMIMO_final/rho_model_kron/H_param_50seq' + str(i+750), 'wb') as fp:
#            pkl.dump([R, list(model.parameters())], fp)
if __name__ == '__main__':
    main()
