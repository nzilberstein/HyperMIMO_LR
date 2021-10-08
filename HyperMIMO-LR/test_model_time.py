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


##########################
##------Parameters------##
##########################
NT = 2
NR = 4

snrdb_list = {16:np.arange(11.0, 22.0), 6:np.arange(10.0, 21.0), 2:np.arange(5.0, 15.0)}

num_layers = 6
train_iter = 50000
learning_rate = 1e-3
batch_size = 100
MMNet_batch_size = 700
time_seq = 5
mod_n = 4

load_pretrained_model = False
corr_flag = True
batch_corr = True
test_set_flag = True

PATH = os.getcwd()


PATH = PATH + '/rho_model_kron/H_param_50seq'
model_filename = PATH + '/model_saved.pth'


def main():
    device = 'cuda'
    generator = sample_generator(1 * train_batch_size, mod_n, NR)
    model = MMNet(num_layers, NT, NR, generator.constellation, device=device)
    model = model.to(device=device)
    

    model.load_state_dict(torch.load(model_filename))
    batch_size = 100
    time_seq = 5
    H0 = torch.empty((batch_size, 2 * NR, 2 * NT, time_seq))
    H1 = torch.empty((batch_size, 2 * NR, 2 * NT))
    H2 = torch.empty((batch_size, 2 * NR, 2 * NT))
    H3 = torch.empty((batch_size, 2 * NR, 2 * NT))
    H4 = torch.empty((batch_size, 2 * NR, 2 * NT))

    with open(PATH + '/rho_model_kron/H_test', 'rb') as fp:
        H = pkl.load(fp)
        
    for ii in range(0, batch_size):
        H0[ii] = H[0 + ii * time_seq:1 + ii*time_seq,:,:]
        H1[ii] = H[1 + ii * time_seq:2 + ii*time_seq,:,:]
        H2[ii] = H[2 + ii * time_seq:3 + ii*time_seq,:,:]
        H3[ii] = H[3 + ii * time_seq:4 + ii*time_seq,:,:]
        H4[ii] = H[4 + ii * time_seq:5 + ii*time_seq,:,:]


    accs_NN = []
    accs_NN.append(model_eval(NT, model, snrdb_list[NT][0], snrdb_list[NT][-1], 500 , generator, 'cuda', iterations=500, test_set_flag = test_set_flag, test_set = H_0.repeat_interleave(5, dim= 0).double()))
    accs_NN.append(model_eval(NT, model, snrdb_list[NT][0], snrdb_list[NT][-1], 500 , generator, 'cuda', iterations=500, test_set_flag = test_set_flag, test_set = H_1.repeat_interleave(5, dim= 0).double()))
    accs_NN.append(model_eval(NT, model, snrdb_list[NT][0], snrdb_list[NT][-1], 500 , generator, 'cuda', iterations=500, test_set_flag = test_set_flag, test_set = H_2.repeat_interleave(5, dim= 0).double()))
    accs_NN.append(model_eval(NT, model, snrdb_list[NT][0], snrdb_list[NT][-1], 500 , generator, 'cuda', iterations=500, test_set_flag = test_set_flag, test_set = H_3.repeat_interleave(5, dim= 0).double()))
    accs_NN.append(model_eval(NT, model, snrdb_list[NT][0], snrdb_list[NT][-1], 500 , generator, 'cuda', iterations=500, test_set_flag = test_set_flag, test_set = H_4.repeat_interleave(5, dim= 0).double()))    
    print(accs_NN)

    with open(PATH + 'H_seq_reg_5hops_time_750mat', 'wb') as fp:
        pkl.dump(accs_NN, fp)

if __name__ == '__main__':
    main()
