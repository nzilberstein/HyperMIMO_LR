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
from settings import *

######--------UTILS----------#######

def model_eval(NT, model, snr_min, snr_max, test_batch_size, generator, device, correlated_flag = None, batch_corr = None, rho_low = None, rho_high = None, rho = None, Cu = None, test_set_flag = None, test_set = None, iterations=150):
    SNR_dBs = np.linspace(np.int(snr_min), np.int(snr_max), np.int(snr_max - snr_min + 1))
    accs_NN = []#np.zeros(shape=SNR_dBs.shape)
    bs = test_batch_size
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)
    criterion = nn.MSELoss().to(device=device)
    for i in range(SNR_dBs.shape[0]):
        acum = 0.
        for jj in range(iterations):
            if (test_set_flag):
                H = torch.tensor(test_set)
                y, x, j_indices, noise_sigma = generator.give_batch_data_Hinput(H, NT, snr_db_min=SNR_dBs[i], snr_db_max=SNR_dBs[i], batch_size=test_batch_size)
            else:
                H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=SNR_dBs[i], snr_db_max=SNR_dBs[i], batch_size=test_batch_size, correlated_flag = correlated_flag, rho=rho)

            H = H.to(device=device)
            y = y.to(device=device)
            noise_sigma = noise_sigma.to(device=device)
            x = x.to(device=device)
            j_indices = j_indices.to(device=device)
            with torch.no_grad():
                    list_batch_x_predicted = model.forward(H, y, noise_sigma)
                    loss_last, SER_final = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion)               
                    results = [loss_last.detach().item(), 1 - SER_final]
                    acum   += SER_final
        acum = acum / iterations
        accs_NN.append((SNR_dBs[i], 1. - acum))# += acc[1]/iterations
    return accs_NN

def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
    #Convierte a complejo
    x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
    #Lo expande a los 4 posibles simbolos para comparar
    x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
    x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

    #Calcula la resta
    x_real = torch.pow(x_real - real_QAM_const, 2)
    x_imag = torch.pow(x_imag - imag_QAM_const, 2)
    x_dist = x_real + x_imag
    x_indices = torch.argmin(x_dist, dim=-1)

    accuracy = (x_indices == j_indices).sum().to(dtype=torch.float32)
    return accuracy.item()/j_indices.numel()


# def sym_detection(x_hat, x, generator, j_indices):
#     accuracy = (generator.demodulate(x.to(device='cuda')) == generator.demodulate(x_hat.to(device='cuda'))).sum().to(dtype=torch.float64)
#     return accuracy.item()/ (2*j_indices.numel())


def loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion, ser_only=False):
    if (ser_only):
        SER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
        return SER_final
    else:
        x_out = torch.cat(list_batch_x_predicted, dim=0)
        x = x.repeat(num_layers, 1)
        loss = criterion(x_out.double(), x.double())
        SER_final = sym_detection(list_batch_x_predicted[-1].double(), j_indices, real_QAM_const, imag_QAM_const)
        return loss, SER_final


def batch_matvec_mul(A,b):
    '''Multiplies a matrix A of size batch_sizexNxK
       with a vector b of size batch_sizexK
       to produce the output of size batch_sizexN
    '''    
    C = torch.matmul(A, torch.unsqueeze(b, dim=2))
    return torch.squeeze(C, -1) 

def batch_identity_matrix(row, cols, batch_size):
    eye = torch.eye(row, cols)
    eye = eye.reshape((1, row, cols))
    
    return eye.repeat(batch_size, 1, 1)

def batch_trace(H):
    return H.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def getTotalParams(model):
    return sum(p.numel() for p in model.parameters())

def createPermutationMatrix(N):
    E=torch.eye(N)  #N X N identity matrix 

    a = np.linspace(0,N-1,N).astype(int)
    #the permutation in Cauchy 2 line form
    np.random.shuffle(a)
    # permutation=np.concat([np.linspace(0,N-1,N).astype(int),np.random.shuffle(a)])   #butterfly permutation example

    P = torch.zeros([N,N]) #initialize the permutation matrix

    for i in range(0,N):
        P[i]=E[a[i]]
        
    return P

def createPermutationMatrix(N):
    E=torch.eye(N)  #N X N identity matrix 

    a = np.linspace(0,N-1,N).astype(int)
    #the permutation in Cauchy 2 line form
    np.random.shuffle(a)
    # permutation=np.concat([np.linspace(0,N-1,N).astype(int),np.random.shuffle(a)])   #butterfly permutation example

    P = torch.zeros([N,N]) #initialize the permutation matrix

    for i in range(0,N):
        P[i]=E[a[i]]
        
    return P

def permuteBatchMatrix(H, batch_size, N, device):
    HP = torch.empty(H.shape)
    for ii in range(batch_size):
        P = createPermutationMatrix(N).to(device=device).double()
        HP[ii,:,:] = torch.matmul(H[ii,:,:].to(device=device), P)
        HP[ii,:,:] = torch.matmul(P.permute(1,0), H[ii,:,:].to(device=device))
    return HP