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

######--------UTILS----------#######

def model_eval(H, model, snr_min, snr_max, test_batch_size, generator, device, iterations=150):
    SNR_dBs = np.linspace(np.int(snr_min), np.int(snr_max), np.int(snr_max - snr_min + 1))
    accs_NN = []#np.zeros(shape=SNR_dBs.shape)
    bs = test_batch_size
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)
    for i in range(SNR_dBs.shape[0]):
        acum = 0.
        y, x, j_indices, noise_sigma = generator.give_batch_data(H, NT, snr_db_min=SNR_dBs[i], snr_db_max=SNR_dBs[i], batch_size=test_batch_size)
        H = H.to(device=device)
        y = y.to(device=device)
        noise_sigma = noise_sigma.to(device=device)
        with torch.no_grad():
            for j in range(iterations):
                list_batch_x_predicted = model.forward(H, y, noise_sigma)
                x = x.to(device=device)
                j_indices = j_indices.to(device=device)
                loss_last, SER_final = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, generator)                
                results = [loss_last.detach().item(), 1 - SER_final]
                acum   += SER_final / iterations
        accs_NN.append((SNR_dBs[i], 1. - acum))# += acc[1]/iterations
    return accs_NN

def model_eval_ZF(H, H_inv, H_tilde, model, snr_min, snr_max, test_batch_size, generator, device, iterations=150):
    SNR_dBs = np.linspace(np.int(snr_min), np.int(snr_max), np.int(snr_max - snr_min + 1))
    accs_NN = []#np.zeros(shape=SNR_dBs.shape)
    bs = test_batch_size
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)
    criterion = nn.MSELoss().to(device=device)
    for i in range(SNR_dBs.shape[0]):
        acum = 0.
        y, x, j_indices, noise_sigma = generator.give_batch_data(H, NT, snr_db_min=SNR_dBs[i], snr_db_max=SNR_dBs[i], batch_size=test_batch_size)
        H = H.to(device=device)
        y = y.to(device=device)
        
        y = batch_matvec_mul(H_inv, y)
        
        noise_sigma = noise_sigma.to(device=device)
        with torch.no_grad():
            for j in range(iterations):
                x = x.to(device=device)
                j_indices = j_indices.to(device=device)
                SER_final = sym_detection(y, j_indices, real_QAM_const, imag_QAM_const)
                results = [1 - SER_final]
                acum   += SER_final / iterations
        accs_NN.append((SNR_dBs[i], 1. - acum))# += acc[1]/iterations
    return accs_NN

def model_eval_Hdaga(H, H_inv, H_tilde, model, snr_min, snr_max, test_batch_size, generator, device, iterations=150):
    SNR_dBs = np.linspace(np.int(snr_min), np.int(snr_max), np.int(snr_max - snr_min + 1))
    accs_NN = []#np.zeros(shape=SNR_dBs.shape)
    bs = test_batch_size
    real_QAM_const = generator.real_QAM_const.to(device=device)
    imag_QAM_const = generator.imag_QAM_const.to(device=device)
    for i in range(SNR_dBs.shape[0]):
        acum = 0.
        y, x, j_indices, noise_sigma = generator.give_batch_data(H, NT, snr_db_min=SNR_dBs[i], snr_db_max=SNR_dBs[i], batch_size=test_batch_size)
        H = H.to(device=device)
        y = y.to(device=device)
        
        y = batch_matvec_mul(H_inv, y)
        
        noise_sigma = noise_sigma.to(device=device)
        with torch.no_grad():
            for j in range(iterations):
                list_batch_x_predicted = model.forward(H_tilde, y, noise_sigma)
                x = x.to(device=device)
                j_indices = j_indices.to(device=device)
                loss_last, SER_final = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, generator)                
                results = [loss_last.detach().item(), 1 - SER_final]
                acum   += SER_final / iterations
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
    return accuracy.item()/x_indices.numel()


# def sym_detection(x_hat, x, generator, j_indices):
#     accuracy = (generator.demodulate(x.to(device='cuda')) == generator.demodulate(x_hat.to(device='cuda'))).sum().to(dtype=torch.float64)
#     return accuracy.item()/ (2*j_indices.numel())


def loss_fn(batch_x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, generator, ser_only=False):
    loss = []
    for batch_x_predicted in enumerate(list_batch_x_predicted):
        loss_index = torch.mean(torch.mean(torch.pow((batch_x - batch_x_predicted[1]),2), dim=1))
        loss.append(loss_index)

#     SER_final = sym_detection(list_batch_x_predicted[-1], batch_x, generator, j_indices)
     
    SER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
    
    return loss[-1], SER_final


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


def createH_sequence(Hr_init, Hi_init, time_seq, NT, NR):
    rho_seq = 0.98
#     Hr = torch.empty((batch_size, NR, NT)).normal_(mean=0,std=1/2)
#     Hi = torch.empty((batch_size, NR, NT)).normal_(mean=0,std=1/2)
    Hr = torch.empty((time_seq, NR, NT))
    Hi = torch.empty((time_seq, NR, NT))

    Hr[0,:,:] = Hr_init
    Hi[0,:,:] = Hi_init

#     for bs in range(0, batch_size):
    for ii in range(1, time_seq):

        er = torch.empty((1, NR, NT)).normal_(mean=0,std=1/2)
        ei = torch.empty((1, NR, NT)).normal_(mean=0,std=1/2)

        Hr[ii:ii+1,:,:] = rho_seq * Hr[ii-1:ii,:,:] + np.sqrt(1 - rho_seq**2) * er
        Hi[ii:ii+1,:,:] = rho_seq * Hi[ii-1:ii,:,:] + np.sqrt(1 - rho_seq**2) * ei

    h1 = torch.cat((Hr, -1. * Hi), dim=2)
    h2 = torch.cat((Hi, Hr), dim=2)
    H = torch.cat((h1, h2), dim=1)
    
    return H