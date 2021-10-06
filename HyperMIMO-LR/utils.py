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


def model_eval(NT, model, snr_min, snr_max, test_batch_size, generator, device, correlated_flag = None, batch_corr = None, rho_low = None, rho_high = None, rho = None, test_set_flag = None, test_set = None, iterations=150):
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
                    loss_last, SER_final = loss_fn_eval(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion)               
                    results = [loss_last.detach().item(), 1 - SER_final]
                    acum   += SER_final
        acum = acum / iterations
        accs_NN.append((SNR_dBs[i], 1. - acum))# += acc[1]/iterations
    return accs_NN


def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
    x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
    x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
    x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

    x_real = torch.pow(x_real - real_QAM_const, 2)
    x_imag = torch.pow(x_imag - imag_QAM_const, 2)
    x_dist = x_real + x_imag
    x_indices = torch.argmin(x_dist, dim=-1)

    accuracy = (x_indices == j_indices).sum().to(dtype=torch.float32)
    return accuracy.item()/j_indices.numel()


def loss_fn_MMNet(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion, num_layers, ser_only=False):
    if (ser_only):
        SER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
        return SER_final
    else:
        x_out = torch.cat(list_batch_x_predicted, dim=0)
        x = x.repeat(num_layers, 1)
        loss = criterion(x_out.double(), x.double())
        SER_final = sym_detection(list_batch_x_predicted[-1].double(), j_indices, real_QAM_const, imag_QAM_const)
            
        return loss, SER_final

def complex2real(R, I):
    bs = R.shape[0]
    w1 = torch.cat((R, -1. * I), dim=3)
    w2 = torch.cat((I, R), dim=3)
    thetaMMNet = torch.cat((w1, w2), dim=2)
    
    return thetaMMNet


def loss_fn_eval(batch_x, list_batch_x_predicted,j_indices, real_QAM_const, imag_QAM_const, generator, ser_only=False):
    loss = []

    for batch_x_predicted in enumerate(list_batch_x_predicted):       
        loss_index = torch.mean(torch.mean(torch.pow((batch_x - batch_x_predicted[1]),2), dim=1))
        loss.append(loss_index)
     
    SER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
    
    return loss[-1], SER_final


def loss_fn_theta_fast(thetaHyperNet, thetaHyperNetVec, ThetaMMNet, ThetaVecMMNet, MMNet_batch_size, num_layers, device, ser_only=False):

    loss_theta = torch.mean(torch.norm((thetaHyperNet - ThetaMMNet.to(device=device)),1, dim=1), dim=0)
    loss_theta_vec = torch.mean(torch.norm((thetaHyperNetVec - ThetaVecMMNet.to(device=device)),1, dim=1), dim=0)
    
    return (loss_theta + loss_theta_vec)



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
    np.random.shuffle(a)

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


def getTheta(Theta, num_layers, NT, NR):
    ThetaReal = torch.empty(num_layers, NT, NR)
    ThetaImag = torch.empty(num_layers, NT, NR)
    ThetaVec = torch.empty(num_layers, 2 * NT, 1)
    
    for ii in range(num_layers):
        ThetaReal[ii, :, :] = Theta[0 + 3 * ii]
        ThetaImag[ii, :, :] = Theta[0 + 3 * ii + 1]
        ThetaVec[ii,:,:] = Theta[0 + 3 * ii + 2]
        
    return ThetaReal, ThetaImag, ThetaVec

def getBatchHMMnet(batch_size, num_layers, PATH, NT, NR):
    H = torch.empty(batch_size, 2*NR, 2*NT)
    Theta_MMNet = []
    Theta_Vec = []
    ThetaReal_MMNet = torch.empty(batch_size, num_layers, NT, NR)
    ThetaImag_MMNet = torch.empty(batch_size, num_layers, NT, NR)
    ThetaVec_MMNet = torch.empty(batch_size, num_layers, NT)
    
    for bs in range(batch_size):
        with open(PATH + str(bs), 'rb') as fp:
            Haux, Theta = pkl.load(fp)
            
        H[bs,:,:] = Haux[0,:,:]
        
        ThetaReal, ThetaImag, ThetaVec = getTheta(Theta, num_layers, NT, NR)
        ThetaReal_MMNet[bs,:, :,:] = ThetaReal
        ThetaImag_MMNet[bs,:, :,:] = ThetaImag
        Theta_MMNet.append(Theta)
        Theta_Vec.append(ThetaVec)
        
    return H, Theta_MMNet, ThetaReal_MMNet, ThetaImag_MMNet, Theta_Vec

def processThetaMMNet(ThetaMMNet, Theta_vec, batch_size, num_layers, NT, NR):
    thetaMMNet = torch.zeros((batch_size, num_layers, 2*NT, 2*NR))
    thetaVec = torch.zeros((batch_size, num_layers, 2*NT, 1))

    for layer in range(0, num_layers):
        for bs in range(batch_size): #cambiar el 10
            Wr = ThetaMMNet[bs][0 + 3 * layer]
            Wi = ThetaMMNet[bs][0 + 3 * layer + 1]
            w1 = torch.cat((Wr, -1. * Wi), dim=2)
            w2 = torch.cat((Wi, Wr), dim=2)
            thetaMMNet[bs,layer,:,:] = torch.cat((w1, w2), dim=1)
            thetaVec[bs, layer,:,:] = Theta_vec[bs][layer]
    thetaMMNet = torch.reshape(thetaMMNet, shape=(batch_size * num_layers, 2*NT, 2*NR))
    thetaMMNet = torch.reshape(thetaMMNet, shape=(-1, 2*NT * 2*NR))
    thetaVec = torch.reshape(thetaVec, shape=(batch_size * num_layers, 2*NT, 1))
    thetaVec = torch.squeeze(thetaVec, dim=-1)
    
    return thetaMMNet, thetaVec


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