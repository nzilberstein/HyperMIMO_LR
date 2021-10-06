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

num_layers_mainNN = 6

class HyperNetwork(nn.Module):
    def __init__(self, NT, NR, device):
        super(HyperNetwork, self).__init__()
        self.input_shape = 2 * NT * 2 * NR
        self.num_layers = num_layers_mainNN 
        self.output_shape = self.num_layers * (2 * NT * 2 * NR + 2 * NT)
        self.device = device
        self.hidden1 = 100

        
        self.layer1 = nn.Linear(self.input_shape , self.input_shape).double()
        self.layer2 = nn.Linear(self.input_shape , self.hidden1).double()
        self.layer3 = nn.Linear(self.hidden1, self.output_shape).double()

        self.layer1.weight = torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2.weight = torch.nn.init.xavier_uniform_(self.layer2.weight)
        self.layer3.weight = torch.nn.init.xavier_uniform_(self.layer3.weight)


    def process_forward(self, H):

        Hflat = torch.flatten(H, start_dim=1)
        output1 = self.layer1(Hflat.double())
        norm_output1 = F.elu(output1).double()
        output2 = self.layer2(norm_output1.double())
        norm_output2 = F.elu(output2).double()

        paramsMainNet = self.layer3(norm_output2.double())
        
        return F.elu(paramsMainNet).double()

    def forward(self, H):
        return self.process_forward(H)

