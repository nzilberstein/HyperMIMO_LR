# Robust MIMO Detection using Hypernetworks with Learned Regularizers

This repository contains the implementation of the "Robust MIMO Detection using Hypernetworks with Learned Regularizers".


It is separated in folders, so each detector generates an output and then in the folder results is a function to generate them. For each detector,

  1. First you will need to train the model
  2. Then generate the results. The output will be in pkl format
  3. Move everyone to the main folder to run the jupyter plots.ipynb



General Requirements:

Python3.5 or above version
Pytorch CUDA
Numpy
Matplotlib
Gurobi optimizer with Python engine


Obs:
Sample_generator file and DetNet are based on the code from: https://github.com/krpratik/RE-MIMO ("RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection")
