# Robust MIMO Detection using Hypernetworks with Learned Regularizers

This repository contains the implementation of the "Robust MIMO Detection using Hypernetworks with Learned Regularizers".


It is separated in folders, so each detector generates an output and then in the folder results is a function to generate them. For each detector,

  1. First you will need to train the model
  2. Then generate the results. The output will be in pkl format. See that in each folder there is a test_model.py and test_model_time.py
      a. Test_model.py generates the SER as a function of SNR for the whole set (results in Fig. 2(a))
      b. Test_model_time.py generates the results for Fig. 2(b) and (c)
  4. Move the files containing the results to the main folder and run the jupyter plots.ipynb to obtain the plots of the paper.



General Requirements:

Python3.5 or above version
Pytorch CUDA
Numpy
Matplotlib
Gurobi optimizer with Python engine


Obs:
Sample_generator file and DetNet are based on the code from: https://github.com/krpratik/RE-MIMO ("RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection")
