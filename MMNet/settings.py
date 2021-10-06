import torch
import numpy as np

#torch.manual_seed(123)
#np.random.seed(123)


num_layers = 6
NT = 2
NR = 4
snrdb_classical_list = {16:np.arange(11.0, 22.0), 2: np.arange(5.0,15.0), 6: np.arange(10.0,21.0)}
mod_n = 4