import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import random
import os
from sklearn import preprocessing
from .pos_encoding import positional_encoding
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##################################################################################
# Neural network class
# Inputs: 
#   G: number of genes/features
#   S_hidden_list: list of hidden layer sizes for f_S 
#                  (e.g. [50] means f_S has one hidden layer of size 50)
#   A_hidden_list: list of hidden layer sizes for f_A 
#                  (e.g. [10,10] means f_A has two hidden layers, both of size 10)
#   activation_fn: activation function
##################################################################################

class GASTON(nn.Module):
    """
    Neural network class. Has two attributes: 
    (1) spatial embedding f_S : R^2 -> R, and 
    (2) expression function f_A : R -> R^G. 
    Each of these is parametrized by a neural network.
    
    Parameters
    ----------
    G
        number of genes/features
    S_hidden_list
        list of hidden layer sizes for f_S 
        (e.g. [50] means f_S has one hidden layer of size 50)
    A_hidden_list
        list of hidden layer sizes for f_A 
        (e.g. [10,10] means f_A has two hidden layers, both of size 10)
    activation_fn
        activation function for neural network
    pos_encoding
        positional encoding option
    embed_size
        positional encoding embedding size
    sigma
        positional encoding sigma hyperparameter
    """
    
    def __init__(
        self, 
        G, 
        S_hidden_list, 
        A_hidden_list,
        activation_fn=nn.ReLU(),
        pos_encoding=False,
        embed_size=4,
        sigma=0.1,
    ):
        super(GASTON, self).__init__()

        self.pos_encoding = pos_encoding
        self.embed_size = embed_size
        self.sigma = sigma

        input_size = 2*embed_size if self.pos_encoding else 2
        
        # create spatial embedding f_S
        S_layer_list=[input_size] + S_hidden_list + [1]
        S_layers=[]
        for l in range(len(S_layer_list)-1):
            # add linear layer
            S_layers.append(nn.Linear(S_layer_list[l], S_layer_list[l+1]))
            # add activation function except for last layer
            if l != len(S_layer_list)-2:
                S_layers.append(activation_fn)
                
        self.spatial_embedding=nn.Sequential(*S_layers)
        
        # create expression function f_A
        A_layer_list=[1] + A_hidden_list + [G]
        A_layers=[]
        for l in range(len(A_layer_list)-1):
            # add linear layer
            A_layers.append(nn.Linear(A_layer_list[l], A_layer_list[l+1]))
            # add activation function except for last layer
            if l != len(A_layer_list)-2:
                A_layers.append(activation_fn)
            
        self.expression_function=nn.Sequential(*A_layers)

    def forward(self, x):
        z = self.spatial_embedding(x) # relative depth
        return self.expression_function(z)

##################################################################################
# Train NN
# Inputs: 
#   model: GASTON object
#   S: torch Tensor (N x 2) containing spot locations
#   A: torch Tensor (N x G) containing features
#   epochs: number of epochs to train
#   batch_size: batch size



#   A_hidden_list: list of hidden layer sizes for f_A 
#                  (e.g. [10,10] means f_A has two hidden layers, both of size 10)
#   activation_fn: activation function
##################################################################################

def train(S, A, 
          gaston_model=None, S_hidden_list=None, A_hidden_list=None, activation_fn=nn.ReLU(),
          epochs=1000, batch_size=None, 
          checkpoint=100, save_dir=None, loss_reduction='mean',
          optim='sgd', lr=1e-3, weight_decay=0, momentum=0, seed=0, save_final=False,
          pos_encoding=False, embed_size=4, sigma=0.1):
    """
    Train GASTON model from scratch
    
    Parameters
    ----------
    model
        GASTON object
    S
        torch Tensor (N x 2) containing spot locations
    A
        torch Tensor (N x G) containing features
    epochs
        number of epochs to train
    batch_size
        batch size of neural network
    checkpoint
        save the current NN when the epoch is a multiple of checkpoint
    save_dir
        folder to save NN at checkpoints
    loss_reduction
        either 'mean' or 'sum' for MSELoss
    optim
        optimizer to use (currently supports either 'sgd' or 'adam')
    lr
        learning rate for the optimizer
    weight_decay
        weight decay parameter for optimizer
    momentum
        momentum parameter, if using SGD optimizer
    """
    set_seeds(seed)
    N,G=A.shape
    if gaston_model == None:
        gaston_model=GASTON(A.shape[1], S_hidden_list, A_hidden_list, activation_fn=activation_fn, pos_encoding=pos_encoding, embed_size=embed_size, sigma=sigma)
    
    if optim=='sgd':
        opt = torch.optim.SGD(gaston_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim=='adam':
        opt = torch.optim.Adam(gaston_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim=='adagrad':
        opt = torch.optim.Adagrad(gaston_model.parameters(), weight_decay=weight_decay)    
    loss_list=np.zeros(epochs)

    S_init = torch.clone(S)
    if gaston_model.pos_encoding:
        S = positional_encoding(S, gaston_model.embed_size, gaston_model.sigma)

    loss_function=torch.nn.MSELoss(reduction=loss_reduction)
    
    for epoch in range(epochs):
        if epoch%checkpoint==0:
            #print(f'epoch: {epoch}')
            if save_dir is not None:
                torch.save(gaston_model, f'{save_dir}/model_epoch_{epoch}.pt')
        
        if batch_size is not None:
            # take non-overlapping random samples of size batch_size
            permutation = torch.randperm(N)
            for i in range(0, N, batch_size):
                opt.zero_grad()
                indices = permutation[i:i+batch_size]

                S_ind=S[indices,:]
                S_ind.requires_grad_()

                A_ind=A[indices,:]

                loss = loss_function(gaston_model(S_ind), A_ind)
                loss_list[epoch] += loss.item()

                loss.backward()
                opt.step()
        else:
            opt.zero_grad()
            S.requires_grad_()

            loss = loss_function(gaston_model(S), A)
            loss_list[epoch] += loss.item()

            loss.backward()
            opt.step()

    if save_final:
        torch.save(gaston_model, f'{save_dir}/final_model.pt')
        np.savetxt(f'{save_dir}/loss_list.txt', loss_list)
        with open(f'{save_dir}/min_loss.txt', 'w') as f:
            f.write(str(min(loss_list)) + "\n")
        torch.save(A, f'{save_dir}/Atorch.pt')
        if gaston_model.pos_encoding:
            torch.save(S_init, f'{save_dir}/Storch.pt')
        else:
            torch.save(S, f'{save_dir}/Storch.pt')
    
    return gaston_model, loss_list
    

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_loss(mod, St, At):
    N,G=At.shape
    errr=(mod(St) - At)**2
    return torch.mean(errr)


def load_rescale_input_data(S, A):
  assert S.shape[0] == A.shape[0], 'Input and output files do not have same number of rows! Some spots are missing or do not have expression PC values!'
  
  scaler = preprocessing.StandardScaler().fit(A)
  A_scaled = scaler.transform(A)
  A_torch = torch.tensor(A_scaled,dtype=torch.float32)

  scaler = preprocessing.StandardScaler().fit(S)
  S_scaled = scaler.transform(S)
  S_torch = torch.tensor(S_scaled,dtype=torch.float32)

  return S_torch, A_torch
