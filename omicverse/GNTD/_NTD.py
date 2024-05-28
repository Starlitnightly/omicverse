import torch
from torch.nn import Embedding, Linear, LeakyReLU, PReLU, Parameter

class NTD(torch.nn.Module):
    def __init__(self, n_x, n_y, n_g, rank, random_state=1234567):
        super().__init__()
        torch.manual_seed(random_state)
        
        # Define embedding layer along x, y, g modes
        self.embedding_x = Embedding(n_x, rank)
        self.embedding_y = Embedding(n_y, rank)
        self.embedding_g = Embedding(n_g, rank)
        # Define nonlinear mapping layer along x, y, g modes
        self.lin_x_1 = Linear(rank, rank)
        self.lin_y_1 = Linear(rank, rank)
        self.lin_g_1 = Linear(rank, rank)
        self.prelu = PReLU(init=0.9)

    def forward(self, x_index, y_index, g_index):
        
        # Linear factors
        x = self.embedding_x(x_index)
        y = self.embedding_y(y_index)
        g = self.embedding_g(g_index)
        
        # Nonlinear factors
        x = self.lin_x_1(x)
        x = self.prelu(x)
        y = self.lin_y_1(y)
        y = self.prelu(y)
        g = self.lin_g_1(g)
        g = self.prelu(g)
        
        # Nonlinear aggregation 
        o = torch.einsum('im,jm,km->ijk', g, x, y)
        o = o.relu_()
        
        return x, y, g, o