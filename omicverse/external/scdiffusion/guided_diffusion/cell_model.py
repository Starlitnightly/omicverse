import torch
import torch.nn as nn

from .nn import (
    linear,
    timestep_embedding,
)

class TimeEmbedding(nn.Module):  
    def __init__(self, hidden_dim):  
        super(TimeEmbedding, self).__init__()  
        self.time_embed = nn.Sequential(  
            nn.Linear(hidden_dim, hidden_dim),  
            nn.SiLU(),  
            nn.Linear(hidden_dim, hidden_dim),  
        )  
        self.hidden_dim = hidden_dim
  
    def forward(self, t):  
        return self.time_embed(timestep_embedding(t, self.hidden_dim).squeeze(1))  

class ResidualBlock(nn.Module):  
    def __init__(self, in_features, out_features, time_features):  
        super(ResidualBlock, self).__init__()  
        self.fc = nn.Linear(in_features, out_features)  
        self.norm = nn.LayerNorm(out_features) 
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            linear(
                time_features,
                out_features,
            ),
        ) 
        self.act = nn.SiLU()  
        self.drop = nn.Dropout(0)  
  
    def forward(self, x, emb):  
        h = self.fc(x)  
        h = h + self.emb_layer(emb)  
        h = self.norm(h)  
        h = self.act(h)  
        h = self.drop(h)  
        return h  
  
class Cell_Unet(nn.Module):  
    def __init__(self, input_dim=2, hidden_num=[2000,1000,500,500], dropout=0.1):  
        super(Cell_Unet, self).__init__()  
        self.hidden_num = hidden_num  
  
        self.time_embedding = TimeEmbedding(hidden_num[0])  
  
        # Create layers dynamically  
        self.layers = nn.ModuleList()  

        self.layers.append(ResidualBlock(input_dim, hidden_num[0], hidden_num[0]))

        for i in range(len(hidden_num)-1):  
            self.layers.append(ResidualBlock(hidden_num[i], hidden_num[i+1], hidden_num[0]))  
  
        self.reverse_layers = nn.ModuleList()  
        for i in reversed(range(len(hidden_num)-1)):  
            self.reverse_layers.append(ResidualBlock(hidden_num[i+1], hidden_num[i], hidden_num[0]))  
  
        self.out1 = nn.Linear(hidden_num[0], int(hidden_num[1]*2))  
        self.norm_out = nn.LayerNorm(int(hidden_num[1]*2))
        self.out2 = nn.Linear(int(hidden_num[1]*2), input_dim, bias=True)

        self.act = nn.SiLU()  
        self.drop = nn.Dropout(dropout)  
  
    def forward(self, x_input, t, y=None):  
        emb = self.time_embedding(t)  
        x = x_input.float()  
  
        # Forward pass with history saving  
        history = []  
        for layer in self.layers:  
            x = layer(x, emb)  
            history.append(x)  
        
        history.pop()
  
        # Reverse pass with skip connections  
        for layer in self.reverse_layers:  
            x = layer(x, emb)  
            x = x + history.pop()  # Skip connection  
  
        x = self.out1(x)  
        x = self.norm_out(x)
        x = self.act(x)  
        x = self.out2(x)  
        return x  


class Cell_classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_num=[2000,1000,500,200], num_class=11, dropout = 0.1):
        super().__init__()
        self.num_class = num_class
        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.drop_rate = dropout

        self.time_embed = nn.Sequential(
            linear(hidden_num[0], hidden_num[0]),
            nn.SiLU(),
            linear(hidden_num[0], hidden_num[0]),
        )

        self.fc1 = nn.Linear(input_dim, hidden_num[0], bias=True)
        self.emb_layers1 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[0],
            ),
        )
        self.norm1 = nn.BatchNorm1d(hidden_num[0])
        
        self.fc2 = nn.Linear(hidden_num[0], hidden_num[1], bias=True)
        self.emb_layers2 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[1],
            ),
        )
        self.norm2 = nn.BatchNorm1d(hidden_num[1])

        self.fc3 = nn.Linear(hidden_num[1], hidden_num[2], bias=True)
        self.emb_layers3 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[2],
            ),
        )
        self.norm3 = nn.BatchNorm1d(hidden_num[2])

        self.act = torch.nn.SiLU()
        self.drop = nn.Dropout(self.drop_rate)
        self.out = nn.Linear(hidden_num[2], num_class, bias=True)


    def forward(self, x_input, t):
        emb = self.time_embed(timestep_embedding(t, self.hidden_num[0]).squeeze(1))

        x = self.fc1(x_input)
        x = x+self.emb_layers1(emb)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)                  
        x = x+self.emb_layers2(emb)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)                  
        x = self.norm3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.out(x)
        return x
