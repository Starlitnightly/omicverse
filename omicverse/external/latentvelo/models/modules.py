import torch as th
from torch.nn import functional as F
import torch.nn as nn
from .gclayer import GraphConvolution as GCLayer
import numpy as np    


class MLP(nn.Module):
    """
    Simple dense neural network
    """
    def __init__(self, input, hidden, output, activation = nn.ELU(), bn = False):
        super(MLP, self).__init__()

        self.activation = activation
        self.input = input
        self.output = output
        self.hidden = hidden
        
        self.layers = []
        self.layers.append(nn.Linear(input, hidden[0]))
        if bn:
            self.layers.append(nn.BatchNorm1d(hidden[0]))
        self.layers.append(activation)
        
        for size in hidden[1:]:
            self.layers.append(nn.Linear(size, size))
            if bn:
                self.layers.append(nn.BatchNorm1d(size))
            self.layers.append(activation)
        self.layers.append(nn.Linear(hidden[-1], output))
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self, z):

        output = self.layers[0](z)
        for layer in self.layers[1:]:
            output = layer(output)
        return output


class GCN(nn.Module):
    """
    Graph convolutional network to utilize cell nearest neighbor graph
    """
    def __init__(self, input, output, gcn_layers, combine, activation = nn.ELU()):
        super(GCN, self).__init__()

        self.activation = activation
        self.combine = combine
        self.input = input
        
        self.gcn_layers = []
        for i in range(gcn_layers):
            self.gcn_layers.append(GCLayer(input, input, bias=True))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        if combine:
            self.linear = nn.Linear(2*input, output)
        else:
            self.linear = nn.Linear(input, output)
            
    def forward(self, z, adj):

        if self.combine:
            zs, zu = z[...,:self.input], z[...,self.input:]
            
            ys = self.activation(self.gcn_layers[0](self.activation(zs), adj))
            yu = self.activation(self.gcn_layers[0](self.activation(zu), adj))
            
            for gcn in self.gcn_layers[1:]:
                ys = self.activation(gcn(ys, adj))
                yu = self.activation(gcn(yu, adj))
            
            return self.linear(th.cat((ys, yu), dim=-1))
        
        else:
            y = self.activation(self.gcn_layers[0](self.activation(z), adj))
            
            for gcn in self.gcn_layers[1:]:
                y = self.activation(gcn(y, adj))
            
            return self.linear(y)

class GCNCombined(nn.Module):
    
    def __init__(self, input, output, gcn_layers, activation = nn.ELU()):
        super(GCNCombined, self).__init__()

        self.activation = activation
        self.input = input
        
        self.gcn_layers = []
        for i in range(gcn_layers):
            self.gcn_layers.append(GCLayer(input, input, bias=True))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)
        
        self.linear = nn.Linear(input, output)
            
    def forward(self, z, adj):
        
        y = self.activation(self.gcn_layers[0](self.activation(z), adj))
        
        for gcn in self.gcn_layers[1:]:
            y = self.activation(gcn(y, adj))
        
        return self.linear(y)




def create_encoder(observed, latent, encoder_hidden, batch_correction, batches, shared, encoder_bn):
    """
    Create encoders
    """
    if batch_correction:
        if shared:
            encoder_z0 = MLP(observed+batches, [encoder_hidden, encoder_hidden], 2*latent, bn=encoder_bn)
        else:
            encoder_z0_s = MLP(observed+batches, [encoder_hidden, encoder_hidden], 2*latent, bn=encoder_bn)
            encoder_z0_u = MLP(observed+batches, [encoder_hidden, encoder_hidden], 2*latent, bn=encoder_bn)
    else:
        if shared:
            encoder_z0 = MLP(observed, [encoder_hidden], 2*latent, bn=encoder_bn)
        else:
            encoder_z0_s = MLP(observed, [encoder_hidden], 2*latent, bn=encoder_bn)
            encoder_z0_u = MLP(observed, [encoder_hidden], 2*latent, bn=encoder_bn)
    
    if not batch_correction:
        if shared:
            encoder_z = GCN(2*latent, 2*latent, gcn_layers=2, combine=False)
        else:
            encoder_z_s = GCN(2*latent, 2*latent, gcn_layers=2, combine=False)
            encoder_z_u = GCN(2*latent, 2*latent, gcn_layers=2, combine=False)

    if batch_correction:
        return encoder_z0_s, encoder_z0_u, None, None
    else:
        return encoder_z0_s, encoder_z0_u, encoder_z_s, encoder_z_u

def create_decoder(observed, latent, decoder_hidden, linear_decoder, batch_correction, batches, shared, decoder_bn):
    """
    Create decoders
    """
    if batch_correction:
        if linear_decoder:
            if shared:
                decoder = []
                for i in range(batches):
                    decoder.append(nn.Linear(latent, observed))
                decoder=nn.ModuleList(decoder)
            else:
                decoder_s = []
                decoder_u = []
                for i in range(batches):
                    decoder_s.append(nn.Linear(latent, observed))
                    decoder_u.append(nn.Linear(latent, observed))
                decoder_s=nn.ModuleList(decoder_s)
                decoder_u=nn.ModuleList(decoder_u)
                
        else:
            if shared:
                decoder = MLP(latent+batches, [decoder_hidden], self.observed, bn=decoder_bn)
            else:
                decoder_s = MLP(latent+batches, [decoder_hidden], observed, bn=decoder_bn)
                decoder_u = MLP(latent+batches, [decoder_hidden], observed, bn=decoder_bn)
                    
    else:
        if linear_decoder:
            if shared:
                decoder = nn.Linear(latent, observed)
            else:
                decoder_s = nn.Linear(latent, observed)
                decoder_u = nn.Linear(latent, observed)
        else:
            if shared:
                decoder = MLP(latent, [decoder_hidden], observed, bn=decoder_bn)
            else:
                decoder_s = MLP(latent, [decoder_hidden], observed, bn=decoder_bn)
                decoder_u = MLP(latent, [decoder_hidden], observed, bn=decoder_bn)    

    return decoder_s, decoder_u

class ATACGCN(nn.Module):
    
    def __init__(self, input, output, gcn_layers, combine, activation = nn.ELU()):
        super(ATACGCN, self).__init__()

        self.activation = activation
        self.combine = combine
        self.input = input
        
        self.gcn_layers = []
        for i in range(gcn_layers):
            self.gcn_layers.append(GCLayer(input, input, bias=True))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        if combine:
            self.linear = nn.Linear(3*input, output)
        else:
            self.linear = nn.Linear(input, output)
            
    def forward(self, z, adj):

        if self.combine:
            zs, zu, za = z[...,:self.input], z[...,self.input:self.input*2], z[...,self.input*2:self.input*3]
            
            ys = self.activation(self.gcn_layers[0](self.activation(zs), adj))
            yu = self.activation(self.gcn_layers[0](self.activation(zu), adj))
            ya = self.activation(self.gcn_layers[0](self.activation(za), adj))
            
            for gcn in self.gcn_layers[1:]:
                ys = self.activation(gcn(ys, adj))
                yu = self.activation(gcn(yu, adj))
                ya = self.activation(gcn(ya, adj))
            
            return self.linear(th.cat((ys, yu, za), dim=-1))
        
        else:
            y = self.activation(self.gcn_layers[0](self.activation(z), adj))
            
            for gcn in self.gcn_layers[1:]:
                y = self.activation(gcn(y, adj))
            
            return self.linear(y)
