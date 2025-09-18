import torch as th
import torch.nn as nn
from .modules import MLP

# Detect device - use CUDA if available, otherwise use CPU
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class VelocityFieldReg(nn.Module):
    """
    Velocity field for the dynamics of zu, zs, and zr
    """
    def __init__(self, latent, h_dim, zr_dim, include_time = False, linear_splicing=True):
        super(VelocityFieldReg, self).__init__()
        
        self.latent=latent
        self.zr_dim = zr_dim
        self.h_dim = h_dim
        self.include_time=include_time

        if include_time:
            self.unspliced_net = MLP(latent+zr_dim+1, [latent+5], latent)
        else:
            self.unspliced_net = MLP(latent+zr_dim, [latent+5], latent)

        if linear_splicing:
            self.spliced_net = nn.Linear(2*latent, latent)
        else:
            self.spliced_net = MLP(2*latent, [latent+5], latent)
        
        self.reg_net = MLP(latent + zr_dim + h_dim, [latent+5], zr_dim)
        
        for m in self.unspliced_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)
        
        for m in self.spliced_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)
        
        for m in self.reg_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)
    
    def forward(self, t, z):
        t = t.repeat(z.shape[0], 1)
        
        zs, zu, zr, h = z[:,:self.latent], z[:,self.latent:2*self.latent], z[:,2*self.latent:2*self.latent+self.zr_dim], z[:,2*self.latent+self.zr_dim:]
        
        if self.include_time:
            unspliced_drift = self.unspliced_net(th.cat((zu, zr, t), dim=-1))
        else:
            unspliced_drift = self.unspliced_net(th.cat((zu, zr), dim=-1))
        
        spliced_drift = self.spliced_net(th.cat((zs, zu), dim=-1))
        reg_drift = self.reg_net(th.cat((zs, zr, h), dim=-1))
        
        return th.cat((spliced_drift, unspliced_drift, reg_drift,
                       th.zeros(spliced_drift.shape[0], self.h_dim).to(device)), dim=-1)
    
    def drift(self, z):
        
        zs, zu, zr, h, t = z[:,:self.latent], z[:,self.latent:2*self.latent], z[:,2*self.latent:2*self.latent+self.zr_dim], z[:,2*self.latent+self.zr_dim:2*self.latent+self.zr_dim+self.h_dim], z[:,2*self.latent+self.zr_dim+self.h_dim:]
        
        if self.include_time:
            unspliced_drift = self.unspliced_net(th.cat((zu, zr, t), dim=-1))
        else:
            unspliced_drift = self.unspliced_net(th.cat((zu, zr), dim=-1))
        
        spliced_drift = self.spliced_net(th.cat((zs, zu), dim=-1))
        reg_drift = self.reg_net(th.cat((zs, zr, h), dim=-1))
        
        return th.cat((spliced_drift, unspliced_drift, reg_drift), dim=-1)



class ATACRegVelocityField(nn.Module):

    def __init__(self, latent, h_dim, zr_dim, include_time = False):
        super(ATACRegVelocityField, self).__init__()
        
        self.latent=latent
        self.zr_dim=zr_dim
        self.include_time=include_time
        self.h_dim=h_dim
        
        if include_time:
            self.unspliced_net = MLP(latent+latent+1, [latent+5], latent)
        else:
            self.unspliced_net = MLP(latent+latent+zr_dim, [latent+5], latent)
        
        self.spliced_net = nn.Linear(2*latent, latent) 
        self.atac_net = MLP(latent + zr_dim + 1, [latent+5], latent)
        
        self.reg_net = MLP(latent + zr_dim + h_dim, [latent+5], zr_dim)
        
        for m in self.unspliced_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.spliced_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.atac_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.reg_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)
                
        
    def forward(self, t, z):
        t = t.repeat(z.shape[0], 1)
        
        zs, zu, za, zr, h = z[:,:self.latent], z[:,self.latent:2*self.latent], z[:,2*self.latent:2*self.latent+self.latent], z[:,3*self.latent:3*self.latent+self.zr_dim], z[:,3*self.latent+self.zr_dim:3*self.latent+self.zr_dim+self.h_dim]
        
        # dimensions of z are latent, control, control_dynparam
        if self.include_time:
            unspliced_drift = self.unspliced_net(th.cat((zu, za, t), dim=-1))
        else:
            unspliced_drift = self.unspliced_net(th.cat((zu, za, zr), dim=-1)) #0*zt
        
        spliced_drift = self.spliced_net(th.cat((zs, zu), dim=-1)) #, h
        #atac_drift = self.atac_net(th.cat((zs, za), dim=-1)) #, h
        atac_drift = self.atac_net(th.cat((za, zr, t), dim=-1)) #, h

        
        reg_drift = self.reg_net(th.cat((zs, zr, h), dim=-1))
        
        return th.cat((spliced_drift, unspliced_drift, atac_drift, reg_drift,
                       th.zeros(spliced_drift.shape[0], self.h_dim).to(device)), dim=-1)
    
    def drift(self, z):
        
        zs, zu, za, zr, h, t = z[:,:self.latent], z[:,self.latent:2*self.latent], z[:,2*self.latent:3*self.latent], z[:,3*self.latent:3*self.latent+self.zr_dim], z[:,3*self.latent+self.zr_dim:3*self.latent+self.zr_dim+self.h_dim], z[:,3*self.latent+self.zr_dim+self.h_dim:]  
        
        if self.include_time:
            unspliced_drift = self.unspliced_net(th.cat((zu, za, zr, t), dim=-1))
        else:
            unspliced_drift = self.unspliced_net(th.cat((zu, za, zr), dim=-1))
        
        spliced_drift = self.spliced_net(th.cat((zs, zu), dim=-1))
        atac_drift = self.atac_net(th.cat((za, zr, t), dim=-1))
        reg_drift = self.reg_net(th.cat((zs, zr, h), dim=-1))
        
        return th.cat((spliced_drift, unspliced_drift, atac_drift, reg_drift), dim=-1)
