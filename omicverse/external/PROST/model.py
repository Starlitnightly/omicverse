import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.nn.functional as F
from .layers import GraphAttentionLayer,SpGraphAttentionLayer
from .utils import mclust, sparse_mx_to_torch_sparse_tensor



class PROST_NN(nn.Module):
    def __init__(self, nfeat, embedding_size, cuda=False):
        super(PROST_NN, self).__init__()
        self.embedding_size = embedding_size
        self.cuda = cuda
        if self.cuda:
            if torch.cuda.is_available(): 
                print("Using cuda acceleration")
            else:
                raise ValueError("Cuda is unavailable, please set 'cuda=False'")

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.beta=0.5

        self.gal = GraphAttentionLayer(nfeat, embedding_size, 0.05, 0.15).to(self.device)
          
    def get_q(self, z):
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.beta) + 1e-8)
        q = q**(self.beta+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def KL_div(self, p, q):
        loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
        return loss
  
    def forward(self, x, adj):
        z = self.gal(x, adj)
        q = self.get_q(z)
        return z, q

    def train_(self, X, adj, init="mclust",n_clusters=7,res=0.1,tol=1e-3,lr=0.1, 
                max_epochs=500, seed = 818, update_interval=3):

        optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=5e-4)
        
        X = torch.FloatTensor(X).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device) 


        with torch.no_grad():
            features = self.gal(X, adj)
     
        #----------------------------------------------------------------           
        if init=="kmeans":
            print("\nInitializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())
            
        elif init=="mclust":
            print("\nInitializing cluster centers with mclust, n_clusters known")
            data = features.detach().cpu().numpy()
            self.n_clusters = n_clusters
            self.seed = seed
            y_pred = mclust(data, num_cluster = self.n_clusters, random_seed = self.seed)
            y_pred = y_pred.astype(int)

        elif init=="louvain":
            print("\nInitializing cluster centers with louvain, resolution = ", res)
            adata = sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
            
        elif init=="leiden":
            print("\nInitializing cluster centers with leiden, resolution = ", res)
            adata=sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.leiden(adata, resolution=res)
            y_pred = adata.obs['leiden'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.embedding_size))

        features = pd.DataFrame(features.detach().cpu().numpy()).reset_index(drop = True)
        Group = pd.Series(y_pred, index=np.arange(0,features.shape[0]), name="Group")
        Mergefeature = pd.concat([features,Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())       
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        
        #---------------------------------------------------------------- 
        with tqdm(total=max_epochs) as t:
            for epoch in range(max_epochs):            
                t.set_description('Epoch')
                self.train()
                
                if epoch%update_interval == 0:
                    _, Q = self.forward(X, adj)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    t.update(update_interval)
                    
                z,q = self(X, adj)
                p = self.target_distribution(Q.detach())
                
                loss = self.KL_div(p, q)

                optimizer.zero_grad()              
                loss.backward()
                optimizer.step()
    
                t.set_postfix(loss = loss.data.cpu().numpy())
                
                #Check stop criterion
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
                y_pred_last = y_pred
                if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    print("Total epoch:", epoch)
                    break

    def predict(self, X, adj):
        X = torch.FloatTensor(X).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
        z, q = self(X, adj)

        return z, q
    
    
class PROST_NN_sparse(nn.Module):
    def __init__(self, nfeat, embedding_size, cuda=False):
        super(PROST_NN_sparse, self).__init__()
        self.embedding_size = embedding_size
        self.cuda = cuda
        if self.cuda:
            if torch.cuda.is_available(): 
                print("Using cuda acceleration")
            else:
                raise ValueError("Cuda is unavailable, please set 'cuda=False'")

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.beta=0.5

        self.gal = SpGraphAttentionLayer(nfeat, embedding_size, 0.05, 0.15).to(self.device)

    def get_q(self, z):
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.beta) + 1e-8)
        q = q**(self.beta+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def KL_div(self, p, q):
        loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
        return loss
  
    def forward(self, x, adj):
        z = self.gal(x, adj)
        q = self.get_q(z)
        return z, q

    def train_(self, X, adj, init="mclust",n_clusters=7,res=0.1,tol=1e-3,lr=0.1, 
                max_epochs=500, seed = 818, update_interval=3):

        optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=5e-4)
        
        X = torch.FloatTensor(X).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(self.device) 


        with torch.no_grad():
            features = self.gal(X, adj)
     
        #----------------------------------------------------------------           
        if init=="kmeans":
            print("\nInitializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())
            
        elif init=="mclust":
            print("\nInitializing cluster centers with mclust, n_clusters known")
            data = features.detach().cpu().numpy()
            self.n_clusters = n_clusters
            self.seed = seed
            y_pred = mclust(data, num_cluster = self.n_clusters, random_seed = self.seed)
            y_pred = y_pred.astype(int)

        elif init=="louvain":
            print("\nInitializing cluster centers with louvain, resolution = ", res)
            adata = sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
            
        elif init=="leiden":
            print("\nInitializing cluster centers with leiden, resolution = ", res)
            adata=sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.leiden(adata, resolution=res)
            y_pred = adata.obs['leiden'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.embedding_size))

        features = pd.DataFrame(features.detach().cpu().numpy()).reset_index(drop = True)
        Group = pd.Series(y_pred, index=np.arange(0,features.shape[0]), name="Group")
        Mergefeature = pd.concat([features,Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())       
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        
        #----------------------------------------------------------------
        
        with tqdm(total=max_epochs) as t:
            for epoch in range(max_epochs):            
                t.set_description('Epoch')
                self.train()
                
                if epoch%update_interval == 0:
                    _, Q = self.forward(X, adj)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    t.update(update_interval)
                    
                z,q = self(X, adj)
                p = self.target_distribution(Q.detach())
                
                loss = self.KL_div(p, q)
   
                optimizer.zero_grad()              
                loss.backward()
                optimizer.step()
    
                t.set_postfix(loss = loss.data.cpu().numpy())
                
                #Check stop criterion
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
                y_pred_last = y_pred
                if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    print("Total epoch:", epoch)
                    break

    def predict(self, X, adj):
        X = torch.FloatTensor(X).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
        z, q = self(X, adj)

        return z, q
    