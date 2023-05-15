import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy.sparse as sp
from sklearn.preprocessing import normalize

torch_install=False

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)


class scnocd(object):
    
    def check_torch(self):
        global torch_install
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            torch_install=True
            print('torch have been install version:',torch.__version__)
        except ImportError:
            raise ImportError(
                'Please install the pytorch: `conda install -c conda-forge pytorch` or `pip install pytorch`.'
            )

    def __init__(self,adata):
        self.check_torch()
        global torch_install
        if torch_install==True:
            global_imports("torch")
            global_imports("torch.nn","nn")
            global_imports("torch.nn.functional","F")
            globals()['nocd'] = __import__("Pyomic.nocd",fromlist=['nocd'])
        self.adata_raw=adata
        self.adata=adata.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      
    def matrix_transform(self):
        
        try:
            self.adata.obsp['connectivities']
        except NameError:
            var_exists = False
            print('......You need to calculate the neighbor by sc.pp.neighbors')
            return None
        else:
            var_exists = True
        
        try:
            self.adata.obs['leiden']
        except NameError:
            var_exists = False
            print('......You need to calculate the leiden by sc.tl.leiden')
            return None
        else:
            var_exists = True
            
        self.X=sp.csr_matrix(self.adata.X)
        self.A=self.adata.obsp['connectivities']
        self.Z_gt=pd.get_dummies(self.adata.obs['leiden']).values
        self.Z_gt=self.Z_gt.astype(np.float32)
        
        self.N, self.K = self.Z_gt.shape

 
    def matrix_normalize(self,cuda=False):
        self.x_norm = normalize(self.X)
        self.x_norm=nocd.utils.to_sparse_tensor(self.x_norm,cuda=cuda)

     
    def GNN_configure(self,hidden_size=128,
                     weight_decay=1e-2,
                     dropout=0.5,
                     batch_norm=True,
                     lr=1e-3,
                     max_epochs=500,
                     display_step=25,
                     balance_loss=True,
                     stochastic_loss=True,
                     batch_size=20000):
        
        self.hidden_sizes = [hidden_size]    # hidden sizes of the GNN
        self.weight_decay = weight_decay     # strength of L2 regularization on GNN weights
        self.dropout = dropout           # whether to use dropout
        self.batch_norm = batch_norm       # whether to use batch norm
        self.lr = lr              # learning rate
        self.max_epochs = max_epochs        # number of epochs to train
        self.display_step = display_step       # how often to compute validation loss
        self.balance_loss = balance_loss     # whether to use balanced loss
        self.stochastic_loss = stochastic_loss  # whether to use stochastic or full-batch training
        self.batch_size = batch_size      # batch size (only for stochastic training)

    
    def GNN_preprocess(self,num_workers=5,
                      ):
        self.sampler = nocd.sampler.get_edge_sampler(self.A, self.batch_size, self.batch_size, num_workers=num_workers)
        self.gnn = nocd.nn.GCN(self.x_norm.shape[1], self.hidden_sizes, self.K, 
                               batch_norm=self.batch_norm, dropout=self.dropout).to(self.device)
        self.adj_norm = self.gnn.normalize_adj(self.A)
        self.decoder = nocd.nn.BerpoDecoder(self.N, self.A.nnz, balance_loss=self.balance_loss)
        self.opt = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)

       
    def get_nmi(self,thresh=0.5):
        """Compute Overlapping NMI of the communities predicted by the GNN."""
        self.gnn.eval()
        Z = F.relu(self.gnn(self.x_norm, self.adj_norm))
        Z_pred = Z.cpu().detach().numpy() > thresh
        nmi = nocd.metrics.overlapping_nmi(Z_pred, self.Z_gt)
        return nmi
    
    
    def GNN_model(self):
        val_loss = np.inf
        validation_fn = lambda: val_loss
        early_stopping = nocd.train.NoImprovementStopping(validation_fn, patience=10)
        self.model_saver = nocd.train.ModelSaver(self.gnn)

        for epoch, batch in enumerate(self.sampler):
            if epoch > self.max_epochs:
                break
            if epoch % 25 == 0:
                with torch.no_grad():
                    self.gnn.eval()
                    # Compute validation loss
                    Z = F.relu(self.gnn(self.x_norm, self.adj_norm))
                    val_loss = self.decoder.loss_full(Z, self.A)
                    print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, nmi = {self.get_nmi():.2f}')

                    # Check if it's time for early stopping / to save the model
                    early_stopping.next_step()
                    if early_stopping.should_save():
                        self.model_saver.save()
                    if early_stopping.should_stop():
                        print(f'Breaking due to early stopping at epoch {epoch}')
                        break

            # Training step
            self.gnn.train()
            self.opt.zero_grad()
            Z = F.relu(self.gnn(self.x_norm, self.adj_norm))
            ones_idx, zeros_idx = batch
            if self.stochastic_loss:
                loss = self.decoder.loss_batch(Z, ones_idx, zeros_idx)
            else:
                loss = self.decoder.loss_full(Z, A)
            loss += nocd.utils.l2_reg_loss(self.gnn, scale=self.weight_decay)
            loss.backward()
            self.opt.step()

           
    def GNN_result(self,thresh=0.5):
        thresh = thresh

        Z = F.relu(self.gnn(self.x_norm, self.adj_norm))
        self.Z_pred = Z.cpu().detach().numpy() > thresh
        self.model_saver.restore()
        print(f'Final nmi = {self.get_nmi(thresh):.3f}')

      
    def GNN_plot(self,figsize=[10,10],markersize=0.05):
        plt.figure(figsize=figsize)
        z = np.argmax(self.Z_pred, 1)
        o = np.argsort(z)
        nocd.utils.plot_sparse_clustered_adjacency(self.A, self.K, z, o, markersize=markersize)

       
    def calculate_nocd(self):
        zpred=self.Z_pred+0
        pr=np.argmax(zpred, axis=-1)
        prf=pd.DataFrame(pr)
        prf.index=self.adata.obs.index

        pred=pd.DataFrame(zpred)
        pred.index=self.adata.obs.index
        m=(pred==1).sum(axis=1)
        n=pd.DataFrame(m)
        n[n>1]=-1
        
        df=n.merge(prf, how='inner', left_index=True, right_index=True)
        df1=df.sort_values("0_x")
        
        k=n.loc[~(n==1).all(axis=1)]
        df1.drop(df1.head(len(k)).index,inplace=True) 
        df2=df1.drop(["0_x"],axis=1)
        df3=df2.rename(columns={"0_y":0})
        
        con=pd.concat([k,df3],axis=0)
        
        print('......add nocd result to adata.obs')
        self.adata.obs['nocd']=con
    
    
    def cal_nocd(self):
        #pred matrix
        pred_pd=pd.DataFrame(self.Z_pred+0)
        pred_pd.index=self.adata.obs.index

        #nocd result
        nocd_res=[]
        for cell in pred_pd.index:
            nocd_type=''
            for i in pred_pd.loc[cell][pred_pd.loc[cell]!=0].index:
                if nocd_type=='':
                    nocd_type+=str(i)
                else:
                    nocd_type+='-'+str(i)
            nocd_res.append(nocd_type)
        nocd_res
        print('......add nocd result to adata.obs')
        self.adata.obs['nocd_n']=nocd_res
        