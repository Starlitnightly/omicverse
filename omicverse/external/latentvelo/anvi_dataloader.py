import torch as th
import numpy as np
import scanpy as sc
import scipy as scp

from sklearn.model_selection import train_test_split

from .utils import normalize, sparse_mx_to_torch_sparse_tensor
from .dataloader_utils import handle_adata_view, ensure_csr_csc

class ANVIDataset(th.utils.data.Dataset):
    def __init__(self, adata, batch_size, shuffle=True, test=0.1, random_seed=42):

        # Handle AnnData views - convert to regular AnnData to avoid COO matrix issues
        adata = handle_adata_view(adata)
        # Ensure sparse matrices use CSR/CSC formats (no COO) to avoid indexing errors
        adata = ensure_csr_csc(adata)

        if shuffle:
            np.random.seed(random_seed)
            selected = np.random.choice(adata.shape[0], size = adata.shape[0], replace=False)
            adata = adata[selected]

        velo_genes_mask = np.zeros(adata.layers['spliced'].shape)
        velo_genes_mask[:,adata.var['velocity_genes'].values] = 1
        adata.layers['velo_genes_mask'] = velo_genes_mask
        print(velo_genes_mask[0].sum().astype(int), 'velocity genes used')
        
        adj = adata.obsp['adj']

        if 'exp_time' not in adata.obs.columns.values:
            adata.obs['exp_time'] = 0
        
        self.batches = []
        self.adj = []
        self.index_train = []
        self.index_test = []
        batch_ids = []
        index_test_ids = []
        
        for i in range(0, adata.shape[0], batch_size):
            
            index_train, index_test = train_test_split(np.arange(adata.X[i:i+batch_size,:].shape[0]), test_size=test, shuffle=False, random_state=i + 42)
            
            adj_i = adj[i:i+batch_size, i:i+batch_size]
            self.adj.append(normalize(0.9*adj_i + scp.sparse.eye(adj_i.shape[0])))
            
            self.batches.append([i, i+batch_size])
            self.index_train.append(index_train)
            self.index_test.append(index_test)
            index_test_ids.append(adata[i:i+batch_size,:].obs.index[index_test])

        index_test = np.concatenate(index_test_ids)
        adata.uns['index_test'] = index_test
        self.adata = adata.copy()
        self.IDs = np.arange(len(self.batches), dtype = int)
        
    def __len__(self):
        return self.IDs.shape[0]
    
    def __getitem__(self, index):
        selected = self.IDs[index]

        i = self.batches[selected][0]
        j = self.batches[selected][1]
        
        return {'S': th.Tensor(self.adata[i:j].layers['spliced_counts'].astype(float)),
                'U': th.Tensor(self.adata[i:j].layers['unspliced_counts'].astype(float)),
                'normedS': th.Tensor(self.adata[i:j].layers['spliced'].astype(float)),
                'normedU': th.Tensor(self.adata[i:j].layers['unspliced'].astype(float)),
                'maskS': th.Tensor(self.adata[i:j].layers['mask_spliced'].astype(float)),
                'maskU': th.Tensor(self.adata[i:j].layers['mask_unspliced'].astype(float)),
                'spliced_size_factor': th.Tensor(self.adata[i:j].obs['spliced_size_factor'].astype(float)),
                'unspliced_size_factor': th.Tensor(self.adata[i:j].obs['unspliced_size_factor'].astype(float)),
                'root': th.Tensor(self.adata[i:j].obs['root'].astype(int)).long(),
                'adj': self.adj[selected],
                'velo_genes_mask': th.Tensor(self.adata[i:j].layers['velo_genes_mask'].astype(float)),
                'adata': self.adata[i:j],
                'batch_onehot': th.Tensor(self.adata[i:j].obsm['batch_onehot']).long(),
                'batch_id': th.Tensor(self.adata[i:j].obs['batch_id']).float(),
                'celltype': th.Tensor(self.adata[i:j].obsm['celltype']).long(),
                'exp_time': th.Tensor(self.adata[i:j].obs['exp_time']).float(),
                'index_train': th.Tensor(self.index_train[selected]).long(),
                'index_test': th.Tensor(self.index_test[selected]).long(),
                'celltype_id': th.Tensor(self.adata[i:j].obs['celltype_id'])}
