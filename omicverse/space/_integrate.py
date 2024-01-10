
"""
__author__ = "Xiang Zhou"
__email__ = "xzhou@amss.ac.cn"
__citation__ = Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894–906 (2023).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import sklearn.neighbors


import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..STAligner.mnn_utils import create_dictionary_mnn
from ..STAligner.STALIGNER import STAligner



def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']


    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    #X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)       
    #cells = np.array(X.index)
    cells = np.array(adata.obs_names) # LeiHu update, reducing the demand of memory.
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!") 
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G

    
class pySTAligner(object):

    def __init__(self,adata, hidden_dims=[512, 30],n_epochs=1000, lr=0.001,batch_key='batch_name', key_added='STAligner',
                             gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                             random_seed=666, iter_comb=None, knn_neigh=100, Batch_list=None,
                device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        
        self.device = device    
        
        
        section_ids = np.array(adata.obs[batch_key].unique())

        comm_gene = adata.var_names
        data_list = []
        for adata_tmp in Batch_list:
            adata_tmp = adata_tmp[:, comm_gene]
            edge_index = np.nonzero(adata_tmp.uns['adj'])
            data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp.X.todense())))

        loader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.loader=loader  
        self.adata = adata
        self.data_list = data_list

        # hyper-parameters  
        self.lr=lr
        self.section_ids = section_ids
        self.n_epochs = n_epochs
        self.weight_decay=weight_decay
        self.hidden_dims = hidden_dims
        self.key_added = key_added
        self.gradient_clipping = gradient_clipping
        self.random_seed = random_seed
        self.margin = margin
        self.verbose = verbose
        self.iter_comb = iter_comb
        self.knn_neigh = knn_neigh
        self.Batch_list = Batch_list
        self.batch_key = batch_key
        
        self.model = STAligner(hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if verbose:
            print(self.model)

    def train(self):

        seed = self.random_seed
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        print('Pretrain with STAGATE...')
        for epoch in tqdm(range(0, 500)):
            for batch in self.loader:
                self.model.train()
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                z, out = self.model(batch.x, batch.edge_index)

                loss = F.mse_loss(batch.x, out)  # +adv_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()


        with torch.no_grad():
            z_list = []
            for batch in self.data_list:
                z, _ = self.model.cpu()(batch.x, batch.edge_index)
                z_list.append(z.cpu().detach().numpy())
        self.adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)
        self.model = self.model.to(self.device)


        print('Train with STAligner...')
        for epoch in tqdm(range(500, self.n_epochs)):
            if epoch % 100 == 0 or epoch == 500:
                if self.verbose:
                    print('Update spot triplets at epoch ' + str(epoch))
                    
                with torch.no_grad():
                    z_list = []
                    for batch in self.data_list:
                        z, _ = self.model.cpu()(batch.x, batch.edge_index)
                        z_list.append(z.cpu().detach().numpy())

                self.adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)
                self.model = self.model.to(self.device)

                pair_data_list = []

                for comb in self.iter_comb:
                    #print(comb)
                    i, j = comb[0], comb[1]
                    batch_pair = self.adata[self.adata.obs[self.batch_key].isin([self.section_ids[i], self.section_ids[j]])]
                    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAGATE', batch_name=self.batch_key,
                                                           k=self.knn_neigh,
                                                           iter_comb=None, verbose=0)
                    
                    batchname_list = batch_pair.obs[self.batch_key]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(self.section_ids)):
                        cellname_by_batch_dict[self.section_ids[batch_id]] = batch_pair.obs_names[
                            batch_pair.obs[self.batch_key] == self.section_ids[batch_id]].values
                    
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for batch_pair_name in mnn_dict.keys():  # pairwise compare for multiple batches
                        for anchor in mnn_dict[batch_pair_name].keys():
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair_name][anchor][0]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(batch_pair.obs_names), range(0, batch_pair.shape[0])))
                    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
                    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
                    negative_ind = list(map(lambda _: batch_as_dict[_], negative_list))

                    edge_list_1 = np.nonzero(self.Batch_list[i].uns['adj'])

                    max_ind = edge_list_1[0].max()
                    edge_list_2 = np.nonzero(self.Batch_list[j].uns['adj'])

                    edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
                    edge_list = [edge_list_1, edge_list_2]

                    edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]), np.append(edge_list[0][1], edge_list[1][1])]

                    pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           anchor_ind=torch.LongTensor(np.array(anchor_ind)),
                                           positive_ind=torch.LongTensor(np.array(positive_ind)),
                                           negative_ind=torch.LongTensor(np.array(negative_ind)),
                                           x=batch_pair.X)) #torch.FloatTensor(batch_pair.X.todense())
                
                # for temp in pair_data_list:
                #     temp.to(device)
                pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=True)

            for batch in pair_loader:
                self.model.train()
                self.optimizer.zero_grad()
                
                batch.x = torch.FloatTensor(batch.x[0].todense())
                batch = batch.to(self.device)
                z, out = self.model(batch.x, batch.edge_index)
                mse_loss = F.mse_loss(batch.x, out)

                anchor_arr = z[batch.anchor_ind,]
                positive_arr = z[batch.positive_ind,]
                negative_arr = z[batch.negative_ind,]

                triplet_loss = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='sum')
                tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

                loss = mse_loss + tri_output
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                self.optimizer.step()



    def predicted(self):
        self.model.eval()
        with torch.no_grad():
            z_list = []
            for batch in self.data_list:
                z, _ = self.model.cpu()(batch.x, batch.edge_index)
                z_list.append(z.cpu().detach().numpy())

        self.adata.obsm[self.key_added] = np.concatenate(z_list, axis=0)
        return self.adata
    