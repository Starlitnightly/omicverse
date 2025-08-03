import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from .mnn_utils import create_dictionary_mnn
from .STALIGNER import STAligner

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def train_STAGATE(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAligner',
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=0, save_reconstrction=False,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = STAligner(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out)  # F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    z, out = model(data.x, data.edge_index)

    STAGATE_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = STAGATE_rep

    if save_loss:
        adata.uns['STAGATE_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['STAGATE_ReX'] = ReX

    return adata


def train_STAligner(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAligner',
                    gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                    random_seed=666, iter_comb=None, knn_neigh=100,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())
    edgeList = adata.uns['edgeList']
    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                prune_edge_index=torch.LongTensor(np.array([])),
                x=torch.FloatTensor(adata.X.todense()))
    data = data.to(device)

    model = STAligner(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Pretrain with STAGATE...')
    for epoch in tqdm(range(0, 500)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)

        loss = F.mse_loss(data.x, out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    with torch.no_grad():
        z, _ = model(data.x, data.edge_index)
    adata.obsm['STAGATE'] = z.cpu().detach().numpy()

    print('Train with STAligner...')
    for epoch in tqdm(range(500, n_epochs)):
        if epoch % 100 == 0 or epoch == 500:
            if verbose:
                print('Update spot triplets at epoch ' + str(epoch))
            adata.obsm['STAGATE'] = z.cpu().detach().numpy()

            # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
            # not all points have MNN achors
            mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                       iter_comb=iter_comb, verbose=0)

            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                        adata.obs['batch_name'] == section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    ## np.random.choice(mnn_dict[batch_pair][anchor])
                    positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        mse_loss = F.mse_loss(data.x, out)

        anchor_arr = z[anchor_ind,]
        positive_arr = z[positive_ind,]
        negative_arr = z[negative_ind,]

        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

        loss = mse_loss + tri_output
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    #
    model.eval()
    adata.obsm[key_added] = z.cpu().detach().numpy()
    return adata


def train_STAligner_subgraph(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAligner',
                             gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                             random_seed=666, iter_comb=None, knn_neigh=100, Batch_list=None,
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.
    To deal with large-scale data with multiple slices and reduce GPU memory usage, each slice is considered as a subgraph for training.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []
    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]
        edge_index = np.nonzero(adata_tmp.uns['adj'])
        data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp.X.todense())))

    loader = DataLoader(data_list, batch_size=1, shuffle=True)

    model = STAligner(hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Pretrain with STAGATE...')
    for epoch in tqdm(range(0, 500)):
        for batch in loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            z, out = model(batch.x, batch.edge_index)

            loss = F.mse_loss(batch.x, out)  # +adv_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

    with torch.no_grad():
        z_list = []
        for batch in data_list:
            z, _ = model.cpu()(batch.x, batch.edge_index)
            z_list.append(z.cpu().detach().numpy())
    adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)
    model = model.to(device)

    print('Train with STAligner...')
    for epoch in tqdm(range(500, n_epochs)):
        if epoch % 100 == 0 or epoch == 500:
            if verbose:
                print('Update spot triplets at epoch ' + str(epoch))

            with torch.no_grad():
                z_list = []
                for batch in data_list:
                    z, _ = model.cpu()(batch.x, batch.edge_index)
                    z_list.append(z.cpu().detach().numpy())
            adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)
            model = model.to(device)

            pair_data_list = []
            for comb in iter_comb:
                #print(comb)
                i, j = comb[0], comb[1]
                batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]
                mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAGATE', batch_name='batch_name',
                                                           k=knn_neigh,
                                                           iter_comb=None, verbose=0)

                batchname_list = batch_pair.obs['batch_name']
                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cellname_by_batch_dict[section_ids[batch_id]] = batch_pair.obs_names[
                        batch_pair.obs['batch_name'] == section_ids[batch_id]].values

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

                edge_list_1 = np.nonzero(Batch_list[i].uns['adj'])
                max_ind = edge_list_1[0].max()
                edge_list_2 = np.nonzero(Batch_list[j].uns['adj'])
                edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
                edge_list = [edge_list_1, edge_list_2]
                edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]), np.append(edge_list[0][1], edge_list[1][1])]
                pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           anchor_ind=torch.LongTensor(np.array(anchor_ind)),
                                           positive_ind=torch.LongTensor(np.array(positive_ind)),
                                           negative_ind=torch.LongTensor(np.array(negative_ind)),
                                           x=torch.FloatTensor(batch_pair.X.todense())))

            # for temp in pair_data_list:
            #     temp.to(device)
            pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=True)

        for batch in pair_loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            z, out = model(batch.x, batch.edge_index)
            mse_loss = F.mse_loss(batch.x, out)

            anchor_arr = z[batch.anchor_ind,]
            positive_arr = z[batch.positive_ind,]
            negative_arr = z[batch.negative_ind,]

            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='sum')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            loss = mse_loss + tri_output
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for batch in data_list:
            z, _ = model.cpu()(batch.x, batch.edge_index)
            z_list.append(z.cpu().detach().numpy())
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)
    return adata
