from typing import List, Tuple, Optional
import numpy as np
from anndata import AnnData
import scanpy as sc
import pandas as pd
from tqdm import tqdm
#
from scipy.optimize import nnls
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
#from .helper import intersect, kl_divergence_backend, to_dense_array, extract_exp, extract_data_matrix, process_anndata, construct_graph, distances_cal, dist_cal, scale_num
from .helper import *
    
def ot_alignment(
    adata1: AnnData,  
    adata2: AnnData,
    alpha: float = 0.1, 
    dissimilarity: str = 'scaled_euc',
    n_components: int = 30,  
    G_init = None,
    p_distribution = None, 
    q_distribution = None, 
    numItermax: int = 200, 
    norm: str = 'l2', 
    backend = None,  
    return_obj: bool = False,
    verbose: bool = True, 
    k: int = 10,
    graph_mode: str = "connectivity",
    aware_spatial: bool = True,
    aware_multi: bool = True,
    aware_power: int = 2,
    **kwargs) :
    
    """
    Calculates and returns optimal alignment spot and single cell data. 
    
    Args:
        adata1: Spatial transcriptomic data.
        adata2: Single cell multi-omics data.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        dissimilarity: Expression dissimilarity measure: "count_euc", "data_euc", "scaled_euc", "pca_euc", "kl".
        n_components: Number of pca dimensions selected when processing input data
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        p_distribution (array-like, optional): Distribution of adata1 spots, otherwise default is uniform.
        q_distribution (array-like, optional): Distribution of adata2 cells, otherwise default is uniform.
        numItermax: Max number of iterations.
        norm: Determines what sort of normalization to run on low dimensional representation of adata2, Default="l2".
        backend: Type of backend to run calculations.
        return_obj: Determines whether to use returns objective function output of FGW-OT.
        verbose: Prints loss when optimizing the optimal transport formulation. Default=True.
        k: Number of neighbors to be used when constructing kNN graph.
        graph_mode: "connectivity" or "distance". Determines whether to use a connectivity graph or a distance graph.Default="connectivity".
        aware_power: Type aware parameter. The greater the parameter, the greater the distance between different areas/types of spots/cells in the graph.
        aware_spatial: Determines whether to adjust the distance between spots according to areas (or other meta info)
        aware_multi: Determines whether to adjust the distance between cells according to types (or other meta info)
        
   
    Returns:
        - Alignment results between spots and cells.

        If ``return_obj = True``, additionally returns:
        
        - Objective function output of FGW-OT.
    """
    import ot
    if backend is None:
        backend=ot.backend.NumpyBackend()
    nx = backend
    n1 = adata1.shape[0]
    n2 = adata2.shape[0]
    
    # Calculate expression dissimilarity
    if dissimilarity == 'pca_euc':
        print('Calculating dissimilarity using euclidean distance on pca low dimensional...')
        # Merge tow anndata
        ad1=adata1.copy()
        ad1.X=ad1.layers['counts']
        ad2=adata2.copy()
        ad2.X=ad2.layers['counts']
        adata_merge = ad1.concatenate(ad2,index_unique=None)
        # Process the merged anndata
        adata_merge = process_anndata(adata_merge,ndims = n_components)
        reduc_spot = adata_merge.obsm["X_pca"][0:n1,]
        reduc_single = adata_merge.obsm["X_pca"][n1:n1+n2,]
        # reduc_spot = adata_merge.X[0:n1,]
        # reduc_single = adata_merge.X[n1:n1+n2,]
        M = ot.dist(reduc_spot, reduc_single, metric='euclidean')
        M /= M.max()
        M = nx.from_numpy(M)
        del ad1,ad2

    if dissimilarity == 'scaled_euc':
        print('Calculating dissimilarity using euclidean distance on scaled data...')
        ad1=adata1.copy()
        ad1.X=ad1.layers['counts']
        ad2=adata2.copy()
        ad2.X=ad2.layers['counts']
        adata_merge = ad1.concatenate(ad2,index_unique=None)
        # Process the merged anndata
        adata_merge = process_anndata(adata_merge,ndims = n_components,scale=True,pca=False)
        reduc_spot = adata_merge.X[0:n1,]
        reduc_single = adata_merge.X[n1:n1+n2,]
        M = ot.dist(reduc_spot, reduc_single, metric='euclidean')
        M /= M.max()
        M = nx.from_numpy(M)
        del ad1,ad2

    if dissimilarity == 'data_euc':
        print('Calculating dissimilarity using euclidean distance on normalized data...')
        # Merge tow anndata
        ad1=adata1.copy()
        ad1.X=ad1.layers['counts']
        ad2=adata2.copy()
        ad2.X=ad2.layers['counts']
        adata_merge = ad1.concatenate(ad2,index_unique=None)
        # Process the merged anndata
        adata_merge = process_anndata(adata_merge,ndims = n_components,scale=False,pca=False)
        reduc_spot = adata_merge.X[0:n1,]
        reduc_single = adata_merge.X[n1:n1+n2,]
        M = ot.dist(reduc_spot, reduc_single, metric='euclidean')
        M /= M.max()
        M = nx.from_numpy(M)
        del ad1,ad2

    if dissimilarity == 'count_euc':
        print('Calculating dissimilarity using euclidean distance on count data...')
        # Calculate expression dissimilarity
        ad1=adata1.copy()
        ad1.X=ad1.layers['counts']
        ad2=adata2.copy()
        ad2.X=ad2.layers['counts']
        #adata_merge = ad1.concatenate(ad2,index_unique=None)
        A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(ad1,rep = None))), nx.from_numpy(to_dense_array(extract_data_matrix(ad2,rep = None)))
        M = ot.dist(A_X,B_X)
        M /= M.max()
        M = nx.from_numpy(M)
        del ad1,ad2
        
    if dissimilarity == 'kl':
        print('Computing dissimilarity using kl divergence...')
        ad1=adata1.copy()
        ad1.X=ad1.layers['counts']
        ad2=adata2.copy()
        ad2.X=ad2.layers['counts']
        A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(ad1,rep = None))), nx.from_numpy(to_dense_array(extract_data_matrix(ad2,rep = None)))
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence_backend(s_A, s_B)
        M = nx.from_numpy(M) #r:ncellA*c:ncellB
        del ad1,ad2

    # Construct the graph
    location_array = adata1.obsm['spatial'].values
    reduction_array = adata2.obsm['reduction'].values
    reduction_array =normalize(reduction_array, norm=norm, axis=1)
    #Xgraph = construct_graph(location_array, k=k, mode=graph_mode, type_aware=spot_meta, aware_power=0)###break up
    print('Constructing '+str(graph_mode)+"...")
    print('k = '+str(k))
    Xgraph = construct_graph(location_array, k=k, mode=graph_mode)
    ygraph = construct_graph(reduction_array, k=k, mode=graph_mode)

    # Adjust the distance according to meta info
    type_aware1 = None
    type_aware2 = None
    if aware_spatial:
        print('aware_spatial = True')
        type_aware_dict = {
        'spot': pd.Series(adata1.obs.index.tolist(),index=adata1.obs.index.tolist()),
        'spot_type': pd.Series(adata1.obs['type'],index=adata1.obs.index.tolist())
        }
        type_aware1 = pd.DataFrame(type_aware_dict)
    if aware_multi:
        print('aware_multi = True')
        type_aware_dict = {
        'single': pd.Series(adata2.obs.index.tolist(),index=adata2.obs.index.tolist()),
        'single_type': pd.Series(adata2.obs['type'],index=adata2.obs.index.tolist())
        }
        type_aware2 = pd.DataFrame(type_aware_dict)
    print('aware power = '+str(aware_power))
    Cx = distances_cal(Xgraph,type_aware=type_aware1, aware_power=aware_power)
    Cy = distances_cal(ygraph,type_aware=type_aware2, aware_power=aware_power)
    Cx = nx.from_numpy(Cx)
    Cy = nx.from_numpy(Cy)

    # Init distributions
    if p_distribution is None:
        p = np.ones((n1,)) / n1
        p = nx.from_numpy(p)
    else:
        p = nx.from_numpy(p_distribution)
    if q_distribution is None:
        q = np.ones((n2,)) / n2
        q = nx.from_numpy(q)
    else:
        q = nx.from_numpy(q_distribution)

    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
   
    print('Running OT...')
    print('alpha = '+str(alpha))
    pi, logw = my_ot(M, Cx, Cy, p, q, G_init = G_init, loss_fun='square_loss', 
                     alpha= alpha, log=True, numItermax=numItermax,verbose=verbose)
    pi = nx.to_numpy(pi)
    print("OT done!")
    #obj = nx.to_numpy(logw['fgw_dist'])
    out_data = pd.DataFrame(pi)
    out_data.columns = adata2.obsm['reduction'].index
    out_data.index =  adata1.obsm['spatial'].index

    # Filter the results
    out_data['spot']=out_data.index
    out_data = out_data.melt(id_vars=['spot'], var_name ='cell', value_name='value')
    out_data = out_data.sort_values(by="value",ascending=False)

    if return_obj:
        return pi, out_data
    return out_data



def assign_coord(
    adata1: AnnData,
    adata2: AnnData,
    out_data: pd.DataFrame, 
    non_zero_probabilities: bool = True,
    no_repeated_cells: bool = True,
    top_num: int = None,
    expected_num: pd.DataFrame = None,
    random: bool = False,
    normalize: bool = True,
    ) :
    """
    Assign coordinates for single cells according to the output of ot alignment.
    
    Args:
        adata1: Spatial transcriptomic data.
        adata2: Single cell multi-omics data.
        out_data:  Alignment results between spots and cells from previous step.
        non_zero_probabilities: Determines whether to remove 0 frome alignment results. Default=True.
        no_repeated_cells: Determines whether to allow a cell to be used multiple times when allocating coordinates. Default=True.
        top_num: The maximum number of cells allocated in a spot, Default=5.
        expected_num: DataFrame specifying the expected number of cells per spot. Default is None.
        random: Determines whether to randomly assign cell coordinates or assign coordinates based on pearson correlation coefficient. Default=False.
        expected_num: DataFrame specifying the expected number of cells per spot. Default is None.
    
    Returns:
    Returns:
        - Spatial assignment of cells.
    """
    
    print('Assigning spatial coordinates to cells...')
    print('random = '+str(random))
    if top_num is not None:
        print("Maximum number of cells assigned to spots: "+str(top_num))
        cell_num = {spot: top_num for spot in pd.unique(adata1.obs_names)}
    if expected_num is not None:
        print("Determine the number of cells allocated to each spot based on the input information")
        cell_num = expected_num['cell_num'].to_dict()

    if non_zero_probabilities :
        out_data = out_data[out_data['value']>0]
    if no_repeated_cells :   
        out_data = out_data.sort_values(by="value",ascending=False)
        out_data = out_data[out_data.duplicated('cell') == False]

    if normalize:
        print('Normalizing the alignment results...')
        
        adata1_copy=adata1.copy()
        adata1_copy.X=adata1_copy.layers['counts']
        adata2_copy=adata2.copy()
        adata2_copy.X=adata2_copy.layers['counts']

        adata1_copy = process_anndata(adata1_copy,ndims=50,scale=False,pca=False)
        adata2_copy = process_anndata(adata2_copy,ndims=50,scale=False,pca=False)
    else:
        adata1_copy=adata1.copy()
        adata2_copy=adata2.copy()
        
    common_genes = intersect(adata1_copy.var.index, adata2_copy.var.index)
    adata1_copy = adata1_copy[:, common_genes]
    adata2_copy = adata2_copy[:, common_genes]
    
    adata1_copy = adata1_copy[adata1_copy.obs.index.isin(pd.unique(out_data.spot))]
    adata2_copy = adata2_copy[adata2_copy.obs.index.isin(pd.unique(out_data.cell))]

    res2 = pd.DataFrame(columns=adata2_copy.var_names, index=adata2_copy.obs['type'].astype('category').cat.categories)
    
    for clust in adata2_copy.obs['type'].astype('category').cat.categories: 
        res2.loc[clust] = adata2_copy[adata2_copy.obs['type'].isin([clust]),:].X.mean(0)
    res2 = np.array(res2)
    res2 = np.transpose(res2)  
    data = extract_exp(adata1_copy)
    ratio_df = pd.DataFrame(columns=adata2_copy.obs['type'].astype('category').cat.categories, index=pd.unique(out_data.spot))   
    for spot in pd.unique(out_data.spot):
        res1 = data.loc[spot]
        res1 = res1.T.values  # (gene_num, 1)
        res1 = res1.reshape(res1.shape[0],)
        ratio_sub = nnls(res2, res1)[0]
        
        ratio_sum = np.sum([ratio_sub], axis=1)[0]
        if ratio_sum == 0:  
            ratio_sub = [0] * len(ratio_sub)
        else:
            ratio_sub = (ratio_sub / ratio_sum).tolist()
        ratio_sub = np.round(np.array(ratio_sub)*cell_num[spot])
        ratio_df.loc[spot] = ratio_sub

    meta1_dict = {
        'spot': pd.Series(adata1.obs.index.tolist(),index=adata1.obs.index.tolist()),
        'spot_type': pd.Series(adata1.obs['type'],index=adata1.obs.index.tolist())
    }
    meta1 = pd.DataFrame(meta1_dict)
    out_data = pd.merge(out_data, meta1, on='spot',how="left")

    meta2_dict = {
        'spot': pd.Series(adata2.obs.index.tolist(),index=adata2.obs.index.tolist()),
        'spot_type': pd.Series(adata2.obs['type'],index=adata2.obs.index.tolist())
    }
    meta2 = pd.DataFrame(meta2_dict)
    meta2.columns = ['cell','cell_type']
    out_data = pd.merge(out_data, meta2, on='cell',how="left")


    # Assign cell to spot
    decon_df = pd.DataFrame(columns=out_data.columns)
    for spot in pd.unique(out_data.spot):
        spot_ratio = ratio_df.loc[spot]
        spot_ot = out_data.loc[out_data['spot'] == spot]
        decon_spot1 = pd.DataFrame(columns=out_data.columns)
        for cluster_id in range(0,len(spot_ratio)):
            cluster = spot_ratio.index[cluster_id]
            decon_num = spot_ratio[cluster_id]
            decon_spot_ot = spot_ot.loc[spot_ot['cell_type'] == cluster][0:int(decon_num)]
            decon_spot1 = pd.concat([decon_spot1,decon_spot_ot])
        decon_num = decon_spot1.shape[0]
        if decon_num < cell_num[spot]:
            rest_spot_ot = spot_ot.drop(decon_spot1.index)
            rest_spot_ot = rest_spot_ot.sort_values(by="value",ascending=False)
            decon_spot2 = rest_spot_ot.iloc[0:(cell_num[spot]-decon_num)]
            decon_spot = pd.concat([decon_spot1,decon_spot2])
        elif decon_num > 0 :
            decon_spot = decon_spot1
        decon_df = pd.concat([decon_df,decon_spot])
        
    out_data = decon_df.groupby('spot').apply(lambda x: x.nlargest(cell_num[x.name], 'value'))

    # Adjust cell coord
    if random:
        ## Calculate radius
        coord = adata1.obsm['spatial'].copy()
        a = coord[['x', 'y']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        coord['spot'] = coord.index
        
        ## Calculate coord randomly  
        if len(list(out_data.index.names)) > 1:
            out_data.index = out_data.index.droplevel()
        df_meta = pd.merge(out_data,coord,on='spot',how="left")
        all_coord = df_meta[['x', 'y']].to_numpy()

        mean_radius = np.mean(radius)
        all_radius = [mean_radius] * all_coord.shape[0]

        length = np.random.uniform(0, all_radius)
        angle = np.pi * np.random.uniform(0, 2, all_coord.shape[0])
        x = all_coord[:, 0] + length * np.cos(angle)
        y = all_coord[:, 1] + length * np.sin(angle)
        cell_coord = {'Cell_xcoord': np.around(x, 2).tolist(), 'Cell_ycoord': np.around(y, 2).tolist()}
        df_cc = pd.DataFrame(cell_coord)
        df_meta = pd.concat([df_meta, df_cc], axis=1)
    else:
        ## Calculate radius
        coord = adata1.obsm['spatial'].copy()
        a = coord[['x', 'y']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        mean_radius = np.mean(radius)

        if len(list(out_data.index.names)) > 1:
            out_data.index = out_data.index.droplevel()

        ## Calculate dist of two spots
        dist = pd.DataFrame(columns=coord.index,index = coord.index)
        dist['spot1']=dist.index
        dist = dist.melt(id_vars=['spot1'], var_name ='spot2', value_name='value')
        coord1 = coord.copy()
        coord1.columns = ['x1','y1']
        coord1['spot1']=coord1.index
        coord2 = coord.copy()
        coord2.columns = ['x2','y2']
        coord2['spot2']=coord2.index
        dist = pd.merge(dist, coord1, on='spot1',how="left")
        dist = pd.merge(dist, coord2, on='spot2',how="left")
        
        ## Filter dist dataframe with n_neighber to speed up
        a = coord[['x', 'y']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        indices = pd.DataFrame(indices)
        nn_dic = {indices.index.tolist()[i]:coord.index.tolist()[i] for i in range(coord.shape[0])}
        indices = indices.replace(nn_dic)
        indices = indices.rename(columns={0:'spot1'})
        nn_dist = indices.melt(id_vars=['spot1'], var_name ='num', value_name='spot2')
        nn_dist["spot_pair"] = nn_dist["spot1"]+nn_dist["spot2"]
        dist["spot_pair"] = dist["spot1"]+dist["spot2"]
        dist = dist[dist['spot_pair']. isin(nn_dist["spot_pair"])]
        dist = dist.drop(['spot_pair'],axis=1)
        
        ## Filter spot that asigned no cells
        spot_num = coord.shape[0]
        spot_has_cell_num = len(np.unique(out_data.spot))
        dist = dist[dist.spot1.isin(pd.unique(out_data.spot))&dist.spot2.isin(pd.unique(out_data.spot))]
        print('There are '+str(spot_num)+' spots and '+str(spot_has_cell_num)+' of them were assigned cells.')
        
        
        dist_dis = []
        for i in dist.index:
            dist_dis.append(dist_cal(dist.loc[i].x1,dist.loc[i].y1,dist.loc[i].x2,dist.loc[i].y2))
        dist.value = dist_dis

        ## Select nearest neighbors of each spot
        dist_closest = dist[dist['value'] > 0]
        dist_closest = dist_closest[dist_closest['value'] < 1.5*min(dist_closest.value)]
        num_closest = pd.value_counts(dist_closest.spot2).max()

        ## Make gene expression data of mappeds spot and cells
        exp_adata2 = extract_exp(adata2_copy)
        exp_adata2_copy = exp_adata2.copy()
        exp_adata2_copy['cell'] = exp_adata2_copy.index
        exp_mapped = pd.merge(out_data[['spot','cell']], exp_adata2_copy, on='cell',how="left")
        exp_mapped = exp_mapped.drop(['cell'],axis=1)
        exp_mapped = exp_mapped.groupby('spot')[np.setdiff1d(exp_mapped.columns, 'spot')].mean()

        df_meta = pd.DataFrame(columns = list(out_data.columns)+['x','y','Cell_xcoord','Cell_ycoord'] )
        for each_spot in tqdm(np.unique(out_data.spot)):
            each_spot_x = coord.loc[each_spot].x
            each_spot_y = coord.loc[each_spot].y

            ### Claculate dist to neighbors of each spot
            dist_of_each_spot = dist[dist.spot1==each_spot]
            dist_of_each_spot = dist_of_each_spot[dist_of_each_spot['value'] < 3*mean_radius]
            dist_of_each_spot = dist_of_each_spot[dist_of_each_spot['value'] > 0]
            if dist_of_each_spot.shape[0] == 0 :
                dist_of_each_spot = dist[dist.spot1==each_spot]
                
            ### Add pseudo cell when neighbors are insufficient
            if dist_of_each_spot.shape[0] < num_closest:
                x_sum = (dist_of_each_spot.shape[0]+1)*each_spot_x
                y_sum = (dist_of_each_spot.shape[0]+1)*each_spot_y
                x_pseudo = x_sum-sum(dist_of_each_spot.x2)
                y_pseudo = y_sum-sum(dist_of_each_spot.y2)
                value_pseudo = dist_cal(each_spot_x,each_spot_y,x_pseudo,y_pseudo)
                pseudo_data = [each_spot,each_spot,value_pseudo,each_spot_x,each_spot_y,x_pseudo,y_pseudo]
                pseudo_data = pd.DataFrame(pseudo_data,columns=[each_spot],index=dist_of_each_spot.columns).T
                dist_of_each_spot =  pd.concat([dist_of_each_spot,pseudo_data])
                
            if dist_of_each_spot.shape[0] > num_closest:
                dist_of_each_spot.nsmallest(num_closest,"value",keep='all')
            

            ### Extract ot output of each spot
            spot_cell_ot = out_data[out_data.spot==each_spot].copy()
            exp_mapped = exp_mapped[exp_adata2.columns]
            spot_cell_ot.loc[:,'Cell_xcoord'] = each_spot_x
            spot_cell_ot.loc[:,'Cell_ycoord'] = each_spot_y
            spot_cell_ot.loc[:,'x'] = each_spot_x
            spot_cell_ot.loc[:,'y'] = each_spot_y

            ### Align cells according to pearson correlation coefficient calculated with neighbor spots
            
            for cell_self in spot_cell_ot.cell:
                exp_cell = exp_adata2.loc[cell_self].values
                neighbor_pearson = []
                for neighbor_spot in dist_of_each_spot.spot2:
                    exp_spot = exp_mapped.loc[neighbor_spot].values
                    pc = pearsonr(exp_cell,exp_spot)
                    neighbor_pearson.append(pc[0])

                if len(neighbor_pearson)>2:
                    neighbor_pearson_scaled = scale_num(neighbor_pearson)###scale to 0-1
                elif len(neighbor_pearson)>1:
                    neighbor_pearson_scaled = neighbor_pearson
                else:
                    neighbor_pearson_scaled = 0
                    
                dist_of_each_spot=dist_of_each_spot.copy()
                
                dist_of_each_spot.loc[:,'x_difference'] = dist_of_each_spot.x2 - dist_of_each_spot.x1
                dist_of_each_spot.loc[:,'y_difference'] = dist_of_each_spot.y2 - dist_of_each_spot.y1
                
                x_map = np.mean(dist_of_each_spot.x_difference * neighbor_pearson_scaled + dist_of_each_spot.x1)
                y_map = np.mean(dist_of_each_spot.y_difference * neighbor_pearson_scaled + dist_of_each_spot.y1)
                spot_cell_ot.loc[spot_cell_ot.cell==cell_self,'Cell_xcoord'] = x_map
                spot_cell_ot.loc[spot_cell_ot.cell==cell_self,'Cell_ycoord'] = y_map

            ### Adjust coord to make cells more distributed
            if spot_cell_ot.shape[0] > 1:
                x_midpoint = np.mean(spot_cell_ot.Cell_xcoord)
                y_midpoint = np.mean(spot_cell_ot.Cell_ycoord)
                spot_cell_ot.Cell_xcoord = spot_cell_ot.Cell_xcoord + each_spot_x - x_midpoint
                spot_cell_ot.Cell_ycoord = spot_cell_ot.Cell_ycoord + each_spot_y - y_midpoint
                x_dif = spot_cell_ot.Cell_xcoord - each_spot_x
                y_dif = spot_cell_ot.Cell_ycoord - each_spot_y
                #### Restrict coord to the scope of the spot
                squ = x_dif * x_dif + y_dif * y_dif
                # if spot_cell_ot.shape[0] > 2:
                ratio = mean_radius/max(squ ** 0.5)
                spot_cell_ot.Cell_xcoord = x_dif * ratio + each_spot_x
                spot_cell_ot.Cell_ycoord = y_dif * ratio + each_spot_y
                # else :
                #     ratio = mean_radius/max(squ ** 0.5)
                #     spot_cell_ot.Cell_xcoord = x_dif * ratio /2 + each_spot_x
                #     spot_cell_ot.Cell_ycoord = y_dif * ratio /2 + each_spot_y
                
            df_meta = pd.concat([df_meta, spot_cell_ot])
            
    print('Assignment done!')
            
    return df_meta

def my_ot(M, C1, C2, p, q, G_init = None, loss_fun='square_loss', alpha=0.1, 
          armijo=False, log=False,numItermax=200,numItermaxEmd=10e6, use_gpu = False, **kwargs):
    """
    Adapted fused_gromov_wasserstein with G_init (inital mapping).
    Also added capability of utilizing different POT backends to speed up computation.
    
    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """
    import ot
    p, q = ot.utils.list_to_array(p, q)
    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = ot.backend.get_backend(p0, q0, C10, C20, M0)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1/nx.sum(G_init)) * G_init
        if use_gpu:
            G0 = G0.cuda()
    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)
    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)
    if log:
        if ot.__version__ < '0.9.0':
            res, log = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, numItermaxEmd=numItermaxEmd, **kwargs)
        else:
            res, log = ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, numItermaxEmd=numItermaxEmd, **kwargs)
        
        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        log['u'] = log['u']
        log['v'] = log['v']
        return res, log
    else:
        if ot.__version__ < '0.9.0':
            return ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
        else:
            return ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
        
