import torch,random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances,pairwise_distances_chunked,confusion_matrix
import scanpy as sc
from scipy.sparse import csr_matrix as csr
from .utils import coords2adjacentmat

def space_project(
    sdata_inte,
    idx_source,
    idx_target,
    raw_layer,
    source_sample,
    target_sample,
    coords_source,
    coords_target,
    output_path,
    source_sample_ctype_col,
    target_cell_pc_feature = None,
    source_cell_pc_feature = None,
    k2 = 1,
    ifplot = True,
    umap_feature = 'X_umap',
    ave_dist_fold = 2,
    batch_t = '',
    alignment_shift_adjustment = 50,
    color_dict = None,
    adjust_shift = False,
    metric_t = 'cosine',
    working_memory_t = 1000
    ):
    sdata_ref = sdata_inte[idx_target,:].copy()
    source_feat = sdata_inte[idx_source,:].layers[raw_layer].toarray()

    project_ind = np.zeros([np.sum(idx_target),k2]).astype(int)
    project_weight = np.zeros_like(project_ind).astype(float)
    cdists = np.zeros_like(project_ind).astype(float)
    physical_dist = np.zeros_like(project_ind).astype(float)
    all_avg_feat = np.zeros([np.sum(idx_target),source_feat.shape[1]]).astype(float)

    if source_sample_ctype_col is not None:
        for ctype_t in np.unique(sdata_inte[idx_target].obs[source_sample_ctype_col]):
            print(f'Start to project {ctype_t} cells:')
            idx_ctype_t = np.isin(sdata_inte[idx_target].obs[source_sample_ctype_col],ctype_t)
            ave_dist_t,_,_,_ = average_dist(coords_target[idx_ctype_t,:].copy(),working_memory_t=working_memory_t)
            dist_thres = ave_dist_fold * ave_dist_t + alignment_shift_adjustment
            if adjust_shift:
                coords_shift = group_shift(target_cell_pc_feature[idx_ctype_t,:], source_cell_pc_feature, coords_target[idx_ctype_t,:], coords_source, working_memory_t = working_memory_t, metric_t = metric_t)
                coords_source_t = coords_source + coords_shift
                print(coords_shift)
            else:
                coords_source_t = coords_source.copy()
            project_ind[idx_ctype_t,:],project_weight[idx_ctype_t,:],cdists[idx_ctype_t,:],physical_dist[idx_ctype_t,:],all_avg_feat[idx_ctype_t,:] = physical_dist_priority_project(
                feat_target = target_cell_pc_feature[idx_ctype_t,:],
                feat_source = source_cell_pc_feature,
                coords_target = coords_target[idx_ctype_t,:],
                coords_source = coords_source_t,
                source_feat = source_feat,
                k2 = 1,
                pdist_thres = dist_thres,
                metric_t = metric_t,
                working_memory_t = working_memory_t)
    else:
        ave_dist_t,_,_,_ = average_dist(coords_target.copy(),working_memory_t=working_memory_t,strategy_t='delaunay')
        dist_thres = ave_dist_fold * ave_dist_t + alignment_shift_adjustment
        project_ind,project_weight,cdists,physical_dist,all_avg_feat = physical_dist_priority_project(
                feat_target = target_cell_pc_feature,
                feat_source = source_cell_pc_feature,
                coords_target = coords_target,
                coords_source = coords_source,
                source_feat = source_feat,
                k2 = 1,
                pdist_thres = dist_thres,
                working_memory_t = working_memory_t)

    umap_target = sdata_inte[idx_target,:].obsm[umap_feature]
    umap_source = sdata_inte[idx_source,:].obsm[umap_feature]

    sdata_ref.layers[f'{source_sample}_raw'] = csr(all_avg_feat)
    sdata_ref.layers[f'{target_sample}_norm1e4'] = csr(sc.pp.normalize_total(sdata_ref,target_sum=1e4,layer = f'{raw_layer}',inplace=False)['X'])
    sdata_ref.layers[f'{source_sample}_norm1e4'] = csr(sc.pp.normalize_total(sdata_ref,target_sum=1e4,layer = f'{source_sample}_raw',inplace=False)['X'])
    y_true_t = np.array(sdata_inte[idx_target].obs[source_sample_ctype_col].values) if source_sample_ctype_col is not None else None
    y_source = np.array(sdata_inte[idx_source].obs[source_sample_ctype_col].values) if source_sample_ctype_col is not None else None
    y_pred_t = y_source[project_ind[:,0]] if source_sample_ctype_col is not None else None
    torch.save([physical_dist,project_ind,coords_target,coords_source,y_true_t,y_pred_t,y_source,output_path,source_sample_ctype_col,umap_target,umap_source,source_sample,target_sample,cdists,k2],f'{output_path}/mid_result{batch_t}.pt')
    if ifplot == True:
        evaluation_project(
            physical_dist = physical_dist,
            project_ind = project_ind,
            coords_target = coords_target,
            coords_source = coords_source,
            y_true_t = y_true_t,
            y_pred_t = y_pred_t,
            y_source = y_source,
            output_path = output_path,
            source_sample_ctype_col = source_sample_ctype_col,
            umap_target = umap_target,
            umap_source = umap_source,
            source_sample = source_sample,
            target_sample = target_sample,
            cdists = cdists,
            batch_t = batch_t,
            color_dict = color_dict)
    return sdata_ref,[project_ind,project_weight,cdists,physical_dist]

def average_dist(coords,quantile_t = 0.99,working_memory_t = 1000,strategy_t = 'convex'):
    coords_t = pd.DataFrame(coords)
    coords_t.drop_duplicates(inplace = True)
    coords = np.array(coords_t)
    if coords.shape[0] > 5:
        delaunay_graph_t = coords2adjacentmat(coords,output_mode='raw',strategy_t = strategy_t)
        edges = np.array(delaunay_graph_t.edges())
        def reduce_func(chunk_t, start):
            return chunk_t
        dists = pairwise_distances_chunked(coords, coords, metric='euclidean', n_jobs=-1,working_memory = working_memory_t,reduce_func = reduce_func)
        edge_dist = []
        start_t = 0
        for dist_mat_t in dists:
            end_t = start_t + dist_mat_t.shape[0]
            idx_chunk = (start_t <= edges[:,0]) & (edges[:,0] < end_t)
            edge_t = edges[idx_chunk,:]
            edge_dist_t = np.array([dist_mat_t[node - start_t,val] for [node,val] in edge_t])
            edge_dist.extend(edge_dist_t)
            start_t = end_t
        filter_thres = np.quantile(edge_dist,quantile_t)
        for i,j in edges[edge_dist > filter_thres,:]:
            delaunay_graph_t.remove_edge(i,j)
        result_t = np.mean(np.array(edge_dist)[edge_dist <= filter_thres])
        return result_t,filter_thres,edge_dist,delaunay_graph_t
    else:
        dists = pairwise_distances(coords, coords, metric='euclidean', n_jobs=-1)
        result_t = np.mean(dists.flatten())
        return result_t,'','',''

def group_shift(feat_target, feat_source, coords_target_t, coords_source_t, working_memory_t = 1000, pencentile_t = 0.8, metric_t = 'cosine'):
    from sklearn.metrics import pairwise_distances_chunked
    print(f'Using {metric_t} distance to calculate group shift:')
    feat_similarity_ctype = np.vstack(list(pairwise_distances_chunked(feat_target, feat_source, metric=metric_t, n_jobs=-1, working_memory=working_memory_t)))
    num_anchor = int(feat_similarity_ctype.shape[0] * pencentile_t)
    anchor_rank = np.argpartition(feat_similarity_ctype, num_anchor - 1, axis=-1)[:,:num_anchor]
    anchors = []
    for i in range(num_anchor):
        anchors.extend(anchor_rank[:,i].tolist())
        anchors = list(set(anchors))
        if len(anchors) >= num_anchor:
            break
    coords_shift = np.median(coords_target_t,axis=0) - np.median(coords_source_t[np.array(anchors),:],axis=0)
    return coords_shift

def physical_dist_priority_project(feat_target, feat_source, coords_target, coords_source, source_feat = None, k2 = 1, k_extend = 20, pdist_thres = 200, working_memory_t = 1000, metric_t = 'cosine'):
    def reduce_func_cdist_priority(chunk_cdist, start):
        chunk_pdist = pairwise_distances(coords_target[start:(chunk_cdist.shape[0] + start),:],coords_source, metric='euclidean', n_jobs=-1)
        idx_pdist_t = chunk_pdist < pdist_thres
        idx_pdist_sum = idx_pdist_t.sum(1)
        idx_lessk2 = (idx_pdist_sum>= k2)
        cosine_knn_ind = np.zeros([chunk_cdist.shape[0],k2]).astype(int)
        cosine_knn_weight = np.zeros_like(cosine_knn_ind).astype(float)
        cosine_knn_cdist = np.zeros_like(cosine_knn_ind).astype(float)
        cosine_knn_physical_dist = np.zeros_like(cosine_knn_ind).astype(float)
        
        idx_narrow = np.where(idx_lessk2)[0]
        idx_narrow_reverse = np.where(np.logical_not(idx_lessk2))[0]

        for i in idx_narrow:
            idx_pdist_t_i = idx_pdist_t[i,:]
            idx_i = np.where(idx_pdist_t[i,:])[0]
            knn_ind_t = idx_i[np.argpartition(chunk_cdist[i,idx_pdist_t_i], k2 - 1, axis=-1)[:k2]]
            _,weight_cell,cdist_cosine = cosine_IDW(chunk_cdist[i,knn_ind_t],k2 = k2,need_filter=False)
            cosine_knn_ind[[i],:] = knn_ind_t
            cosine_knn_weight[[i],:] = weight_cell
            cosine_knn_cdist[[i],:] = cdist_cosine
            cosine_knn_physical_dist[[i],:] = chunk_pdist[i,knn_ind_t]
        if len(idx_narrow_reverse) > 0:
            for i in idx_narrow_reverse:
                idx_pdist_extend = np.argpartition(chunk_pdist[i,:], k_extend - 1, axis=-1)[:k_extend]
                knn_ind_t = idx_pdist_extend[np.argpartition(chunk_cdist[i,idx_pdist_extend], k2 - 1, axis=-1)[:k2]]
                _,weight_cell,cdist_cosine = cosine_IDW(chunk_cdist[i,knn_ind_t],k2 = k2,need_filter=False)
                cosine_knn_ind[[i],:] = knn_ind_t
                cosine_knn_weight[[i],:] = weight_cell
                cosine_knn_cdist[[i],:] = cdist_cosine
                cosine_knn_physical_dist[[i],:] = chunk_pdist[i,knn_ind_t]
        return cosine_knn_ind,cosine_knn_weight,cosine_knn_cdist,cosine_knn_physical_dist
    
    print(f'Using {metric_t} distance to calculate cell low dimensional distance:')
    dists = pairwise_distances_chunked(feat_target, feat_source, metric=metric_t, n_jobs=-1,working_memory = working_memory_t,reduce_func=reduce_func_cdist_priority)
    cosine_knn_inds = []
    cosine_k2nn_weights = []
    cosine_k2nn_cdists = []
    cosine_k2nn_physical_dists = []
    for output in tqdm(dists):
        cosine_knn_inds.append(output[0])
        cosine_k2nn_weights.append(output[1])
        cosine_k2nn_cdists.append(output[2])
        cosine_k2nn_physical_dists.append(output[3])

    all_cosine_knn_inds = np.concatenate(cosine_knn_inds)
    all_cosine_k2nn_weights = np.concatenate(cosine_k2nn_weights)
    all_cosine_k2nn_cdists = np.concatenate(cosine_k2nn_cdists)
    all_cosine_k2nn_physical_dists = np.concatenate(cosine_k2nn_physical_dists)
    
    if source_feat is not None:
        mask_idw = sparse_mask(all_cosine_k2nn_weights,all_cosine_knn_inds, source_feat.shape[0])
        all_avg_feat = mask_idw.dot(source_feat)
        return all_cosine_knn_inds,all_cosine_k2nn_weights,all_cosine_k2nn_cdists,all_cosine_k2nn_physical_dists,all_avg_feat
    else:
        return all_cosine_knn_inds,all_cosine_k2nn_weights,all_cosine_k2nn_cdists,all_cosine_k2nn_physical_dists


def sparse_mask(idw_t, ind : np.ndarray, n_cols : int, dtype=np.float64): # ind is indices with shape (num data points, indices), in the form of output of numpy.argpartition function
    # build csr matrix from scratch
    rows = np.repeat(np.arange(ind.shape[0]), ind.shape[1]) # gives like [1,1,1,2,2,2,3,3,3]
    cols = ind.flatten() # the col indices that should be 1
    data = idw_t.flatten() # Set to `1` each (row,column) pair
    return csr_matrix((data, (rows, cols)), shape=(ind.shape[0], n_cols), dtype=dtype)

def cosine_IDW(cosine_dist_t,k2=5,eps = 1e-6,need_filter = True,ifavg = False):
    if need_filter:
        idx_cosdist_t = np.argpartition(cosine_dist_t, k2 - 1, axis=-1)[:k2]
        cdist_cosine_t = cosine_dist_t[idx_cosdist_t]
    else:
        idx_cosdist_t = 0
        cdist_cosine_t = cosine_dist_t
    if ifavg:
        weight_cell_t = np.array([1/k2] * k2)
    else:
        weight_cell_t = IDW(cdist_cosine_t,eps)
    return idx_cosdist_t, weight_cell_t, cdist_cosine_t

def IDW(df_value,eps = 1e-6):
    weights = 1.0 /(df_value + eps).T
    weights /= weights.sum(axis=0)
    return weights.T

def evaluation_project(
    physical_dist,
    project_ind,
    coords_target,
    coords_source,
    y_true_t,
    y_pred_t,
    y_source,
    output_path,
    source_sample_ctype_col,
    umap_target = None,
    umap_source = None,
    source_sample = None,
    target_sample = None,
    cdists = None,
    batch_t = '',
    exclude_group = 'Other',
    color_dict = None,
    umap_examples = False):
    print(f'Generate evaluation plots:')
    plt.rcParams.update({'pdf.fonttype':42, 'font.size' : 15})
    plt.rcParams['axes.grid'] = False
    ### histogram ###
    cdist_hist(physical_dist.flatten(),range_t = [0,2000])
    plt.savefig(f'{output_path}/physical_dist_hist{batch_t}.pdf')
    cdist_hist(cdists.flatten(),range_t = [0,2])
    plt.savefig(f'{output_path}/cdist_hist{batch_t}.pdf')

    ### confusion matrix ###
    if source_sample_ctype_col is not None:
        if exclude_group is not None:
            idx_t = y_true_t != exclude_group
            y_true_t_use = y_true_t[idx_t]
            y_pred_t_use = y_pred_t[idx_t]
        else:
            y_true_t_use = y_true_t
            y_pred_t_use = y_pred_t
        confusion_mat_plot(y_true_t_use,y_pred_t_use)
        plt.savefig(f'{output_path}/confusion_mat_raw_with_label_{source_sample_ctype_col}{batch_t}.pdf')
        confusion_mat_plot(y_true_t_use,y_pred_t_use,withlabel = False)
        plt.savefig(f'{output_path}/confusion_mat_raw_without_label_{source_sample_ctype_col}{batch_t}.pdf')

    ### link plot 3d ###
    if color_dict is not None and source_sample_ctype_col is not None:
        color_target = [color_dict[x] for x in y_true_t]
        color_source = [color_dict[x] for x in y_source]
    else:
        color_target="#9295CA" 
        color_source='#E66665'
    link_plot_3d(project_ind, coords_target, coords_source, k = 1,figsize_t = [10,10], 
                sample_n=200, link_color_mask = None, 
                color_target = color_target, color_source = color_source,
                color_true = "#222222")
    plt.savefig(f'{output_path}/link_plot{batch_t}.pdf', dpi=300)

    ### Umap ###
    if umap_examples:
        cdist_check(cdists.copy(),project_ind.copy(),umap_target,umap_source,labels_t=[target_sample,source_sample],random_seed_t=0,figsize_t=[40,32])
        plt.savefig(f'{output_path}/umap_examples{batch_t}.pdf',dpi = 300)

#################### Visualization ####################

def cdist_hist(data_t,range_t = None,step = None):
    plt.figure(figsize=[5,5])
    plt.hist(data_t, bins='auto',alpha = 0.5,color = '#1073BC')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if type(range_t) != type(None):
        if type(step) != type(None):
            plt.xticks(np.arange(range_t[0], range_t[1] + 0.001, step),fontsize=20)
        else:
            plt.xticks(fontsize=20)
            plt.xlim(range_t[0], range_t[1])
    else:
        plt.xticks(fontsize=20)
    plt.tight_layout()

def confusion_mat_plot(y_true_t, y_pred_t, filter_thres = None, withlabel = True, fig_x = 60, fig_y = 20):
    plt.rcParams.update({'axes.labelsize' : 30,'pdf.fonttype':42,'axes.titlesize' : 30,'font.size': 15,'legend.markerscale' : 3})
    plt.rcParams['axes.grid'] = False
    TPrate = np.round(np.sum(y_pred_t == y_true_t) / len(y_true_t),2)
    uniq_t = np.unique(y_true_t,return_counts=True)
    if type(filter_thres) == type(None):
        labels_t = uniq_t[0]
    else:
        labels_t = uniq_t[0][uniq_t[1] >= filter_thres]
    plt.figure(figsize=[fig_x,fig_y])
    for idx_t, i in enumerate(['count','true','pred']):
        if i == 'count':
            normalize_t = None
            title_t = 'Counts (TP%%: %.2f)' % TPrate
        elif i == 'true':
            normalize_t = 'true'
            title_t = 'Sensitivity'
        elif i == 'pred':
            normalize_t = 'pred'
            title_t = 'Precision'
        plt.subplot(1,3,idx_t + 1)
        confusion_mat = confusion_matrix(y_true_t,y_pred_t,labels = labels_t, normalize = normalize_t)
        if i == 'count':
            vmax_t = np.max(confusion_mat)
        else:
            vmax_t = 1
        confusion_mat = pd.DataFrame(confusion_mat,columns=labels_t,index=labels_t)
        if withlabel:
            annot = np.diag(np.diag(confusion_mat.values.copy(),0),0)
            annot = np.round(annot,2)
            annot = annot.astype('str')
            annot[annot=='0.0']=''
            annot[annot=='0']=''
            sns.heatmap(confusion_mat,cmap = 'RdBu',center = 0,annot=annot,fmt='',square = True,vmax = vmax_t)
        else:
            sns.heatmap(confusion_mat,cmap = 'RdBu',center = 0,square = True,vmax = vmax_t)
        plt.title(title_t)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()

def cdist_check(cdist_t,cdist_idx,umap_coords0,umap_coords1, labels_t = ['query','ref'],random_seed_t = 2,figsize_t = [40,32],output_path_t = None):
    plt.rcParams.update({'xtick.labelsize' : 20,'ytick.labelsize':20, 'axes.labelsize' : 30, 'axes.titlesize' : 40,'axes.grid': False})
    random.seed(random_seed_t)
    sampled_points = np.sort(random.sample(list(range(0,cdist_idx.shape[0])),20))
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=figsize_t)
    axs = axs.flatten()
    for i in range(len(sampled_points)):
        idx_check = sampled_points[i]
        axs[i].scatter(umap_coords0[:,0],umap_coords0[:,1],s = 0.5,c = '#1f77b4',rasterized=True)
        axs[i].scatter(umap_coords1[:,0],umap_coords1[:,1],s = 0.5,c = '#E47E8B',rasterized=True)
        axs[i].scatter(umap_coords0[idx_check,0],umap_coords0[idx_check,1],s = 220,linewidth = 4,c = '#1f77b4',edgecolors = '#000000',label = labels_t[0],rasterized=False)
        axs[i].scatter(umap_coords1[cdist_idx[idx_check,0],0],umap_coords1[cdist_idx[idx_check,0],1],s = 220,linewidth=4,c = '#E47E8B',edgecolors = '#000000',label = labels_t[1], rasterized=False)
        axs[i].legend(scatterpoints=1,markerscale=2, fontsize=30)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title('cdist = ' + str(format(cdist_t[idx_check,0],'.2f')))
    if output_path_t is not None:
        plt.savefig(f'{output_path_t}/umap_examples.pdf',dpi = 300)
        plt.close('all')

def link_plot_3d(assign_mat, coords_target, coords_source, k, figsize_t = [15,20], sample_n=1000, link_color_mask=None, color_target="#9295CA", color_source='#E66665', color_true = "#999999", color_false = "#999999", remove_background = True):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    assert k == 1
    ax = plt.figure(figsize=figsize_t).add_subplot(projection='3d')
    xylim = max(coords_source.max(), coords_target.max())
    ax.set_xlim(0, xylim)
    ax.set_ylim(0, xylim)
    ax.set_zlim(-0.1, 1.1)
    ax.set_box_aspect([1,1,0.6])
    ax.view_init(elev=25)

    coordsidx_transfer_source_link = assign_mat[:, 0]
    
    coords_transfer_source_link = coords_source[coordsidx_transfer_source_link,:]
    t1 = np.vstack((coords_transfer_source_link[:,0],coords_transfer_source_link[:,1])) # source
    t2 = np.vstack((coords_target[:,0],coords_target[:,1])) # target
    
    downsample_indices = np.random.choice(range(coords_target.shape[0]), sample_n)
    
    if link_color_mask is not None:
        final_true_indices = np.intersect1d(downsample_indices, np.where(link_color_mask)[0])
        final_false_indices = np.intersect1d(downsample_indices, np.where(~link_color_mask)[0])
        segs = [[(*t2[:, i], 0), (*t1[:, i], 1)] for i in final_false_indices]
        line_collection = Line3DCollection(segs, colors=color_false, lw=0.5, linestyles='dashed')
        line_collection.set_rasterized(True)
        ax.add_collection(line_collection)
    else:
        final_true_indices = downsample_indices
        
    segs = [[(*t2[:, i], 0), (*t1[:, i], 1)] for i in final_true_indices]
    line_collection = Line3DCollection(segs, colors=color_true, lw=0.5, linestyles='dashed')
    line_collection.set_rasterized(True)
    ax.add_collection(line_collection)
    
    ### target - z = 0
    ax.scatter(xs = coords_target[:,0],ys = coords_target[:,1], zs=0, s = 2, c =color_target, alpha = 0.8, ec='none', rasterized=True, depthshade=False)
    ### source - z = 1
    ax.scatter(xs = coords_source[:,0],ys = coords_source[:,1], zs=1, s = 2, c =color_source, alpha = 0.8, ec='none', rasterized=True, depthshade=False)
    if remove_background:
        # Remove axis
        ax.axis('off')
    # Remove background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
