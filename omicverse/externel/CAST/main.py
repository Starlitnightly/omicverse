from .CAST_Mark import *
from .CAST_Stack import *
from .CAST_Projection import *
from .utils import *
from .visualize import *
from .model.model_GCNII import Args, CCA_SSG

def CAST_MARK(coords_raw_t,exp_dict_t,output_path_t,task_name_t = None,
              gpu_t = None,args = None,epoch_t = None, if_plot = True, 
              graph_strategy = 'convex',device = 'cuda:0'):
    ### setting
    try:
        import dgl
    except:
        print('Maybe you need to using `pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`')
        raise ImportError('Please install the dgl package from https://www.dgl.ai/pages/start.html')
    #gpu_t = 0 if torch.cuda.is_available() and gpu_t is None else -1
    #device = 'cuda:0' if gpu_t == 0 else 'cpu'
    samples = list(exp_dict_t.keys())
    task_name_t = task_name_t if task_name_t is not None else 'task1'
    inputs = []

    ### construct delaunay graphs and input data
    print(f'Constructing delaunay graphs for {len(samples)} samples...')
    for sample_t in samples:
        graph_dgl_t = delaunay_dgl(sample_t,coords_raw_t[sample_t],output_path_t,if_plot=if_plot,strategy_t = graph_strategy).to(device)
        feat_torch_t = torch.tensor(exp_dict_t[sample_t], dtype=torch.float32, device=device)
        inputs.append((sample_t, graph_dgl_t, feat_torch_t))
    
    ### parameters setting
    if args is None:
        args = Args(
            dataname=task_name_t, # name of the dataset, used to save the log file
            gpu = gpu_t, # gpu id, set to zero for single-GPU nodes
            epochs=400, # number of epochs for training
            lr1= 1e-3, # learning rate
            wd1= 0, # weight decay
            lambd= 1e-3, # lambda in the loss function, refer to online methods
            n_layers=9, # number of GCNII layers, more layers mean a deeper model, larger reception field, at a cost of VRAM usage and computation time
            der=0.5, # edge dropout rate in CCA-SSG
            dfr=0.3, # feature dropout rate in CCA-SSG
            use_encoder=True, # perform a single-layer dimension reduction before the GNNs, helps save VRAM and computation time if the gene panel is large
            encoder_dim=512, # encoder dimension, ignore if `use_encoder` set to `False`
        )
    args.epochs = epoch_t if epoch_t is not None else args.epochs

    ### Initialize the model
    in_dim = inputs[0][-1].size(-1)
    model = CCA_SSG(in_dim=in_dim, encoder_dim=args.encoder_dim, n_layers=args.n_layers, use_encoder=args.use_encoder).to(args.device)

    ### Training
    print(f'Training on {args.device}...')
    embed_dict, loss_log, model = train_seq(graphs=inputs, args=args, dump_epoch_list=[], out_prefix=f'{output_path_t}/{task_name_t}_seq_train', model=model)

    ### Saving the results
    torch.save(embed_dict, f'{output_path_t}/demo_embed_dict.pt')
    torch.save(loss_log, f'{output_path_t}/demo_loss_log.pt')
    torch.save(model, f'{output_path_t}/demo_model_trained.pt')
    print(f'Finished.')
    print(f'The embedding, log, model files were saved to {output_path_t}')
    return embed_dict

def CAST_STACK(coords_raw,embed_dict,output_path,graph_list,params_dist= None,tmp1_f1_idx = None, mid_visual = False, sub_node_idxs = None, rescale = False, corr_q_r = None, if_embed_sub = False, early_stop_thres = None, renew_mesh_trans = True):
    ### setting parameters
    query_sample = graph_list[0]
    ref_sample = graph_list[1]
    prefix_t = f'{query_sample}_align_to_{ref_sample}'
    result_log = dict()
    coords_raw, result_log['ref_rescale_factor'] = rescale_coords(coords_raw,graph_list,rescale = rescale)

    if sub_node_idxs is None:
        sub_node_idxs = {
            query_sample: np.ones(coords_raw[query_sample].shape[0],dtype=bool),
            ref_sample: np.ones(coords_raw[ref_sample].shape[0],dtype=bool)
        }

    if params_dist is None:
        params_dist = reg_params(dataname = query_sample,
                                    gpu = 0,
                                    #### Affine parameters
                                    iterations=500,
                                    dist_penalty1=0,
                                    bleeding=500,
                                    d_list = [3,2,1,1/2,1/3],
                                    attention_params = [None,3,1,0],
                                    #### FFD parameters                                    
                                    dist_penalty2 = [0],
                                    alpha_basis_bs = [500],
                                    meshsize = [8],
                                    iterations_bs = [400],
                                    attention_params_bs = [[tmp1_f1_idx,3,1,0]],
                                    mesh_weight = [None])
    if params_dist.alpha_basis == []:
        params_dist.alpha_basis = torch.Tensor([1/3000,1/3000,1/100,5,5]).reshape(5,1).to(params_dist.device)
    round_t = 0
    plt.rcParams.update({'pdf.fonttype':42})
    plt.rcParams['axes.grid'] = False

    ### Generate correlation matrix of the graph embedding
    if corr_q_r is None: 
        if if_embed_sub:
            corr_q_r = corr_dist(embed_dict[query_sample].cpu()[sub_node_idxs[query_sample]], embed_dict[ref_sample].cpu()[sub_node_idxs[ref_sample]]) 
        else:
            corr_q_r = corr_dist(embed_dict[query_sample].cpu(), embed_dict[ref_sample].cpu())
    else:
        corr_q_r = corr_q_r
    
    # Plot initial coordinates
    kmeans_plot_multiple(embed_dict,graph_list,coords_raw,prefix_t,output_path,k=15,dot_size = 10) if mid_visual else None
    corr_heat(coords_raw[query_sample][sub_node_idxs[query_sample]],coords_raw[ref_sample][sub_node_idxs[ref_sample]],corr_q_r,output_path,filename=prefix_t+'_corr') if mid_visual else None
    plot_mid(coords_raw[query_sample],coords_raw[ref_sample],output_path,f'{prefix_t}_raw')

    ### Initialize the coordinates and tensor
    corr_q_r = torch.Tensor(corr_q_r).to(params_dist.device)
    params_dist.mean_q = coords_raw[query_sample].mean(0)
    params_dist.mean_r = coords_raw[ref_sample].mean(0)
    coords_query = torch.Tensor(coords_minus_mean(coords_raw[query_sample])).to(params_dist.device)
    coords_ref = torch.Tensor(coords_minus_mean(coords_raw[ref_sample])).to(params_dist.device)

    ### Pre-location
    theta_r1_t = prelocate(coords_query,coords_ref,max_minus_value_t(corr_q_r),params_dist.bleeding,output_path,d_list=params_dist.d_list,prefix = prefix_t,index_list=[sub_node_idxs[k_t] for k_t in graph_list],translation_params = params_dist.translation_params,mirror_t=params_dist.mirror_t)
    params_dist.theta_r1 = theta_r1_t
    coords_query_r1 = affine_trans_t(params_dist.theta_r1,coords_query)
    plot_mid(coords_query_r1.cpu(),coords_ref.cpu(),output_path,prefix_t + '_prelocation') if mid_visual else None ### consistent scale with ref coords

    ### Affine
    output_list = Affine_GD(coords_query_r1,
                        coords_ref,
                        max_minus_value_t(corr_q_r),
                        output_path,
                        params_dist.bleeding,
                        params_dist.dist_penalty1,
                        alpha_basis = params_dist.alpha_basis,
                        iterations = params_dist.iterations,
                        prefix=prefix_t,
                        attention_params = params_dist.attention_params,
                        coords_log = True,
                        index_list=[sub_node_idxs[k_t] for k_t in graph_list],
                        mid_visual = mid_visual,
                        early_stop_thres = early_stop_thres,
                        ifrigid=params_dist.ifrigid)

    similarity_score,it_J,it_theta,coords_log = output_list
    params_dist.theta_r2 = it_theta[-1]
    result_log['affine_J'] = similarity_score
    result_log['affine_it_theta'] = it_theta
    result_log['affine_coords_log'] = coords_log
    result_log['coords_ref'] = coords_ref

    # Affine results
    affine_reg_params([i.cpu().numpy() for i in it_theta],similarity_score,params_dist.iterations,output_path,prefix=prefix_t)# if mid_visual else None
    if if_embed_sub:
        embed_stack_t = np.row_stack((embed_dict[query_sample].cpu().detach().numpy()[sub_node_idxs[query_sample]],embed_dict[ref_sample].cpu().detach().numpy()[sub_node_idxs[ref_sample]]))
    else:
        embed_stack_t = np.row_stack((embed_dict[query_sample].cpu().detach().numpy(),embed_dict[ref_sample].cpu().detach().numpy()))
    coords_query_r2 = affine_trans_t(params_dist.theta_r2,coords_query_r1)
    register_result(coords_query_r2.cpu().detach().numpy(),
                    coords_ref.cpu().detach().numpy(),
                    max_minus_value_t(corr_q_r).cpu(),
                    params_dist.bleeding,
                    embed_stack_t,
                    output_path,
                    k=20,
                    prefix=prefix_t,
                    scale_t=1,
                    index_list=[sub_node_idxs[k_t] for k_t in graph_list])# if mid_visual else None
    
    if params_dist.iterations_bs[round_t] != 0:
        ### B-Spline free-form deformation 
        padding_rate = params_dist.PaddingRate_bs # by default, 0
        coords_query_r2_min = coords_query_r2.min(0)[0] # The x and y min of the query coords
        coords_query_r2_tmp = coords_minus_min_t(coords_query_r2) # min of the x and y is 0
        max_xy_tmp = coords_query_r2_tmp.max(0)[0] # max_xy withouth padding
        adj_min_qr2 = coords_query_r2_min - max_xy_tmp * padding_rate # adjust the min_qr2
        setattr(params_dist,'img_size_bs',[(max_xy_tmp * (1+padding_rate * 2)).cpu()]) # max_xy
        params_dist.min_qr2 = [adj_min_qr2]
        t1 = BSpline_GD(coords_query_r2 - params_dist.min_qr2[round_t],
                        coords_ref - params_dist.min_qr2[round_t],
                        max_minus_value_t(corr_q_r),
                        params_dist.iterations_bs[round_t],
                        output_path,
                        params_dist.bleeding,
                        params_dist.dist_penalty2[round_t],
                        params_dist.alpha_basis_bs[round_t],
                        params_dist.diff_step,
                        params_dist.meshsize[round_t],
                        prefix_t + '_' + str(round_t),
                        params_dist.mesh_weight[round_t],
                        params_dist.attention_params_bs[round_t],
                        coords_log = True,
                        index_list=[sub_node_idxs[k_t] for k_t in graph_list],
                        mid_visual = mid_visual,
                        max_xy = params_dist.img_size_bs[round_t],
                        renew_mesh_trans = renew_mesh_trans)

        # B-Spline FFD results
        register_result(t1[0].cpu().numpy(),(coords_ref - params_dist.min_qr2[round_t]).cpu().numpy(),max_minus_value_t(corr_q_r).cpu(),params_dist.bleeding,embed_stack_t,output_path,k=20,prefix=prefix_t+ '_' + str(round_t) +'_BSpine_' + str(params_dist.iterations_bs[round_t]),index_list=[sub_node_idxs[k_t] for k_t in graph_list])# if mid_visual else None
        # register_result(t1[0].cpu().numpy(),(coords_ref - coords_query_r2.min(0)[0]).cpu().numpy(),max_minus_value_t(corr_q_r).cpu(),params_dist.bleeding,embed_stack_t,output_path,k=20,prefix=prefix_t+ '_' + str(round_t) +'_BSpine_' + str(params_dist.iterations_bs[round_t]),index_list=[sub_node_idxs[k_t] for k_t in graph_list])# if mid_visual else None
        result_log['BS_coords_log1'] = t1[4]
        result_log['BS_J1'] = t1[3]
        if renew_mesh_trans:
            setattr(params_dist,'mesh_trans_list',[t1[1]])
        else:
            setattr(params_dist,'mesh_trans_list',[[t1[1][-1]]])

    ### Save results
    torch.save(params_dist,os.path.join(output_path,f'{prefix_t}_params.data'))
    torch.save(result_log,os.path.join(output_path,f'{prefix_t}_result_log.data'))
    coords_final = dict()
    _, coords_q_final = reg_total_t(coords_raw[query_sample],coords_raw[ref_sample],params_dist)
    coords_final[query_sample] = coords_q_final.cpu() / result_log['ref_rescale_factor'] ### rescale back to the original scale
    coords_final[ref_sample] = coords_raw[ref_sample] / result_log['ref_rescale_factor'] ### rescale back to the original scale
    plot_mid(coords_final[query_sample],coords_final[ref_sample],output_path,f'{prefix_t}_align')
    torch.save(coords_final,os.path.join(output_path,f'{prefix_t}_coords_final.data'))
    return coords_final

def CAST_PROJECT(
    sdata_inte, # the integrated dataset
    source_sample, # the source sample name
    target_sample, # the target sample name
    coords_source, # the coordinates of the source sample
    coords_target, # the coordinates of the target sample
    scaled_layer = 'log2_norm1e4_scaled', # the scaled layer name in `adata.layers`, which is used to be integrated
    raw_layer = 'raw', # the raw layer name in `adata.layers`, which is used to be projected into target sample
    batch_key = 'protocol', # the column name of the samples in `obs`
    use_highly_variable_t = True, # if use highly variable genes
    ifplot = True, # if plot the result
    n_components = 50, # the `n_components` parameter in `sc.pp.pca`
    umap_n_neighbors = 50, # the `n_neighbors` parameter in `sc.pp.neighbors`
    umap_n_pcs = 30, # the `n_pcs` parameter in `sc.pp.neighbors`
    min_dist = 0.01, # the `min_dist` parameter in `sc.tl.umap`
    spread_t = 5, # the `spread` parameter in `sc.tl.umap`
    k2 = 1, # select k2 cells to do the projection for each cell
    source_sample_ctype_col = 'level_2', # the column name of the cell type in `obs`
    output_path = '', # the output path
    umap_feature = 'X_umap', # the feature used for umap
    pc_feature = 'X_pca_harmony', # the feature used for the projection
    integration_strategy = 'Harmony', # 'Harmony' or None (use existing integrated features)
    ave_dist_fold = 3, # the `ave_dist_fold` is used to set the distance threshold (average_distance * `ave_dist_fold`)
    save_result = True, # if save the results
    ifcombat = True, # if use combat when using the Harmony integration
    alignment_shift_adjustment = 50, # to adjust the small alignment shift for the distance threshold)
    color_dict = None, # the color dict for the cell type
    adjust_shift = False, # if adjust the alignment shift by group
    metric_t = 'cosine',
    working_memory_t = 1000 # the working memory for the pairwise distance calculation
    ):

    #### integration
    if integration_strategy == 'Harmony':
        sdata_inte = Harmony_integration(
            sdata_inte = sdata_inte,
            scaled_layer = scaled_layer,
            use_highly_variable_t = use_highly_variable_t,
            batch_key = batch_key,
            umap_n_neighbors = umap_n_neighbors,
            umap_n_pcs = umap_n_pcs,
            min_dist = min_dist,
            spread_t = spread_t,
            source_sample_ctype_col = source_sample_ctype_col,
            output_path = output_path,
            n_components = n_components,
            ifplot = True,
            ifcombat = ifcombat)
    elif integration_strategy is None:
        print(f'Using the pre-integrated data {pc_feature} and the UMAP {umap_feature}')

    #### Projection
    idx_source = sdata_inte.obs[batch_key] == source_sample
    idx_target = sdata_inte.obs[batch_key] == target_sample
    source_cell_pc_feature = sdata_inte[idx_source, :].obsm[pc_feature]
    target_cell_pc_feature = sdata_inte[idx_target, :].obsm[pc_feature]
    sdata_ref,output_list = space_project(
        sdata_inte = sdata_inte,
        idx_source = idx_source,
        idx_target = idx_target,
        raw_layer = raw_layer,
        source_sample = source_sample,
        target_sample = target_sample,
        coords_source = coords_source,
        coords_target = coords_target,
        output_path = output_path,
        source_sample_ctype_col = source_sample_ctype_col,
        target_cell_pc_feature = target_cell_pc_feature,
        source_cell_pc_feature = source_cell_pc_feature,
        k2 = k2,
        ifplot = ifplot,
        umap_feature = umap_feature,
        ave_dist_fold = ave_dist_fold,
        alignment_shift_adjustment = alignment_shift_adjustment,
        color_dict = color_dict,
        metric_t = metric_t,
        adjust_shift = adjust_shift,
        working_memory_t = working_memory_t
        )

    ### Save the results
    if save_result == True:
        sdata_ref.write_h5ad(f'{output_path}/sdata_ref.h5ad')
        torch.save(output_list,f'{output_path}/projection_data.pt')
    return sdata_ref,output_list