from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse import issparse

from .dp_related import rotate_by_theta

def bin_data(counts_mat, gaston_labels, gaston_isodepth, 
              cell_type_df, gene_labels, num_bins=70, num_bins_per_domain=None,
             idx_kept=None, umi_threshold=500, pc=0, pc_exposure=True, extra_data=[]):
    
    if idx_kept is None:
        idx_kept=np.where(np.sum(counts_mat,0) > umi_threshold)[0]
    gene_labels_idx=gene_labels[idx_kept]
    
    exposure=np.sum(counts_mat,axis=1)
    
    cmat=counts_mat[:,idx_kept]
    
    if cell_type_df is not None:
        cell_type_mat=cell_type_df.to_numpy()
        cell_type_names=np.array(cell_type_df.columns)
    else:
        N=len(exposure)
        cell_type_mat=np.ones((N,1))
        cell_type_names=['All']

    N,G=cmat.shape

    # BINNING
    if num_bins_per_domain is not None:
        bins=np.array([])
        L=len(np.unique(gaston_labels))
        
        for l in range(L):
            isodepth_l=gaston_isodepth[np.where(gaston_labels==l)[0]]
            
            if l>0:
                isodepth_lm1=gaston_isodepth[np.where(gaston_labels==l-1)[0]]
                isodepth_left=0.5*(np.min(isodepth_l) + np.max(isodepth_lm1))
            else:
                isodepth_left=np.min(isodepth_l)-0.01
                
            if l<L-1:
                isodepth_lp1=gaston_isodepth[np.where(gaston_labels==l+1)[0]]
                isodepth_right=0.5*(np.max(isodepth_l) + np.min(isodepth_lp1))
            else:
                isodepth_right=np.max(isodepth_l)+0.01
            
            bins_l=np.linspace(isodepth_left, isodepth_right, num=num_bins_per_domain[l]+1)
            if l!=0:
                bins_l=bins_l[1:]
            bins=np.concatenate((bins, bins_l))
    else:
        isodepth_min, isodepth_max=np.floor(np.min(gaston_isodepth))-0.5, np.ceil(np.max(gaston_isodepth))+0.5
        bins=np.linspace(isodepth_min, isodepth_max, num=num_bins+1)

    unique_binned_isodepths=np.array( [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)] )
    binned_isodepth_inds=np.digitize(gaston_isodepth, bins)-1 #ie [1,0,3,15,...]
    binned_isodepths=unique_binned_isodepths[binned_isodepth_inds]
    
    # remove bins not used
    unique_binned_isodepths=np.delete(unique_binned_isodepths,
                                   [np.where(unique_binned_isodepths==t)[0][0] for t in unique_binned_isodepths if t not in binned_isodepths])

    N_1d=len(unique_binned_isodepths)
    binned_count=np.zeros( (N_1d,G) )
    binned_exposure=np.zeros( N_1d )
    to_subtract=np.zeros( N_1d )
    binned_labels=np.zeros(N_1d)
    binned_cell_type_mat=np.zeros((N_1d, len(cell_type_names)))
    binned_number_spots=np.zeros(N_1d)

    binned_count_per_ct={ct: np.zeros( (N_1d,G) ) for ct in cell_type_names}
    binned_exposure_per_ct={ct: np.zeros( N_1d ) for ct in cell_type_names}
    to_subtract_per_ct={ct:np.zeros( N_1d ) for ct in cell_type_names}
    binned_extra_data=[np.zeros(N_1d) for i in range(len(extra_data))]
    map_1d_bins_to_2d={} # map b -> [list of cells in bin b]
    for ind, b in enumerate(unique_binned_isodepths):
        bin_pts=np.where(binned_isodepths==b)[0]
        
        binned_count[ind,:]=np.sum(cmat[bin_pts,:],axis=0)
        binned_exposure[ind]=np.sum(exposure[bin_pts])
        if pc>0:
            to_subtract[ind]=np.log(10**6 * (len(bin_pts)/np.sum(exposure[bin_pts])))
        binned_labels[ind]= int(mode( gaston_labels[bin_pts],keepdims=False).mode)
        binned_cell_type_mat[ind,:] = np.sum( cell_type_mat[bin_pts,:], axis=0)
        binned_number_spots[ind]=len(bin_pts)
        map_1d_bins_to_2d[b]=bin_pts

        for i, eb in enumerate(extra_data):
            binned_extra_data[i][ind]=np.mean(extra_data[i][bin_pts])
        
        for ct_ind, ct in enumerate(cell_type_names):
            
            ct_spots=np.where(cell_type_mat[:,ct_ind] > 0)[0]
            ct_spots_bin = [t for t in ct_spots if t in bin_pts]
            ct_spots_bin_proportions=cell_type_mat[ct_spots_bin,ct_ind]
            
            if len(ct_spots_bin)>0:
                
                if issparse(cmat):
                    binned_count_per_ct[ct][ind,:]=np.sum(cmat[ct_spots_bin,:].T * np.tile(ct_spots_bin_proportions,(G,1)).T, axis=0)
                else:
                    binned_count_per_ct[ct][ind,:]=np.sum(cmat[ct_spots_bin,:] * np.tile(ct_spots_bin_proportions,(G,1)).T, axis=0)
                #binned_count_per_ct[ct][ind,:]=np.sum(cmat[ct_spots_bin,:].T * np.tile(ct_spots_bin_proportions,(G,1)).T, axis=0)
                binned_exposure_per_ct[ct][ind]=np.sum(exposure[ct_spots_bin] * ct_spots_bin_proportions)
                if pc>0:
                    to_subtract_per_ct[ct]=np.log(10**6 * len(ct_spots_bin) / np.sum(exposure[ct_spots_bin]))
            
    # subtract single constant if we add PC
    to_subtract=np.median(to_subtract)
    to_subtract_per_ct={ct:np.median(to_subtract_per_ct[ct]) for ct in cell_type_names}
            
    L=len(np.unique(gaston_labels))
    segs=[np.where(binned_labels==i)[0] for i in range(L)]

    to_return={}
    
    to_return['L']=len(np.unique(gaston_labels))
    to_return['umi_threshold']=umi_threshold
    to_return['gaston_labels']=gaston_labels
    to_return['counts_mat_idx']=cmat
    to_return['cell_type_mat']=cell_type_mat
    to_return['cell_type_names']=cell_type_names
    to_return['idx_kept']=idx_kept
    to_return['gene_labels_idx']=gene_labels_idx
    
    to_return['binned_isodepths']=binned_isodepths
    to_return['unique_binned_isodepths']=unique_binned_isodepths
    to_return['binned_count']=binned_count
    to_return['binned_exposure']=binned_exposure
    to_return['to_subtract']=to_subtract
    to_return['binned_labels']=binned_labels
    to_return['binned_cell_type_mat']=binned_cell_type_mat
    to_return['binned_number_spots']=binned_number_spots
    
    to_return['binned_count_per_ct']=binned_count_per_ct
    to_return['binned_exposure_per_ct']=binned_exposure_per_ct
    to_return['to_subtract_per_ct']=to_subtract_per_ct
    to_return['binned_extra_data']=binned_extra_data
    
    to_return['map_1d_bins_to_2d']=map_1d_bins_to_2d
    to_return['segs']=segs

    return to_return

    
def plot_gene_pwlinear(gene_name, pw_fit_dict, gaston_labels, gaston_isodepth, binning_output,
                       cell_type_list=None, ct_colors=None, spot_threshold=0.25, pt_size=10, 
                       colors=None, linear_fit=True, lw=2, domain_list=None, ticksize=20, figsize=(7,3),
                      offset=10**6, xticks=None, yticks=None, alpha=1, domain_boundary_plotting=False, 
                      save=False, save_dir="./", variable_spot_size=False, show_lgd=False,
                      lgd_bbox=(1.05,1), extract_values = False):
    
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    binned_labels=binning_output['binned_labels']
    
    binned_count_list=[]
    binned_exposure_list=[]
    to_subtract_list=[]
    ct_ind_list=[]
    
    if cell_type_list is None:
        binned_count_list.append(binning_output['binned_count'])
        binned_exposure_list.append(binning_output['binned_exposure'])
        to_subtract_list.append(binning_output['to_subtract'])
        
    else:
        for ct in cell_type_list:
            binned_count_list.append(binning_output['binned_count_per_ct'][ct])
            binned_exposure_list.append(binning_output['binned_exposure_per_ct'][ct])
            to_subtract_list.append(binning_output['to_subtract_per_ct'][ct])
            ct_ind_list.append( np.where(binning_output['cell_type_names']==ct)[0][0] )
    
    segs=binning_output['segs']
    L=len(segs)

    fig,ax=plt.subplots(figsize=figsize)

    if domain_list is None:
        domain_list=range(L)

    values_list = []
    for seg in domain_list:
        for i in range(len(binned_count_list)):
            pts_seg=np.where(binned_labels==seg)[0]
            binned_count=binned_count_list[i]
            binned_exposure=binned_exposure_list[i]
            to_subtract=np.log( offset*1 / np.mean(binned_exposure) )
            ct=None
            if cell_type_list is not None:
                ct=cell_type_list[i]
                # if restricting cell types, then restrict spots also
                binned_cell_type_mat=binning_output['binned_cell_type_mat']
                ct_ind=ct_ind_list[i]
                pts_seg=[p for p in pts_seg if binned_cell_type_mat[p,ct_ind] / binned_cell_type_mat[p,:].sum() > spot_threshold]
                
                # set colors for cell types
                if ct_colors is None:
                    c=None
                else:
                    c=ct_colors[ct]
            else:
                # set colors for domains
                if colors is None:
                    c=None
                else:
                    c=colors[seg]
                
            xax=unique_binned_isodepths[pts_seg]
            # print(binned_count.shape)
            yax=np.log((binned_count[pts_seg,gene] / binned_exposure[pts_seg]) * offset + 1)

            if extract_values:
                values_list.append(np.column_stack((xax, yax)))
            
            s=pt_size
            if variable_spot_size:
                s=s*binning_output['binned_number_spots'][pts_seg]
            plt.scatter(xax, yax, color=c, s=s, alpha=alpha,label=ct)

            if linear_fit:
                if ct is None:
                    slope_mat, intercept_mat, _, _ = pw_fit_dict['all_cell_types']
                else:
                    slope_mat, intercept_mat, _, _ = pw_fit_dict[ct]

                slope=slope_mat[gene,seg]
                intercept=intercept_mat[gene,seg]
                plt.plot(unique_binned_isodepths[pts_seg], np.log(offset) + intercept + slope*unique_binned_isodepths[pts_seg], color='grey', alpha=1, lw=lw )

    if xticks is None:
        plt.xticks(fontsize=ticksize)
    else:
        plt.xticks(xticks,fontsize=ticksize)
        
    if yticks is None:
        plt.yticks(fontsize=ticksize)
    else:
        plt.yticks(yticks,fontsize=ticksize)
        
    if domain_boundary_plotting and len(domain_list)>1:
        binned_labels=binning_output['binned_labels']
        
        left_bps=[]
        right_bps=[]

        for i in range(len(binned_labels)-1):
            if binned_labels[i] != binned_labels[i+1]:
                left_bps.append(unique_binned_isodepths[i])
                right_bps.append(unique_binned_isodepths[i+1])
        
        for i in range(len(left_bps)):
            plt.axvline((left_bps[i]+right_bps[i])*0.5, color='black', ls='--', linewidth=1.5, alpha=0.2)

    sns.despine()
    if show_lgd:
        plt.legend(bbox_to_anchor=lgd_bbox)
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{gene_name}_pwlinear.pdf", bbox_inches="tight")
        plt.close()

    if extract_values:
        all_values = np.vstack(values_list)
        values_filename = f"{save_dir}/{gene_name}_raw_all.txt"
        save_values({gene_name: all_values}, values_filename)

def save_values(values_dict, filename):
    with open(filename, 'w') as file:
        for key, values in values_dict.items():
            file.write(f"{key}\n")
            np.savetxt(file, values, delimiter='\t', fmt='%.6f')

def get_gene_plot_values(gene_name, binning_output, offset=10**6):
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    binned_labels=binning_output['binned_labels']
    
    binned_count_list=binning_output['binned_count']
    binned_exposure_list=binning_output['binned_exposure']

    domain_list=range(len(binning_output['segs']))

    values = []
        
    for seg in domain_list:
        for i in range(len(binned_count_list)):
            pts_seg=np.where(binned_labels==seg)[0]
            binned_count=binned_count_list[i]
            binned_exposure=binned_exposure_list[i]
                
            xax=unique_binned_isodepths[pts_seg]
            yax=np.log((binned_count[gene,pts_seg] / binned_exposure[pts_seg]) * offset + 1)

            values.append(np.column_stack((xax, yax)))
    
    return np.vstack(values)

# NxG counts matrix
# plot raw expression values of gene
def plot_gene_raw(gene_name, gene_labels, counts_mat, coords_mat, 
                       offset=10**6, figsize=(6,6), colorbar=True, vmax=None, vmin=None, s=16, rotate=None,
                       cmap='RdPu'):

    if rotate is not None:
        coords_mat=rotate_by_theta(coords_mat,rotate)
    gene_idx=np.where(gene_labels==gene_name)[0]

    exposure = np.sum(counts_mat, axis=1, keepdims=False)
    raw_expression = np.squeeze(counts_mat[:, gene_idx])

    expression = np.log((raw_expression / exposure) * offset + 1)

    fig,ax=plt.subplots(figsize=figsize)

    im1 = ax.scatter(coords_mat[:, 0], 
        coords_mat[:, 1],
        c = expression,
        cmap = cmap, s=s, vmax=vmax, vmin=vmin)

    if colorbar:
        cbar=plt.colorbar(im1)
        cbar.ax.tick_params(labelsize=10)

    plt.axis('off')

# plot piecewise linear gene function learned by GASTON
def plot_gene_function(gene_name, coords_mat, pw_fit_dict, gaston_labels, gaston_isodepth, 
                       binning_output, offset=10**6, figsize=(6,6), colorbar=True, 
                       contours=False, contour_levels=4, contour_lw=1, contour_fs=10, s=16,
                      rotate=None,cmap='RdPu'):

    if rotate is not None:
        coords_mat=rotate_by_theta(coords_mat,rotate)
    
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    slope_mat, intercept_mat, _, _ = pw_fit_dict['all_cell_types']
    if gene_name in binning_output['gene_labels_idx']:
        gene=np.where(gene_labels_idx==gene_name)[0]

    outputs = np.zeros(gaston_isodepth.shape[0])
    for i in range(gaston_isodepth.shape[0]):
        dom = int(gaston_labels[i])
        slope=slope_mat[gene,dom]
        intercept=intercept_mat[gene,dom]
        outputs[i] = np.log(offset) + intercept + slope * gaston_isodepth[i]

    fig,ax=plt.subplots(figsize=figsize)

    im1 = ax.scatter(coords_mat[:, 0], 
        coords_mat[:, 1],
        c = outputs,
        cmap = cmap, s=s)


    if contours:
        CS=ax.tricontour(coords_mat[:,0], coords_mat[:,1], outputs, levels=contour_levels, linewidths=contour_lw, colors='k', linestyles='solid')
        ax.clabel(CS, CS.levels, inline=True, fontsize=contour_fs)
    if colorbar:
        cbar=plt.colorbar(im1)
        cbar.ax.tick_params(labelsize=10)

    plt.axis('off')
