import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.collections import LineCollection


from .binning_and_plotting import bin_data

from matplotlib.colors import ListedColormap
import seaborn as sns

# domain_boundary_label='domain boundaries \n from expression or \n cell type proportion'
domain_boundary_label='Domain boundaries'

# example of ct_colors: {'Oligodendrocytes': 'C6', 'Granule': 'mediumseagreen', 'Purkinje': 'red', 'Bergmann': 'C4', 'MLI1': 'gold', 'MLI2': 'goldenrod',  'Astrocytes': 'C0', 'Golgi': 'C9', 'Fibroblast': 'C5'}

# example of ct_pseudocounts: {3: 1} -- ie domain 4 needs CT pseudocount of 1

############################################################################################################

def domain_cts_svg(cell_type_df, gaston_labels, gaston_isodepth, domain_ct_threshold=0.7,
                  num_bins=60, num_bins_per_domain=[10,16,7,17]):
    N=len(gaston_labels)
    # need dummy counts mat and dummy gene_labels to create binning_output
    binning_output=bin_data(np.ones((N,10)), gaston_labels, gaston_isodepth, 
                         cell_type_df, np.array(['test' for i in range(10)]), num_bins=num_bins, num_bins_per_domain=num_bins_per_domain)
    ct_dict=get_domain_cts(binning_output, domain_ct_threshold) # {0: [list of CTs], 1: [list of CTs], ...}
    ct_list=[]
    for d in ct_dict:
        ct_list+=list(ct_dict[d])
    return np.unique(ct_list)

def get_domain_cts(binning_output, domain_ct_threshold, exclude_ct=[]):
    domain_ct_markers={}
    gaston_labels=binning_output['gaston_labels'] # unbinned labels
    cell_type_mat=binning_output['cell_type_mat'] # unbinned cell types
    cell_type_names=binning_output['cell_type_names']
    L=len(np.unique(gaston_labels))
    
    ct_names_not_excluded=np.array([ct for ct in cell_type_names if ct not in exclude_ct])
    ct_inds_not_excluded=np.array([i for i,ct in enumerate(cell_type_names) if ct not in exclude_ct])
    ct_mat_not_excluded=cell_type_mat[:, ct_inds_not_excluded]
    
    for t in range(L):
        pts_t=np.where(gaston_labels==t)[0]
        
        ct_counts_t=np.sum(ct_mat_not_excluded[pts_t,:],0)

        argsort_cts=np.argsort(ct_counts_t)[::-1]
        i=0
        while np.sum(ct_counts_t[argsort_cts[:i]]) / np.sum(ct_counts_t) < domain_ct_threshold:
            i+=1
        domain_t_cts=ct_names_not_excluded[argsort_cts[:i]]
        domain_ct_markers[t]=domain_t_cts
    return domain_ct_markers

############################################################################################################

# Make a plot of cell type proportion vs isodepth
# gaston_labels: N x 1 array of domain labels
# gaston_isodepth: N x 1 array of isodepth values per spot
# cell_type_df: N x C dataframe of (binary) cell type labels per spot
# (1)num_bins OR (2)num_bins_per_domain: 
#         specify either (1) number of total bins OR (2) number of bins in each spatial domain
# ct_list: list of cell types to plot
# ct_colors: list of colors for cell types in ct_list
# color_palette: matplotlib color palette, if ct_colors is not given
# ct_pseudocounts: dictionary of pseudocount to add to cell type proportion 
#         {3: 1} -- means domain 3 needs CT pseudocount of 1, so cell type c proportion is (c+1)/(n+1)
# domain_ct_threshold: we bold the cell types that cover at least domain_ct_threshold spots in each domain i
# ticksize: size of xticks, yticks
# domain_boundary_label: label for domain boundaries in legend
# exclude_ct: list of cell types to ignore
# width1, width2: width of cell types in their domain vs other domains
# lgd_width, lgd_fontsize: width/fontsize of legend
# include_lgd, lgd_frameon: boolean whether to include legend, and whether to include legend frame
# lgd_bbox: where to place legend
def plot_ct_props(cell_type_df, gaston_labels, gaston_isodepth, 
                  num_bins=60, num_bins_per_domain=[10,16,7,17],
                  ct_list=None, ct_colors=None, color_palette=None, 
                   ct_pseudocounts=None, linewidth=8, figsize=(15,6),
                   domain_ct_threshold=0.6, ticksize=25, domain_boundary_label=domain_boundary_label, 
                   exclude_ct=[],width1=8, width2=4,
                  lgd_width=4, include_lgd=True, lgd_fontsize=25, lgd_bbox=(1.75,1), lgd_frameon=True,
                 return_ct_raw=False):

    N=len(gaston_labels)
    # need dummy counts mat and dummy gene_labels to create binning_output
    binning_output=bin_data(np.ones((N,10)), gaston_labels, gaston_isodepth, 
                         cell_type_df, np.array(['test' for i in range(10)]), 
                         num_bins=num_bins, num_bins_per_domain=num_bins_per_domain)
    # print(binning_output)
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    binned_labels=binning_output['binned_labels']
    ct_count_mat=binning_output['binned_cell_type_mat'].T # len(unique_cell_types) x binned_labels
    cell_type_names=binning_output['cell_type_names']
    
    L=len(np.unique(binned_labels))
    
    left_bps=[]
    right_bps=[]

    for i in range(len(binned_labels)-1):
        if binned_labels[i] != binned_labels[i+1]:
            left_bps.append(unique_binned_isodepths[i])
            right_bps.append(unique_binned_isodepths[i+1])
            
    if ct_list is None:
        # ct_inds=np.array([i for i,ct in enumerate(cell_type_names) if ct not in exclude_ct])
        # ct_list=cell_type_names[ct_inds]
        domain_ct_markers=get_domain_cts(binning_output, domain_ct_threshold, exclude_ct=exclude_ct)
        ct_list=[]
        for v in domain_ct_markers.values():
            ct_list += list(v)
        ct_list=list(set(ct_list))
        ct_inds=np.array([np.where(cell_type_names==ct)[0][0] for ct in ct_list])
    else:
        ct_inds=np.array([np.where(cell_type_names==i)[0][0] for i in ct_list if i not in exclude_ct])
        ct_list=cell_type_names[ct_inds]

    if color_palette is None:
        l=len(ct_list)
        color_palette=ListedColormap([sns.color_palette("Spectral", as_cmap=True)(i/l) for i in range(l)])
    
    pc_mat=np.zeros( ct_count_mat.shape )
    if ct_pseudocounts is not None:
        for l in ct_pseudocounts:
            pc=ct_pseudocounts[l]
            pc_mat[:,np.where(binned_labels==l)[0]]=pc

    ct_count_mat=(ct_count_mat+pc_mat)[ct_inds]
    ct_count_prop=normalize(ct_count_mat,axis=0,norm='l1')

    fig,ax=plt.subplots(figsize=figsize)
        
    if ct_colors is None:
        ct_colors_list=[color_palette(i) for i in range(len(ct_list))]
    # print(len(ct_colors_list))
    c=0
    
    for i,ct in enumerate(ct_list):
        widths=np.ones(len(unique_binned_isodepths))*width2
        for l in range(L):
            if ct in domain_ct_markers[l]:
                pts_l=np.where(binned_labels==l)[0]
                widths[pts_l]=width1
        for s in range(len(widths)-1):
            if widths[s]==width1 and widths[s+1] < width1:
                widths[s]=width2
        
        x,y=unique_binned_isodepths, ct_count_prop[i,:]
        points = np.vstack((x, y)).T.reshape(-1, 1, 2)
        segments = np.hstack((points[:-1], points[1:]))
        if ct_colors is None:
            lc = LineCollection(segments, alpha=1, color=ct_colors_list[c], lw=widths, label=ct)
        else:
            lc = LineCollection(segments, alpha=1, color=ct_colors[ct], lw=widths, label=ct)
        line = ax.add_collection(lc)
        c+=1
        
    # AXES tick size
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    for i in range(L-1):
        if i==0:
            plt.axvline((left_bps[i]+right_bps[i])*0.5, color='black', ls='--', linewidth=3, label=domain_boundary_label)
        else:
            plt.axvline((left_bps[i]+right_bps[i])*0.5, color='black', ls='--', linewidth=3)

    if include_lgd:
        lgd=plt.legend(fontsize=lgd_fontsize, bbox_to_anchor=lgd_bbox, labelcolor='linecolor', frameon=lgd_frameon)
        try:
            for lh in lgd.legendHandles: 
                lh.set_alpha(1)
                lh.set_linewidth(lgd_width)
        except:
            for lh in lgd.legend_handles:
                lh.set_alpha(1)
                lh.set_linewidth(lgd_width)
        for text in lgd.get_texts():
            text.set_alpha(1)
            text.set_size(lgd_fontsize)
    
    
    plt.ylim((0, 1.05))
    sns.despine()

    if return_ct_raw:
        return ct_list, ct_count_prop, unique_binned_isodepths
