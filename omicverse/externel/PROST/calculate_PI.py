import pandas as pd
import numpy as np

import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import scanpy as sc
from tqdm import trange, tqdm

import multiprocessing as mp
from .utils import pre_process, make_image, gene_img_flatten, minmax_normalize, gau_filter_for_single_gene



def prepare_for_PI(adata, grid_size=20, percentage=0.1, platform="visium"):    
    selected_gene_idxs, postcount = pre_process(adata, percentage, var_stabilization = False)

    if platform=="visium" or platform=="ST":
        try:
            locates = adata.obs[["array_row","array_col"]]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        except:
            locates = adata.obsm["spatial"]
            locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        if np.min(locates) == 0:
            locates += 1
        _, image_idx = make_image(postcount[0], locates, platform, get_image_idx = True, grid_size=grid_size)
        adata = adata[:, selected_gene_idxs]
        sc.pp.filter_genes(adata, min_cells=3)
        adata.obs['image_idx_1d'] = image_idx

    else:
        locates = adata.obsm["spatial"].astype(float)
        locates = locates.values if isinstance(locates, pd.DataFrame) else locates
        _, shape = make_image(postcount[0], locates, platform, grid_size=grid_size)
        assert shape[0]>1 and shape[1]>1, f"Gene image size is {shape[0]} * {shape[1]} after interpolation. Please set a smaller grid size!!"
        print (f"Spatial gene expression is interpolated into images of size [{shape[0]} * {shape[1]}]")
        adata = adata[:, selected_gene_idxs]
        sc.pp.filter_genes(adata, min_cells=50)
        adata.uns['shape'] = shape
    adata.uns['grid_size'] = grid_size
    adata.uns['locates'] = locates
    return adata


def minmax_scaler(adata,layer='counts'):
    if layer=='X' or layer=='raw':
        if sp.issparse(adata.X):
            data = adata.X.A.T
        else:
            data = adata.X.T      
    else:
        if sp.issparse(adata.layers[layer]):
            data = adata.layers[layer].A.T
        else:
            data = adata.layers[layer].T   

    print('\nNormalize each geneing...')
    nor_counts = data.copy().T
    _nor_maxdata = np.max(nor_counts, 0)
    _nor_mindata = np.min(nor_counts, 0)
    nor_counts = (nor_counts - _nor_mindata) / (_nor_maxdata - _nor_mindata)
    adata.uns['nor_counts'] = nor_counts.T
    return adata


def gau_filter_for_single_gene(arglist):
    gene_data, locates, platform, image_idx_1d = arglist

    I,_ = make_image(gene_data, locates, platform) 
    I = gaussian_filter(I, sigma = 1, truncate = 2)
    if platform=="visium":
        I_1d = I.T.flatten()
        output = I_1d[image_idx_1d-1]
    else:
        output = I.flatten()
    return output


def gau_filter(adata, platform="visium", multiprocess=False):
    gene_data = adata.uns['nor_counts']
    locates = adata.uns['locates']
    N_gene = len(gene_data)

    print('\nGaussian filtering...')
    if platform=="visium":
        image_idx_1dd = adata.obs['image_idx_1d'].astype(int).values

    def sel_data():  # data generater
        for gene_i in range(N_gene):
            if platform=="visium":
                yield [gene_data[gene_i], locates, platform, image_idx_1dd]
            else:
                yield [gene_data[gene_i], locates, platform, '']
                
    if multiprocess:
        num_cores = int(mp.cpu_count() / 2)         # default core is half of total
        with mp.Pool(processes=num_cores) as pool:
            gau_fea = list(tqdm(pool.imap(gau_filter_for_single_gene, sel_data()), total=N_gene))
    else:
        gau_fea = list(tqdm(map(gau_filter_for_single_gene, sel_data()), total=N_gene))
        
    adata.uns['gau_fea'] = np.array(gau_fea, dtype=np.float64) 
    return adata


def _iget_binary(arglists):
    import cv2
    fig1, locates, platform, method, r1 = arglists

    if platform=="visium":
        Im, _ = make_image(fig1, locates, platform)
        if method == "iterative":    
            m, n = Im.shape       
            zd = float(np.max(Im))
            zx = float(np.min(Im))
            Th = float((zd+zx))/2
            while True:
                S0 = 0.0; n0 = 0.0; S1 = 0.0; n1 = 0.0
                flag = Im >= Th
                S1 = Im[flag].sum()
                n1 = flag.sum()
                S0 = Im[~flag].sum()
                n0 = (~flag).sum()
                T0 = S0/n0; T1 = S1/n1
                if abs(Th - ((T0 + T1)/2)) < 0.0001:
                    break
                else:
                    Th = (T0 + T1)/2 
                        
        elif method == "otsu":
            thres_list = np.arange(0.01,0.995,0.025)
            temp_std = np.zeros(thres_list.shape)
            for iii in range(len(thres_list)):
                temp_thres = thres_list[iii]
                q1 = fig1 > temp_thres
                b1 = fig1 <= temp_thres
                qv = r1[q1]
                bv = r1[b1]
                if len(qv) >= len(r1) * 0.15:
                    temp_std[iii] = (len(qv) * np.std(qv) + len(bv) * np.std(bv)) / len(fig1)
                else:
                    temp_std[iii] = 1e4
            Th = thres_list[temp_std == np.min(temp_std)]
    #--------------------------------------------------------------------------        
    else:               
        if method == "iterative":        
            zd = float(np.nanmax(fig1))
            zx = float(np.nanmin(fig1))
            Th = float((zd+zx))/2
            while True:                       
                S1 = np.sum(fig1[fig1>=Th])
                n1 = len(fig1[fig1>=Th])
                S0 = np.sum(fig1[fig1<Th])
                n0 = len(fig1[fig1<Th])
                T0 = S0/n0; T1 = S1/n1
                if abs(Th - ((T0 + T1)/2)) < 0.0001:
                    break
                else:
                    Th = (T0 + T1)/2 

        elif method == "otsu":
            # for ii in trange(len(gene_data)):
            img = fig1.reshape(locates)
            Th2, a_img = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_OTSU) 

    return fig1 >= Th


def get_binary(adata, platform="visium", method = "iterative", multiprocess=False):
    gene_data = adata.uns['gau_fea']
    if sp.issparse(adata.X):
        raw_gene_data = adata.X.A.T
    else:
        raw_gene_data = adata.X.T

    if platform=="visium":
        locates = adata.uns['locates']
    else:
        locates = adata.uns['shape']
    
    print('\nBinary segmentation for each gene:')
    
    N_gene = len(gene_data)
    def sel_data():  # data generater
        for gene_i in range(N_gene):
            yield [gene_data[gene_i, :], locates, platform, method, raw_gene_data[gene_i,:]]

    if multiprocess:
        num_cores = int(mp.cpu_count() / 2)         # default core is half of total
        with mp.Pool(processes=num_cores) as pool:
            output = list(tqdm(pool.imap(_iget_binary, sel_data()), total=N_gene))
    else:
        output = list(tqdm(map(_iget_binary, sel_data()), total=N_gene))
    output = np.array(output, dtype=np.float64) + 0.0
    adata.uns['binary_image'] = output
    return adata


def get_sub(adata, kernel_size = 5, platform="visium",del_rate = 0.01): 
    import cv2
    from skimage.measure import label
    gene_data = adata.uns['binary_image']
    locates = adata.uns['locates']
        
    print('\nSpliting subregions for each gene:')
    #--------------------------------------------------------------------------
    if platform=="visium":
        image_idx_1d = adata.obs['image_idx_1d']
        output = np.zeros(gene_data.shape)
        del_index = np.ones(gene_data.shape[0])
        for i in tqdm(range(len(gene_data))):
            temp_data = gene_data[i, :]
            temp_i, _ = make_image(temp_data, locates)      
            kernel = np.ones((kernel_size,kernel_size), np.uint8)
            temp_i = cv2.morphologyEx(temp_i, cv2.MORPH_CLOSE, kernel) # close
            region_label = label(temp_i)
            T = np.zeros(region_label.shape)
            classes = np.max(np.unique(region_label)) + 1      
            len_list = np.zeros(classes)     
            for j in range(classes):
                len_list[j] = len(region_label[region_label == j])
            cond = len_list >= gene_data.shape[1] * 0.01        
            if len(np.where(cond[1:] == True)[0]) == 0:
                del_index[i] = 0
            indexes = np.where(cond == True)[0]       
            for j in range(len(indexes)):
                tar_num = indexes[j]
                tar_locs = region_label == tar_num
                T[tar_locs] = j
            targe_image = T * (temp_i > 0)
            classes_n = np.max(np.unique(targe_image)).astype(int) + 1       
            len_list_n = np.zeros(classes_n)        
            for j in range(classes_n):
                len_list_n[j] = len(targe_image[targe_image == j])            
            if len(len_list_n) > 1:                        
                if np.max(len_list_n[1:]) < gene_data.shape[1] * del_rate:
                    del_index[i] = 0
            else:
                del_index[i] = 0
            output[i, :] = gene_img_flatten(targe_image, image_idx_1d.values)      
    #--------------------------------------------------------------------------
    else:
        output = np.zeros((gene_data.shape[0], adata.uns['shape'][0]*adata.uns['shape'][1]))
        del_index = np.ones(gene_data.shape[0])
        for i in tqdm(range(len(gene_data))):
            temp_data = gene_data[i, :]
            temp_i = temp_data.reshape(adata.uns['shape'])     
            kernel = np.ones((kernel_size,kernel_size), np.uint8)
            temp_i = cv2.morphologyEx(temp_i, cv2.MORPH_CLOSE, kernel)
            region_label = label(temp_i)
            T = np.zeros(region_label.shape)
            classes = np.max(region_label) + 1      
            len_list = np.zeros(classes)
            for j in range(classes):
                len_list[j] = len(region_label[region_label == j])
            cond = len_list >= gene_data.shape[1] * 0.002        
            if len(np.where(cond[1:] == True)[0]) == 0:
                del_index[i] = 0
            indexes = np.where(cond == True)[0]       
            for j in range(len(indexes)):
                tar_num = indexes[j]
                tar_locs = region_label == tar_num
                T[tar_locs] = j
            targe_image = T * (temp_i > 0)
            classes_n = np.max(np.unique(targe_image)).astype(int) + 1       
            len_list_n = np.zeros(classes_n)        
            for j in range(classes_n):
                len_list_n[j] = len(targe_image[targe_image == j])            
            if len(len_list_n) > 1:                        
                if np.max(len_list_n[1:]) < gene_data.shape[1] * del_rate:
                    del_index[i] = 0
            else:
                del_index[i] = 0
            output[i, :] = targe_image.flatten()
    #--------------------------------------------------------------------------
    adata.uns['subregions'] = output
    adata.uns['del_index'] = del_index.astype(int)  
    return adata


def cal_prost_index(adata, platform="visium"):
    data = adata.uns['nor_counts']
    subregions = adata.uns['subregions']
    del_idx = adata.uns['del_index']
    
    print('\nComputing PROST Index for each gene:')
    #--------------------------------------------------------------------------
    if platform=="visium":
        SEP = np.zeros(len(data))
        SIG = np.zeros(len(data))
        region_number = np.zeros(len(data))
        
        for i in tqdm(range(len(data))): 
            temp_raw = data[i, :]
            temp_label = subregions[i, :]
            back_value = temp_raw[temp_label == 0]
            back_value = back_value[back_value > 0]
            if back_value.size == 0:
                back_value = 0  
            class_mean = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_var = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_std = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_len = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
         
            for ii in range(max(np.unique(temp_label)).astype(int) + 1):
                Temp = temp_raw[temp_label == ii]
                if Temp.size == 0:
                    class_value = 0
                else:
                    class_value = Temp
                class_mean[ii] = np.mean(class_value)
                class_var[ii] = np.var(class_value)
                class_std[ii] = np.std(class_value)
                if isinstance(class_value, int):
                    if class_value == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1               
                else:
                    if class_value.size == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1
                        
            target_class = np.where(class_mean > 0)[0]
            class_mean = class_mean[target_class]
            class_std = class_std[target_class]
            class_var = class_var[target_class]
            class_len = class_len[target_class]
            
            # Calculate Separability and Significance
            SEP[i] = 1 - sum((class_len * class_var)) / ((len(temp_raw)-1) * np.var(temp_raw))
            SIG[i] = (np.mean(class_mean) - np.mean(back_value)) / sum(class_std / class_mean) 
            region_number[i] = len(class_len)
            del class_mean, class_var, class_len, class_std
                     
        # Pattern Index    
        PI = minmax_normalize(SEP) * minmax_normalize(SIG)
        PI = PI * del_idx   
        adata.var["SEP"] = SEP
        adata.var["SIG"] = SIG
        adata.var["PI"] = PI
    #--------------------------------------------------------------------------
    else:
        locates = adata.uns['locates']
        SEP = np.zeros(len(data))
        SIG = np.zeros(len(data))
        for i in tqdm(range(len(data))): 
            temp_raw = data[i, :]
            temp_img,_ = make_image(temp_raw, locates, platform)
            temp_raw = temp_img.flatten()
            temp_label = subregions[i, :]
            back_value = temp_raw[temp_label == 0]
            back_value = back_value[back_value > 0]
            
            if back_value.size == 0:
                back_value = 0  
            class_mean = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_var = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_std = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
            class_len = np.zeros(max(np.unique(temp_label)).astype(int) + 1)
         
            for ii in range(max(np.unique(temp_label)).astype(int) + 1):
                Temp = temp_raw[temp_label == ii]
                if Temp.size == 0:
                    class_value = 0
                else:
                    class_value = Temp
                class_mean[ii] = np.nanmean(class_value)
                class_var[ii] = np.nanvar(class_value)
                class_std[ii] = np.nanstd(class_value)
                if isinstance(class_value, int):
                    if class_value == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1               
                else:
                    if class_value.size == 0:
                        class_len[ii] = 0
                    else:              
                        class_len[ii] = len(class_value) - 1
                        
            target_class = np.where(class_mean > 0)[0]
            class_mean = class_mean[target_class]
            class_std = class_std[target_class]
            class_var = class_var[target_class]
            class_len = class_len[target_class]
            
            # Calculate Separability and Significance
            SEP[i] = 1 - sum((class_len * class_var)) / ((len(temp_raw)-1) * np.nanvar(temp_raw))
            SIG[i] = (np.mean(class_mean) - np.mean(back_value)) / sum(class_std / class_mean)      
            del class_mean, class_var, class_len, class_std
            
        # Pattern Index    
        PI = minmax_normalize(SEP) * minmax_normalize(SIG)
        PI = PI * del_idx
        adata.var["SEP"] = SEP
        adata.var["SIG"] = SIG
        adata.var["PI"] = PI
        
        adata.uns['shape'] = []
    #--------------------------------------------------------------------------
    adata.uns['nor_counts'] = []
    adata.uns['binary_image'] = []
    adata.uns['subregions'] = []
    adata.uns['del_index'] = []
    
    return adata
