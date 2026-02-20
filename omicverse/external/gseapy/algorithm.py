# -*- coding: utf-8 -*-

import sys, logging
import numpy as np
from math import ceil
from .stats import multiple_testing_correction



def enrichment_score(gene_list, correl_vector, gene_set, weighted_score_type=1, 
                     nperm=1000, seed=None, single=False, scale=False):
    """This is the most important function of GSEApy. It has the same algorithm with GSEA and ssGSEA.

    :param gene_list:       The ordered gene list gene_name_list, rank_metric.index.values
    :param gene_set:        gene_sets in gmt file, please use gsea_gmt_parser to get gene_set.
    :param weighted_score_type:  It's the same with gsea's weighted_score method. Weighting by the correlation
                            is a very reasonable choice that allows significant gene sets with less than perfect coherence.
                            options: 0(classic),1,1.5,2. default:1. if one is interested in penalizing sets for lack of
                            coherence or to discover sets with any type of nonrandom distribution of tags, a value p < 1
                            might be appropriate. On the other hand, if one uses sets with large number of genes and only
                            a small subset of those is expected to be coherent, then one could consider using p > 1.
                            Our recommendation is to use p = 1 and use other settings only if you are very experienced
                            with the method and its behavior.

    :param correl_vector:   A vector with the correlations (e.g. signal to noise scores) corresponding to the genes in
                            the gene list. Or rankings, rank_metric.values
    :param nperm:           Only use this parameter when computing esnull for statistical testing. Set the esnull value
                            equal to the permutation number.
    :param seed:            Random state for initializing gene list shuffling. Default: seed=None

    :return:

     ES: Enrichment score (real number between -1 and +1)

     ESNULL: Enrichment score calculated from random permutations.

     Hits_Indices: Index of a gene in gene_list, if gene is included in gene_set.

     RES: Numerical vector containing the running enrichment score for all locations in the gene list .

    """
    N = len(gene_list)
    # Test whether each element of a 1-D array is also present in a second array
    # It's more intuitive here than original enrichment_score source code.
    # use .astype to covert bool to integer
    tag_indicator = np.isin(gene_list, gene_set, assume_unique=True).astype(int)  # notice that the sign is 0 (no tag) or 1 (tag)

    if weighted_score_type == 0 :
        correl_vector = np.repeat(1, N)
    else:
        correl_vector = np.abs(correl_vector)**weighted_score_type

    # get indices of tag_indicator
    hit_ind = np.flatnonzero(tag_indicator).tolist()
    # if used for compute esnull, set esnull equal to permutation number, e.g. 1000
    # else just compute enrichment scores
    # set axis to 1, because we have 2D array
    axis = 1
    tag_indicator = np.tile(tag_indicator, (nperm+1,1))
    correl_vector = np.tile(correl_vector,(nperm+1,1))
    # gene list permutation
    rs = np.random.RandomState(seed)
    for i in range(nperm): rs.shuffle(tag_indicator[i])
    # np.apply_along_axis(rs.shuffle, 1, tag_indicator)

    Nhint = tag_indicator.sum(axis=axis, keepdims=True)
    sum_correl_tag = np.sum(correl_vector*tag_indicator, axis=axis, keepdims=True)
    # compute ES score, the code below is identical to gsea enrichment_score method.
    no_tag_indicator = 1 - tag_indicator
    Nmiss =  N - Nhint
    norm_tag =  1.0/sum_correl_tag
    norm_no_tag = 1.0/Nmiss

    RES = np.cumsum(tag_indicator * correl_vector * norm_tag - no_tag_indicator * norm_no_tag, axis=axis)

    if scale: RES = RES / N
    if single:
        es_vec = RES.sum(axis=axis)
    else:
        max_ES, min_ES =  RES.max(axis=axis), RES.min(axis=axis)
        es_vec = np.where(np.abs(max_ES) > np.abs(min_ES), max_ES, min_ES)
    # extract values
    es, esnull, RES = es_vec[-1], es_vec[:-1], RES[-1,:]

    return es, esnull, hit_ind, RES



def enrichment_score_tensor(gene_mat, cor_mat, gene_sets, weighted_score_type, nperm=1000,
                            seed=None, single=False, scale=False):
    """Next generation algorithm of GSEA and ssGSEA. Works for 3d array

        :param gene_mat:        the ordered gene list(vector) with or without gene indices matrix.
        :param cor_mat:         correlation vector or matrix  (e.g. signal to noise scores)
                                corresponding to the genes in the gene list or matrix.
        :param dict gene_sets:  gmt file dict.
        :param float weighted_score_type:     weighting by the correlation.
                                options: 0(classic), 1, 1.5, 2. default:1 for GSEA and 0.25 for ssGSEA.
        :param int nperm:       permutation times.
        :param bool scale:      If True, normalize the scores by number of genes_mat.
        :param bool single:     If True, use ssGSEA algorithm, otherwise use GSEA.
        :param seed:              Random state for initialize gene list shuffling.
                                Default: seed=None
        :return: a tuple contains::

                 | ES: Enrichment score (real number between -1 and +1), for ssGSEA, set scale eq to True.
                 | ESNULL: Enrichment score calculated from random permutation.
                 | Hits_Indices: Indices of genes if genes are included in gene_set.
                 | RES: The running enrichment score for all locations in the gene list.

    """
    rs = np.random.RandomState(seed)
    # gene_mat -> 1d: prerank, ssSSEA or 2d: GSEA
    keys = sorted(gene_sets.keys())

    if weighted_score_type == 0:
        # don't bother doing calcuation, just set to 1
        cor_mat = np.ones(cor_mat.shape)
    elif weighted_score_type > 0:
        pass
    else:
        logging.error("Using negative values of weighted_score_type, not allowed")
        raise ValueError("weighted_score_type should be postive numerics")

    cor_mat = np.abs(cor_mat)
    if cor_mat.ndim ==1:
        # ssGSEA or Prerank
        # genestes->M, genes->N, perm-> axis=2
        N, M = len(gene_mat), len(keys)
        # generate gene hits matrix
        # for 1d ndarray of gene_mat, set assume_unique=True,
        # means the input arrays are both assumed to be unique,
        # which can speed up the calculation.
        tag_indicator = np.vstack([np.isin(gene_mat, gene_sets[key], assume_unique=True) for key in keys])
        tag_indicator = tag_indicator.astype(int)
        # index of hits
        hit_ind = [ np.flatnonzero(tag).tolist() for tag in tag_indicator ]
        # generate permutated hits matrix
        perm_tag_tensor = np.repeat(tag_indicator, nperm+1).reshape((M,N,nperm+1))
        # shuffle matrix, last matrix is not shuffled when nperm > 0
        if nperm: np.apply_along_axis(lambda x: np.apply_along_axis(rs.shuffle,0,x),1, perm_tag_tensor[:,:,:-1])
        # missing hits
        no_tag_tensor = 1 - perm_tag_tensor
        # calculate numerator, denominator of each gene hits
        rank_alpha = (perm_tag_tensor*cor_mat[np.newaxis,:,np.newaxis])** weighted_score_type

    elif cor_mat.ndim == 2:
        # GSEA
        # 2d ndarray, gene_mat and cor_mat are shuffled already
        # reshape matrix
        cor_mat = cor_mat.T
        # gene_mat is a tuple contains (gene_name, permuate_gene_name_indices)
        genes, genes_ind = gene_mat
        # genestes->M, genes->N, perm-> axis=2
        # don't use assume_unique=True in 2d array when use np.isin().
        # elements in gene_mat are not unique, or will cause unwanted results
        tag_indicator = np.vstack([np.isin(genes, gene_sets[key], assume_unique=True) for key in keys])
        tag_indicator = tag_indicator.astype(int)
        perm_tag_tensor = np.stack([tag.take(genes_ind).T for tag in tag_indicator], axis=0)
        #index of hits
        hit_ind = [ np.flatnonzero(tag).tolist() for tag in perm_tag_tensor[:,:,-1] ]
        # nohits
        no_tag_tensor = 1 - perm_tag_tensor
        # calculate numerator, denominator of each gene hits
        rank_alpha = (perm_tag_tensor*cor_mat[np.newaxis,:,:])** weighted_score_type
    else:
        logging.error("Program die because of unsupported input")
        raise ValueError("Correlation vector or matrix (cor_mat) is not supported")

    # Nhint = tag_indicator.sum(1)
    # Nmiss =  N - Nhint
    axis=1
    P_GW_denominator = np.sum(rank_alpha, axis=axis, keepdims=True)
    P_NG_denominator = np.sum(no_tag_tensor, axis=axis, keepdims=True)
    REStensor = np.cumsum(rank_alpha / P_GW_denominator - no_tag_tensor / P_NG_denominator, axis=axis)
    # ssGSEA: scale es by gene numbers ?
    # https://gist.github.com/gaoce/39e0907146c752c127728ad74e123b33
    if scale: REStensor = REStensor / len(gene_mat)
    if single:
        #ssGSEA
        esmatrix = REStensor.sum(axis=axis)
    else:
        #GSEA
        esmax, esmin = REStensor.max(axis=axis), REStensor.min(axis=axis)
        esmatrix = np.where(np.abs(esmax)>np.abs(esmin), esmax, esmin)

    es, esnull, RES = esmatrix[:,-1], esmatrix[:,:-1], REStensor[:,:,-1]

    return es, esnull, hit_ind, RES


def ranking_metric_tensor(exprs, method, permutation_num, pos, neg, classes,
                          ascending, seed=None, skip_last=False):
    """Build shuffled ranking matrix when permutation_type eq to phenotype.
       Works for 3d array.

       :param exprs:   gene_expression DataFrame, gene_name indexed.
       :param str method:  calculate correlation or ranking. methods including:
                           1. 'signal_to_noise' (s2n) or 'abs_signal_to_noise' (abs_s2n).
                           2. 't_test'.
                           3. 'ratio_of_classes' (also referred to as fold change).
                           4. 'diff_of_classes'.
                           5. 'log2_ratio_of_classes'.
       :param int permuation_num: how many times of classes is being shuffled
       :param str pos: one of labels of phenotype's names.
       :param str neg: one of labels of phenotype's names.
       :param list classes:  a list of phenotype labels, to specify which column of
                             dataframe belongs to what class of phenotype.
       :param bool ascending:  bool. Sort ascending vs. descending.
       :param seed: random_state seed
       :param bool skip_last: (internal use only) whether to skip the permutation of the last rankings.  

       :return:
                returns two 2d ndarray with shape (nperm, gene_num).

                | cor_mat_indices: the indices of sorted and permutated (exclude last row) ranking matrix.
                | cor_mat: sorted and permutated (exclude last row) ranking matrix.

    """
    rs = np.random.RandomState(seed)
    # S: samples, G: gene number
    G, S = exprs.shape
    # genes = exprs.index.values
    expr_mat = exprs.values.T
    perm_cor_tensor = np.tile(expr_mat, (permutation_num,1,1))
    if skip_last:
        # random shuffle on the first dim, the last matrix (expr_mat) is not shuffled
        for arr in perm_cor_tensor[:-1]: rs.shuffle(arr)
    else:
        for arr in perm_cor_tensor: rs.shuffle(arr)   
    # metrics     
    classes = np.array(classes)
    pos = classes == pos
    neg = classes == neg
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    pos_cor_mean = perm_cor_tensor[:,pos,:].mean(axis=1)
    neg_cor_mean = perm_cor_tensor[:,neg,:].mean(axis=1)
    pos_cor_std = perm_cor_tensor[:,pos,:].std(axis=1, ddof=1)
    neg_cor_std = perm_cor_tensor[:,neg,:].std(axis=1, ddof=1)

    if method in ['signal_to_noise', 's2n']:
        cor_mat = (pos_cor_mean - neg_cor_mean)/(pos_cor_std + neg_cor_std)
    elif method in ['abs_signal_to_noise', 'abs_s2n']:
        cor_mat = np.abs((pos_cor_mean - neg_cor_mean)/(pos_cor_std + neg_cor_std))
    elif method == 't_test':
        denom = np.sqrt((pos_cor_std**2)/n_pos  + (neg_cor_std**2)/n_neg)
        cor_mat = (pos_cor_mean - neg_cor_mean)/ denom
    elif method == 'ratio_of_classes':
        cor_mat = pos_cor_mean / neg_cor_mean
    elif method == 'diff_of_classes':
        cor_mat  = pos_cor_mean - neg_cor_mean
    elif method == 'log2_ratio_of_classes':
        cor_mat  =  np.log2(pos_cor_mean / neg_cor_mean)
    else:
        logging.error("Please provide correct method name!!!")
        raise LookupError("Input method: %s is not supported"%method)
    # return matix[nperm+1, perm_cors]
    cor_mat_ind = cor_mat.argsort()
    # ndarray: sort in place
    cor_mat.sort()
    # genes_mat = genes.take(cor_mat_ind)
    if ascending: return cor_mat_ind, cor_mat
    # descending order of ranking and genes
    # return genes_mat[:,::-1], cor_mat[:,::-1]
    return cor_mat_ind[:, ::-1], cor_mat[:, ::-1]

def ranking_metric(df, method, pos, neg, classes, ascending):
    """The main function to rank an expression table. works for 2d array.

       :param df:      gene_expression DataFrame.
       :param method:  The method used to calculate a correlation or ranking. Default: 'log2_ratio_of_classes'.
                       Others methods are:

                       1. 'signal_to_noise' (s2n) or 'abs_signal_to_noise' (abs_s2n)

                          You must have at least three samples for each phenotype to use this metric.
                          The larger the signal-to-noise ratio, the larger the differences of the means (scaled by the standard deviations);
                          that is, the more distinct the gene expression is in each phenotype and the more the gene acts as a “class marker.”
                       

                       2. 't_test'

                          Uses the difference of means scaled by the standard deviation and number of samples.
                          Note: You must have at least three samples for each phenotype to use this metric.
                          The larger the tTest ratio, the more distinct the gene expression is in each phenotype
                          and the more the gene acts as a “class marker.”

                       3. 'ratio_of_classes' (also referred to as fold change).

                          Uses the ratio of class means to calculate fold change for natural scale data.

                       4. 'diff_of_classes'

                          Uses the difference of class means to calculate fold change for natural scale data

                       5. 'log2_ratio_of_classes'

                          Uses the log2 ratio of class means to calculate fold change for natural scale data.
                          This is the recommended statistic for calculating fold change for log scale data.


       :param str pos: one of labels of phenotype's names.
       :param str neg: one of labels of phenotype's names.
       :param dict classes: column id to group mapping.
       :param bool ascending:  bool or list of bool. Sort ascending vs. descending.
       :return:

            returns a pd.Series of correlation to class of each variable. Gene_name is index, and value is rankings.

            visit here for more docs: http://software.broadinstitute.org/gsea/doc/GSEAUserGuideFrame.html
    """

    # exclude any zero stds.
    df_mean = df.groupby(by=classes, axis=1).mean()
    df_std =  df.groupby(by=classes, axis=1).std()
    n_pos = np.sum(classes == pos)
    n_neg = np.sum(classes == neg)
    if method in ['signal_to_noise', 's2n']:
        ser = (df_mean[pos] - df_mean[neg])/(df_std[pos] + df_std[neg])
    elif method in ['abs_signal_to_noise', 'abs_s2n']:
        ser = ((df_mean[pos] - df_mean[neg])/(df_std[pos] + df_std[neg])).abs()
    elif method == 't_test':
        ser = (df_mean[pos] - df_mean[neg])/ np.sqrt(df_std[pos]**2/n_pos+df_std[neg]**2/n_neg)
    elif method == 'ratio_of_classes':
        ser = df_mean[pos] / df_mean[neg]
    elif method == 'diff_of_classes':
        ser  = df_mean[pos] - df_mean[neg]
    elif method == 'log2_ratio_of_classes':
        ser  =  np.log2(df_mean[pos] / df_mean[neg])
    else:
        logging.error("Please provide correct method name!!!")
        raise LookupError("Input method: %s is not supported"%method)
    ser = ser.sort_values(ascending=ascending)

    return ser


def gsea_compute_tensor(data, gmt, n, weighted_score_type, permutation_type,
                 method, pheno_pos, pheno_neg, classes, ascending,
                 processes=1, seed=None, single=False, scale=False):
    """compute enrichment scores and enrichment nulls.
       This function will split large array into smaller pieces to advoid memroy overflow.

        :param data: preprocessed expression dataframe or a pre-ranked file if prerank=True.
        :param dict gmt: all gene sets in .gmt file. need to call load_gmt() to get results.
        :param int n: permutation number. default: 1000.
        :param str method: ranking_metric method. see above.
        :param str pheno_pos: one of labels of phenotype's names.
        :param str pheno_neg: one of labels of phenotype's names.
        :param list classes: a list of phenotype labels, to specify which column of dataframe belongs to what category of phenotype.
        :param float weighted_score_type: default:1
        :param bool ascending: sorting order of rankings. Default: False.
        :param seed: random seed. Default: np.random.RandomState()
        :param bool scale: if true, scale es by gene number.

        :return: a tuple contains::

                | zipped results of es, nes, pval, fdr.
                | nested list of hit indices of input gene_list.
                | nested list of ranked enrichment score of each input gene_sets.
                | list of enriched terms

    """
    from joblib import delayed, Parallel
    w = weighted_score_type
    subsets = sorted(gmt.keys())
    genes_mat, cor_mat = data.index.values, data.values
    base = 5 if data.shape[0] >= 5000 else 10
    ## phenotype permutation
    np.random.seed(seed) # control the ranodm numbers

    if permutation_type == "phenotype":
        # shuffling classes and generate random correlation rankings
        logging.debug("Start to permutate classes..............................")

        if (n + 1) % base == 0: # n+1: last permute is for orignial ES calculation
            num_bases = [ base ] * ((n + 1) // base)
            skip_last = [0] * ( n // base) + [1] # last is not permuted
        else:
            num_bases = [ base ] * ((n + 1) // base) + [ (n +1) % base]
            skip_last = [0] * ((n + 1) // base) + [ (n +1) % base]
        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(num_bases))
        genes_ind = []
        cor_mat = []
        # split permutation array into smaller blocks to save memory
        temp_rnk = Parallel(n_jobs=processes)(delayed(ranking_metric_tensor)(
            data, method, b, pheno_pos, pheno_neg, classes, ascending, se, skip) 
            for b, skip, se in zip(num_bases, skip_last, random_seeds))

        for k, temp in enumerate(temp_rnk):
            gi, cor = temp
            genes_ind.append(gi)
            cor_mat.append(cor)
        genes_ind, cor_mat = np.vstack(genes_ind), np.vstack(cor_mat)
        # convert to tuple
        genes_mat = (data.index.values, genes_ind)
    
    logging.debug("Start to compute es and esnulls........................")
    # Prerank, ssGSEA, GSEA
    es = []
    RES = []
    hit_ind = []
    esnull = []
    temp_esnu = []
    
    # split gmt dataset, too
    block = ceil(len(subsets) / base)
    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=block)
    # split large array into smaller blocks to avoid memory overflow
    i, m = 1, 0
    gmt_block = []
    while i <= block:
        # you have to reseed, or all your processes are sharing the same seed value
        rs = random_seeds[i-1]
        gmtrim = {k: gmt.get(k) for k in subsets[m:base * i]}
        gmt_block.append(gmtrim)
        m = base * i
        i += 1
    ## if permutation_type == "phenotype": n = 0
    ## NOTE for GSEA: cor_mat is 2d array, it won't permute again when call enrichment_score_tensor
    temp_esnu = Parallel(n_jobs=processes)(delayed(enrichment_score_tensor)(
                    genes_mat, cor_mat, gmtrim, w, n, rs, single, scale) 
                    for gmtrim, rs in zip(gmt_block, random_seeds))

    # esn is a list, don't need to use append method.
    for si, temp in enumerate(temp_esnu):
        # e, enu, hit, rune = temp.get()
        e, enu, hit, rune = temp
        esnull.append(enu)
        es.append(e)
        RES.append(rune)
        hit_ind += hit
    # concate results
    es, esnull, RES = np.hstack(es), np.vstack(esnull), np.vstack(RES)

    return gsea_significance(es, esnull), hit_ind, RES, subsets



def gsea_compute(data, gmt, n, weighted_score_type, permutation_type,
                 method, pheno_pos, pheno_neg, classes, ascending,
                 processes=1, seed=None, single=False, scale=False):
    """compute enrichment scores and enrichment nulls.

        :param data: preprocessed expression dataframe or a pre-ranked file if prerank=True.
        :param dict gmt: all gene sets in .gmt file. need to call load_gmt() to get results.
        :param int n: permutation number. default: 1000.
        :param str method: ranking_metric method. see above.
        :param str pheno_pos: one of labels of phenotype's names.
        :param str pheno_neg: one of labels of phenotype's names.
        :param list classes: a list of phenotype labels, to specify which column of dataframe belongs to what category of phenotype.
        :param float weighted_score_type: default:1
        :param bool ascending: sorting order of rankings. Default: False.
        :param seed: random seed. Default: np.random.RandomState()
        :param bool scale: if true, scale es by gene number.

        :return: a tuple contains::

                | zipped results of es, nes, pval, fdr.
                | nested list of hit indices of input gene_list.
                | nested list of ranked enrichment score of each input gene_sets.
                | list of enriched terms

    """
    from joblib import delayed, Parallel
    
    w = weighted_score_type
    subsets = sorted(gmt.keys())
    es = []
    RES=[]
    hit_ind=[]
    esnull = [ [] for a in range(len(subsets)) ]
    np.random.seed(seed) # control the ranodm numbers
    logging.debug("Start to compute enrichment scores......................")

    if permutation_type == "phenotype":
        logging.debug("Start to permutate classes..............................")
        # this version won't split large array into smaller ones
        genes_mat, cor_mat = ranking_metric_tensor(exprs=data, method=method,
                                                   permutation_num=n+1,
                                                   pos=pheno_pos, neg=pheno_neg,
                                                   classes=classes,
                                                   ascending=ascending, seed=seed, 
                                                   skip_last=True)

        # compute es, esnulls. hits, RES
        logging.debug("Start to compute enrichment nulls.......................")
        es, esnull, hit_ind, RES = enrichment_score_tensor(gene_mat=genes_mat,
                                                           cor_mat=cor_mat,
                                                           gene_sets=gmt,
                                                           weighted_score_type=w,
                                                           nperm=n, seed=seed,
                                                           single=False, scale=False,)

    else:
        # Prerank, ssGSEA, GSEA with gene_set permutation
        gl, cor_vec = data.index.values, data.values
        logging.debug("Start to compute es and esnulls........................")
        ## this version don't split large array into smaller ones
        # es, esnull, hit_ind, RES = enrichment_score_tensor(gene_mat=gl,
        #                                                    cor_mat=cor_vec,
        #                                                    gene_sets=gmt,
        #                                                    weighted_score_type=w,
        #                                                    nperm=n, rs=rs
        #                                                    single=single, scale=scale)
        temp_esnu=[]
        # you have to reseed, or all your processes are sharing the same seed value
        # np.random.seed(seed)
        random_seeds= np.random.randint(np.iinfo(np.int32).max, size=len(subsets))
        temp_esnu = Parallel(n_jobs=processes)(delayed(enrichment_score)(
                        gl, cor_vec, gmt.get(subset), w, n, 
                        rs, single, scale) 
                        for subset, rs in zip(subsets, random_seeds))        
        # esn is a list, don't need to use append method.
        for si, temp in enumerate(temp_esnu):
            e, enu, hit, rune = temp
            esnull[si] = enu
            es.append(e)
            RES.append(rune)
            hit_ind.append(hit)

    return gsea_significance(es, esnull), hit_ind, RES, subsets


def normalize(es, esnull):
    """normalize the ES(S,pi) and the observed ES(S), separately rescaling
       the positive and negative scores by dividing the mean of the ES(S,pi).
       
       return: NES, NESnull
    """

    nEnrichmentScores =np.zeros(es.shape)
    nEnrichmentNulls=np.zeros(esnull.shape)
    # esnullmean = np.zeros(es.shape)
    # # calculate nESnulls
    # for i in range(esnull.shape[0]):
    #     # NES
    #     enrNull = esnull[i]
    #     if es[i] >= 0:
    #         mes = enrNull[enrNull >= 0].mean()
    #         nEnrichmentScores[i] = es[i] / mes
    #     else:
    #         mes = enrNull[enrNull < 0 ].mean()
    #         nEnrichmentScores[i] = - es[i] / mes
    #     esnullmean[i] = mes

    #     # NESnull
    #     for j in range(esnull.shape[1]):
    #         if esnull[i,j] >= 0:
    #             nEnrichmentNulls[i,j] = esnull[i,j] / esnullmean[i]
    #         else:
    #             nEnrichmentNulls[i,j] = - esnull[i,j] / esnullmean[i]

    esnull_pos = np.ma.MaskedArray(esnull, mask=(esnull<0)).mean(axis=1)
    esnull_neg = np.ma.MaskedArray(esnull, mask=(esnull>=0)).mean(axis=1)
    esnull_pos = np.array(esnull_pos)
    esnull_neg = np.array(esnull_neg)
    # NES
    nEnrichmentScores  = np.where(es>=0, es/esnull_pos, -es/esnull_neg)
    # NES_NULL
    nEnrichmentNulls = np.where(esnull>=0, esnull/esnull_pos[:,np.newaxis],
                                          -esnull/esnull_neg[:,np.newaxis])
    
    return nEnrichmentScores, nEnrichmentNulls

def gsea_pval(es, esnull):
    """Compute nominal p-value.

    From article (PNAS):
    estimate nominal p-value for S from esnull by using the positive
    or negative portion of the distribution corresponding to the sign
    of the observed ES(S).
    """

    # to speed up, using numpy function to compute pval in parallel.
    condlist = [ es < 0, es >=0]
    choicelist = [(esnull < es.reshape(len(es),1)).sum(axis=1)/ (esnull < 0).sum(axis=1),
                  (esnull >= es.reshape(len(es),1)).sum(axis=1)/ (esnull >= 0).sum(axis=1)]
    pvals = np.select(condlist, choicelist)

    return pvals


def gsea_fdr(nEnrichmentScores, nEnrichmentNulls):
    """Create a histogram of all NES(S,pi) over all S and pi.
       Use this null distribution to compute an FDR q value.
       
    :param nEnrichmentScores:  normalized ES
    :param nEnrichmentNulls:   normalized ESnulls
    :return: FDR
    """

    # FDR null distribution histogram
    # vals = reduce(lambda x,y: x+y, nEnrichmentNulls, [])
    # nvals = np.array(sorted(vals))
    # or
    nvals = np.sort(nEnrichmentNulls.flatten())
    nnes = np.sort(nEnrichmentScores)
    fdrs = []
    # FDR computation
    for i in range(len(nEnrichmentScores)):
        nes = nEnrichmentScores[i]
        # use the same pval method to calculate fdr
        if nes >= 0:
            allPos = int(len(nvals) - np.searchsorted(nvals, 0, side="left"))
            allHigherAndPos = int(len(nvals) - np.searchsorted(nvals, nes, side="left"))
            nesPos = len(nnes) - int(np.searchsorted(nnes, 0, side="left"))
            nesHigherAndPos = len(nnes) - int(np.searchsorted(nnes, nes, side="left"))
            # allPos = (nvals >= 0).sum()
            # allHigherAndPos = (nvals >= nes).sum()
            # nesPos = (nnes >=0).sum()
            # nesHigherAndPos = (nnes >= nes).sum()
        else:
            allPos = int(np.searchsorted(nvals, 0, side="left"))
            allHigherAndPos = int(np.searchsorted(nvals, nes, side="right"))
            nesPos = int(np.searchsorted(nnes, 0, side="left"))
            nesHigherAndPos = int(np.searchsorted(nnes, nes, side="right"))
            # allPos = (nvals < 0).sum()
            # allHigherAndPos = (nvals < nes).sum()
            # nesPos = (nnes < 0).sum()
            # nesHigherAndPos = (nnes < nes).sum()

        try:
            pi_norm = allHigherAndPos / float(allPos)
            pi_obs = nesHigherAndPos / float(nesPos)
            fdr = pi_norm / pi_obs
            fdrs.append(fdr if fdr < 1 else 1.0)
        except:
            fdrs.append(1000000000.0)

    logging.debug("Statistical testing finished.............................")

    return fdrs


def gsea_significance(enrichment_scores, enrichment_nulls):
    """Compute nominal pvals, normalized ES, and FDR q value.

        For a given NES(S) = NES* >= 0. The FDR is the ratio of the percentage of all (S,pi) with
        NES(S,pi) >= 0, whose NES(S,pi) >= NES*, divided by the percentage of
        observed S wih NES(S) >= 0, whose NES(S) >= NES*, and similarly if NES(S) = NES* <= 0.
    """
    # For a zero by zero division (undetermined, results in a NaN),
    np.seterr(divide='ignore', invalid='ignore')
    # import warnings
    # warnings.simplefilter("ignore")
    es = np.array(enrichment_scores)
    esnull = np.array(enrichment_nulls)
    logging.debug("Start to compute pvals..................................")
    # P-values.
    pvals = gsea_pval(es, esnull).tolist()

    logging.debug("Start to compute nes and nesnull........................")
    # NES
    nEnrichmentScores, nEnrichmentNulls = normalize(es, esnull)

    logging.debug("Start to compute fdrs..................................")
    # FDR
    fdrs = gsea_fdr(nEnrichmentScores, nEnrichmentNulls)

    #TODO: use multiple testing correction for ssgsea? ssGSEA2.0 use BH correction.
    # https://github.com/broadinstitute/ssGSEA2.0/blob/master/src/ssGSEA2.0.R
    # line 969
    # fdrs, _ = multiple_testing_correction(pvals, alpha=0.05)

    return zip(enrichment_scores, nEnrichmentScores, pvals, fdrs)




