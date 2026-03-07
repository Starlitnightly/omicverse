"""
The function cytotrace2 is a Python implementation of CytoTRACE 2, a deep learning-based tool for cell potency prediction. CytoTRACE 2 predicts
"""
import concurrent.futures
import math
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc


from .._settings import add_reference
from .._registry import register_function

#from ..external.cytotrace2.gen_utils import *
#from cytotrace2_py.common.argument_parser import *

#pylint: disable=too-many-arguments
#pylint: disable=too-many-locals
def process_subset(idx, chunked_expression, smooth_batch_size, smooth_cores_to_use, 
                   species, use_model_dir, output_dir, max_pcs, seed):
    r"""Run CytoTRACE2 prediction for one data chunk.

    Parameters
    ----------
    idx : int
        Index of the current chunk. Used to generate temporary file suffixes.
    chunked_expression : pandas.DataFrame
        Cell-by-gene expression matrix for the current chunk.
    smooth_batch_size : int
        Batch size used during diffusion-based smoothing.
    smooth_cores_to_use : int
        Number of CPU cores used by smoothing steps.
    species : str
        Species label passed to CytoTRACE2 preprocessing (for example,
        ``'mouse'`` or ``'human'``).
    use_model_dir : str
        Directory containing CytoTRACE2 model weight files.
    output_dir : str
        Directory used for intermediate and final output files.
    max_pcs : int
        Maximum number of principal components for KNN smoothing.
    seed : int
        Random seed for reproducible stochastic steps.

    Returns
    -------
    pandas.DataFrame
        Predicted potency table for the chunk after KNN smoothing.
    """
    from ..external.cytotrace2.gen_utils import preprocess, top_var_genes, predict, smoothing_by_diffusion, binning

    # map and rank
    cell_names, gene_names, ranked_data = preprocess(chunked_expression, species)

    # top variable genes
    top_col_inds = top_var_genes(ranked_data)
    top_col_names = gene_names[top_col_inds]

    # predict by unrandomized chunked batches
    predicted_df = predict(ranked_data, cell_names, use_model_dir , chunked_expression.shape[0])

    smooth_score = smoothing_by_diffusion(predicted_df, ranked_data, 
                                          top_col_inds, smooth_batch_size,  seed)

    binned_score_pred_df = binning(predicted_df, smooth_score)

    # Transpose the matrix and create a DataFrame
    ranked_df = pd.DataFrame(ranked_data.T,  columns = cell_names)

    # Set the column names
    ranked_df.index = gene_names
    suffix = '_'+str(idx)
    binned_score_pred_df.to_csv(output_dir+'/binned_df'+suffix+'.txt',sep='\t', index=True)
    ranked_df.to_csv(output_dir+'/ranked_df'+suffix+'.txt', sep='\t', index=True)
	
    with open(output_dir+'/top_var_genes'+suffix+'.txt', 'w',
              encoding='utf-8') as f:
        for item in top_col_names:
            f.write(f"{item}\n")

    if chunked_expression.shape[0] < 100:
        print('cytotrace2: Fewer than 100 cells in dataset. Skipping KNN smoothing step.')
        smooth_by_knn_df = binned_score_pred_df.copy()
    else:
        #run_script = pkg_resources.resource_filename("cytotrace2_py","resources/smoothDatakNN.R")
        #knn_path = output_dir+'/smoothbykNNresult'+suffix+'.txt'
        #out = subprocess.run(['Rscript', run_script, '--output-dir', output_dir,
               #'--suffix', suffix, '--max-pcs', str(max_pcs), '--seed', str(seed)], check=True)
        #smooth_by_knn_df = pd.read_csv(knn_path, index_col = 0, sep='\t')
        from ..external.cytotrace2.smoothDatakNN import smooth_data_kNN
        smooth_data_kNN(output_dir=output_dir, suffix=suffix, max_pcs=max_pcs, seed=seed)
        # 读取平滑处理后的结果
        knn_path = os.path.join(output_dir, f'smoothbykNNresult{suffix}.txt')
        smooth_by_knn_df = pd.read_csv(knn_path, index_col=0, sep='\t')

    return smooth_by_knn_df

def calculate_cores_to_use(chunk_number,smooth_chunk_number,max_cores,disable_parallelization):
    r"""Choose prediction and smoothing worker counts.

    Parameters
    ----------
    chunk_number : int
        Number of prediction chunks generated from the input dataset.
    smooth_chunk_number : int
        Number of smoothing sub-chunks within each prediction chunk.
    max_cores : int or None
        Optional user-provided cap on total CPU cores.
    disable_parallelization : bool
        If ``True``, force single-core execution.

    Returns
    -------
    tuple[int, int]
        ``(pred_cores_to_use, smooth_cores_to_use)``.
    """

    pred_cores_to_use = 1
    smooth_cores_to_use = 1
    if smooth_chunk_number == 1 and chunk_number == 1:
        print("cytotrace2: The number of cells in your dataset is \
              less than the specified batch size.\n")
        print("    Model prediction will not be parallelized.")

    elif not disable_parallelization:
        # Calculate number of available processors
        num_proc = os.cpu_count()
        print("cytotrace2: "+str(num_proc)+" cores detected")
        if num_proc == 1:
            print("cytotrace2: Only one core detected. CytoTRACE 2 will not be run in parallel.")
        elif max_cores is None:
            pred_cores_to_use = min(chunk_number,num_proc-1)
            smooth_cores_to_use = min(smooth_chunk_number,max(math.floor((num_proc-1)/pred_cores_to_use),1))
            print('cytotrace2: Running '+str(pred_cores_to_use)+' prediction batch(es) in parallel using '+str(smooth_cores_to_use)+' cores for smoothing per batch.')
        else:
            max_cores = min(max_cores,num_proc-1)
            pred_cores_to_use = min(chunk_number,max_cores)
            smooth_cores_to_use = min(smooth_chunk_number,
                                      max(math.floor(max_cores/pred_cores_to_use),1))
            print('cytotrace2: Running '+str(pred_cores_to_use)+' prediction batch(es) in parallel using '+str(smooth_cores_to_use)+' cores for smoothing per batch.')

    return pred_cores_to_use, smooth_cores_to_use


@register_function(
    aliases=["细胞潜能预测", "cytotrace2", "cell_potency", "发育潜能", "CytoTRACE2"],
    category="single",
    description="CytoTRACE 2: Deep learning-based cell potency prediction from single-cell RNA-seq data",
    prerequisites={
        'optional_functions': ['preprocess']
    },
    requires={},
    produces={
        'obs': ['CytoTRACE2_Score', 'CytoTRACE2_Potency', 'CytoTRACE2_Relative',
                'preKNN_CytoTRACE2_Score', 'preKNN_CytoTRACE2_Potency']
    },
    auto_fix='none',
    examples=[
        "# Basic CytoTRACE2 analysis",
        "results = ov.single.cytotrace2(adata, use_model_dir='models/5_models_weights')",
        "# Mouse data with custom parameters",
        "results = ov.single.cytotrace2(adata, use_model_dir='models/',",
        "                               species='mouse', batch_size=5000)",
        "# Human data analysis",
        "results = ov.single.cytotrace2(adata, use_model_dir='models/',",
        "                               species='human', max_pcs=100, seed=42)",
        "# Disable parallelization for small datasets",
        "results = ov.single.cytotrace2(adata, use_model_dir='models/',",
        "                               disable_parallelization=True)"
    ],
    related=["single.TrajInfer", "single.pyVIA", "pp.preprocess"]
)
def cytotrace2(
               adata,
               use_model_dir,
               species = "mouse",
               #full_model = False,
               batch_size = 10000,
               smooth_batch_size = 1000,
               disable_parallelization = False,
               max_cores = None,
               max_pcs = 200,
               seed = 14,
               output_dir = 'cytotrace2_results'):
    r"""Predict developmental potency with CytoTRACE2.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression object. Uses ``adata.X`` as the expression matrix.
    use_model_dir : str
        Directory containing CytoTRACE2 pretrained model weights.
    species : str, default='mouse'
        Species label used by gene preprocessing. Supported values are
        typically ``'mouse'`` and ``'human'``.
    batch_size : int, default=10000
        Number of cells per prediction batch.
    smooth_batch_size : int, default=1000
        Number of cells per smoothing batch in diffusion/KNN post-processing.
    disable_parallelization : bool, default=False
        Whether to disable multiprocessing.
    max_cores : int or None, default=None
        Maximum cores used when parallelization is enabled. ``None`` lets the
        function infer an available value.
    max_pcs : int, default=200
        Number of principal components used in KNN smoothing.
    seed : int, default=14
        Random seed used for chunking and stochastic operations.
    output_dir : str, default='cytotrace2_results'
        Directory for intermediate files and final result table.

    Returns
    -------
    pandas.DataFrame
        CytoTRACE2 result table indexed by cell ID, including
        ``CytoTRACE2_Score``, potency category, and relative rank score.

    Notes
    -----
    This function also writes prediction columns into ``adata.obs``.
    """
    from ..external.cytotrace2.gen_utils import preprocess, top_var_genes, predict, smoothing_by_diffusion, binning

    # Make output directory 
    out = os.system('mkdir -p '+output_dir)

    # Load data
    print('cytotrace2: Input parameters')
    #print('    Input file: '+input_path)
    print('    Species: '+species)
    #print('    Full model: '+str(full_model))
    print('    Parallelization enabled: '+str(not disable_parallelization))
    print('    User-provided limit for number of cores to use: '+str(max_cores))
    print('    Batch size: '+str(batch_size))
    print('    Smoothing batch size: '+str(smooth_batch_size))
    print('    Max PCs: '+str(max_pcs))
    print('    Seed: '+str(seed))
    print('    Output directory: '+output_dir)


    #expression =  adata.X.toarray()
    #if sparse.issparse(adata.X):
    from scipy.sparse import issparse
    if issparse(adata.X):
        expression=pd.DataFrame(adata.X.toarray(),index=adata.obs.index,columns=adata.var.index)
    else:
        expression = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
    print('cytotrace2: Dataset characteristics')
    print('    Number of input genes: ',str(expression.shape[1]))
    print('    Number of input cells: ',str(expression.shape[0]))
    
    # Check if the input species is accurate
    # Calculate the proportion of row names that are all uppercase (assumed to be human) or not all uppercase (assumed to be mouse)
    is_human = sum([name.isupper() for name in expression.columns]) / expression.shape[1] > 0.9
    is_mouse = sum([not name.isupper() for name in expression.columns]) / expression.shape[1] > 0.9

    if is_human and species == 'mouse':
        warnings.warn("Species is most likely human. Please revise the 'species' input to the function.")

    if is_mouse and species == 'human':
        warnings.warn("Species is most likely mouse. Please revise the 'species' input to the function.")

    np.random.seed(seed)
    if batch_size > len(expression):
        print("cytotrace2: The passed batch_size is greater than the number of cells in the subsample. \n    Now setting batch_size to "+str(len(expression))+".")
        batch_size = len(expression)
    if batch_size > 10000:
        print(".   Please consider reducing the batch size to 10000 for runtime and memory efficiency.")
    elif len(expression) > 10000 and batch_size > 10000:
        print("cytotrace2: Please consider reducing the batch_size to 10000 for runtime and memory efficiency.")
    
    print('cytotrace2: Preprocessing')
    
    # Calculate chunk number
    chunk_number = math.ceil(len(expression) / batch_size)
    smooth_chunk_number = math.ceil(batch_size/smooth_batch_size)
    if len(expression) < 1000:
        chunk_number = 1
        smooth_chunk_number = 1

    # Determine multiprocessing parameters
    pred_cores_to_use, smooth_cores_to_use = calculate_cores_to_use(chunk_number, smooth_chunk_number, max_cores, disable_parallelization)

    #if full_model == False:
    #    use_model_dir = pkg_resources.resource_filename("omicverse","data_files/5_models_weights/")
    #else:
    #    use_model_dir = pkg_resources.resource_filename("omicverse","data_files/17_models_weights/")
    
    #original_names = adata.var_names
    original_names = adata.obs_names
    subsamples_indices = np.arange(len(expression))
    np.random.shuffle(subsamples_indices)
    subsamples = np.array_split(subsamples_indices, chunk_number)
    
    predictions = []
   
    # Process each chunk separately
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=pred_cores_to_use) as executor:
        for idx in range(chunk_number):
            chunked_expression = expression.iloc[subsamples[idx], :]
            print('cytotrace2: Initiated processing batch ' + str(idx + 1) + '/' + str(chunk_number) + ' with ' + str(chunked_expression.shape[0]) + ' cells')
            results.append(executor.submit(process_subset, idx,
                                        chunked_expression, smooth_batch_size,
                                        smooth_cores_to_use, species,
                                        use_model_dir, output_dir, max_pcs, seed))
        
        for f in concurrent.futures.as_completed(results):
            smooth_by_knn_df = f.result()
            predictions.append(smooth_by_knn_df)
    
    

    for idx in range(chunk_number):
        suffix = '_'+str(idx)
        temp_file_list = [output_dir+'/binned_df'+suffix+'.txt',output_dir+'/ranked_df'+suffix+'.txt',
                                output_dir+'/smoothbykNNresult'+suffix+'.txt',output_dir+'/top_var_genes'+suffix+'.txt']
        for fin in temp_file_list:
            if os.path.isfile(fin):
                os.remove(fin)
    
    predicted_df_final = pd.concat(predictions, ignore_index=False)
    predicted_df_final = predicted_df_final.loc[original_names]
    ranges = np.linspace(0, 1, 7)  
    labels = [
        'Differentiated',
        'Unipotent',
        'Oligopotent',
        'Multipotent',
        'Pluripotent',
        'Totipotent']
    
    predicted_df_final['CytoTRACE2_Potency'] = pd.cut(predicted_df_final['CytoTRACE2_Score'], bins=ranges, labels=labels, include_lowest=True)

    from scipy.stats import rankdata
    ranked_scores = rankdata(predicted_df_final['CytoTRACE2_Score'])
    relative_scores = (ranked_scores-min(ranked_scores))/(max(ranked_scores)-min(ranked_scores))
    predicted_df_final['CytoTRACE2_Relative'] = relative_scores

    adata.obs['CytoTRACE2_Score'] = predicted_df_final['CytoTRACE2_Score']
    adata.obs['CytoTRACE2_Potency'] = predicted_df_final['CytoTRACE2_Potency']
    adata.obs['CytoTRACE2_Relative'] = predicted_df_final['CytoTRACE2_Relative']
    adata.obs['preKNN_CytoTRACE2_Score']=predicted_df_final['preKNN_CytoTRACE2_Score']
    adata.obs['preKNN_CytoTRACE2_Potency']=predicted_df_final['preKNN_CytoTRACE2_Potency']
    print('cytotrace2: Results saved to adata.obs \
          \n    CytoTRACE2_Score: CytoTRACE2 score \
          \n    CytoTRACE2_Potency: CytoTRACE2 potency \
          \n    CytoTRACE2_Relative: CytoTRACE2 relative score \
          \n    preKNN_CytoTRACE2_Score: CytoTRACE2 score before kNN smoothing \
          \n    preKNN_CytoTRACE2_Potency: CytoTRACE2 potency before kNN smoothing')
    #predicted_df_final = predicted_df_final[["CytoTRACE2_Score", "CytoTRACE2_Potency" , "CytoTRACE2_Relative", "preKNN_CytoTRACE2_Score", "preKNN_CytoTRACE2_Potency"]]

    predicted_df_final.to_csv(output_dir+'/cytotrace2_results.txt',sep='\t')

    #print('cytotrace2: Plotting outputs')

    #plot_script = pkg_resources.resource_filename("cytotrace2_py","resources/plot_cytotrace2_results.R")
    
    #if annotation_path != "":
    #    out = subprocess.run(['Rscript', plot_script, '--expression-path', input_path, '--result-path', output_dir+'/cytotrace2_results.txt', '--annotation-path', annotation_path, '--plot-dir', output_dir+'/plots'], check=True)
    #else:
    #    out = subprocess.run(['Rscript', plot_script, '--expression-path', input_path, '--result-path', output_dir+'/cytotrace2_results.txt', '--plot-dir', output_dir+'/plots'], capture_output=True)

    print('cytotrace2: Finished.')
    add_reference(adata,'CytoTRACE2','cell potency prediction with CytoTRACE2')
    return predicted_df_final
