import concurrent.futures
import math
import numpy as np
import os
import pandas as pd
import scanpy as sc
import subprocess
import torch
import warnings

from sklearn.model_selection import train_test_split

from ..externel.cytotrace2.gen_utils import *
#from cytotrace2_py.common.argument_parser import *


def process_subset(idx, chunked_expression, smooth_batch_size, smooth_cores_to_use, 
                   species, use_model_dir, output_dir, max_pcs, seed):

    # map and rank
    cell_names, gene_names, ranked_data = preprocess(chunked_expression, species)

    # top variable genes
    top_col_inds = top_var_genes(ranked_data)
    top_col_names = gene_names[top_col_inds]
    
    # predict by unrandomized chunked batches
    predicted_df = predict(ranked_data, cell_names, use_model_dir , chunked_expression.shape[0])
	
    smooth_score = smoothing_by_diffusion(predicted_df, ranked_data, top_col_inds, smooth_batch_size,  seed)
    
    binned_score_pred_df = binning(predicted_df, smooth_score)

    # Transpose the matrix and create a DataFrame
    ranked_df = pd.DataFrame(ranked_data.T,  columns = cell_names)
	
    # Set the column names
    ranked_df.index = gene_names
    suffix = '_'+str(idx)
    binned_score_pred_df.to_csv(output_dir+'/binned_df'+suffix+'.txt',sep='\t', index=True)
    ranked_df.to_csv(output_dir+'/ranked_df'+suffix+'.txt', sep='\t', index=True)
	
    with open(output_dir+'/top_var_genes'+suffix+'.txt', 'w') as f:
        for item in top_col_names:
            f.write("%s\n" % item)

    if chunked_expression.shape[0] < 100:
        print('cytotrace2: Fewer than 100 cells in dataset. Skipping KNN smoothing step.')
        smooth_by_knn_df = binned_score_pred_df.copy()
    else:
        #run_script = pkg_resources.resource_filename("cytotrace2_py","resources/smoothDatakNN.R")
        #knn_path = output_dir+'/smoothbykNNresult'+suffix+'.txt'
        #out = subprocess.run(['Rscript', run_script, '--output-dir', output_dir,
                               #'--suffix', suffix, '--max-pcs', str(max_pcs), '--seed', str(seed)], check=True)
        #smooth_by_knn_df = pd.read_csv(knn_path, index_col = 0, sep='\t')
        from ..externel.cytotrace2.smoothDatakNN import smooth_data_kNN
        smooth_data_kNN(output_dir=output_dir, suffix=suffix, max_pcs=max_pcs, seed=seed)
        # 读取平滑处理后的结果
        knn_path = os.path.join(output_dir, f'smoothbykNNresult{suffix}.txt')
        smooth_by_knn_df = pd.read_csv(knn_path, index_col=0, sep='\t')

    return smooth_by_knn_df

def calculate_cores_to_use(chunk_number,smooth_chunk_number,max_cores,disable_parallelization):

    pred_cores_to_use = 1
    smooth_cores_to_use = 1
    if smooth_chunk_number == 1 and chunk_number == 1:
        print("cytotrace2: The number of cells in your dataset is less than the specified batch size.\n")
        print("    Model prediction will not be parallelized.")

    elif not disable_parallelization:
        # Calculate number of available processors
        num_proc = os.cpu_count()
        print("cytotrace2: "+str(num_proc)+" cores detected")
        if num_proc == 1:
            print("cytotrace2: Only one core detected. CytoTRACE 2 will not be run in parallel.")
        elif max_cores == None:
            pred_cores_to_use = min(chunk_number,num_proc-1)
            smooth_cores_to_use = min(smooth_chunk_number,max(math.floor((num_proc-1)/pred_cores_to_use),1))
            print('cytotrace2: Running '+str(pred_cores_to_use)+' prediction batch(es) in parallel using '+str(smooth_cores_to_use)+' cores for smoothing per batch.')
        else:
            max_cores = min(max_cores,num_proc-1)
            pred_cores_to_use = min(chunk_number,max_cores)
            smooth_cores_to_use = min(smooth_chunk_number,max(math.floor(max_cores/pred_cores_to_use),1))
            print('cytotrace2: Running '+str(pred_cores_to_use)+' prediction batch(es) in parallel using '+str(smooth_cores_to_use)+' cores for smoothing per batch.')

    return pred_cores_to_use, smooth_cores_to_use


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
    """
    CytoTRACE 2: A deep learning-based tool for cell potency prediction

    Arguments:
        adata: AnnData object containing the scRNA-seq data.
        use_model_dir: Path to the directory containing the pre-trained model files.
        species: The species of the input data. Default is "mouse".
        batch_size: The number of cells to process in each batch. Default is 10000.
        smooth_batch_size: The number of cells to process in each batch for smoothing. Default is 1000.
        disable_parallelization: If True, disable parallel processing. Default is False.
        max_cores: Maximum number of CPU cores to use for parallel processing. If None, all available cores will be used. Default is None.
        max_pcs: Maximum number of principal components to use. Default is 200.
        seed: Random seed for reproducibility. Default is 14.
        output_dir: Directory to save the results. Default is 'cytotrace2_results'.

    
    """

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
        expression=adata.X
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
        batch_size <- len(expression)
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

    ranked_scores = scipy.stats.rankdata(predicted_df_final['CytoTRACE2_Score'])
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
    return predicted_df_final
