import concurrent.futures
import sys
import os
import math
import numpy as np 
import pandas as pd
import torch
import scipy
import random
import warnings
from sklearn.preprocessing import MinMaxScaler

from .models import *

# functions for one-to-one best match mapping of orthology
def top_hit_human(x):
    r = x.sort_values('%id. target Mouse gene identical to query gene')
    return r.iloc[-1]
def top_hit_mouse(x):
    r = x.sort_values('%id. query gene identical to target Mouse gene')
    return r.iloc[-1]

# data loader
def generic_data_loader(expression, batch_size, train=False):
    
    test_tensor = torch.utils.data.TensorDataset(torch.Tensor(expression))
    data_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return data_loader

# one model predictor
def validation(data_loader, model, device):
    prob_pred_list = []
    order_pred_list = []

    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    order_vector = torch.Tensor(np.arange(6).reshape(6,1)/5).to(device)
    for batch_idx, X in enumerate(data_loader):
        X = X[0].to(device)

        model_output = model(X)
        _, prob_pred = model_output
        prob_pred = prob_pred.squeeze(2)
        prob_pred = softmax(prob_pred)
        prob_order = torch.matmul(prob_pred,order_vector)

        prob_pred_list.append(prob_pred.detach().cpu().numpy())
        order_pred_list.append(prob_order.squeeze(1).detach().cpu().numpy())
        

    return np.concatenate(prob_pred_list,0), np.concatenate(order_pred_list,0)


# dispersion function
def disp_fn(x):
    if len(np.unique(x)) == 1:
        return 0
    else:
        return np.var(x)/np.mean(x)


# choosing top variable genes
def top_var_genes(ranked_data):

    dispersion_index = [disp_fn(ranked_data[:, i]) for i in range(ranked_data.shape[1])]
    top_col_inds = np.argsort(dispersion_index)[-1000:]
                        
    return top_col_inds



def build_mapping_dict():
    import pkg_resources
    fn_mart_export = pkg_resources.resource_filename("omicverse", "data_files/mart_export.txt")
    human_mapping = pd.read_csv(fn_mart_export,sep='\t').dropna().reset_index()
    fn_features = pkg_resources.resource_filename("omicverse", "data_files/features_model_training_17.csv")
    features = pd.read_csv(fn_features)['0']
    mapping_unique = human_mapping.groupby('Gene name').apply(top_hit_human)
    mapping_unique = mapping_unique.groupby('Mouse gene name').apply(top_hit_mouse)
    mt_dict = dict(zip(mapping_unique['Gene name'].values,mapping_unique['Mouse gene name'].values))
    fn_alias_list = pkg_resources.resource_filename("omicverse", "data_files/human_alias_list.txt")
    human_mapping_alias_and_previous_symbols = pd.read_csv(fn_alias_list,sep='\t')
    mt_dict_alias_and_previous_symbols = dict(zip(human_mapping_alias_and_previous_symbols['Alias or Previous Gene name'].values,
                                                  human_mapping_alias_and_previous_symbols['Mouse gene name'].values))
    
    for gene in features[~features.isin(mt_dict.values())].values:
        if gene.upper() in mt_dict.keys():
            print(gene)
        mt_dict[gene.upper()] = gene

    return mt_dict, mt_dict_alias_and_previous_symbols, features


def load(input_path):
    print("cytotrace2: Loading dataset")
    expression = pd.read_csv(input_path,sep='\t',index_col=0).T # read data
    # expression = expression.loc[:,~expression.columns.duplicated()].copy() #drop duplicate gene names, to avoid random information loss make sure the genes are unique
    if expression.columns.duplicated().any():
        raise ValueError("   Please make sure the gene names are unique.")
    if expression.index.duplicated().any():
        raise ValueError("   Please make sure the cell names are unique.")
    return expression


def preprocess(expression, species):
    gene_names = expression.columns 
    mt_dict, mt_dict_alias_and_previous_symbols, features = build_mapping_dict()
    # mapping to orthologs

    if species == "human":
        mapped_genes = gene_names.map(mt_dict)
        mapped_genes = mapped_genes.astype(object)
    
        unmapped_genes = {value: index for index, value in enumerate(gene_names) if value in gene_names[mapped_genes.isna()]}
        mapped_genes.values[mapped_genes.isna()] = gene_names[mapped_genes.isna()].map(mt_dict_alias_and_previous_symbols)
        expression.columns = mapped_genes.values
        num_genes_mapped = len([i for i in mapped_genes if i in features.values])
        print("    Mapped "+str(num_genes_mapped)+" input gene names to mouse orthologs")    
        duplicate_genes = expression.columns[expression.columns.duplicated()].dropna().values
        idx = [unmapped_genes[i.upper()] for i in duplicate_genes if i.upper() in unmapped_genes.keys()]
        expression = expression.iloc[:, [j for j, c in enumerate(expression.columns) if j not in idx]]
        
    else:   
        mapped_genes = gene_names
        expression.columns = mapped_genes
            
    expression = expression[expression.columns[~expression.columns.isna()]]
    
    # check the number of input genes mapped to model features
    intersection = set(expression.columns).intersection(features)
    print("    "+str(len(intersection))+" input genes are present in the model features.")
    if len(intersection) < 9000:
        warnings.warn("    Please verify the input species is correct.\n    In case of a correct species input, be advised that model performance might be compromised due to gene space differences.")
    expression = pd.DataFrame(index=features).join(expression.T).T
    expression = expression.fillna(0)
    cell_names = expression.index
    gene_names = expression.columns

    #print("    Ranking gene expression values within each cell")
    ranked_data = scipy.stats.rankdata(expression.values*-1,axis=1,method='average') # ranking data
    return cell_names, gene_names, ranked_data
    
def predict(ranked_data, cell_names, model_dir, batch_size = 10000):
    
    device = "cpu"
    n_labels = 6
    n_genes = ranked_data.shape[1]
    dropout = 0
    
    all_preds_test = []
    all_order_test = []
    
    all_models_path = pd.Series(np.array([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(model_dir)) for f in fn]))
    all_models_path = all_models_path.astype(str)  # 确保所有元素都是字符串
    all_models_path = all_models_path[all_models_path.str.endswith('.pt')]
    

    data_loader = generic_data_loader(ranked_data, batch_size)
    
    #print('    Started prediction')
    for model_path in all_models_path:
        pytorch_model = torch.load(model_path, map_location=torch.device('cpu'))
    
        hidden_size = pytorch_model['layers.0.weight'].shape[1]
    
        model = BinaryEncoder(num_layers=n_labels, input_size=n_genes, dropout=dropout, hidden_size=hidden_size,num_labels=1)
        model = model.to(device)
        model.load_state_dict(pytorch_model)
    
        prob_test, order_test = validation(data_loader, model, device)
    
        all_preds_test.append(prob_test.reshape(-1,6,1))
        all_order_test.append(order_test.reshape(-1,1))
    
    predicted_order = np.mean(np.concatenate(all_order_test,1),1)
    predicted_potency = np.argmax(np.concatenate(all_preds_test,2).mean(2),1)
    labels = ['Differentiated','Unipotent','Oligopotent','Multipotent','Pluripotent','Totipotent']
    labels_dict = dict(zip([0,1,2,3,4,5],labels))
    predicted_df = pd.DataFrame({'preKNN_CytoTRACE2_Score':predicted_order,'preKNN_CytoTRACE2_Potency':predicted_potency})
    predicted_df['preKNN_CytoTRACE2_Potency'] = predicted_df['preKNN_CytoTRACE2_Potency'].map(labels_dict)
    predicted_df.index = cell_names

    return predicted_df
# CytoTRACE-based smoothing functions
def get_markov_matrix(ranked_data, top_col_inds):
    num_samples, num_genes = ranked_data.shape
    
    sub_mat = ranked_data[:, top_col_inds]

    D = np.corrcoef(sub_mat)  # Pairwise pearson-r corrs

    D[np.arange(num_samples), np.arange(num_samples)] = 0
    D[np.where(D != D)] = 0
    cutoff = max(np.mean(D), 0)
    D[np.where(D < cutoff)] = 0

    A = D / (D.sum(1, keepdims=True) + 1e-5)
    return A


def rescale_fn(adjusted_score, original_score, deg=1, ratio=None):
    params = np.polyfit(adjusted_score, original_score, deg)
    smoothed_score = 0
    for i, p in enumerate(params):
        smoothed_score += p * np.power(adjusted_score, deg - i)

    if not ratio is None:
        # Revert to original SD
        smoothed_range = np.std(smoothed_score)
        original_range = np.std(original_score) * ratio

        smoothed_score = (smoothed_score - smoothed_score.mean()) / (smoothed_range + 1e-9)
        smoothed_score = smoothed_score * original_range + original_score.mean()
    return smoothed_score

def smooth_subset(chunk_ranked_data,chunk_predicted_df,top_col_inds,maxiter):
    markov_mat = get_markov_matrix(chunk_ranked_data, top_col_inds)
    score = chunk_predicted_df["preKNN_CytoTRACE2_Score"]
    init_score = score.copy()
    prev_score = score.copy()
    traj = []

    for _ in range(int(maxiter)):
        cur_score = 0.9 * markov_mat.dot(prev_score) + 0.1 * init_score
        traj.append(np.mean(np.abs(cur_score - prev_score)) / (np.mean(init_score) + 1e-6))
        if np.mean(np.abs(cur_score - prev_score)) / (np.mean(init_score) + 1e-6) < 1e-6:
            break
        prev_score = cur_score

    return cur_score

def smoothing_by_diffusion(predicted_df, ranked_data, top_col_inds, smooth_batch_size=1000, smooth_cores_to_use=1, seed = 14,
                           maxiter=1e4,rescale=True, rescale_deg=1, rescale_ratio=None):
    # Set seed for reproducibility
    np.random.seed(seed)
    if smooth_batch_size > len(ranked_data):
        print("The passed subsample size is greater than the number of cells in the subsample. \n    Now setting subsample size to "+str(len(ranked_data))+". \n    Please consider reducing the smooth_batch_size to a number in range 1000 - 3000\n    for runtime and memory efficiency. ")
        smooth_batch_size <- len(ranked_data)
    elif len(ranked_data) > 1000 and smooth_batch_size > 1000:
        print("Please consider reducing the smooth_batch_size to a number in range 1000 - 3000 for runtime and memory efficiency.")

    #print('    Started smoothing')

    # Calculate chunk number
    chunk_number = math.ceil(len(ranked_data) / smooth_batch_size)

    original_names = predicted_df.index
    subsamples_indices = np.arange(len(ranked_data))
    np.random.shuffle(subsamples_indices)
    subsamples = np.array_split(subsamples_indices, chunk_number)
    
    # Extract sample names for each subsample
    # shuffled_names = [predicted_df.index[subsample] for subsample in subsamples]

    smoothed_scores = []
    smooth_results = []
    # Process each chunk separately
    with concurrent.futures.ProcessPoolExecutor(max_workers=smooth_cores_to_use) as executor:
        for subsample in subsamples:
            chunk_ranked_data = ranked_data[subsample, :]
            chunk_predicted_df = predicted_df.iloc[subsample, :]
            smooth_results.append(executor.submit(smooth_subset,chunk_ranked_data,chunk_predicted_df,top_col_inds,maxiter))
        for f in concurrent.futures.as_completed(smooth_results):
            cur_score = f.result()
            smoothed_scores.append(cur_score)

    # Concatenate the smoothed scores for all chunks
    smoothed_scores_concatenated = pd.concat(smoothed_scores)
   
    return smoothed_scores_concatenated[original_names]






def binning(predicted_df, scores): # scores is smoothed scores
    labels = ['Differentiated',
      'Unipotent',
      'Oligopotent',
      'Multipotent',
      'Pluripotent',
      'Totipotent']
    
    #print('    Started binning')
    pred_potencies = predicted_df["preKNN_CytoTRACE2_Potency"]
    unique_potency = np.unique(pred_potencies)
    score = 'preKNN_CytoTRACE2_Score'
    df_pred_potency = pd.DataFrame({'preKNN_CytoTRACE2_Potency':pred_potencies,'preKNN_CytoTRACE2_Score':scores})
    limits = np.arange(7)/6
    for potency_i, potency in enumerate(labels):
        lower = limits[potency_i]
        upper = limits[potency_i+1]
        if potency in unique_potency:
            data_order =  df_pred_potency[df_pred_potency['preKNN_CytoTRACE2_Potency']==potency]['preKNN_CytoTRACE2_Score'].sort_values()
            index = data_order.index
            n = len(index)
            scaler = MinMaxScaler(feature_range=(lower+1e-8, upper-1e-8))
            order = scaler.fit_transform(np.arange(n).reshape(-1,1))[:,0]
            df_pred_potency.loc[index,score] = order 

    predicted_df["preKNN_CytoTRACE2_Score"] = df_pred_potency[score][predicted_df.index]

            
    return predicted_df
