import argparse
import random,os,sys
import numpy as np
import pandas as pd
import argparse
import pickle
import torch
from tqdm import tqdm
import os

import sys 
sys.path.append("../pretrain/") 
from load import *

####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('--ckpt_path', type=str, default=None, help='ckpt path')
parser.add_argument('--ckpt_name', type=str, default=None, help='ckpt path')
parser.add_argument('--highres', type=int, default=0, help='high res')
parser.add_argument('--type', type=str, default='encoder', help='encoder or all')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns

def main():
    random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('../data/processed.pkl', 'rb') as f:
        alldata = pickle.load(f) # 107446 instances across 561 cell lines and 223 drugs were generated.
    mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = alldata['mutation_feature'],alldata['drug_feature'],alldata['gexpr_feature'],alldata['methylation_feature'],alldata['data_idx']
    
    gene_list_df = pd.read_csv('../public_data/single_cell_data/for_AE/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    gexpr_feature, to_fill_columns = main_gene_selection(gexpr_feature,gene_list)
    
    data = gexpr_feature.values
    data = data/data.sum(1,keepdims=True)*10000
    data = np.log1p(data)
    gexpr_feature = pd.DataFrame(data,index=gexpr_feature.index,columns=gexpr_feature.columns)
    np.save('../data/normalized19264.npy',gexpr_feature.values)
    
    print(gexpr_feature.shape)
    pretrainmodel,pretrainconfig = load_model_frommmf(args.ckpt_path,device)
    pretrainmodel.eval()
    geneexpemb=[]
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            if pretrainconfig['rawcount'] == False:
                pretrain_gene_x = torch.tensor(gexpr_feature.iloc[i,:]).unsqueeze(0).cuda()
                data_gene_ids = torch.arange(19264, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
            else:
                totalcount = gexpr_feature.iloc[i,:].sum()
                pretrain_gene_x = torch.tensor(gexpr_feature.iloc[i,:].tolist()+[totalcount+args.highres,totalcount]).unsqueeze(0).cuda()
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])

            if args.type=='encoder':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        pretrainconfig['pad_token_id'])
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x,x_padding)
                
                geneemb, _ = torch.max(geneemb, dim=1)  # B 128
                geneexpemb.append(geneemb.detach().cpu().numpy())
            else:
                print('Not implemented')
                
    geneexpemb = np.squeeze(np.array(geneexpemb))
    print(geneexpemb.shape)
    if args.type=='encoder':
        np.save('../data/{}_embedding.npy'.format(args.ckpt_name),geneexpemb)
    else:
        np.save('../data/{}_embedding.npy'.format(args.ckpt_name+'_'+'all'),geneexpemb)
    

if __name__=='__main__':
    main()
    
