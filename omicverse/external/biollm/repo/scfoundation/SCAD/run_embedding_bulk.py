import argparse
import random,os,sys
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import os
import scanpy as sc
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
parser.add_argument('--data_path', type=str, default='encoder', help='encoder or all')
parser.add_argument('--scdata', type=str, default='./data/original/scrna_ccle_jhu006_exprs_19264.csv', help='encoder or all')
parser.add_argument('--hvg', type=int, default=0, help='hvg or not')
parser.add_argument('--hvgfile', type=str, default='./data/split_norm/Gefitinib_DEG.npy', help='hvg or not')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


# 1. CPM (count per million) for scrna_ccle_jhu006_exprs.tsv and scrna_ccle_scc47_expr.tsv, the original CPM expression count matrix can be downloaded from "https://singlecell.broadinstitute.org/single_cell/study/SCP542/pan-cancer-cell-line-heterogeneity#study-download"
# 2. Normalize counts per cell (target_sum = 1e4) for GSE108383_exprs.PLX4720_A375.tsv, GSE108383_exprs_PLX4720_451Lu.tsv, and GSE149215_exprs.Etoposide_PC9.tsv
# 3. column names are "NCBI Gene ID", which can map to gene "Approved symbol" via file SCAD/data/processing/HGNC_symbol_all_genes.tsv


def main():

    random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_r=pd.read_csv(args.data_path,index_col=0)
    gexpr_feature = data_r
    
    adata = sc.AnnData(gexpr_feature)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    gexpr_feature = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names)
    
    
    pretrainmodel,pretrainconfig = load_model_frommmf(args.ckpt_path,device)
    pretrainmodel.eval()
    geneexpemb=[]
    
    strname = args.data_path[:-4] +'_'+ args.ckpt_name
    if args.type !='encoder':
        strname = strname+'_'+args.type
    if args.highres != 0:
        strname = strname+'_tgthighres'+str(args.tgthighres)
        
    if args.hvg==1:
        hvgmask = torch.tensor(np.load(args.hvgfile)).cuda()
        strname = strname+'_hvg'

    print('save at {}_embedding.npy'.format(strname))
    
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            if pretrainconfig['rawcount'] == False:
                tmpdata = gexpr_feature.iloc[i,:].tolist()
                pretrain_gene_x = torch.tensor(tmpdata).unsqueeze(0).cuda()
                data_gene_ids = torch.arange(19264, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
            else:
                totalcount = np.log10(gexpr_feature.iloc[i,:].sum())
                tmpdata = gexpr_feature.iloc[i,:].tolist()
                pretrain_gene_x = torch.tensor(tmpdata+[totalcount,totalcount]).unsqueeze(0).cuda()
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
                geneemb1 = geneemb[:,-1,:]
                geneemb2 = geneemb[:,-2,:]
                geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                
                geneexpemb.append(geneembmerge.detach().cpu().numpy())
                
            else:
                pretrainmodel.to_final = None
                value_labels = pretrain_gene_x > 0
                mask_labels = pretrain_gene_x < -10000
                x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])
                encoder_position_gene_ids, _ = gatherData(data_gene_ids, value_labels,pretrainconfig['pad_token_id'])
                
                geneemb = pretrainmodel(x=x.float(), padding_label=x_padding, encoder_position_gene_ids=encoder_position_gene_ids, 
                                        encoder_labels=value_labels, decoder_data=pretrain_gene_x.float(),
                mask_gene_name=False, mask_labels=mask_labels, decoder_position_gene_ids=data_gene_ids, decoder_data_padding_labels=mask_labels)
                
                geneemb1 = geneemb[:,-1,:]
                geneemb2 = geneemb[:,-2,:]
                geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                geneexpemb.append(geneembmerge.detach().cpu().numpy())
    geneexpemb = np.squeeze(np.array(geneexpemb))
    print(geneexpemb.shape)
    np.save('{}_embedding.npy'.format(strname),geneexpemb)
    

if __name__=='__main__':
    main()
    
