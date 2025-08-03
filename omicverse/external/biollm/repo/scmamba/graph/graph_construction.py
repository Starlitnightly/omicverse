# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:graph_construction.py
# @Software:PyCharm
# @Created Time:2023/10/26 2:27 PM
import json

##
import pandas as pd
import dgl
import os

import torch
import torch as th
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.decomposition import PCA

class BioLinkBert():
    def __init__(self,feat_dim=128,model_path='/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/BioSimCSE-BioLinkBERT-BASE'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.pca=PCA(n_components=feat_dim)

    def tokenize(self,sentences):
        self.model.eval()
        feat = []
        if not isinstance(sentences,list):
            sentences=sentences.array.tolist()
        sentences_len=np.mean([sentences[0].split(' ').__len__(),sentences[-1].split(' ').__len__()])
        batchsize=20000//int(sentences_len)
        for i in tqdm(range(sentences.__len__() // batchsize + 1)):
            with th.no_grad():
                encoded_sent = self.tokenizer(sentences[i * batchsize:(i + 1) * batchsize], padding=True,
                                                truncation=True, return_tensors='pt').to(DEVICE)
                tmp_emb = self.model(**encoded_sent).pooler_output.detach().cpu().tolist()
                feat += tmp_emb
        feat=np.array(feat)
        feat_out=self.pca.fit_transform(feat)
        return feat_out

def load_nodes(path,columns=['id', 'name', 'def', 'synonym'],node_type='CL',valid_node_list=None):
    node_info = pd.read_csv(path)
    node_info = pd.DataFrame(node_info, columns=columns)
    node_info = node_info.groupby('id', as_index=False).agg(
        {'def': 'first','name':'first'})  # merge the data based on 'id', aggregate 'def' column by first appearance strategy
    if node_type=="GENE":
        valid_idx = np.where(node_info.name.isin(valid_node_list))[0]
        valid_name = node_info.name[valid_idx]
        valid_feat=node_info['def'][valid_idx]
        valid_db_id=node_info.id[valid_idx]
        valid_feat=[sent.split('<loc>')[0] for sent in valid_feat]
    else:
        valid_db_id=node_info['id']
        valid_feat=node_info['def']
        valid_name=node_info['name']
    return valid_db_id,valid_name,valid_feat

def Entity_Feat_Generatation(tokenizer,node_files):
    for node_file in node_files:
        node_file_name=node_file.split('.')[0]
        if not os.path.exists(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/{node_file_name}_feat.node.npy'):
            print(f'Processing {node_file}...')
            idx,name,feat=load_nodes(path=DATA_PATH+f'/{DATASET_NAME}/Nodes/'+node_file,node_type=node_file_name,valid_node_list=VALID_GENE)
            feat=tokenizer.tokenize(feat)
            np.save(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/{node_file_name}_feat.node.npy', feat)
            np.save(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/{node_file_name}_name.node.npy', name)
            np.save(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/{node_file_name}_idx.node.npy', idx)

def Database2Nodeid():
    # ID of database, GENE first
    db_id_files = [f for f in os.listdir(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/') if f.endswith('_idx.node.npy')]
    GENE_name = np.load(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/GENE_name.node.npy', allow_pickle=True)
    GENE_idx=np.load(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/GENE_idx.node.npy', allow_pickle=True)
    # the idx of gene_name in the VALID_GENE, can be used as node_id directly.
    gene_node_id = np.array([np.argwhere(VALID_GENE == gene)[0][0] for gene in GENE_name])

    total_db_id = np.array([])
    total_feat = np.load(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/' + 'GENE_feat.node.npy', allow_pickle=True)  # gene first
    for db_id_file in db_id_files:
        if db_id_file == 'GENE_idx.node.npy':
            continue
        tmp_db_id = np.load(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/' + db_id_file, allow_pickle=True)
        tmp_name = db_id_file.split('_')[0]
        tmp_feat = np.load(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/' + tmp_name + '_feat.node.npy', allow_pickle=True)
        total_db_id = np.concatenate((total_db_id, tmp_db_id))
        total_feat = np.concatenate((total_feat, tmp_feat))
    # total_feat:[num_nodes,FEAT_DIM], total_db_id[num_nodes-num_gene,]

    # idx of node, same order corresponding to db_id
    gene2nid = {db: nid for db, nid in zip(GENE_idx, gene_node_id)}  # gene name is the index in edge.csv

    total_map = {db: nid + VALID_GENE.__len__() for nid, db in enumerate(total_db_id)}  # map dict:{NCBI0001: node_id}
    total_map = {**gene2nid, **total_map}

    if not os.path.exists(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/' + 'total_nid_map.npy'):
        np.save(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/total_nid_map.npy', total_map)
    if not os.path.exists(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/total_feat.npy'):
        np.save(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/total_feat.npy', total_feat)

    return total_map,total_feat,gene_node_id

def Graph_Construction(edge_files,total_map):

    # Edge construction
    total_edges={}
    edge_feats_id={}
    for edge_file in edge_files:
        edges=pd.read_csv(DATA_PATH+f'/{DATASET_NAME}/Edges/'+edge_file)
        edges=pd.DataFrame(edges, columns=['source', 'relation', 'target', 'condition'])

        # remove the edge with invalid scr/dst
        edges=edges.loc[[s&t for s,t in zip(edges['source'].isin(total_map),edges['target'].isin(total_map))]]

        edge_types=edges.relation.unique()
        for cur_edge_type in edge_types: # enumerate all the edge type
            cur_edge=edges.loc[edges['relation']==cur_edge_type]
            src = list(map(total_map.get, cur_edge['source']))
            dst = list(map(total_map.get, cur_edge['target']))
            con = list(map(total_map.get, cur_edge['condition'],[-1]*src.__len__()))
            if ('N', cur_edge_type, 'N') not in total_edges.keys():
                total_edges[('N', cur_edge_type, 'N')] = (th.tensor(src), th.tensor(dst))
                edge_feats_id[('N', cur_edge_type, 'N')]=th.tensor(con)
            else:
                total_edges[('N', cur_edge_type, 'N')]=tuple(
                    (th.cat([total_edges[('N', cur_edge_type, 'N')][0], th.tensor(src)]),
                    th.cat([total_edges[('N', cur_edge_type, 'N')][1], th.tensor(dst)]))
                )

                edge_feats_id[('N', cur_edge_type, 'N')] = th.cat([edge_feats_id[('N', cur_edge_type, 'N')],th.tensor(con)])
                assert edge_feats_id[('N', cur_edge_type, 'N')].size(0)==total_edges[('N', cur_edge_type, 'N')][0].size(0)
    # Graph construction
    node_num=max(total_map.values())+1
    graph=dgl.heterograph(total_edges,num_nodes_dict={'N':node_num})
    return graph,edge_feats_id

def Graph_feature_construction(graph,edge_feats_id,gene_node_id):
    node_feat=torch.tensor(np.load(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/total_feat.npy',allow_pickle=True)).float()

    # edge feature
    for etype in graph.canonical_etypes:
        tmp_efeat=torch.zeros((edge_feats_id[etype].size(0),FEAT_DIM))
        flag=[edge_feats_id[etype]!=-1]
        tmp_efeat[flag]=node_feat[edge_feats_id[etype][flag]]
        graph.edges[etype].data['feat']=tmp_efeat

    # node feature
    gene_db_feat=node_feat[:gene_node_id.__len__()]
    gene_node_feat=torch.zeros((VALID_GENE.__len__(),FEAT_DIM))
    gene_node_feat[gene_node_id]=gene_db_feat
    final_node_feat=torch.cat([gene_node_feat,node_feat[gene_node_id.__len__():]],dim=0)
    graph.nodes['N'].data['feat']=final_node_feat[:graph.num_nodes('N')]
    return graph








if __name__=='__main__':
    DEVICE = "cuda:2" if th.cuda.is_available() else "cpu"
    DATA_PATH = '/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/data'
    BERT_PATH = '/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/BioSimCSE-BioLinkBERT-BASE'
    DATASET_NAME='Public_DB_1115'
    FEAT_DIM=128

    if not os.path.exists(DATA_PATH + f'/{DATASET_NAME}/Preprocessed'):
        os.mkdir(DATA_PATH + f'/{DATASET_NAME}/Preprocessed')
    print(f"current device: {DEVICE}")

    with open(DATA_PATH + "/gene2vec_names_list.pkl", 'rb') as gf:
        VALID_GENE = np.array(pickle.load(gf))
        gf.close()

    # ent feature generation
    tokenizer=BioLinkBert(feat_dim=FEAT_DIM)
    node_files=[f for f in os.listdir(DATA_PATH+f'/{DATASET_NAME}/Nodes/') if f.endswith('node.csv')]
    edge_files=[f for f in os.listdir(DATA_PATH+f'/{DATASET_NAME}/Edges/') if f.endswith('edge.csv')]
    # node_files=['CL.node.csv','GeneOntology.node.csv', 'OG.node.csv', 'GENE.node.csv']
    # edge_files = ['CellMarker.edge.csv', 'GOA.edge.csv', 'CL.edge.csv', 'OG.edge.csv', 'GeneOntology.edge.csv']

    Entity_Feat_Generatation(tokenizer,node_files)

    # ID of database, GENE first
    total_map,total_feat,gene_node_id=Database2Nodeid()

    # Graph construction
    g,edge_feats_id=Graph_Construction(edge_files,total_map=total_map)

    # graph feature
    g=Graph_feature_construction(graph=g,edge_feats_id=edge_feats_id,gene_node_id=gene_node_id)

    dgl.save_graphs(DATA_PATH + f'/{DATASET_NAME}/Preprocessed/{DATASET_NAME}_graph.wfeat.dgl',g)























