# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:graph_Prep.py
# @Software:PyCharm
# @Created Time:2023/11/2 5:17 PM
##
import dgl
import numpy as np
import torch as th
from sklearn.decomposition import PCA
import pandas as pd
import pickle


def graph_feat_insert(prep_data_path):
    pca=PCA(n_components=128)
    CL_np=pca.fit_transform(np.load(prep_data_path+'CL_feat.npy'))
    GO_np=pca.fit_transform(np.load(prep_data_path+'GO_feat.npy'))
    gene_np=pca.fit_transform(np.load(prep_data_path+'gene_feat.npy'))

    g=dgl.load_graphs(prep_data_path+'gene_centric_hg.dgl')[0][0]
    g.nodes['CL_term'].data['feat']=torch.tensor(CL_np)[:g.num_nodes('CL_term')]
    g.nodes['GO_term'].data['feat']=torch.tensor(GO_np)[:g.num_nodes('GO_term')]
    g.nodes['gene'].data['feat']=torch.tensor(gene_np)[:g.num_nodes('gene')]


    # dgl.save_graphs(data_path+'gene_centric_hg.wfeat.dgl',g)
    return g
def gene_id_maping(node_path,gene_file,prep_path):
    gene_attr=pd.read_csv(node_path+'/GENE.node.csv')
    graph = dgl.load_graphs(prep_path + 'gene_centric_hg.wfeat.dgl')[0][0]
    gene_feat=gene_attr.groupby('id',as_index=False).agg({'def':'first'})# merge the data based on 'id', aggregate 'def' column by first appearance strategy
    with open(gene_file, 'rb') as gf:
        gene_list = np.array(pickle.load(gf))
        gf.close()
    unvalid_gene_index=np.where(~gene_feat.id.isin(gene_list))[0]#unvalid idx of gene_feat
    unvalid_gene_index=unvalid_gene_index[unvalid_gene_index<=(graph.num_nodes('gene')-1)]# in case of overflow

    valid_gene_index = np.where(gene_feat.id.isin(gene_list))
    valid_gene_name = gene_feat['id'].array[valid_gene_index]# valid gene name of genefeat
    sub_graph = graph.remove_nodes( unvalid_gene_index[0], 'gene')
    return sub_graph,valid_gene_name
if __name__=='__main__':
    preprocessed_path = r'/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/data/public_DB/Preprocessed/'
    raw_node_path = r'/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/data/public_DB/Nodes'
    gene_token_id_file=r'/home/share/huada/home/jiangwenjian/proj/scbert_lmdb/KG/data/gene2vec_names_list.pkl'
    subg,vgn= gene_id_maping(raw_node_path, gene_token_id_file,preprocessed_path)
    print(subg)


# Data path
db='gene_centric'#['universe','gene_centric']


if db=="universe":
    node_attr_files=[DATA_PATH+'/public_DB/Nodes/'+n for n in os.listdir(DATA_PATH+'/Nodes') if n.endswith('node.csv')]
    edge_files=[DATA_PATH+'/public_DB/Edges/'+n for n in os.listdir(DATA_PATH+'/Edges') if n.endswith('edge.csv')]
    # node merge
    nodes_feat=pd.DataFrame(columns=['id', 'name', 'def', 'synonym'])
    for node_file in node_attr_files:
        tmp_node_feat=pd.read_csv(node_file)
        tmp_node_feat=pd.DataFrame(tmp_node_feat,columns=['id', 'name', 'def', 'synonym'])
        nodes_feat=pd.concat([nodes_feat,tmp_node_feat],ignore_index=True)

    # edge merge
    edges=pd.DataFrame(columns=['source', 'relation', 'target', 'condition'])
    for edge_file in edge_files:
        tmp_edge=pd.read_csv(edge_file)
        tmp_edge=pd.DataFrame(tmp_edge,columns=['source', 'relation', 'target', 'condition'])
        edges=pd.concat([edges,tmp_edge],ignore_index=True)
elif db=="gene_centric":
    gene_attr=pd.read_csv(DATA_PATH+'/public_DB/Nodes/GENE.node.csv')
    gene_attr=pd.DataFrame(gene_attr,columns=['id', 'name', 'def', 'synonym'])



    CL_attr=pd.read_csv(DATA_PATH+'/public_DB/Nodes/CL.node.csv')
    CL_attr = pd.DataFrame(CL_attr, columns=['id', 'name', 'def', 'synonym'])
    GO_attr=pd.read_csv(DATA_PATH+'/public_DB/Nodes/GeneOntology.node.csv')
    GO_attr = pd.DataFrame(GO_attr, columns=['id', 'name', 'def', 'synonym'])
    UBERON_attr=pd.read_csv(DATA_PATH+'/public_DB/Nodes/OG.node.csv')
    UBERON_attr = pd.DataFrame(UBERON_attr, columns=['id', 'name', 'def', 'synonym'])

    edge_gene2CL=pd.read_csv(DATA_PATH+'/public_DB/Edges/CellMarker.edge.csv')
    edge_gene2CL=pd.DataFrame(edge_gene2CL, columns=['source', 'relation', 'target', 'condition'])
    edge_gene2GO = pd.read_csv(DATA_PATH + '/public_DB/Edges/GOA.edge.csv')
    edge_gene2GO = pd.DataFrame(edge_gene2GO, columns=['source', 'relation', 'target', 'condition'])


##
# node to idx
# node to idx
gene_feat=gene_attr.groupby('id',as_index=False).agg({'def':'first'})# merge the data based on 'id', aggregate 'def' column by first appearance strategy
gene2gid={gene:i for i,gene in enumerate(gene_feat['id'])}

CL_feat=CL_attr.groupby('id',as_index=False).agg({'def':'first'})
CL2gid={cl:i for i,cl in enumerate(CL_feat['id'])}

GO_feat=GO_attr.groupby('id',as_index=False).agg({'def':'first'})
GO2gid={go:i for i,go in enumerate(GO_feat['id'])}

UBERON_feat=UBERON_attr.groupby('id',as_index=False).agg({'def':'first'})
UBERON2gid={og:i for i,og in enumerate(UBERON_feat['id'])}

##
# edge construction
edges={}
node_feat={}
edgetype_gene2CL=edge_gene2CL.relation.unique()

for edgetype in edgetype_gene2CL:
    cur_edge=edge_gene2CL.loc[edge_gene2CL['relation']==edgetype]
    src_gene2CL=list(map(gene2gid.get,cur_edge['source']))
    dst_gene2CL=list(map(CL2gid.get,cur_edge['target']))
    src_valid=[src for src,dst in zip(src_gene2CL,dst_gene2CL) if (src and dst) is not None]
    dst_valid = [dst for src, dst in zip(src_gene2CL, dst_gene2CL) if (src and dst) is not None]
    con_gene2CL=list(map(UBERON2gid.get,cur_edge['condition']))
    if src_valid.__len__() != 0:
        edges[('gene',edgetype,'CL_term')]=(th.tensor(src_valid),th.tensor(dst_valid))

edgetype_gene2GO=edge_gene2GO.relation.unique()
for edgetype in edgetype_gene2GO:
    cur_edge = edge_gene2GO.loc[edge_gene2GO['relation'] == edgetype]
    src_gene2GO=list(map(gene2gid.get,cur_edge['source']))
    dst_gene2GO=list(map(GO2gid.get,cur_edge['target']))
    # con_gene2GO = list(map(UBERON2gid.get, cur_edge['condition']))
    src_valid=[src for src,dst in zip(src_gene2GO,dst_gene2GO) if (src and dst) is not None]
    dst_valid = [dst for src, dst in zip(src_gene2GO, dst_gene2GO) if (src and dst) is not None]
    if src_valid.__len__()!=0:
        edges[('gene', edgetype, 'GO_term')] = (th.tensor(src_valid), th.tensor(dst_valid))
graph=dgl.heterograph(edges)

##
# node feature
# gene emb
gene_batch=5000

gene_emb=[]
model.eval()
for i in tqdm(range(gene_feat.__len__()//gene_batch+1)):
    with th.no_grad():
        encoded_gene_sent = tokenizer(gene_sent[i*gene_batch:(i+1)*gene_batch], padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        tmp_emb=model(**encoded_gene_sent).pooler_output.detach().cpu().tolist()
        gene_emb+=tmp_emb
gene_feat['emb']=gene_emb
print(gene_feat.describe())
gene_feat.to_csv('gene_feat.csv',index=None)

# CL_emb
CL_batch=5000
CL_sent=[sent.split('<loc>')[0] for sent in CL_feat['def']]
CL_emb=[]
model.eval()
for i in tqdm(range(CL_feat.__len__()//CL_batch+1)):
    with th.no_grad():
        encoded_CL_sent = tokenizer(CL_sent[i*CL_batch:(i+1)*CL_batch], padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        tmp_emb=model(**encoded_CL_sent).pooler_output.detach().cpu().tolist()
        CL_emb+=tmp_emb
CL_feat['emb']=CL_emb
print(CL_feat.describe())
CL_feat.to_csv('CL_feat.csv',index=None)

# GO_emb
GO_batch=1000
GO_sent=[sent.split('<loc>')[0] for sent in GO_feat['def']]
GO_emb=[]
model.eval()
for i in tqdm(range(GO_feat.__len__()//GO_batch+1)):
    with th.no_grad():
        encoded_GO_sent = tokenizer(GO_sent[i*GO_batch:(i+1)*GO_batch], padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        #print(encoded_GO_sent.input_ids.size())
        tmp_emb=model(**encoded_GO_sent).pooler_output.detach().cpu().tolist()
        GO_emb+=tmp_emb
GO_feat['emb']=GO_emb
print(GO_feat.describe())
GO_feat.to_csv('GO_feat.csv',index=None)

# UBERON_emb
UBERON_batch=1000
UBERON_sent=[sent.split('<loc>')[0] for sent in UBERON_feat['def']]
UBERON_emb=[]
model.eval()
for i in tqdm(range(UBERON_feat.__len__()//UBERON_batch+1)):
    with th.no_grad():
        encoded_UBERON_sent = tokenizer(UBERON_sent[i*UBERON_batch:(i+1)*UBERON_batch], padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        #print(encoded_GO_sent.input_ids.size())
        tmp_emb=model(**encoded_UBERON_sent).pooler_output.detach().cpu().tolist()
        UBERON_emb+=tmp_emb
UBERON_feat['emb']=UBERON_emb
print(UBERON_feat.describe())
UBERON_feat.to_csv('UBERON_feat.csv',index=None)






