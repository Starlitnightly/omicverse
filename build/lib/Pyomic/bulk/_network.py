import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import networkx as nx

def string_interaction(gene,species):
    r"""A Python library to analysis the protein-protein interaction network by string-db

    Parameters
    ----------
    gene
        The gene list to analysis PPI
    species
        NCBI taxon identifiers (e.g. Human is 9606, see: STRING organisms).
    
    Returns
    -------
    res
        the dataframe of protein-protein interaction
    
    """
    import requests ## python -m pip install requests
    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"
    request_url = "/".join([string_api_url, output_format, method])
    my_genes = gene

    params = {

        "identifiers" : "%0d".join(my_genes), # your protein
        "species" : species, # species NCBI identifier 
        "caller_identity" : "www.awesome_app.org" # your app name

    }
    response = requests.post(request_url, data=params)
    res=pd.DataFrame(columns=['stringId_A',\
                             'stringId_B',\
                             'preferredName_A',\
                             'preferredName_B',\
                             'ncbiTaxonId',\
                             'score',\
                             'nscore',\
                             'fscore',\
                             'pscore',\
                             'ascore',\
                             'escore',\
                             'dscore',\
                             'tscore'])
    num=0
    for line in response.text.strip().split("\n"):
        l = line.strip().split("\t")
        res.loc[num]={'stringId_A':l[0],\
                             'stringId_B':l[1],\
                             'preferredName_A':l[2],\
                             'preferredName_B':l[3],\
                             'ncbiTaxonId':l[4],\
                             'score':l[5],\
                             'nscore':l[6],\
                             'fscore':l[7],\
                             'pscore':l[8],\
                             'ascore':l[9],\
                             'escore':l[10],\
                             'dscore':l[11],\
                             'tscore':l[12]}
        num+=1
    return res



def string_map(gene,species):
    r"""A Python library to find the gene name in string-db

    Parameters
    ----------
    gene
        The gene list to analysis PPI
    species
        NCBI taxon identifiers (e.g. Human is 9606, see: STRING organisms).
    
    Returns
    -------
    res
        the dataframe of query gene and new gene
    
    """
    import requests ## python -m pip install requests
    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"
    params = {

        "identifiers" : "\r".join(gene), # your protein list
        "species" : species, # species NCBI identifier 
        "limit" : 1, # only one (best) identifier per input protein
        "echo_query" : 1, # see your input identifiers in the output
        "caller_identity" : "www.awesome_app.org" # your app name

    }
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    res=pd.DataFrame(columns=['queryItem','queryIndex','stringId','ncbiTaxonId','taxonName','preferredName','annotation'])
    num=0
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        res.loc[num]={'queryItem':l[0],\
          'queryIndex':l[1],\
          'stringId':l[2],\
          'ncbiTaxonId':l[3],\
          'taxonName':l[4],\
          'preferredName':l[5],\
          'annotation':l[6]}
        num+=1
    return res


def max_interaction(gene,species):
    gene_len=len(gene)
    times=gene_len//1000
    shengyu=gene_len-times*1000
    if shengyu!=0:
        times+=1
    ge=[]
    for i in range(times):
        ge.append(gene[1000*i:1000*(i+1)])
    b=[]
    for p in itertools.combinations(range(times),2):
        b.append(string_interaction(ge[p[0]]+ge[p[1]],species))
    res=pd.concat(b,axis=0,ignore_index=True)
    res=res.drop_duplicates()
    return res
    

def generate_G(gene,species,score=0.4):
    r"""A Python library to get the PPI network in string-db

    Parameters
    ----------
    gene
        The gene list to analysis PPI
    species
        NCBI taxon identifiers (e.g. Human is 9606, see: STRING organisms).
    score
        The threshold of protein A and B interaction
    
    Returns
    -------
    G
        the networkx object of PPI in query gene list
    
    """
    import itertools
    a=string_interaction(gene,species)
    b=a.drop_duplicates()
    b.head()
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(set(b['preferredName_A'].tolist()+b['preferredName_B'].tolist()))

    #Connect nodes
    for i in b.index:
        col_label = b.loc[i]['preferredName_A']
        row_label = b.loc[i]['preferredName_B']
        if(float(b.loc[i]['score'])>score):
            G.add_edge(col_label,row_label)
    return G