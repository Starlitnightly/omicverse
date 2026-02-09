import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from ..utils import plot_network
from .._registry import register_function
from typing import Union, Tuple, Any
import matplotlib

@register_function(
    aliases=["STRING交互分析", "string_interaction", "ppi_analysis", "蛋白质交互分析"],
    category="bulk",
    description="Analyze protein-protein interaction network using STRING database",
    examples=[
        "# Get PPI network for gene list",
        "interactions = ov.bulk.string_interaction(gene_list, species=9606)",
        "# Species codes: Human=9606, Mouse=10090, Yeast=4932",
        "# Returns DataFrame with interaction scores"
    ],
    related=["bulk.pyPPI", "bulk.string_map", "utils.plot_network"]
)
def string_interaction(gene:list,species:int) -> pd.DataFrame:
    r"""Analyze protein-protein interaction network using STRING database.
  
    Arguments:
        gene: List of gene names for PPI analysis.
        species: NCBI taxon identifiers (e.g. Human is 9606, see STRING organisms).
    
    Returns:
        res: DataFrame containing protein-protein interaction data with columns stringId_A, stringId_B, preferredName_A, preferredName_B, ncbiTaxonId, score, nscore, fscore, pscore, ascore, escore, dscore, tscore.
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
    res=pd.DataFrame()
    res['stringId_A']=[j.strip().split("\t")[0] for j in response.text.strip().split("\n")]
    res['stringId_B']=[j.strip().split("\t")[1] for j in response.text.strip().split("\n")]
    res['preferredName_A']=[j.strip().split("\t")[2] for j in response.text.strip().split("\n")]
    res['preferredName_B']=[j.strip().split("\t")[3] for j in response.text.strip().split("\n")]
    res['ncbiTaxonId']=[j.strip().split("\t")[4] for j in response.text.strip().split("\n")]
    res['score']=[j.strip().split("\t")[5] for j in response.text.strip().split("\n")]
    res['nscore']=[j.strip().split("\t")[6] for j in response.text.strip().split("\n")]
    res['fscore']=[j.strip().split("\t")[7] for j in response.text.strip().split("\n")]
    res['pscore']=[j.strip().split("\t")[8] for j in response.text.strip().split("\n")]
    res['ascore']=[j.strip().split("\t")[9] for j in response.text.strip().split("\n")]
    res['escore']=[j.strip().split("\t")[10] for j in response.text.strip().split("\n")]
    res['dscore']=[j.strip().split("\t")[11] for j in response.text.strip().split("\n")]
    res['tscore']=[j.strip().split("\t")[12] for j in response.text.strip().split("\n")]
    return res



def string_map(gene:list,species:int)->pd.DataFrame:
    r"""Map gene names to STRING database identifiers.

    Arguments:
        gene: List of gene names for PPI analysis
        species: NCBI taxon identifiers (e.g. Human is 9606, see STRING organisms)
    
    Returns:
        res: DataFrame containing gene mapping information with columns:
             queryItem, queryIndex, stringId, ncbiTaxonId, taxonName,
             preferredName, annotation
    
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
    response = requests.post(request_url, data=params)
    res=pd.DataFrame(columns=['queryItem','queryIndex','stringId','ncbiTaxonId','taxonName','preferredName','annotation'])
    res['queryItem']=[j.strip().split("\t")[0] for j in response.text.strip().split("\n")]
    res['queryIndex']=[j.strip().split("\t")[1] for j in response.text.strip().split("\n")]
    res['stringId']=[j.strip().split("\t")[2] for j in response.text.strip().split("\n")]
    res['ncbiTaxonId']=[j.strip().split("\t")[3] for j in response.text.strip().split("\n")]
    res['taxonName']=[j.strip().split("\t")[4] for j in response.text.strip().split("\n")]
    res['preferredName']=[j.strip().split("\t")[5] for j in response.text.strip().split("\n")]
    res['annotation']=[j.strip().split("\t")[6] for j in response.text.strip().split("\n")]

    return res


def max_interaction(gene:list,species:int)->pd.DataFrame:
    r"""Handle large gene lists by chunking for STRING database interaction analysis.

    Arguments:
        gene: List of gene names for PPI analysis
        species: NCBI taxon identifiers (e.g. Human is 9606, see STRING organisms)
    
    Returns:
        res: DataFrame containing protein-protein interaction data
    
    """
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
    

def generate_G(gene:list,species:int,score:float=0.4) -> nx.Graph:
    r"""Generate protein-protein interaction network from STRING database.

    Arguments:
        gene: List of gene names for PPI analysis
        species: NCBI taxon identifiers (e.g. Human is 9606, see STRING organisms)
        score: Threshold for protein interaction confidence (default: 0.4)

    Returns:
        G: NetworkX Graph object containing PPI network
    
    """
    
    a=string_interaction(gene,species)
    b=a.drop_duplicates()
    b.head()
    G = nx.Graph()
    G.add_nodes_from(set(b['preferredName_A'].tolist()+b['preferredName_B'].tolist()))

    #Connect nodes
    for i in b.index:
        col_label = b.loc[i]['preferredName_A']
        row_label = b.loc[i]['preferredName_B']
        if(float(b.loc[i]['score'])>score):
            G.add_edge(col_label,row_label)
    return G

@register_function(
    aliases=["PPI网络分析", "pyPPI", "protein_interaction", "蛋白质互作网络"],
    category="bulk",
    description="Protein-protein interaction network analysis and visualization using STRING database",
    examples=[
        "# Initialize PPI analysis",
        "gene_list = ['TP53', 'BRCA1', 'EGFR', 'MYC']",
        "gene_type_dict = dict(zip(gene_list, ['Type1']*2 + ['Type2']*2))",
        "gene_color_dict = dict(zip(gene_list, ['red']*2 + ['blue']*2))",
        "ppi = ov.bulk.pyPPI(gene=gene_list, species=9606,",
        "                     gene_type_dict=gene_type_dict,",
        "                     gene_color_dict=gene_color_dict)",
        "# Perform interaction analysis",
        "G = ppi.interaction_analysis()",
        "# Plot network",
        "fig, ax = ppi.plot_network(figsize=(8,8), node_size=1000)"
    ],
    related=["bulk.string_interaction", "bulk.string_map", "utils.plot_network"]
)
class pyPPI(object):

    def __init__(self,gene: list,species: int,gene_type_dict: dict,gene_color_dict: dict,
                 score: float = 0.4) -> None:
        r"""Initialize protein-protein interaction analysis.

        Arguments:
            gene: List of gene names for PPI analysis
            species: NCBI taxon identifiers (e.g. Human is 9606, see STRING organisms)
            gene_type_dict: Dictionary mapping gene names to gene types
            gene_color_dict: Dictionary mapping gene names to colors for visualization
            score: Threshold for protein interaction confidence (default: 0.4)
        """
        self.gene=gene 
        self.species=species
        self.score=score
        self.gene_type_dict=gene_type_dict
        self.gene_color_dict=gene_color_dict

    def interaction_analysis(self) -> nx.Graph:
        r"""Perform protein-protein interaction analysis.
        
        Returns:
            G: NetworkX Graph object containing PPI network for query genes
        """
        G=generate_G(self.gene,
                          self.species,
                          self.score)
        self.G=G 
        return G
    
    def plot_network(self, **kwargs: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        r"""Plot protein-protein interaction network.

        Arguments:
            **kwargs: Additional keyword arguments passed to plot_network function

        Returns:
            fig: Figure object containing PPI network plot
            ax: Axes object containing PPI network plot
        """
        return plot_network(self.G, self.gene_type_dict, self.gene_color_dict, **kwargs)
    
    
        