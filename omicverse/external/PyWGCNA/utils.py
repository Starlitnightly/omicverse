import pickle
import os

import pandas as pd
import requests
import matplotlib.pyplot as plt
import networkx as nx

from .comparison import *
from ..._registry import register_function

# bcolors
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


# read WGCNA obj
@register_function(
    aliases=["读取WGCNA", "readWGCNA", "load_wgcna", "WGCNA加载"],
    category="bulk",
    description="Read and load a saved WGCNA object from pickle file",
    examples=[
        "# Load saved WGCNA object",
        "wgcna_obj = ov.bulk.readWGCNA('wgcna_analysis.p')",
        "# Use loaded object for further analysis",
        "hub_genes = wgcna_obj.top_n_hub_genes('lightgreen', n=10)"
    ],
    related=["bulk.pyWGCNA", "bulk.pyWGCNA.saveWGCNA"]
)
def readWGCNA(file):
    r"""Read and load a saved WGCNA object from pickle file.

    Arguments:
        file: Name or path of WGCNA object file.

    Returns:
        wgcna: PyWGCNA object loaded from the pickle file.
    """
    if not os.path.isfile(file):
        raise ValueError('WGCNA object not found at given path!')

    picklefile = open(file, 'rb')
    wgcna = pickle.load(picklefile)

    print(f"{BOLD}{OKBLUE}Reading {wgcna.name} WGCNA done!{ENDC}")
    return wgcna


# compare serveral networks
def compareNetworks(PyWGCNAs):
    """
    Compare serveral PyWGCNA objects
                
    :param PyWGCNAs: list of PyWGCNA objects
    :type PyWGCNAs: list of PyWGCNA class

    :return: compare object
    :rtype: Compare class
    """
    geneModules = {}
    for PyWGCNA in PyWGCNAs:
        geneModules[PyWGCNA.name] = PyWGCNA.datExpr.var
    compare = Comparison(geneModules=geneModules)
    compare.compareNetworks()

    return compare


# compare WGCNA to single cell
def compareSingleCell(PyWGCNAs, sc):
    """
    Compare WGCNA and gene marker from single cell experiment

    :param PyWGCNAs: WGCNA object
    :type PyWGCNAs: PyWGCNA class
    :param sc: gene marker table which has ....
    :type sc: pandas dataframe

    :return: compare object
    :rtype: Compare class

    """
    geneModules = {}
    for PyWGCNA in PyWGCNAs:
        geneModules[PyWGCNA.name] = PyWGCNA.datExpr.var
    geneModules["single_cell"] = sc
    compare = Comparison(geneModules=geneModules)
    compare.compareNetworks()

    return compare


def getGeneList(dataset='mmusculus_gene_ensembl',
                attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'],
                maps=['gene_id', 'gene_name', 'go_id'],
                server_domain="http://ensembl.org/biomart"):
    """
    get table that map gene ensembl id to gene name from biomart

    :param dataset: name of the dataset we used from biomart; mouse: mmusculus_gene_ensembl and human: hsapiens_gene_ensembl
        you can find more information here: https://bioconductor.riken.jp/packages/3.4/bioc/vignettes/biomaRt/inst/doc/biomaRt.html#selecting-a-biomart-database-and-dataset
    :type dataset: string
    :param attributes: List the types of data we want
    :type attributes: list
    :param maps: mapping between attributes and column names of gene information you want to show
    :type maps: list
    :param server_domain: URL of ensembl biomart server that you want to use to pull out the information (options: [‘’, ‘uswest’, ‘asia’])
    :type server_domain: string
    
    :return: table extracted from biomart related to the datasets including information from attributes
    :rtype: pandas dataframe
    """
    import biomart
    r = requests.get(f"{server_domain}/martview")

    if r.status_code != 200:
        print("The biomart server you requested is currently unavailable! please use other biomart server or try later")
        return

    r.close()
    server = biomart.BiomartServer(server_domain)
    mart = server.datasets[dataset]

    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')

    geneInfo = pd.DataFrame(columns=attributes)
    # Store the data in a dict
    for line in data.splitlines():
        line = line.split('\t')
        tmp = pd.DataFrame(line, index=attributes).T
        dict = {}
        for i in range(len(attributes)):
            dict[attributes[i]] = line[i]
        geneInfo = pd.concat([geneInfo, tmp], ignore_index=True)

    geneInfo.index = geneInfo[attributes[0]]
    geneInfo.drop(attributes[0], axis=1, inplace=True)

    if maps is not None:
        geneInfo.columns = maps[1:]

    return geneInfo


def getGeneListGOid(dataset='mmusculus_gene_ensembl',
                    attributes=['ensembl_gene_id', 'external_gene_name', 'go_id'],
                    Goid='GO:0003700',
                    server_domain="http://ensembl.org/biomart"):
    """
    get table that find gene id and gene name to specific Go term from biomart

    :param dataset: name of the dataset we used from biomart; mouse: mmusculus_gene_ensembl and human: hsapiens_gene_ensembl
        you can find more information here: https://bioconductor.riken.jp/packages/3.4/bioc/vignettes/biomaRt/inst/doc/biomaRt.html#selecting-a-biomart-database-and-dataset
    :type dataset: string
    :param attributes: List the types of data we want
    :type attributes: list
    :param Goid: GO term id you would like to get genes from them
    :type Goid: list or str
    :param server_domain: URL of ensembl biomart server that you want to use to pull out the inforamtion
    :type server_domain: string

    :return: table extracted from biomart related to the datasets including information from attributes with filtering
    :rtype: pandas dataframe
    """
    import biomart
    r = requests.get(f"{server_domain}/martview")

    if r.status_code != 200:
        print("The biomart server you requested is currently unavailable! please use other biomart server or try later")
        return

    r.close()
    server = biomart.BiomartServer(server_domain)
    mart = server.datasets[dataset]

    # mart.show_attributes()
    # mart.show_filters()

    response = mart.search({
        'filters': {
            'go': [Goid]
        },
        'attributes': attributes
    })
    data = response.raw.data.decode('ascii')

    geneInfo = pd.DataFrame(columns=attributes)
    # Store the data in a dict
    for line in data.splitlines():
        line = line.split('\t')
        dict = {}
        for i in range(len(attributes)):
            dict[attributes[i]] = line[i]
        geneInfo = geneInfo.append(dict, ignore_index=True)

    return geneInfo


# read comparison obj
def readComparison(file):
    """
    Read a comparison from a saved pickle file.

    :param file: Name / path of comparison object
    :type file: string

    :return: comparison object
    :rtype: comparison class
    """
    if not os.path.isfile(file):
        raise ValueError('Comparison object not found at given path!')

    picklefile = open(file, 'rb')
    comparison = pickle.load(picklefile)

    print(f"{BOLD}{OKBLUE}Reading comparison done!{ENDC}")
    return comparison
