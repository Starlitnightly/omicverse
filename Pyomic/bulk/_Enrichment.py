import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from scipy import stats
import networkx as nx
import datetime
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy  
from scipy import cluster   
from sklearn import decomposition as skldec 


from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,dendrogram

import ERgene
import os

import gseapy as gp
from gseapy.plot import barplot, dotplot


def enrichment_KEGG(gene_list,
					gene_sets=['KEGG_2019_Human'],
					organism='Human',
					description='test_name',
					outdir='enrichment_kegg',
					cutoff=0.5):

	'''
	Gene enrichment analysis of KEGG

	Parameters
	----------
	gene_list:list
		The gene set to be enrichment analyzed
	gene_sets:list
		The gene_set of enrichr library
		Input Enrichr Libraries (https://maayanlab.cloud/Enrichr/#stats)
	organism:str
		Select from (human, mouse, yeast, fly, fish, worm)
	description:str
		The title of enrichment
	outdir:str
		The savedir of enrichment
	cutoff:float
		Show enriched terms which Adjusted P-value < cutoff.

	Returns
	----------
	res:pandas.DataFrame
		stores your last query
	'''

	enr = gp.enrichr(gene_list=gene_list,
				 gene_sets=gene_sets,
				 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
				 description=description,
				 outdir=outdir,
				 # no_plot=True,
				 cutoff=cutoff # test dataset, use lower value from range(0,1)
				)
	subp=dotplot(enr.res2d, title=description,cmap='seismic')
	print(subp)
	return enr.res2d

def enrichment_GO(gene_list,
					go_mode='Bio',
					organism='Human',
					description='test_name',
					outdir='enrichment_go',
					cutoff=0.5):

	'''
	Gene enrichment analysis of GO

	Parameters
	----------
	gene_list:list
		The gene set to be enrichment analyzed
	go_mode:str
		The module of GO include:'Bio','Cell','Mole'
	organism:str
		Select from (human, mouse, yeast, fly, fish, worm)
	description:str
		The title of enrichment
	outdir:str
		The savedir of enrichment
	cutoff:float
		Show enriched terms which Adjusted P-value < cutoff.

	Returns
	----------
	result:pandas.DataFrame
		stores your last query
	'''
	if(go_mode=='Bio'):
		geneset='GO_Biological_Process_2018'
	if(go_mode=='Cell'):
		geneset='GO_Cellular_Component_2018'
	if(go_mode=='Mole'):
		geneset='GO_Molecular_Function_2018'
	enr = gp.enrichr(gene_list=gene_list,
				 gene_sets=geneset,
				 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
				 description=description,
				 outdir=outdir,
				 # no_plot=True,
				 cutoff=cutoff # test dataset, use lower value from range(0,1)
				)
	subp=dotplot(enr.res2d, title=description,cmap='seismic')
	print(subp)
	return enr.res2d

def enrichment_GSEA(data,
				   gene_sets='KEGG_2016',
				   processes=4,
				   permutation_num=100,
				   outdir='prerank_report_kegg',
				   seed=6):
	'''
	Gene enrichment analysis of GSEA

	Parameters
	----------
	data:pandas.DataFrame
		The result of Find_DEG(function in DeGene.py)
	gene_sets:list
		The gene_set of enrichr library
		Input Enrichr Libraries (https://maayanlab.cloud/Enrichr/#stats)
	processes:int
		CPU number
	permutation_num:int
		Number of permutations for significance computation. Default: 1000.
	outdir:str
		The savedir of enrichment
	seed:int
		Random seed

	Returns
	----------
	result:Return a Prerank obj. 
	All results store to  a dictionary, obj.results,
		 where contains::

			 | {es: enrichment score,
			 |  nes: normalized enrichment score,
			 |  p: P-value,
			 |  fdr: FDR,
			 |  size: gene set size,
			 |  matched_size: genes matched to the data,
			 |  genes: gene names from the data set
			 |  ledge_genes: leading edge genes}
	'''


	rnk=pd.DataFrame(columns=['genename','FoldChange'])
	rnk['genename']=data.index
	rnk['FoldChange']=data['FoldChange'].tolist()
	rnk1=rnk.drop_duplicates(['genename'])
	rnk1=rnk1.sort_values(by='FoldChange', ascending=False)
	
	pre_res = gp.prerank(rnk=rnk1, gene_sets=gene_sets,
					 processes=processes,
					 permutation_num=permutation_num, # reduce number to speed up testing
					 outdir=outdir, format='png', seed=seed)
	pre_res.res2d.sort_index().to_csv('GSEA_result.csv')
	return pre_res

def Plot_GSEA(data,num=0):

	'''
	Plot the GSEA result figure

	Parameters
	----------
	data:prerank obj
		The result of enrichment_GSEA
	num:int
		The sequence of pathway drawn 
		Default:0(the first pathway)

	'''
	terms = data.res2d.index
	from gseapy.plot import gseaplot
	# to save your figure, make sure that ofname is not None
	gseaplot(rank_metric=data.ranking, term=terms[num], **data.results[terms[num]])