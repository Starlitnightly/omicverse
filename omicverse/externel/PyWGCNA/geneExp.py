import numpy as np
import pandas as pd
import os
import anndata as ad


# remove runtime warning (divided by zero)
np.seterr(divide='ignore', invalid='ignore')


class GeneExp:
    """
    A class used to creat gene expression anndata along data trait including both genes and samples information.

    :param species: species of the data you use i.e mouse, human
    :type species: str
    :param level: which type of data you use including gene, transcript (default: gene)
    :type level: str
    :param anndata: if the expression data is in anndata format you should pass it through this parameter. X should be expression matrix. var is a gene information and obs is a sample information.
    :param anndata: anndata
    :param geneExp: expression matrix which genes are in the rows and samples are columns
    :type geneExp: pandas dataframe
    :param geneExpPath: path of expression matrix
    :type geneExpPath: str
    :param sep: separation symbol to use for reading data in geneExpPath properly
    :type sep: str
    :param geneInfo: dataframe that contains genes information it should have a same index as gene expression column names (gene/transcript ID)
    :type geneInfo: pandas dataframe
    :param sampleInfo: dataframe that contains samples information it should have a same index as gene expression index (sample ID)
    :type sampleInfo: pandas dataframe
    """

    def __init__(self, 
                 species=None, 
                 level='gene',
                 anndata=None, 
                 geneExp=None,
                 geneExpPath=None, 
                 sep=',',
                 geneInfo=None,
                 sampleInfo=None):
        self.species = species
        self.level = level
        if geneExpPath is not None:
            if not os.path.isfile(geneExpPath):
                raise ValueError("file does not exist!")
            else:
                expressionList = pd.read_csv(geneExpPath, sep=sep, index_col=0)
        elif geneExp is not None:
            if isinstance(geneExp, pd.DataFrame):
                expressionList = geneExp
            else:
                raise ValueError("geneExp is not data frame!")
        elif anndata is not None:
            if isinstance(anndata, ad.AnnData):
                self.geneExpr = anndata
                return
            else:
                raise ValueError("geneExp is not data frame!")
        else:
            raise ValueError("all type of input can not be empty at the same time!")

        if geneInfo is None:
            geneInfo = pd.DataFrame(index=expressionList.columns)

        if sampleInfo is None:
            sampleInfo = pd.DataFrame(index=expressionList.index)

        self.geneExpr = ad.AnnData(X=expressionList, obs=sampleInfo, var=geneInfo)

    @staticmethod
    def updateGeneInfo(geneExpr, geneInfo=None, path=None, sep=','):
        """
        add/update genes info in expr anndata

        :param geneExpr: gene expression data along with sample and genes/transcript information
        :type geneExpr: anndata
        :param geneInfo: gene information table you want to add to your data
        :type geneInfo: pandas dataframe
        :param path: path of geneInfo
        :type path: str
        :param sep: separation symbol to use for reading data in path properly (default: ',')
        :type sep: str

        :return: updated gene expression data along with sample and genes/transcript information
        :rtype: anndata
        """
        if path is not None:
            if not os.path.isfile(path):
                raise ValueError("path does not exist!")
            geneInfo = pd.read_csv(path, sep=sep, index_col=0)
        elif geneInfo is not None:
            if not isinstance(geneInfo, pd.DataFrame):
                raise ValueError("geneInfo is not pandas dataframe!")
        else:
            raise ValueError("path and geneInfo can not be empty at the same time!")

        same_columns = geneExpr.var.columns.intersection(geneInfo.columns)
        geneExpr.var.drop(same_columns, axis=1, inplace=True)
        geneExpr.var = pd.concat([geneExpr.var, geneInfo], axis=1).loc[geneExpr.var.index, :]

        return geneExpr

    @staticmethod
    def updateSampleInfo(geneExpr, sampleInfo=None, path=None, sep=','):
        """
        add/update metadata in expr anndata

        :param geneExpr: gene expression data along with sample and genes/transcript information
        :type geneExpr: anndata
        :param sampleInfo: Sample information table you want to add to your data
        :type sampleInfo: pandas dataframe
        :param path: path of metaData
        :type path: str
        :param sep: separation symbol to use for reading data in path properly (default: ',')
        :type sep: str

        :return: updated gene expression data along with sample and genes/transcript information
        :rtype: anndata
        """
        if path is not None:
            if not os.path.isfile(path):
                raise ValueError("path does not exist!")
            sampleInfo = pd.read_csv(path, sep=sep, index_col=0)
        elif sampleInfo is not None:
            if not isinstance(sampleInfo, pd.DataFrame):
                raise ValueError("meta data is not pandas dataframe!")
        else:
            raise ValueError("path and metaData can not be empty at the same time!")

        same_columns = geneExpr.obs.columns.intersection(sampleInfo.columns)
        geneExpr.obs.drop(same_columns, axis=1, inplace=True)
        geneExpr.obs = pd.concat([geneExpr.obs, sampleInfo], axis=1).loc[geneExpr.obs.index, :]

        return geneExpr
