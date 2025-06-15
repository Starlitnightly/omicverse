import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import requests
import io

ucd_install=False

class BulkDeconvolve(object):
    r"""
    Bulk RNA-seq deconvolution using UCDeconvolve.
    
    This class provides an interface to the UCDeconvolve platform for deconvolving
    bulk RNA-seq samples into cell-type proportions using curated reference datasets.
    UCDeconvolve offers cloud-based deconvolution with multiple reference datasets
    and algorithms.
    
    The class handles authentication with the UCDeconvolve API and preprocessing
    of input data for deconvolution analysis.
    """

    def check_ucdeconvolve(self):
        r"""
        Check if UCDeconvolve package is installed.
        
        Verifies that the ucdeconvolve package is available and sets the global
        installation flag for subsequent use.

        Arguments:
            None
            
        Returns:
            None: Sets global ucd_install flag
            
        Raises:
            ImportError: If ucdeconvolve package is not installed
        """
        global ucd_install
        try:
            import ucdeconvolve as ucd
            ucd_install=True
            print('ucdeconvolve have been install version:',ucd.__version__)
        except ImportError:
            raise ImportError(
                'Please install the ucdeconvolve: `pip install -U ucdeconvolve`.'
            )

    def __init__(self,token,adata):
        r"""
        Initialize BulkDeconvolve with authentication and data.
        
        Sets up the UCDeconvolve API connection and prepares the input data
        for deconvolution analysis.

        Arguments:
            token: Authentication token for UCDeconvolve API access
            adata: AnnData object containing bulk RNA-seq expression data
            
        Returns:
            None
        """
        self.check_ucdeconvolve()
        global ucd_install
        if ucd_install==True:
            global_imports("ucdeconvolve","ucd")
        ucd.api.authenticate(token)
        self.adata=adata
        ucd.tl.base(adata)

    
