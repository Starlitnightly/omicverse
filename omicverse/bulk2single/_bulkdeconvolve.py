import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import requests
import io

ucd_install=False

class BulkDeconvolve(object):

    def check_ucdeconvolve(self):
        """
        
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
        self.check_ucdeconvolve()
        global ucd_install
        if ucd_install==True:
            global_imports("ucdeconvolve","ucd")
        ucd.api.authenticate(token)
        self.adata=adata
        ucd.tl.base(adata)

    
