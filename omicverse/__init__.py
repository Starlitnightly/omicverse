r"""
Pyomic (A omic framework for multi-omic analysis)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import bulk,single,mofapy2,utils,bulk2single,pp,space
#usually
from .utils._data import read
from .utils._plot import palette,ov_plot_set,plot_set

name = "omicverse"
__version__ = version(name)
omics="""
   ____            _     _    __                  
  / __ \____ ___  (_)___| |  / /__  _____________ 
 / / / / __ `__ \/ / ___/ | / / _ \/ ___/ ___/ _ \ 
/ /_/ / / / / / / / /__ | |/ /  __/ /  (__  )  __/ 
\____/_/ /_/ /_/_/\___/ |___/\___/_/  /____/\___/                                              
"""
print(omics)
print(f'Version: {__version__}, Tutorials: https://omicverse.readthedocs.io/')


