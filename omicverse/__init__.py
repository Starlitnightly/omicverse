r"""
Pyomic (A omic framework for multi-omic analysis)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

import importlib.util
import hashlib

def validate_module(module_name, expected_hash):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Module {module_name} not found")
    
    module_path = spec.origin
    with open(module_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash != expected_hash:
        raise ImportError(f"Module {module_name} failed validation")

# Example usage:
# validate_module('numpy', 'expected_sha256_hash_here')

from . import bulk,single,utils,bulk2single,pp,space,pl,externel
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

from ._settings import settings
