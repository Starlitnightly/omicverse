"""This is scSemiProfiler, a tool for semi-profiling single-cell sequencing data."""

__version__ = '0.1.0'

import pdb,sys,os


from .representative_selection import activeselection
from .get_eg_representatives import get_eg_representatives
from .initial_setup import initsetup
#from .initial_setup import inspect_data

from .inference import scinfer
from .singlecell_process import scprocess 
from .utils import *

__all__=['fast_generator','activeselect','initsetup','scinfer','scprocess','get_eg_representatives']
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

#for i in __all__:
#    __import__(i)

