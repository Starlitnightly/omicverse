r"""
Pyomic (A omic framework for multi-omic analysis)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import bulk,single,mofapy2,utils,bulk2single,pp
#usually
from .utils._data import read
from .utils._plot import palette,ov_plot_set

name = "omicverse"
__version__ = version(name)


