r"""
Pyomic (A omic framework for multi-omic analysis)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import bulk,single,utils,bulk2single,pp,space,pl,externel
#usually
from .utils._data import read
from .utils._plot import palette,ov_plot_set,plot_set


name = "omicverse"
__version__ = version(name)


from ._settings import settings,generate_reference_table


# 导入 matplotlib.pyplot
import matplotlib.pyplot as plt

# 将 plt 作为 omicverse 的一个属性
plt = plt  # 注意：确保没有其他变量名冲突

import numpy as np

np = np  # 注意：确保没有其他变量名冲突

# 导入 pandas
import pandas as pd

# 将 pd 作为 omicverse 的一个属性
pd = pd  # 注意：确保没有其他变量名冲突