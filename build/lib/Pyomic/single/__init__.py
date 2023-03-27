r"""
single (A omic framework for single cell omic analysis)
"""

from ._cosg import cosg
from ._pySCSA import data_preprocess,cell_annotate,cell_anno_print,scanpy_lazy
from ._nocd import scnocd
from ._mofa import mofa,factor_exact,factor_correlation,get_weights
from ._scdrug import autoResolution,writeGEP,Drug_Response