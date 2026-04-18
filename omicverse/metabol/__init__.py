r"""
Metabolomics analysis for omicverse — AnnData-native peak-table workflows.

``ov.metabol`` covers the downstream-of-peak-picking stages of a
metabolomics study: QC, imputation, normalization, transformation,
univariate and multivariate statistics. Input is a peak table (produced
by XCMS / MZmine / MS-DIAL / OpenMS / MetaboAnalyst) loaded into an
AnnData with ``obs=samples``, ``var=metabolites`` — the same convention
as every other omicverse module.

**Not covered**: raw-file processing (mzML → peak table) and advanced
annotation (MS/MS spectral matching). Use ``pyopenms`` / MZmine / MS-DIAL
upstream, then bring the peak table here.

Quick-start
-----------
>>> from omicverse.metabol import pyMetabo, read_metaboanalyst
>>> adata = read_metaboanalyst("human_cachexia.csv")
>>> m = pyMetabo(adata)
>>> (m.impute(method="qrilc")
...    .normalize(method="pqn")
...    .transform(method="pareto")
...    .differential(method="welch_t")
...    .opls_da(n_ortho=1))
>>> m.deg_table.head()                         # univariate hits
>>> m.vip_table().head()                       # multivariate VIP
>>> m.significant_metabolites(padj_thresh=0.05)

Pipeline stages
---------------
I/O
    ``read_metaboanalyst``, ``read_wide``, ``read_lcms``
QC  (MS-specific)
    ``cv_filter``, ``drift_correct``, ``blank_filter``
Imputation
    ``impute`` — kNN, half-min, QRILC, zero
Sample normalization
    ``normalize`` — PQN, TIC, median, MSTUS
Feature transform
    ``transform`` — log, glog, autoscale, Pareto
Differential analysis
    ``differential`` — t-test, Wilcoxon, limma-moderated
Multivariate
    ``plsda``, ``opls_da`` — PLS-DA / OPLS-DA with VIP
Plotting
    ``volcano``, ``s_plot``, ``vip_bar``

Relationship to existing omicverse modules
------------------------------------------
- Multi-omics integration: reuse ``ov.single.pyMOFA`` directly on an
  AnnData stack (gene expression × metabolite concentrations).
- Batch correction: ``ov.bulk.batch_correction`` (pyComBat) works on
  metabolite matrices out-of-the-box.
- WGCNA-style co-expression: ``ov.bulk.pyWGCNA`` ditto.
"""
from __future__ import annotations

from . import plotting
from ._impute import impute
from ._norm import normalize
from ._plsda import PLSDAResult, opls_da, plsda
from ._qc import blank_filter, cv_filter, drift_correct
from ._stats import differential
from ._transform import transform
from .io import read_lcms, read_metaboanalyst, read_wide
from .plotting import s_plot, vip_bar, volcano
from .pymetabo import pyMetabo

__all__ = [
    # class API
    "pyMetabo",
    # I/O
    "read_metaboanalyst",
    "read_wide",
    "read_lcms",
    # QC
    "cv_filter",
    "drift_correct",
    "blank_filter",
    # preprocessing
    "impute",
    "normalize",
    "transform",
    # stats
    "differential",
    # multivariate
    "plsda",
    "opls_da",
    "PLSDAResult",
    # plotting
    "plotting",
    "volcano",
    "s_plot",
    "vip_bar",
]
