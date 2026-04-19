r"""
Microbiome analysis utilities.

Downstream 16S / amplicon analysis on top of the AnnData produced by
:mod:`omicverse.alignment.amplicon_16s_pipeline`
(samples × ASVs, ``var`` carries 7-rank SINTAX taxonomy + ASV sequence).

Classes
-------
pyAlpha : Alpha-diversity (Shannon / Simpson / Chao1 / Observed / Faith PD)
pyBeta  : Beta-diversity distance matrices (Bray-Curtis / Jaccard / Aitchison
          / UniFrac — UniFrac requires ``unifrac`` package + phylogenetic tree)
pyOrdinate : PCoA / NMDS / RDA / CCA ordinations (via scikit-bio / sklearn)
pyDA    : Differential-abundance testing (Wilcoxon, pyDESeq2, ANCOM-BC)

Functions
---------
rarefy                 : Subsample counts to a common depth
filter_by_prevalence   : Drop rare taxa below a prevalence threshold
collapse_taxa          : Collapse ASV counts to a given rank (e.g. genus)
clr / ilr              : Centred / isometric log-ratio transforms

Examples
--------
>>> import omicverse as ov
>>> adata = ov.alignment.amplicon_16s_pipeline(fastq_dir='raw/', ...)
>>> # diversity
>>> alpha = ov.micro.pyAlpha(adata).run(metrics=['shannon', 'observed_asvs'])
>>> beta  = ov.micro.pyBeta(adata).run(metric='braycurtis', rarefy=True)
>>> # ordination
>>> ord_  = ov.micro.pyOrdinate(adata, dist_key='braycurtis').pcoa(n=3)
>>> # differential abundance at genus level
>>> da    = ov.micro.pyDA(adata).wilcoxon(group_key='group', rank='genus')
"""

from ._diversity import pyAlpha, pyBeta, ALPHA_METRICS, BETA_METRICS
from ._ord import pyOrdinate
from ._da import pyDA
from ._pp import rarefy, filter_by_prevalence, collapse_taxa, clr, ilr

__all__ = [
    # diversity
    "pyAlpha",
    "pyBeta",
    "ALPHA_METRICS",
    "BETA_METRICS",
    # ordination
    "pyOrdinate",
    # differential abundance
    "pyDA",
    # preprocessing
    "rarefy",
    "filter_by_prevalence",
    "collapse_taxa",
    "clr",
    "ilr",
]
