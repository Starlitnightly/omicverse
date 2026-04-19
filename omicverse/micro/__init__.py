r"""
Microbiome analysis utilities.

Downstream 16S / amplicon analysis on top of the AnnData produced by
:mod:`omicverse.alignment.amplicon_16s_pipeline`
(samples × ASVs, ``var`` carries 7-rank SINTAX taxonomy + ASV sequence).

Classes
-------
Alpha : Alpha-diversity (Shannon / Simpson / Chao1 / Observed / Faith PD)
Beta  : Beta-diversity distance matrices (Bray-Curtis / Jaccard / Aitchison
          / UniFrac — UniFrac requires ``unifrac`` package + phylogenetic tree)
Ordinate : PCoA / NMDS / RDA / CCA ordinations (via scikit-bio / sklearn)
DA    : Differential-abundance testing (Wilcoxon, pyDESeq2, ANCOM-BC)

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
>>> alpha = ov.micro.Alpha(adata).run(metrics=['shannon', 'observed_otus'])
>>> beta  = ov.micro.Beta(adata).run(metric='braycurtis', rarefy=True)
>>> # ordination
>>> ord_  = ov.micro.Ordinate(adata, dist_key='braycurtis').pcoa(n=3)
>>> # differential abundance at genus level
>>> da    = ov.micro.DA(adata).wilcoxon(group_key='group', rank='genus')
"""

from ._diversity import Alpha, Beta, ALPHA_METRICS, BETA_METRICS
from ._ord import Ordinate
from ._da import DA
from ._pp import rarefy, filter_by_prevalence, collapse_taxa, clr, ilr

__all__ = [
    # diversity
    "Alpha",
    "Beta",
    "ALPHA_METRICS",
    "BETA_METRICS",
    # ordination
    "Ordinate",
    # differential abundance
    "DA",
    # preprocessing
    "rarefy",
    "filter_by_prevalence",
    "collapse_taxa",
    "clr",
    "ilr",
]
