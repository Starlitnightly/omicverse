r"""
single (A omic framework for single cell omic analysis)
"""

from ._cosg import cosg
from ._anno import pySCSA,MetaTiME,scanpy_lazy,scanpy_cellanno_from_dict,get_celltype_marker
from ._nocd import scnocd
from ._mofa import pyMOFAART,pyMOFA,GLUE_pair,factor_exact,factor_correlation,get_weights,glue_pair
from ._scdrug import autoResolution,writeGEP,Drug_Response
from ._cpdb import cpdb_network_cal,cpdb_plot_network,cpdb_plot_interaction,cpdb_interaction_filtered,cpdb_submeans_exacted
from ._scgsea import geneset_aucell,pathway_aucell,pathway_aucell_enrichment,pathway_enrichment,pathway_enrichment_plot
from ._via import pyVIA,scRNA_hematopoiesis
from ._simba import pySIMBA
from ._tosica import pyTOSICA
from ._atac import atac_concat_get_index,atac_concat_inner,atac_concat_outer
from ._batch import batch_correction
from ._cellfategenie import cellfategenie,gene_trends
from ._ltnn import scLTNN,plot_origin_tesmination,find_related_gene
from ._traj import TrajInfer
from ._cefcon import pyCEFCON,convert_human_to_mouse_network,load_human_prior_interaction_network,mouse_hsc_nestorowa16
from ._aucell import aucell
from ._metacell import MetaCell
