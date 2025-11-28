import anndata
from ..utils.registry import register_function


@register_function(
    aliases=["批次效应校正", "batch_correction", "combat", "批次校正", "去批次效应"],
    category="bulk",
    description="Perform batch effect correction using ComBat algorithm for bulk RNA-seq data",
    prerequisites={
        'functions': [],
        'optional_functions': []
    },
    requires={
        'obs': []
    },
    produces={
        'layers': ['batch_correction']
    },
    auto_fix='none',
    examples=[
        "ov.bulk.batch_correction(adata, batch_key='batch')",
        "ov.bulk.batch_correction(adata, batch_key='sample_batch', key_added='combat_corrected')"
    ],
    related=["single.batch_correction", "pp.scale", "pp.regress"]
)
def batch_correction(adata:anndata.AnnData,
                     batch_key=None,
                     key_added:str='batch_correction'):
    r"""Perform batch effect correction using ComBat algorithm.
    
    Arguments:
        adata: AnnData object containing expression data.
        batch_key: Key in adata.obs containing batch information.
        key_added: Name for the corrected data layer. Default: 'batch_correction'.
    
    Returns:
        None: The function modifies adata.layers[key_added] in place with batch-corrected expression data.
    """
    
    try:
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError(
            'Please install the combat: `pip install combat`.'
        )
    adata.layers[key_added]=pycombat(adata.to_df().T,adata.obs[batch_key].values).T
    print(f"Storing batch correction result in adata.layers['{key_added}']")