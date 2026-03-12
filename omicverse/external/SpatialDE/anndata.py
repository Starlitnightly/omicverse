''' Wrapper functions to use SpatialDE directly on AnnData objects
'''
import logging

import pandas as pd

#import NaiveDE

from .aeh import spatial_patterns
from .base import run
from .util import qvalue

def spatialde_test(adata, coord_columns=['x', 'y'], regress_formula='np.log(total_counts)'):
    ''' Run the SpatialDE test on an AnnData object

    Parameters
    ----------

    adata: An AnnData object with counts in the .X field.

    coord_columns: A list with the columns of adata.obs which represent spatial
                   coordinates. Default ['x', 'y'].

    regress_formula: A patsy formula for linearly regressing out fixed effects
                     from columns in adata.obs before fitting the SpatialDE models.
                     Default is 'np.log(total_counts)'.

    Returns
    -------

    results: A table of spatial statistics for each gene.
    '''
    from ..NaiveDE import stabilize, regress_out 
    logging.info('Performing VST for NB counts')
    adata.layers['stabilized'] = stabilize(adata.X.T).T

    logging.info('Regressing out fixed effects')
    adata.layers['residual'] = regress_out(adata.obs,
                                                   adata.layers['stabilized'].T,
                                                   regress_formula).T

    X = adata.obs[coord_columns].values
    expr_mat = pd.DataFrame.from_records(adata.layers['residual'],
                                         columns=adata.var.index,
                                         index=adata.obs.index)

    results = run(X, expr_mat)

    # Clip 0 pvalues
    min_pval = results.query('pval > 0')['pval'].min() / 2
    results['pval'] = results['pval'].clip_lower(min_pval)

    # Correct for multiple testing
    results['qval'] = qvalue(results['pval'], pi0=1.)

    return results


def automatic_expression_histology(adata, filtered_results, C, l,
                                    coord_columns=['x', 'y'], layer='residual', **kwargs):
    ''' Fit the Automatic Expression Histology (AEH) model to
    expression in an AnnData object.

    Parameters
    ----------

    adata: An AnnData object with a layer of stabilized expression values

    filtered_results: A DataFrame with the signifificant subset of results
                      from the SpatialDE significance test.

    C: integer, the number of hidden spatial patterns.

    l: float, the common lengthscale for the hidden spatial patterns.

    coord_columns: A list with the columns of adata.obs which represent spatial
                   coordinates. Default ['x', 'y'].

    layer: A string indicating the layer of adata to fit the AEH model to.
           By defualt uses the 'residual' layer.

    Remaining arguments are passed to SpatialDE.aeh.spatial_patterns()

    Returns
    -------

    (histology_results, patterns)

    histology_results: DataFrame with pattern membership information for each gene.

    patterns: DataFrame with the inferred hidden spatial functions the genes belong to
              evaluated at all points in the data.

    '''
    X = adata.obs[coord_columns].values

    expr_mat = pd.DataFrame.from_records(adata.layers[layer],
                                         columns=adata.var.index,
                                         index=adata.obs.index)

    logging.info('Performing Automatic Expression Histology')
    histology_results, patterns = spatial_patterns(X, expr_mat, filtered_results,
                                                   C, l, **kwargs)

    return histology_results, patterns
