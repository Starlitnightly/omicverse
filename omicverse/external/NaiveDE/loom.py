import numpy as np
import pandas as pd
import patsy
from scipy import stats
from tqdm import tqdm


def lr_tests(ds, alt_model, null_model='~ 1', gene_column='Gene', batch_size=2000, transformation=np.log1p, rcond=-1):
    '''
    Compare alt_model and null_model by a likelihood ratio test for every gene in ds.

    Args:
        ds (LoomConnection):    Dataset
        alt_model (str):    Formula describing alternative model
        null_model (str):   Formula describing null model
        gene_column (str):  Name of the gene labels to use in the ds (default "Gene")
        batch_size (int):   The number of genes to read from disk in each iteration (default 2000)
        transformation (function):  Transformation to apply to expression values before fitting (default np.log1p)
        rcond (float):  Conditioning for the least square fitting (default -1, which has no effect)

    Returns:
        results (DataFrame):    Dataframe with model parameter estimates for each gene, with P values from LRT.
    '''
    sample_info = pd.DataFrame()
    for k in ds.ca.keys():
        sample_info[k] = ds.ca[k]
    
    alt_design = patsy.dmatrix(alt_model, sample_info, return_type='dataframe')
    null_design = patsy.dmatrix(null_model, sample_info, return_type='dataframe')

    n = ds.shape[1]

    genes = []
    betas = []
    pvals = []

    total_batches = np.ceil(ds.shape[0] / batch_size).astype(int)
    for (ix, selection, vals) in tqdm(ds.scan(axis=0, batch_size=batch_size), total=total_batches):
        expression_matrix = transformation(vals[:, :])
        beta_alt, res_alt, rank_alt, s_alt = np.linalg.lstsq(alt_design, expression_matrix.T, rcond=rcond)
        beta_null, res_null, rank_null, s_null = np.linalg.lstsq(null_design, expression_matrix.T, rcond=rcond)

        genes.append(vals.ra[gene_column])

        ll_alt  = -n / 2. * np.log(2 * np.pi) - n / 2. * np.ma.log(res_alt  / n) - n / 2.
        ll_null = -n / 2. * np.log(2 * np.pi) - n / 2. * np.ma.log(res_null / n) - n / 2.

        llr = ll_alt - ll_null

        pval = stats.chi2.sf(2 * llr, df=(beta_alt.shape[0] - beta_null.shape[0]))
        pval = np.ma.MaskedArray(pval, mask=llr.mask).filled(1.)

        betas.append(beta_alt)
        pvals.append(pval)

    results = pd.DataFrame({gene_column: np.hstack(genes)})

    for name, beta in zip(alt_design.columns, np.hstack(betas)):
        results[name] = beta

    results['pval'] = np.hstack(pvals)

    min_pval = results.pval[results.pval != 0].min()
    results['pval'] = results.pval.clip_lower(min_pval)

    return results

