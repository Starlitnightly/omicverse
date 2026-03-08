import numpy as np
import pandas as pd

from scipy import stats


def lr_tests(sample_info, expression_matrix, alt_model, null_model='~ 1', rcond=-1, genes=None):
    ''' Compare alt_model and null_model by a Likelihood Ratio Test for every
    gene in the expression_matrix.
    
    Assumes columns are samples, and rows are genes.
    
    OLS Regression assumes data is variance stabilized. E.g. log scaled.
    
    Returns a DataFrame with alt_model weights and p-values.
    '''
    import patsy
    alt_design  = patsy.dmatrix(alt_model, sample_info, return_type='dataframe')
    null_design = patsy.dmatrix(null_model, sample_info, return_type='dataframe')
    
    beta_alt,  res_alt,  rank_alt,  s_alt = np.linalg.lstsq(alt_design, expression_matrix.T, rcond=rcond)
    beta_null, res_null, rank_null, s_null = np.linalg.lstsq(null_design, expression_matrix.T, rcond=rcond)

    if res_alt.shape[0] == 0:
        res_alt = np.sum(np.square(expression_matrix.T - alt_design.dot(beta_alt))).values

    if res_null.shape[0] == 0:
        res_null = np.sum(np.square(expression_matrix.T - null_design.dot(beta_null))).values

    if genes is None:
        genes = expression_matrix.index
    
    results = pd.DataFrame(beta_alt.T, columns=alt_design.columns, index=genes)
    
    n = expression_matrix.shape[1]
    ll_alt  = -n / 2. * np.log(2 * np.pi) - n / 2. * np.ma.log(res_alt  / n) - n / 2.
    ll_null = -n / 2. * np.log(2 * np.pi) - n / 2. * np.ma.log(res_null / n) - n / 2.
    
    llr = ll_alt - ll_null
    
    pval = stats.chi2.sf(2 * llr, df=beta_alt.shape[0] - beta_null.shape[0])
    pval = np.ma.MaskedArray(pval, mask=llr.mask).filled(1.)
    
    results['pval'] = pval
    results['qval'] = (results['pval'] * results.shape[0]).clip(upper=1.)
    
    return results


def regress_out(sample_info, expression_matrix, covariate_formula, design_formula='1', rcond=-1):
    ''' Implementation of limma's removeBatchEffect function
    '''
    # Ensure intercept is not part of covariates
    import patsy
    covariate_formula += ' - 1'

    covariate_matrix = patsy.dmatrix(covariate_formula, sample_info)
    design_matrix = patsy.dmatrix(design_formula, sample_info)

    design_batch = np.hstack((design_matrix, covariate_matrix))

    coefficients, res, rank, s = np.linalg.lstsq(design_batch, expression_matrix.T, rcond=rcond)
    beta = coefficients[design_matrix.shape[1]:]
    regressed = expression_matrix - covariate_matrix.dot(beta).T

    return regressed


def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data

    See https://f1000research.com/posters/4-1041 for motivation.

    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def anscombe(exppression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data

    See https://f1000research.com/posters/4-1041 for motivation.

    Assumes columns are samples, and rows are genes
    '''
    return stabilize(expression_matrix)


def vst(expression_matrix):
    ''' A VST derived from assumption of a global NB phi parameter for all genes.

    Var(mu) = mu + phi * mu^2

    Defined by symbolic integral as described in http://www.bioconductor.org/packages//2.13/bioc/vignettes/DESeq/inst/doc/vst.pdf

    Unlike the `stabilize` function, results here will be non-negative. This also
    assumes phi > 0.

    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    v = lambda mu, phi: mu + phi * mu ** 2
    phi_hat, _ = optimize.curve_fit(v, expression_matrix.mean(1), expression_matrix.var(1))

    return 2 * np.arcsinh(np.sqrt(phi_hat[0] * expression_matrix)) / np.sqrt(phi_hat[0])
