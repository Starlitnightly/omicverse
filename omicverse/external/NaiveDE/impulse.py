import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import optimize, stats


def impulse(t, h0, h1p, h2, t1, t2, beta):
    ''' A parametric impulse function defined by two conjoined sigmoids.

    See ImpulseDE2 paper.
    '''
    return 1. / (h1p + 1) * \
           (h0 + (h1p - h0) / (1. + np.exp(-np.exp(beta) * (t - t1)))) * \
           (h2 + (h1p - h2) / (1. + np.exp( np.exp(beta) * (t - t2))))


def make_p0(run_data):
    ''' Reasonable parameter initialization for the impulse model.
    '''
    h0_init = run_data.loc[run_data.hour.argmin(), 'expr']
    peak = run_data.loc[run_data.expr.argmax()]
    h1_init = peak['expr']
    h2_init = run_data.loc[run_data.hour.argmax(), 'expr']

    t1_init = (run_data['hour'].min() + peak['hour']) / 2
    t2_init = (run_data['hour'].max() + peak['hour']) / 2
    
    beta_init = -1.
    
    set_start_values = (0., h1_init, 0., t1_init, t2_init, beta_init)
    
    return set_start_values


def make_run_data(gene, t, expression_matrix):
    return pd.DataFrame({
        'hour': t,
        'expr': expression_matrix.loc[gene]
    })


# Helpers for the optimize.leastsq function
def func(p, x, y):
    return impulse(x, *p)

def residuals(p, x, y):
    return func(p, x, y) - y


def impulse_tests(t, expression_matrix, maxfev=50):
    ''' Least squares version of the ImpulseDE2 test

        t : A Series or vector with time values
        expresion_matrix: Assume columns are genes and rows are sampels

        maxfev: Maximum number of function evaluations per gene, higher numbers 
                give better accuracy at the cost of speed. Defualt is 50.
    '''
    t = np.array(t)
    n = expression_matrix.shape[1]

    gene_params = pd.DataFrame(index=expression_matrix.index,
                               columns=['h0', 'h1', 'h2', 't1', 't2', 'beta', \
                                        'llr', 'pval', 'res_alt', 'res_null'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for gene in tqdm(expression_matrix.index):
            run_data = make_run_data(gene, t, expression_matrix)

            opt_res = optimize.leastsq(residuals,
                                       make_p0(run_data),
                                       args=(t, run_data.expr.values),
                                       maxfev=maxfev)

            gene_params.loc[gene, ['h0', 'h1', 'h2', 't1', 't2', 'beta']] = opt_res[0]

            res_alt = np.sum(np.square(residuals(opt_res[0], t, run_data.expr.values)))
            gene_params.loc[gene, 'res_alt'] = res_alt
            ll_alt = -n / 2. * np.log(2 * np.pi) - n / 2. * np.log(res_alt / n) - n / 2.
            
            res_null = np.sum(np.square(run_data.expr.values.mean() - run_data.expr.values))
            gene_params.loc[gene, 'res_null'] = res_null
            ll_null = -n / 2. * np.log(2 * np.pi) - n / 2. * np.log(res_null / n) - n / 2.

            gene_params.loc[gene, 'llr'] = ll_alt - ll_null

    gene_params['llr'] =  gene_params['llr'].replace(-np.inf, 0).fillna(0)
    gene_params['pval'] = stats.chi2.sf(2 * gene_params['llr'], df=5)

    return gene_params

