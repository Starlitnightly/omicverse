import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def xpercent_scale():
    ''' Helper function to format X-axis to percent tick labels
    '''
    plt.gca().set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()])


def FSV_sig(results, ms_results=None, certain_only=False, covariate_names=['log_total_count']):
    ''' Make a plot of Fraction Spatial Variance vs Q-value

    Optionally provide model selection results to the function will color points by model.

    Point size correspond to certinety of the FSV value.
    '''
    plt.yscale('log')
    
    results = results.copy()
    covariates = results.query('g in @covariate_names')
    results = results.query('g not in @covariate_names').copy()
    
    results['FSV95conf'] = 2 * np.sqrt(results['s2_FSV'])
    if ms_results is not None:
        results = results.merge(ms_results[['g', 'model']],
                                how='outer',
                                on='g',
                                suffixes=('', '_bic'))
    else:
        results['model_bic'] = results['model']
    
    # Split by FSV uncertainty levels
    size_map = {0.0: 40, 0.1: 12, 1.0: 1}
    conf_categories = pd.cut(results['FSV95conf'], [0, 1e-1, 1e0, np.inf])
    for conf_class, result_group in results.groupby(conf_categories):
        if certain_only:
            if conf_class.left > .0:
                continue
        
        # Plot non-signficant genes
        tmp = result_group.query('qval > 0.05')
        label = 'Genes (Not Significant)' if conf_class.left == 0.0 else None
        plt.scatter(tmp['FSV'], tmp['pval'],
                    alpha=0.5,
                    rasterized=True,
                    label=label,
                    marker='o',
                    color='k',
                    s=size_map[conf_class.left])

        tmp = result_group.query('qval <= 0.05')

        # Split significant genes by function class
        model_colors = {'SE': 'C0', 'PER': 'C1', 'linear': 'C2'}
        label_map = {'SE': 'General', 'PER': 'Periodic', 'linear': 'Linear'}
        for model_name, model_group in tmp.groupby('model_bic'):
            label = 'Genes ({} function)'.format(label_map[model_name]) if conf_class.left == 0.0 else None
            plt.scatter(model_group['FSV'], model_group['pval'],
                        alpha=0.5,
                        rasterized=True,
                        label=label,
                        marker='o',
                        color=model_colors[model_name],
                        s=size_map[conf_class.left])

    # Plot external covarites for reference
    plt.scatter(covariates['FSV'], covariates['pval'], marker='x', c='k', s=50, label=None)
    
    FDR_lim = results.query('qval < 0.05')['pval'].max()
    plt.axhline(FDR_lim, ls='--', c='k', lw=1, label='FDR = 0.05')
    
    # Label axes
    plt.xlabel('Fraction spatial variance')
    plt.ylabel('P-value')
    plt.gca().invert_yaxis()

    lgd = plt.legend(scatterpoints=3, loc='upper left', bbox_to_anchor=[1., 1.])
    
    xpercent_scale()
