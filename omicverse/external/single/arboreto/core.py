"""
Core functional building blocks, composed in a Dask graph for distributed computation.
"""

import numpy as np
import pandas as pd
import logging

import scipy
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from dask import delayed
from dask.dataframe import from_delayed
from dask.dataframe.utils import make_meta

logger = logging.getLogger(__name__)

DEMON_SEED = 666
ANGEL_SEED = 777
EARLY_STOP_WINDOW_LENGTH = 25

SKLEARN_REGRESSOR_FACTORY = {
    'RF': RandomForestRegressor,
    'ET': ExtraTreesRegressor,
    'GBM': GradientBoostingRegressor
}

# scikit-learn random forest regressor
RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn extra-trees regressor
ET_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn gradient boosting regressor
GBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 500,
    'max_features': 0.1
}

# scikit-learn stochastic gradient boosting regressor
SGBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 5000,  # can be arbitrarily large
    'max_features': 0.1,
    'subsample': 0.9
}


def is_sklearn_regressor(regressor_type):
    """
    :param regressor_type: string. Case insensitive.
    :return: whether the regressor type is a scikit-learn regressor, following the scikit-learn API.
    """
    return regressor_type.upper() in SKLEARN_REGRESSOR_FACTORY.keys()


def is_xgboost_regressor(regressor_type):
    """
    :param regressor_type: string. Case insensitive.
    :return: boolean indicating whether the regressor type is the xgboost regressor.
    """
    return regressor_type.upper() == 'XGB'


def is_oob_heuristic_supported(regressor_type, regressor_kwargs):
    """
    :param regressor_type: on
    :param regressor_kwargs:
    :return: whether early stopping heuristic based on out-of-bag improvement is supported.

    """
    return \
        regressor_type.upper() == 'GBM' and \
        'subsample' in regressor_kwargs and \
        regressor_kwargs['subsample'] < 1.0


def to_tf_matrix(expression_matrix,
                 gene_names,
                 tf_names):
    """
    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names: a list of gene names. Each entry corresponds to the expression_matrix column with same index.
    :param tf_names: a list of transcription factor names. Should be a subset of gene_names.
    :return: tuple of:
             0: A numpy matrix representing the predictor matrix for the regressions.
             1: The gene names corresponding to the columns in the predictor matrix.
    """

    tuples = [(index, gene) for index, gene in enumerate(gene_names) if gene in tf_names]

    tf_indices = [t[0] for t in tuples]
    tf_matrix_names = [t[1] for t in tuples]

    return expression_matrix[:, tf_indices], tf_matrix_names


def fit_model(regressor_type,
              regressor_kwargs,
              tf_matrix,
              target_gene_expression,
              early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
              seed=DEMON_SEED):
    """
    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param tf_matrix: the predictor matrix (transcription factor matrix) as a numpy array.
    :param target_gene_expression: the target (y) gene expression to predict in function of the tf_matrix (X).
    :param early_stop_window_length: window length of the early stopping monitor.
    :param seed: (optional) random seed for the regressors.
    :return: a trained regression model.
    """
    regressor_type = regressor_type.upper()


    if isinstance(target_gene_expression, scipy.sparse.spmatrix):
        target_gene_expression = target_gene_expression.A.flatten()

    assert tf_matrix.shape[0] == target_gene_expression.shape[0]


    def do_sklearn_regression():
        regressor = SKLEARN_REGRESSOR_FACTORY[regressor_type](random_state=seed, **regressor_kwargs)

        with_early_stopping = is_oob_heuristic_supported(regressor_type, regressor_kwargs)

        if with_early_stopping:
            regressor.fit(tf_matrix, target_gene_expression, monitor=EarlyStopMonitor(early_stop_window_length))
        else:
            regressor.fit(tf_matrix, target_gene_expression)

        return regressor

    if is_sklearn_regressor(regressor_type):
        return do_sklearn_regression()
    # elif is_xgboost_regressor(regressor_type):
    #     raise ValueError('XGB regressor not yet supported')
    else:
        raise ValueError('Unsupported regressor type: {0}'.format(regressor_type))


def to_feature_importances(regressor_type,
                           regressor_kwargs,
                           trained_regressor):
    """
    Motivation: when the out-of-bag improvement heuristic is used, we cancel the effect of normalization by dividing
    by the number of trees in the regression ensemble by multiplying again by the number of trees used.

    This enables prioritizing links that were inferred in a regression where lots of

    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param trained_regressor: the trained model from which to extract the feature importances.
    :return: the feature importances inferred from the trained model.
    """

    if is_oob_heuristic_supported(regressor_type, regressor_kwargs):
        n_estimators = len(trained_regressor.estimators_)

        denormalized_importances = trained_regressor.feature_importances_ * n_estimators

        return denormalized_importances
    else:
        return trained_regressor.feature_importances_


def to_meta_df(trained_regressor,
               target_gene_name):
    """
    :param trained_regressor: the trained model from which to extract the meta information.
    :param target_gene_name: the name of the target gene.
    :return: a Pandas DataFrame containing side information about the regression.
    """
    n_estimators = len(trained_regressor.estimators_)

    return pd.DataFrame({'target': [target_gene_name], 'n_estimators': [n_estimators]})


def to_links_df(regressor_type,
                regressor_kwargs,
                trained_regressor,
                tf_matrix_gene_names,
                target_gene_name):
    """
    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param trained_regressor: the trained model from which to extract the feature importances.
    :param tf_matrix_gene_names: the list of names corresponding to the columns of the tf_matrix used to train the model.
    :param target_gene_name: the name of the target gene.
    :return: a Pandas DataFrame['TF', 'target', 'importance'] representing inferred regulatory links and their
             connection strength.
    """

    def pythonic():
        # feature_importances = trained_regressor.feature_importances_
        feature_importances = to_feature_importances(regressor_type, regressor_kwargs, trained_regressor)

        links_df = pd.DataFrame({'TF': tf_matrix_gene_names, 'importance': feature_importances})
        links_df['target'] = target_gene_name

        clean_links_df = links_df[links_df.importance > 0].sort_values(by='importance', ascending=False)

        return clean_links_df[['TF', 'target', 'importance']]

    if is_sklearn_regressor(regressor_type):
        return pythonic()
    elif is_xgboost_regressor(regressor_type):
        raise ValueError('XGB regressor not yet supported')
    else:
        raise ValueError('Unsupported regressor type: ' + regressor_type)


def clean(tf_matrix,
          tf_matrix_gene_names,
          target_gene_name):
    """
    :param tf_matrix: numpy array. The full transcription factor matrix.
    :param tf_matrix_gene_names: the full list of transcription factor names, corresponding to the tf_matrix columns.
    :param target_gene_name: the target gene to remove from the tf_matrix and tf_names.
    :return: a tuple of (matrix, names) equal to the specified ones minus the target_gene_name if the target happens
             to be one of the transcription factors. If not, the specified (tf_matrix, tf_names) is returned verbatim.
    """

    if target_gene_name not in tf_matrix_gene_names:
        clean_tf_matrix = tf_matrix
    else:
        ix = tf_matrix_gene_names.index(target_gene_name)
        if isinstance(tf_matrix, scipy.sparse.spmatrix):
            clean_tf_matrix = scipy.sparse.hstack([tf_matrix[:, :ix],
                                                   tf_matrix[:, ix+1:]])
        else:
            clean_tf_matrix = np.delete(tf_matrix, ix, 1)

    clean_tf_names = [tf for tf in tf_matrix_gene_names if tf != target_gene_name]

    assert clean_tf_matrix.shape[1] == len(clean_tf_names)  # sanity check

    return clean_tf_matrix, clean_tf_names


def retry(fn, max_retries=10, warning_msg=None, fallback_result=None):
    """
    Minimalistic retry strategy to compensate for failures probably caused by a thread-safety bug in scikit-learn:
    * https://github.com/scikit-learn/scikit-learn/issues/2755
    * https://github.com/scikit-learn/scikit-learn/issues/7346

    :param fn: the function to retry.
    :param max_retries: the maximum number of retries to attempt.
    :param warning_msg: a warning message to display when an attempt fails.
    :param fallback_result: result to return when all attempts fail.
    :return: Returns the result of fn if one attempt succeeds, else return fallback_result.
    """
    nr_retries = 0

    result = fallback_result

    for attempt in range(max_retries):
        try:
            result = fn()
        except Exception as cause:
            nr_retries += 1

            msg_head = '' if warning_msg is None else repr(warning_msg) + ' '
            msg_tail = "Retry ({1}/{2}). Failure caused by {0}.".format(repr(cause), nr_retries, max_retries)

            logger.warning(msg_head + msg_tail)
        else:
            break

    return result


def infer_partial_network(regressor_type,
                          regressor_kwargs,
                          tf_matrix,
                          tf_matrix_gene_names,
                          target_gene_name,
                          target_gene_expression,
                          include_meta=False,
                          early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                          seed=DEMON_SEED):
    """
    Ties together regressor model training with regulatory links and meta data extraction.

    :param regressor_type: string. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param tf_matrix: numpy matrix. The feature matrix X to use for the regression.
    :param tf_matrix_gene_names: list of transcription factor names corresponding to the columns of the tf_matrix used to
                                 train the regression model.
    :param target_gene_name: the name of the target gene to infer the regulatory links for.
    :param target_gene_expression: the expression profile of the target gene. Numpy array.
    :param include_meta: whether to also return the meta information DataFrame.
    :param early_stop_window_length: window length of the early stopping monitor.
    :param seed: (optional) random seed for the regressors.
    :return: if include_meta == True, return links_df, meta_df

             link_df: a Pandas DataFrame['TF', 'target', 'importance'] containing inferred regulatory links and their
             connection strength.

             meta_df: a Pandas DataFrame['target', 'meta', 'value'] containing meta information regarding the trained
             regression model.
    """
    def fn():
        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)

        # special case in which only a single TF is passed and the target gene
        # here is the same as the TF (clean_tf_matrix is empty after cleaning):
        if clean_tf_matrix.size==0:
            raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

        try:
            trained_regressor = fit_model(regressor_type, regressor_kwargs, clean_tf_matrix, target_gene_expression,
                                          early_stop_window_length, seed)
        except ValueError as e:
            raise ValueError("Regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e)))

        links_df = to_links_df(regressor_type, regressor_kwargs, trained_regressor, clean_tf_matrix_gene_names,
                               target_gene_name)

        if include_meta:
            meta_df = to_meta_df(trained_regressor, target_gene_name)

            return links_df, meta_df
        else:
            return links_df

    fallback_result = (_GRN_SCHEMA, _META_SCHEMA) if include_meta else _GRN_SCHEMA

    return retry(fn,
                 fallback_result=fallback_result,
                 warning_msg='WARNING: infer_data failed for target {0}'.format(target_gene_name))


def target_gene_indices(gene_names,
                        target_genes):
    """
    :param gene_names: list of gene names.
    :param target_genes: either int (the top n), 'all', or a collection (subset of gene_names).
    :return: the (column) indices of the target genes in the expression_matrix.
    """

    if isinstance(target_genes, list) and len(target_genes) == 0:
        return []

    if isinstance(target_genes, str) and target_genes.upper() == 'ALL':
        return list(range(len(gene_names)))

    elif isinstance(target_genes, int):
        top_n = target_genes
        assert top_n > 0

        return list(range(min(top_n, len(gene_names))))

    elif isinstance(target_genes, list):
        if not target_genes:  # target_genes is empty
            return target_genes
        elif all(isinstance(target_gene, str) for target_gene in target_genes):
            return [index for index, gene in enumerate(gene_names) if gene in target_genes]
        elif all(isinstance(target_gene, int) for target_gene in target_genes):
            return target_genes
        else:
            raise ValueError("Mixed types in target genes.")

    else:
        raise ValueError("Unable to interpret target_genes.")


_GRN_SCHEMA = make_meta({'TF': str, 'target': str, 'importance': float})
_META_SCHEMA = make_meta({'target': str, 'n_estimators': int})


def create_graph(expression_matrix,
                 gene_names,
                 tf_names,
                 regressor_type,
                 regressor_kwargs,
                 client,
                 target_genes='all',
                 limit=None,
                 include_meta=False,
                 early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                 repartition_multiplier=1,
                 seed=DEMON_SEED):
    """
    Main API function. Create a Dask computation graph.

    Note: fixing the GC problems was fixed by 2 changes: [1] and [2] !!!

    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names: list of gene names. Each entry corresponds to the expression_matrix column with same index.
    :param tf_names: list of transcription factor names. Should have a non-empty intersection with gene_names.
    :param regressor_type: regressor type. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param client: a dask.distributed client instance.
                   * Used to scatter-broadcast the tf matrix to the workers instead of simply wrapping in a delayed().
    :param target_genes: either int, 'all' or a collection that is a subset of gene_names.
    :param limit: optional number of top regulatory links to return. Default None.
    :param include_meta: Also return the meta DataFrame. Default False.
    :param early_stop_window_length: window length of the early stopping monitor.
    :param repartition_multiplier: multiplier
    :param seed: (optional) random seed for the regressors. Default 666.
    :return: if include_meta is False, returns a Dask graph that computes the links DataFrame.
             If include_meta is True, returns a tuple: the links DataFrame and the meta DataFrame.
    """

    assert expression_matrix.shape[1] == len(gene_names)
    assert client, "client is required"

    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_names)

    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    # [1] wrap in a list of 1 -> unsure why but Matt. Rocklin does this often...
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)

    delayed_link_dfs = []  # collection of delayed link DataFrames
    delayed_meta_dfs = []  # collection of delayed meta DataFrame

    for target_gene_index in target_gene_indices(gene_names, target_genes):
        target_gene_name = delayed(gene_names[target_gene_index], pure=True)
        target_gene_expression = delayed(expression_matrix[:, target_gene_index], pure=True)

        if include_meta:
            delayed_link_df, delayed_meta_df = delayed(infer_partial_network, pure=True, nout=2)(
                regressor_type, regressor_kwargs,
                future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed)

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)
                delayed_meta_dfs.append(delayed_meta_df)
        else:
            delayed_link_df = delayed(infer_partial_network, pure=True)(
                regressor_type, regressor_kwargs,
                future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed)

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    # gather the DataFrames into one distributed DataFrame
    all_links_df = from_delayed(delayed_link_dfs, meta=_GRN_SCHEMA)
    all_meta_df = from_delayed(delayed_meta_dfs, meta=_META_SCHEMA)

    # optionally limit the number of resulting regulatory links, descending by top importance
    if limit:
        maybe_limited_links_df = all_links_df.nlargest(limit, columns=['importance'])
    else:
        maybe_limited_links_df = all_links_df

    # [2] repartition to nr of workers -> important to avoid GC problems!
    # see: http://dask.pydata.org/en/latest/dataframe-performance.html#repartition-to-reduce-overhead
    n_parts = len(client.ncores()) * repartition_multiplier

    if include_meta:
        return maybe_limited_links_df.repartition(npartitions=n_parts), \
               all_meta_df.repartition(npartitions=n_parts)
    else:
        return maybe_limited_links_df.repartition(npartitions=n_parts)


class EarlyStopMonitor:

    def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
        """
        :param window_length: length of the window over the out-of-bag errors.
        """

        self.window_length = window_length

    def window_boundaries(self, current_round):
        """
        :param current_round:
        :return: the low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(self, current_round, regressor, _):
        """
        Implementation of the GradientBoostingRegressor monitor function API.

        :param current_round: the current boosting round.
        :param regressor: the regressor.
        :param _: ignored.
        :return: True if the regressor should stop early, else False.
        """

        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False