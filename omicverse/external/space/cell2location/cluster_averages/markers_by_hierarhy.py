import numpy as np
import pandas as pd


def markers_by_hierarhy(inf_aver, var_names, hierarhy_df, quantile=[0.05, 0.1, 0.2], mode="exclusive"):
    r"""Find which genes are expressed at which level of cell type hierarchy.
    Assigns expression counts for each gene to higher levels of hierarhy using estimates of average expression for the
    lowest level and substracts that expression from the lowest level. For example, low level annotation can be `Inh_SST
    neurones`, high level `Inh neurones`, very high level `neurones`, top level `all cell types`. The function can deal
    with any number of layers but the order needs to be carefully considered (from broad to specific).

    .. math::
        g_{g} = \min\limits_{f} g_{f,g}

    .. math::
        g_{fn,g} = (\min\limits_{f∈fn} g_{f,g}) - g_{g}

    .. math::
        ...

    .. math::
        g_{f3,g} = (\min\limits_{f∈f3} g_{f,g}) - ... - g_{fn,g} - g_{g}

    .. math::
        g_{f2,g} = (\min\limits_{f∈f2} g_{f,g}) - g_{f3,g} - ... - g_{fn,g} - g_{g}

    .. math::
        g_{f1, g} = g_{f,g} - g_{f2,g} - g_{f3,g} - ... - g_{fn,g} - g_{g}

    Here, :math:`g_{f,g}` represents average expression of each gene in each level 1 cluster.
    :math:`g_{f1,g}` represents average expression of each gene unique to each level 1 cluster.
    :math:`g_{f2,g}` represents average expression of each gene unique to each level 2 cluster.
    :math:`g_{f3,g}` represents average expression of each gene unique to each level 3 cluster.
    :math:`g_{fn,g}` represents average expression of each gene unique to each level n cluster (can be deep).
    :math:`g_{g}` represents average expression of each gene unique to the top level (all cells).

    :param inf_aver: np.ndarray with :math:`g_{g,f}` or with :math:`g_{g,f,s}` where `s` represents posterior samples
    :param var_names: list, array or index with variable names
    :param hierarhy_df: pd.DataFrame that provides mapping between clusters at different levels.
       Index corresponds to level 1 :math:`f1`, first columns to the top level, second columns to the n-th level
       :math:`fn`,
       last column corresponds to the second level :math:`f2`.
       It is crucial the order of cell types :math:`f` in `hierarhy_df` matches the order of cell types in axis
       1 of `inf_aver`.
    :param quantile: list of posterior distribution quantiles to be computed
    :param mode: 'exclusive' or 'tree' mode. In 'exclusive' mode, the number of counts specific to each layer is
        computed (e.g. counts at layer 2 are excluded from layer 1). In 'tree' mode, children nodes inherit the
        expression of their parents (e.g. layer 1 countains the original counts :math:`g_{f,g}`, layer 2 contains
        counts from all parent layers :math:`g_{f2,g} + g_{f3,g} + ... + g_{fn,g} + g_{g}`.

    :return: When input is :math:`g_{g,f}` the output is pd.DataFrame with values for
        :math:`f1, f2, f3, ..., fn, all`in columns. When input is :math:`g_{g,f,s}` where `s` represents posterior sample
        the output is a dictionary with posterior samples for :math:`g_{g,f1-fn+all,s}` and similar dataframes for 'mean'
        and quantiles of the posterior distribution (e.g. 'q0.05').
    """

    results = {}
    names = {}

    if len(inf_aver.shape) == 2:  # using summarised posterior samples
        results["level_1"] = pd.DataFrame(inf_aver, index=var_names, columns=list(hierarhy_df.index))

        for k in np.arange(hierarhy_df.shape[1]) + 2:
            k_names = list(hierarhy_df.iloc[:, k - 2].unique())
            k_level = hierarhy_df.shape[1] + 3 - k
            results[f"level_{k_level}"] = pd.DataFrame(index=var_names, columns=k_names)

            # iterate over clusters at each level (e.g. f2, f3 ...)
            for c in k_names:
                ind = hierarhy_df.iloc[:, k - 2] == c
                c_names = hierarhy_df.index[ind]
                ind_min = results["level_1"][c_names].min(1)
                results[f"level_{k_level}"][c] = ind_min
                results["level_1"][c_names] = (results["level_1"][c_names].T - ind_min).T

        if mode == "tree":
            # when mode is tree, add counts from parent levels
            for plev in np.arange(len(results) - 1):
                p_level = len(results) - plev
                p_names = list(hierarhy_df.iloc[:, plev].unique())

                # iterate over clusters at each level (e.g. f2, f3 ...)
                for p in p_names:
                    ind = hierarhy_df.iloc[:, plev] == p
                    if (plev) == (len(results) - 2):
                        ch_names = hierarhy_df.index[ind]
                    else:
                        ch_names = hierarhy_df.loc[ind, :].iloc[:, plev + 1]
                    results[f"level_{p_level - 1}"][ch_names] = (
                        results[f"level_{p_level - 1}"][ch_names].T + results[f"level_{p_level}"][p].values
                    ).T

        # concatenate to produce a general summary
        sep_inf_aver = pd.concat(list(results.values()), axis=1)

        return sep_inf_aver

    elif len(inf_aver.shape) == 3:  # using all posterior samples
        n_genes = inf_aver.shape[0]
        n_samples = inf_aver.shape[2]

        results["level_1"] = inf_aver.copy()
        names["level_1"] = list(hierarhy_df.index)

        for k in np.arange(hierarhy_df.shape[1]) + 2:
            k_names = list(hierarhy_df.iloc[:, k - 2].unique())
            k_level = hierarhy_df.shape[1] + 3 - k
            results[f"level_{k_level}"] = np.zeros((n_genes, len(k_names), n_samples))
            names[f"level_{k_level}"] = k_names

            # iterate over clusters at each level (e.g. f2, f3 ...)
            for c in k_names:
                ind = hierarhy_df.iloc[:, k - 2] == c
                k_ind = np.isin(k_names, c)

                ind_min = results["level_1"][:, ind, :].min(axis=1).reshape((n_genes, 1, n_samples))
                results[f"level_{k_level}"][:, k_ind, :] = ind_min
                results["level_1"][:, ind, :] = results["level_1"][:, ind, :] - ind_min

        if mode == "tree":
            # when mode is tree, add counts from parent levels
            for plev in np.arange(len(results) - 1):
                p_level = len(results) - plev
                p_names = list(hierarhy_df.iloc[:, plev].unique())

                # iterate over clusters at each level (e.g. f2, f3 ...)
                for p in p_names:
                    ind = hierarhy_df.iloc[:, plev] == p
                    if (plev) == (len(results) - 2):
                        ch_names = hierarhy_df.index[ind]
                    else:
                        ch_names = hierarhy_df.loc[ind, :].iloc[:, plev + 1]
                    ind = np.isin(names[f"level_{p_level - 1}"], ch_names)
                    p_ind = np.isin(p_names, p)

                    results[f"level_{p_level - 1}"][:, ind, :] = results[f"level_{p_level - 1}"][:, ind, :] + results[
                        f"level_{p_level}"
                    ][:, p_ind, :].reshape((n_genes, 1, n_samples))

        sep_inf_aver = np.concatenate(list(results.values()), axis=1)
        from itertools import chain

        sep_inf_aver_names = list(chain(*names.values()))

        out = {
            "samples": sep_inf_aver,
            "mean": pd.DataFrame(
                np.squeeze(np.mean(sep_inf_aver, axis=2)), index=var_names, columns=sep_inf_aver_names
            ),
        }

        for q in quantile:
            out[f"q{q}"] = pd.DataFrame(
                np.squeeze(np.quantile(sep_inf_aver, q=q, axis=2)), index=var_names, columns=sep_inf_aver_names
            )

        # TODO remove redundant layers

        return out
