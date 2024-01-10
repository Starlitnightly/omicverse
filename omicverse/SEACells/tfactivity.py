import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# TF-target computation
def gene_tf_associations(
    gene_peak,
    pwm,
    min_corr=0.0,
    max_pval=0.1,
    min_peaks=0,
    ct_peaks=None,
    gene_set=None,
):
    """TODO."""
    tfs = pwm.var_names
    len(tfs)

    # initialize returns
    gene_tfs = {}
    n_gp_assoc = 0

    for gene, peak_df in tqdm(gene_peak.items(), desc="genes", total=len(gene_peak)):
        if gene_set is None:
            gene_set = gene_peak.keys()

        if gene not in gene_set or not isinstance(peak_df, pd.DataFrame):
            # skip genes with no peaks
            continue

        if ct_peaks is None:
            # metacell use case
            sig_peaks = (peak_df["pval"] < max_pval) & (peak_df["cor"] > min_corr)
            if all(~sig_peaks):
                # no significant peaks
                continue
            # grab list of sig peaks correlated with this gene
            peaks = peak_df.loc[sig_peaks, :].index
        else:
            valid_peaks = [x for x in peak_df.index if x in ct_peaks]

            if len(valid_peaks) == 0:
                continue
            peaks = peak_df.loc[valid_peaks, :].index

        if len(peaks) >= min_peaks:
            # convert pwm scores to bool, 0 = False
            tf_association = pwm[pwm.obs_names.isin(peaks), :].X
            tf_association = tf_association.toarray().astype(bool)

            # Grab TFs that are associated with a peak at least once
            tf_idxs = np.where(np.any(tf_association, axis=0))[0]

            gene_tfs[gene] = {}
            for tf in tf_idxs:
                tf_name = pwm.var_names[tf]
                tf_mask = tf_association[
                    :, tf
                ]  # subset the PWM scores to the relevant tf

                gene_tfs[gene][tf_name] = peaks[tf_mask]

                # grab the peaks for each tf
                n_gp_assoc += 1

    # log number of genes that passed thresholds
    print(
        f"{len(gene_tfs):,} genes and {n_gp_assoc:,} gene-TF combinations with at least {min_peaks} peaks."
    )

    return gene_tfs


def compute_tf_target_mat(pwm, gene_peak, gene_tfs, ct_specific=False):
    """TODO."""
    tfs = pwm.var_names

    # initiate three DFs
    tf_targ_mat = pd.DataFrame(0.0, index=gene_peak.index, columns=tfs)

    pwm_scores = pwm.to_df()

    for gene, tf_dict in tqdm(gene_tfs.items(), total=len(gene_tfs)):
        for tf, peaks in tf_dict.items():
            if ct_specific:
                score = pwm_scores.loc[peaks, tf].sum()
            else:
                # Compute weighted PWM score
                corrs = gene_peak[gene].loc[peaks, "cor"]
                score = (pwm_scores.loc[peaks, tf].values * corrs.values).sum()

            tf_targ_mat.loc[gene, tf] = score

    return tf_targ_mat


# gene selection


def get_de_genes(adata, groups, thresh, group_key, fc_min=1.5, pval_cut=1e-2):
    """TODO."""
    final_dict = {}
    print("threshold:", thresh)
    print()
    for g in tqdm(groups, total=len(groups)):
        group_dict = {}

        ref = [x for x in groups if x != g]
        _compute_de_genes(adata, g, group_dict, ref, group_key, fc_min, pval_cut)

        gene_list = []
        for gene in group_dict:
            if group_dict[gene] >= thresh:
                gene_list.append(gene)

        num_genes = len(gene_list)

        print(g)
        print("\tnum genes:", num_genes)

        gene_names = []
        for gene in gene_list:
            gene_names.append(gene)

        final_dict[g] = gene_names

    return final_dict


def _compute_de_genes(
    adata,
    g,
    group_dict,
    references,
    group_key,
    fc_min=1.5,
    pval_cut=1e-2,
):
    sc.settings.verbosity = 0
    for ref in references:
        de_data = sc.tl.rank_genes_groups(
            adata,
            groupby=group_key,
            groups=[g],
            reference=ref,
            use_raw=False,
            copy=True,
        )

        de_genes = sc.get.rank_genes_groups_df(
            de_data, g, pval_cutoff=pval_cut, log2fc_min=fc_min
        )

        valid_de_df = de_genes.sort_values(by="logfoldchanges", ascending=False)

        for gene in valid_de_df["names"]:
            if gene not in group_dict:
                group_dict[gene] = 1
            else:
                group_dict[gene] += 1
    sc.settings.verbosity = 1


def get_gene_set(
    mc_ad, peak_counts, de_genes, group_key, sub_group, expr_thresh=25, min_peaks=5
):
    """TODO."""
    mc_expr = mc_ad.to_df()
    valid_genes = peak_counts[peak_counts >= min_peaks].index

    highly_expressed = mc_expr.columns[
        (
            mc_expr.groupby(mc_ad.obs[group_key]).mean().loc[sub_group] > expr_thresh
        ).any()
    ]

    gene_set = np.intersect1d(valid_genes, de_genes)
    gene_set = np.intersect1d(gene_set, highly_expressed)

    print("Number of genes...")
    print(f"   ...highly expressed in {sub_group}: {len(highly_expressed)}")
    print(f"   ...correlated {min_peaks}+ peaks: {len(valid_genes)}")
    print(f"   ...differentially expressed: {len(de_genes)}")
    print("...taking intersection...")
    print()
    print(f"TOTAL: {len(gene_set)}")

    return gene_set


# Lasso regression


def fit_lasso_model(
    gtf, zs, tf_set, gene_set, cells, test_size=0.20, cv_fold=5, max_iter=10000
):
    """TODO."""
    X = gtf.loc[gene_set, tf_set]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index=gene_set, columns=tf_set)

    # set up containers
    res = {}
    for key in ["coefs", "rmses", "rsqs", "sp_corr", "x_tests", "intercept", "y_stats"]:
        res[key] = {}

    y = zs.loc[gene_set, :]
    for cell in tqdm(cells, total=len(cells)):
        _train_model(cell, X, y, gene_set, res, test_size, cv_fold, max_iter)

    return res


def _train_model(cell, X, y, gene_set, res, test_size=0.20, cv_fold=5, max_iter=10000):
    y = y.loc[:, cell]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    lasso_cv_model = LassoCV(cv=cv_fold, max_iter=max_iter, n_jobs=16).fit(
        X_train, y_train
    )

    opt_alpha = lasso_cv_model.alpha_
    lasso_tuned = Lasso().set_params(alpha=opt_alpha).fit(X_train, y_train)

    y_pred = lasso_tuned.predict(X_test)

    # save stats
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_sq = r2_score(y_test, y_pred)
    sp_corr = stats.spearmanr(y_test, y_pred)

    res["rmses"][cell] = rmse
    res["rsqs"][cell] = r_sq
    res["sp_corr"][cell] = sp_corr
    res["x_tests"][cell] = X_test
    res["y_stats"][cell] = y_test, y_pred
    res["intercept"][cell] = lasso_tuned.intercept_
    res["coefs"][cell] = pd.Series(lasso_tuned.coef_, index=X.columns).sort_values()


# TF-Activity functions
def non_zero_tfs(cells, ct_res, thresh=20):
    """TODO."""
    tf_count = {}

    for cell in tqdm(cells, total=len(cells)):
        coefs = ct_res["coefs"][cell]
        coefs = coefs[coefs != 0]
        for tf in coefs.index:
            if tf not in tf_count:
                tf_count[tf] = 1
            else:
                tf_count[tf] += 1
    valid_tfs = [tf for tf in tf_count.keys() if tf_count[tf] >= thresh]
    print(f"Number of Qualifying TFs: {len(valid_tfs)}")

    return valid_tfs


def compute_tf_activity(ad_sub, res, valid_tfs, cells):
    """TODO."""
    # cell_order = ad_sub.obs.loc[cells,:].sort_values('palantir_pseudotime').index
    tf_activities = pd.DataFrame(
        0.0, index=valid_tfs, columns=cells
    )  # set up container

    for cell in tqdm(cells, total=len(cells)):
        X_test = res["x_tests"][cell]
        y_test = res["y_stats"][cell][0]
        res["y_stats"][cell][1]
        coefs = res["coefs"][cell].copy()
        intercept = res["intercept"][cell]
        rmse = res["rmses"][cell]

        for tf in valid_tfs:
            if coefs.loc[tf] >= 0:
                sign = 1
            else:
                sign = -1

            new_coefs = coefs.copy()
            new_coefs.loc[tf] = 0
            new_y_pred = (X_test[new_coefs.index] * new_coefs).sum(axis=1) + intercept

            new_rmse = np.sqrt(mean_squared_error(y_test, new_y_pred))
            diff = np.abs(new_rmse - rmse)

            # fill in container
            tf_activities.loc[tf, cell] = diff * sign

    return tf_activities
