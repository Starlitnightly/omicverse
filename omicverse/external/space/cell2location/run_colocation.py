# -*- coding: utf-8 -*-
"""Run full pipeline of regression model for estimating regulatory programmes of cell types and other covariates
which accounting for the effects of experimental batch and technology."""

import gc
import os
import pickle

# +
import time
from os import mkdir
from re import sub

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib import rcParams

#import cell2location.plt as c2lpl
from .plt import plot_factor_spatial
from .utils._spatial_knn import spatial_neighbours, sum_neighbours


def save_plot(path, filename, extension="png"):
    r"""Save current plot to `path` as `filename` with `extension`"""

    plt.savefig(path + filename + "." + extension)
    # fig.clear()
    # plt.close()


def run_colocation(
    sp_data,
    n_neighbours=None,
    model_name="CoLocatedGroupsSklearnNMF",
    verbose=False,
    return_all=True,
    train_args={
        "n_fact": [30],
        "n_iter": 20000,
        "sample_name_col": None,
        "mode": "normal",
        "n_type": "restart",
        "n_restarts": 5,
    },
    model_kwargs={"nmf_kwd_args": {"tol": 0.00001}},
    posterior_args={},
    export_args={"path": "./results", "run_name_suffix": "", "top_n": 10},
):
    r"""Run co-located cell type combination model: train for specified number of factors,
     evaluate the stability, save, export results and save diagnostic plots

    Parameters
        ----------
        sp_data:
             Anndata object with cell2location model output in .uns['mod']

    Returns
        -------
        dict
            dictionary {'mod','sc_data','model_name', 'train_args','posterior_args',
              'export_args', 'run_name', 'run_time'}
    """

    # set default parameters

    d_train_args = {
        "n_fact": [30],
        "n_iter": 20000,
        "learning_rate": 0.01,
        "use_cuda": False,
        "sample_prior": False,
        "cell_abundance_obsm": "q05_cell_abundance_w_sf",
        "factor_names": "factor_names",
        "sample_name_col": None,
        "mode": "normal",
        "n_type": "restart",
        "n_restarts": 5,
        "include_source_location": False,
    }

    d_posterior_args = {
        "n_samples": 1000,
        "evaluate_stability_align": False,
        "evaluate_stability_transpose": True,
        "mean_field_slot": "init_1",
    }

    d_export_args = {
        "path": "./results",
        "plot_extension": "pdf",
        "scanpy_coords_name": "spatial",
        "scanpy_plot_vmax": "p99.2",
        "scanpy_plot_size": 1.3,
        "scanpy_alpha_img": 0.0,
        "save_model": True,
        "run_name_suffix": "",
        "export_q05": False,
        "plot_histology": False,
        "top_n": 10,
        "plot_cell_type_loadings_kwargs": dict(),
    }

    d_model_kwargs = {"init": "random", "random_state": 0, "nmf_kwd_args": {"tol": 0.000001}, "alpha": 0.01}

    # replace defaults with parameters supplied
    for k in train_args.keys():
        d_train_args[k] = train_args[k]
    train_args = d_train_args

    for k in posterior_args.keys():
        d_posterior_args[k] = posterior_args[k]
    posterior_args = d_posterior_args

    for k in export_args.keys():
        d_export_args[k] = export_args[k]
    export_args = d_export_args

    for k in model_kwargs.keys():
        d_model_kwargs[k] = model_kwargs[k]
    model_kwargs = d_model_kwargs

    # start timing
    start = time.time()

    sp_data = sp_data.copy()

    # import the specified version of the model
    if type(model_name) is str:
        import cell2location.models.downstream as models

        Model = getattr(models, model_name)
    else:
        Model = model_name

    ####### Preparing data #######
    # extract cell density parameter
    X_data = np.array(sp_data.obsm[train_args["cell_abundance_obsm"]].values)
    var_names = sp_data.uns["mod"][train_args["factor_names"]]
    obs_names = sp_data.obs_names
    if train_args["sample_name_col"] is None:
        # if slots needed to generate scanpy plots are present, use scanpy spatial slot name:
        sc_spatial_present = np.isin(list(sp_data.uns.keys()), ["spatial"])[0]
        if sc_spatial_present:
            sp_data.obs["sample"] = list(sp_data.uns["spatial"].keys())[0]
        else:
            sp_data.obs["sample"] = "sample"

        train_args["sample_name_col"] = "sample"

    sample_id = sp_data.obs[train_args["sample_name_col"]]

    if n_neighbours is not None:
        neighbours = spatial_neighbours(
            coords=sp_data.obsm["spatial"],
            n_sp_neighbors=n_neighbours,
            radius=None,
            include_source_location=train_args["include_source_location"],
            sample_id=sample_id,
        )
        neighbours_sum = sum_neighbours(X_data, neighbours)
        X_data = np.concatenate([X_data, neighbours_sum], axis=1)
        var_names = list(var_names) + ["neigh_" + i for i in var_names]

    res_dict = {}

    for n_fact in train_args["n_fact"]:
        ####### Creating model #######
        if verbose:
            print("### Creating model ### - time " + str(np.around((time.time() - start) / 60, 2)) + " min")

        # create model class
        n_fact = int(n_fact)
        mod = Model(
            n_fact,
            X_data,
            n_iter=train_args["n_iter"],
            verbose=verbose,
            var_names=var_names,
            obs_names=obs_names,
            fact_names=["fact_" + str(i) for i in range(n_fact)],
            sample_id=sample_id,
            **model_kwargs,
        )

        ####### Print run name #######
        run_name = (
            str(mod.__class__.__name__)
            + "_"
            + str(mod.n_fact)
            + "combinations_"
            + str(mod.n_obs)
            + "locations_"
            + str(mod.n_var)
            + "factors"
            + export_args["run_name_suffix"]
        )
        path_name = (
            str(mod.__class__.__name__)
            + "_"
            + str(mod.n_obs)
            + "locations_"
            + str(mod.n_var)
            + "factors"
            + export_args["run_name_suffix"]
        )

        print("### Analysis name: " + run_name)  # analysis name is always printed

        # create the export directory
        path = export_args["path"] + path_name + "/"
        if not os.path.exists(path):
            os.makedirs(os.path.abspath(path))

        ####### Sampling prior #######
        if train_args["sample_prior"]:
            raise ValueError("Sampling prior not implemented yet")

        ####### Training model #######
        if verbose:
            print("### Training model###")
        if train_args["mode"] == "normal":
            mod.fit(n=train_args["n_restarts"], n_type=train_args["n_type"])

        elif train_args["mode"] == "tracking":
            raise ValueError("tracking training not implemented yet")
        else:
            raise ValueError("train_args['mode'] can be only 'normal' or 'tracking'")

        ####### Evaluate stability of training #######
        fig_path = path + "stability_plots/"
        if not os.path.exists(fig_path):
            mkdir(fig_path)
        if train_args["n_restarts"] > 1:
            n_plots = train_args["n_restarts"] - 1
            ncol = int(np.min((n_plots, 3)))
            nrow = np.ceil(n_plots / ncol)
            plt.figure(figsize=(5 * nrow, 5 * ncol))
            mod.evaluate_stability(
                "cell_type_factors",
                n_samples=posterior_args["n_samples"],
                align=posterior_args["evaluate_stability_align"],
            )
            plt.tight_layout()
            save_plot(
                fig_path, filename=f"cell_type_factors_n_fact{mod.n_fact}", extension=export_args["plot_extension"]
            )
            if verbose:
                plt.show()
            plt.close()

            plt.figure(figsize=(5 * nrow, 5 * ncol))
            mod.evaluate_stability(
                "location_factors",
                n_samples=posterior_args["n_samples"],
                align=posterior_args["evaluate_stability_align"],
            )
            plt.tight_layout()
            save_plot(
                fig_path, filename=f"location_factors_n_fact{mod.n_fact}", extension=export_args["plot_extension"]
            )
            if verbose:
                plt.show()
            plt.close()

        ####### Evaluating parameters / sampling posterior #######
        if verbose:
            print(
                f"### Evaluating parameters / sampling posterior ### - time {np.around((time.time() - start) / 60, 2)} min"
            )
        # extract all parameters from parameter store or sample posterior
        mod.sample_posterior(
            node="all",
            n_samples=posterior_args["n_samples"],
            save_samples=False,
            mean_field_slot=posterior_args["mean_field_slot"],
        )

        # evaluate predictive accuracy of the model
        mod.compute_expected()

        # Plot predictive accuracy
        fig_path = path + "predictive_accuracy/"
        if not os.path.exists(fig_path):
            mkdir(fig_path)
        try:
            plt.figure(figsize=(5.5, 5.5))
            mod.plot_posterior_mu_vs_data()
            plt.tight_layout()
            save_plot(
                fig_path, filename=f"data_vs_posterior_mean_n_fact{mod.n_fact}", extension=export_args["plot_extension"]
            )
            if verbose:
                plt.show()
            plt.close()
        except Exception as e:
            print("Some error in plotting `mod.plot_posterior_mu_vs_data()`\n " + str(e))

        ####### Export summarised posterior & Saving results #######
        if verbose:
            print("### Saving results ###")

        # extract parameters into DataFrames
        mod.sample2df(node_name="nUMI_factors", ct_node_name="cell_type_factors")

        # export results to scanpy object
        sp_data = mod.annotate_adata(sp_data)  # as columns to .obs
        sp_data = mod.export2adata(sp_data, slot_name=f"mod_coloc_n_fact{mod.n_fact}")  # as a slot in .uns

        # print the fraction of cells of each type located to each combination
        ct_loadings = mod.print_gene_loadings(
            loadings_attr="cell_type_fractions",
            gene_fact_name="cell_type_fractions",
            top_n=min(export_args["top_n"], len(var_names)),
        )

        # save
        save_path = path + "factor_markers/"
        if not os.path.exists(save_path):
            mkdir(save_path)
        ct_loadings.to_csv(f"{save_path}n_fact{mod.n_fact}.csv")

        save_path = path + "location_factors_mean/"
        if not os.path.exists(save_path):
            mkdir(save_path)
        mod.location_factors_df.to_csv(f"{save_path}n_fact{mod.n_fact}.csv")

        save_path = path + "cell_type_fractions_mean/"
        if not os.path.exists(save_path):
            mkdir(save_path)
        mod.cell_type_fractions.to_csv(f"{save_path}n_fact{mod.n_fact}.csv")

        if export_args["export_q05"]:
            save_path = path + "q05_param/"
            if not os.path.exists(save_path):
                mkdir(save_path)
            mod.location_factors_q05.to_csv(f"{path}location_factors_q05.csv")
            mod.cell_type_fractions_q05.to_csv(f"{path}cell_type_fractions_q05.csv")

        # A convenient way to explore the composition of cell type combinations / microenvironments is by using a heatmap:
        # make nice names
        mod.cell_type_fractions.columns = [
            sub("mean_cell_type_factors", "", i) for i in mod.cell_type_fractions.columns
        ]

        fig_path = path + "cell_type_fractions_heatmap/"
        if not os.path.exists(fig_path):
            mkdir(fig_path)
        # plot co-occuring cell type combinations
        mod.plot_cell_type_loadings(
            **export_args["plot_cell_type_loadings_kwargs"],
        )

        save_plot(fig_path, filename=f"n_fact{mod.n_fact}", extension=export_args["plot_extension"])
        if verbose:
            plt.show()
        plt.close()

        ####### Plotting posterior of W / cell locations #######
        # Finally we need to examine where in the tissue each cell type combination / microenvironment is located
        rcParams["figure.figsize"] = [5, 6]
        rcParams["axes.facecolor"] = "black"
        if verbose:
            print("### Plotting cell combinations in 2D ###")

        data_samples = sp_data.obs[train_args["sample_name_col"]].unique()
        cluster_plot_names = mod.location_factors_df.columns

        fig_path = path + "spatial/"

        try:
            for s in data_samples:
                # if slots needed to generate scanpy plots are present, use scanpy:
                sc_spatial_present = np.any(np.isin(list(sp_data.uns.keys()), ["spatial"]))

                if sc_spatial_present:
                    sc.settings.figdir = fig_path

                    s_ind = sp_data.obs[train_args["sample_name_col"]] == s
                    s_keys = list(sp_data.uns["spatial"].keys())
                    s_spatial = np.array(s_keys)[[s in i for i in s_keys]][0]

                    # plot cell density in each combination
                    sc.pl.spatial(
                        sp_data[s_ind, :],
                        cmap="magma",
                        library_id=s_spatial,
                        color=cluster_plot_names,
                        ncols=6,
                        size=export_args["scanpy_plot_size"],
                        img_key="hires",
                        alpha_img=export_args["scanpy_alpha_img"],
                        vmin=0,
                        vmax=export_args["scanpy_plot_vmax"],
                        save=f"cell_density_mean_n_fact{mod.n_fact}_s{s}_{export_args['scanpy_plot_vmax']}.{export_args['plot_extension']}",
                        show=False,
                    )

                    if export_args["plot_histology"]:
                        sc.pl.spatial(
                            sp_data[s_ind, :],
                            cmap="magma",
                            library_id=s_spatial,
                            color=cluster_plot_names,
                            ncols=6,
                            size=export_args["scanpy_plot_size"],
                            img_key="hires",
                            alpha_img=1,
                            vmin=0,
                            vmax=export_args["scanpy_plot_vmax"],
                            save=f"cell_density_mean_n_fact{mod.n_fact}_s{s}_{export_args['scanpy_plot_vmax']}.{export_args['plot_extension']}",
                            show=False,
                        )

                else:
                    # if coordinates exist plot
                    if export_args["scanpy_coords_name"] is not None:
                        # move spatial coordinates to obs for compatibility with our plotter
                        sp_data.obs["imagecol"] = sp_data.obsm[export_args["scanpy_coords_name"]][:, 0]
                        sp_data.obs["imagerow"] = sp_data.obsm[export_args["scanpy_coords_name"]][:, 1]

                        p = plot_factor_spatial(
                            adata=sp_data,
                            fact_ind=np.arange(mod.location_factors_df.shape[1]),
                            fact=mod.location_factors_df,
                            cluster_names=cluster_plot_names,
                            n_columns=6,
                            trans="log10",
                            max_col=100,
                            col_breaks=[0, 1, 10, 20, 50],
                            sample_name=s,
                            samples_col=train_args["sample_name_col"],
                            obs_x="imagecol",
                            obs_y="imagerow",
                        )
                        p.save(
                            filename=fig_path
                            + f"cell_density_mean_n_fact{mod.n_fact}_s{s}.{export_args['plot_extension']}"
                        )

        except Exception as e:
            print("Some error in plotting with scanpy or `cell2location.plt.plot_factor_spatial()`\n " + str(e))

        rcParams["axes.facecolor"] = "white"
        matplotlib.rc_file_defaults()

        # save model object and related annotations
        save_path = path + "models/"
        if not os.path.exists(save_path):
            mkdir(save_path)
        if export_args["save_model"]:
            # save the model and other objects
            res_dict_1 = {
                "mod": mod,
                "model_name": model_name,
                "train_args": train_args,
                "posterior_args": posterior_args,
                "export_args": export_args,
                "run_name": run_name,
                "run_time": str(np.around((time.time() - start) / 60, 2)) + " min",
            }
            pickle.dump(res_dict_1, file=open(save_path + f"model_n_fact{mod.n_fact}.p", "wb"))

        else:
            # just save the settings
            res_dict_1 = {
                "model_name": model_name,
                "train_args": train_args,
                "posterior_args": posterior_args,
                "export_args": export_args,
                "run_name": run_name,
                "run_time": str(np.around((time.time() - start) / 60, 2)) + " min",
            }
            pickle.dump(res_dict_1, file=open(save_path + f"model_n_fact{mod.n_fact}.p", "wb"))

        res_dict[f"n_fact{mod.n_fact}"] = res_dict_1

        if verbose:
            print("### Done ### - time " + res_dict["run_time"])

    save_path = path + "anndata/"
    if not os.path.exists(save_path):
        mkdir(save_path)
    # save anndata with exported posterior
    sp_data.write(filename=f"{save_path}sp.h5ad", compression="gzip")

    if return_all:
        return res_dict, sp_data
    else:
        del res_dict
        del res_dict_1
        del mod
        gc.collect()
        return str((time.time() - start) / 60) + " min"
