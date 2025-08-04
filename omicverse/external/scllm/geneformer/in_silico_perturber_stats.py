"""
Geneformer in silico perturber stats generator.

**Usage:**

.. code-block :: python

    >>> from geneformer import InSilicoPerturberStats
    >>> ispstats = InSilicoPerturberStats(mode="goal_state_shift",
    ...    cell_states_to_model={"state_key": "disease",
    ...                          "start_state": "dcm",
    ...                          "goal_state": "nf",
    ...                          "alt_states": ["hcm", "other1", "other2"]})
    >>> ispstats.get_stats("path/to/input_data",
    ...                    None,
    ...                    "path/to/output_directory",
    ...                    "output_prefix")

**Description:**

| Aggregates data or calculates stats for in silico perturbations based on type of statistics specified in InSilicoPerturberStats.
| Input data is raw in silico perturbation results in the form of dictionaries outputted by ``in_silico_perturber``.

"""


import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
from scipy.stats import ranksums
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm, trange

from . import ENSEMBL_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE
from .perturber_utils import flatten_list, validate_cell_states_to_model

logger = logging.getLogger(__name__)


# invert dictionary keys/values
def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def read_dict(cos_sims_dict, cell_or_gene_emb, anchor_token):
    if cell_or_gene_emb == "cell":
        cell_emb_dict = {
            k: v for k, v in cos_sims_dict.items() if v and "cell_emb" in k
        }
        return [cell_emb_dict]
    elif cell_or_gene_emb == "gene":
        if anchor_token is None:
            gene_emb_dict = {k: v for k, v in cos_sims_dict.items() if v}
        else:
            gene_emb_dict = {
                k: v for k, v in cos_sims_dict.items() if v and anchor_token == k[0]
            }
    return [gene_emb_dict]


# read raw dictionary files
def read_dictionaries(
    input_data_directory,
    cell_or_gene_emb,
    anchor_token,
    cell_states_to_model,
    pickle_suffix,
):
    file_found = False
    file_path_list = []
    if cell_states_to_model is None:
        dict_list = []
    else:
        validate_cell_states_to_model(cell_states_to_model)
        cell_states_to_model_valid = {
            state: value
            for state, value in cell_states_to_model.items()
            if state != "state_key"
            and cell_states_to_model[state] is not None
            and cell_states_to_model[state] != []
        }
        cell_states_list = []
        # flatten all state values into list
        for state in cell_states_to_model_valid:
            value = cell_states_to_model_valid[state]
            if isinstance(value, list):
                cell_states_list += value
            else:
                cell_states_list.append(value)
        state_dict = {state_value: dict() for state_value in cell_states_list}
    for file in os.listdir(input_data_directory):
        # process only files with given suffix (e.g. "_raw.pickle")
        if file.endswith(pickle_suffix):
            file_found = True
            file_path_list += [f"{input_data_directory}/{file}"]
    for file_path in tqdm(file_path_list):
        with open(file_path, "rb") as fp:
            cos_sims_dict = pickle.load(fp)
            if cell_states_to_model is None:
                dict_list += read_dict(cos_sims_dict, cell_or_gene_emb, anchor_token)
            else:
                for state_value in cell_states_list:
                    new_dict = read_dict(
                        cos_sims_dict[state_value], cell_or_gene_emb, anchor_token
                    )[0]
                    for key in new_dict:
                        try:
                            state_dict[state_value][key] += new_dict[key]
                        except KeyError:
                            state_dict[state_value][key] = new_dict[key]

    if not file_found:
        logger.error(
            "No raw data for processing found within provided directory. "
            "Please ensure data files end with '{pickle_suffix}'."
        )
        raise
    if cell_states_to_model is None:
        return dict_list
    else:
        return state_dict


# get complete gene list
def get_gene_list(dict_list, mode):
    if mode == "cell":
        position = 0
    elif mode == "gene":
        position = 1
    gene_set = set()
    if isinstance(dict_list, list):
        for dict_i in dict_list:
            gene_set.update([k[position] for k, v in dict_i.items() if v])
    elif isinstance(dict_list, dict):
        for state, dict_i in dict_list.items():
            gene_set.update([k[position] for k, v in dict_i.items() if v])
    else:
        logger.error(
            "dict_list should be a list, or if modeling shift to goal states, a dict. "
            f"{type(dict_list)} is not the correct format."
        )
        raise
    gene_list = list(gene_set)
    if mode == "gene":
        gene_list.remove("cell_emb")
    gene_list.sort()
    return gene_list


def token_tuple_to_ensembl_ids(token_tuple, gene_token_id_dict):
    try:
        return tuple([gene_token_id_dict.get(i, np.nan) for i in token_tuple])
    except TypeError:
        return gene_token_id_dict.get(token_tuple, np.nan)


def n_detections(token, dict_list, mode, anchor_token):
    cos_sim_megalist = []
    for dict_i in dict_list:
        if mode == "cell":
            cos_sim_megalist += dict_i.get((token, "cell_emb"), [])
        elif mode == "gene":
            cos_sim_megalist += dict_i.get((anchor_token, token), [])
    return len(cos_sim_megalist)


def get_fdr(pvalues):
    return list(smt.multipletests(pvalues, alpha=0.05, method="fdr_bh")[1])


def get_impact_component(test_value, gaussian_mixture_model):
    impact_border = gaussian_mixture_model.means_[0][0]
    nonimpact_border = gaussian_mixture_model.means_[1][0]
    if test_value > nonimpact_border:
        impact_component = 0
    elif test_value < impact_border:
        impact_component = 1
    else:
        impact_component_raw = gaussian_mixture_model.predict([[test_value]])[0]
        if impact_component_raw == 1:
            impact_component = 0
        elif impact_component_raw == 0:
            impact_component = 1
    return impact_component


# aggregate data for single perturbation in multiple cells
def isp_aggregate_grouped_perturb(cos_sims_df, dict_list, genes_perturbed):
    names = ["Cosine_sim", "Gene"]
    cos_sims_full_dfs = []
    if isinstance(genes_perturbed, list):
        if len(genes_perturbed) > 1:
            gene_ids_df = cos_sims_df.loc[
                np.isin(
                    [set(idx) for idx in cos_sims_df["Ensembl_ID"]],
                    set(genes_perturbed),
                ),
                :,
            ]
        else:
            gene_ids_df = cos_sims_df.loc[
                np.isin(cos_sims_df["Ensembl_ID"], genes_perturbed), :
            ]
    else:
        logger.error(
            "aggregate_data is for perturbation of single gene or single group of genes. genes_to_perturb should be formatted as list."
        )
        raise

    if gene_ids_df.empty:
        logger.error("genes_to_perturb not found in data.")
        raise

    tokens = gene_ids_df["Gene"]
    symbols = gene_ids_df["Gene_name"]

    for token, symbol in zip(tokens, symbols):
        cos_shift_data = []
        for dict_i in dict_list:
            cos_shift_data += dict_i.get((token, "cell_emb"), [])

        df = pd.DataFrame(columns=names)
        df["Cosine_sim"] = cos_shift_data
        df["Gene"] = symbol
        cos_sims_full_dfs.append(df)

    return pd.concat(cos_sims_full_dfs)


def find(variable, x):
    try:
        if x in variable:  # Test if variable is iterable and contains x
            return True
        elif x == variable:
            return True
    except (ValueError, TypeError):
        return x == variable  # Test if variable is x if non-iterable


def isp_aggregate_gene_shifts(
    cos_sims_df, dict_list, gene_token_id_dict, gene_id_name_dict, token_dtype
):
    cos_shift_data = dict()
    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        for dict_i in dict_list:
            if token_dtype == "nontuple":
                affected_pairs = [k for k, v in dict_i.items() if k[0] == token]
            else:
                affected_pairs = [k for k, v in dict_i.items() if find(k[0], token)]
            for key in affected_pairs:
                if key in cos_shift_data.keys():
                    cos_shift_data[key] += dict_i.get(key, [])
                else:
                    cos_shift_data[key] = dict_i.get(key, [])

    cos_data_mean = {
        k: [np.mean(v), np.std(v), len(v)] for k, v in cos_shift_data.items()
    }
    cos_sims_full_df = pd.DataFrame()
    cos_sims_full_df["Perturbed"] = [k[0] for k, v in cos_data_mean.items()]
    cos_sims_full_df["Gene_name"] = [
        cos_sims_df[cos_sims_df["Gene"] == k[0]]["Gene_name"].item()
        for k, v in cos_data_mean.items()
    ]
    cos_sims_full_df["Ensembl_ID"] = [
        cos_sims_df[cos_sims_df["Gene"] == k[0]]["Ensembl_ID"].item()
        for k, v in cos_data_mean.items()
    ]

    cos_sims_full_df["Affected"] = [k[1] for k, v in cos_data_mean.items()]
    cos_sims_full_df["Affected_gene_name"] = [
        gene_id_name_dict.get(gene_token_id_dict.get(token, np.nan), np.nan)
        for token in cos_sims_full_df["Affected"]
    ]
    cos_sims_full_df["Affected_Ensembl_ID"] = [
        gene_token_id_dict.get(token, np.nan) for token in cos_sims_full_df["Affected"]
    ]
    cos_sims_full_df["Cosine_sim_mean"] = [v[0] for k, v in cos_data_mean.items()]
    cos_sims_full_df["Cosine_sim_stdev"] = [v[1] for k, v in cos_data_mean.items()]
    cos_sims_full_df["N_Detections"] = [v[2] for k, v in cos_data_mean.items()]

    specific_val = "cell_emb"
    cos_sims_full_df["temp"] = list(cos_sims_full_df["Affected"] == specific_val)
    # reorder so cell embs are at the top and all are subordered by magnitude of cosine sim
    cos_sims_full_df = cos_sims_full_df.sort_values(
        by=(["temp", "Cosine_sim_mean"]), ascending=[False, True]
    ).drop("temp", axis=1)

    return cos_sims_full_df


# stats comparing cos sim shifts towards goal state of test perturbations vs random perturbations
def isp_stats_to_goal_state(
    cos_sims_df, result_dict, cell_states_to_model, genes_perturbed
):
    if (
        ("alt_states" not in cell_states_to_model.keys())
        or (len(cell_states_to_model["alt_states"]) == 0)
        or (cell_states_to_model["alt_states"] == [None])
    ):
        alt_end_state_exists = False
    elif (len(cell_states_to_model["alt_states"]) > 0) and (
        cell_states_to_model["alt_states"] != [None]
    ):
        alt_end_state_exists = True

    # for single perturbation in multiple cells, there are no random perturbations to compare to
    if genes_perturbed != "all":
        cos_sims_full_df = pd.DataFrame()

        cos_shift_data_end = []
        token = cos_sims_df["Gene"][0]
        cos_shift_data_end += result_dict[cell_states_to_model["goal_state"]].get(
            (token, "cell_emb"), []
        )
        cos_sims_full_df["Shift_to_goal_end"] = [np.mean(cos_shift_data_end)]
        if alt_end_state_exists is True:
            for alt_state in cell_states_to_model["alt_states"]:
                cos_shift_data_alt_state = []
                cos_shift_data_alt_state += result_dict.get(alt_state).get(
                    (token, "cell_emb"), []
                )
                cos_sims_full_df[f"Shift_to_alt_end_{alt_state}"] = [
                    np.mean(cos_shift_data_alt_state)
                ]

        # sort by shift to desired state
        cos_sims_full_df = cos_sims_full_df.sort_values(
            by=["Shift_to_goal_end"], ascending=[False]
        )
        return cos_sims_full_df

    elif genes_perturbed == "all":
        goal_end_random_megalist = []
        if alt_end_state_exists is True:
            alt_end_state_random_dict = {
                alt_state: [] for alt_state in cell_states_to_model["alt_states"]
            }
        for i in trange(cos_sims_df.shape[0]):
            token = cos_sims_df["Gene"][i]
            goal_end_random_megalist += result_dict[
                cell_states_to_model["goal_state"]
            ].get((token, "cell_emb"), [])
            if alt_end_state_exists is True:
                for alt_state in cell_states_to_model["alt_states"]:
                    alt_end_state_random_dict[alt_state] += result_dict[alt_state].get(
                        (token, "cell_emb"), []
                    )

        # downsample to improve speed of ranksums
        if len(goal_end_random_megalist) > 100_000:
            random.seed(42)
            goal_end_random_megalist = random.sample(
                goal_end_random_megalist, k=100_000
            )
        if alt_end_state_exists is True:
            for alt_state in cell_states_to_model["alt_states"]:
                if len(alt_end_state_random_dict[alt_state]) > 100_000:
                    random.seed(42)
                    alt_end_state_random_dict[alt_state] = random.sample(
                        alt_end_state_random_dict[alt_state], k=100_000
                    )

        names = [
            "Gene",
            "Gene_name",
            "Ensembl_ID",
            "Shift_to_goal_end",
            "Goal_end_vs_random_pval",
        ]
        if alt_end_state_exists is True:
            [
                names.append(f"Shift_to_alt_end_{alt_state}")
                for alt_state in cell_states_to_model["alt_states"]
            ]
            names.append(names.pop(names.index("Goal_end_vs_random_pval")))
            [
                names.append(f"Alt_end_vs_random_pval_{alt_state}")
                for alt_state in cell_states_to_model["alt_states"]
            ]
        cos_sims_full_df = pd.DataFrame(columns=names)

        n_detections_dict = dict()
        for i in trange(cos_sims_df.shape[0]):
            token = cos_sims_df["Gene"][i]
            name = cos_sims_df["Gene_name"][i]
            ensembl_id = cos_sims_df["Ensembl_ID"][i]
            goal_end_cos_sim_megalist = result_dict[
                cell_states_to_model["goal_state"]
            ].get((token, "cell_emb"), [])
            n_detections_dict[token] = len(goal_end_cos_sim_megalist)
            mean_goal_end = np.mean(goal_end_cos_sim_megalist)
            pval_goal_end = ranksums(
                goal_end_random_megalist, goal_end_cos_sim_megalist
            ).pvalue

            if alt_end_state_exists is True:
                alt_end_state_dict = {
                    alt_state: [] for alt_state in cell_states_to_model["alt_states"]
                }
                for alt_state in cell_states_to_model["alt_states"]:
                    alt_end_state_dict[alt_state] = result_dict[alt_state].get(
                        (token, "cell_emb"), []
                    )
                    alt_end_state_dict[f"{alt_state}_mean"] = np.mean(
                        alt_end_state_dict[alt_state]
                    )
                    alt_end_state_dict[f"{alt_state}_pval"] = ranksums(
                        alt_end_state_random_dict[alt_state],
                        alt_end_state_dict[alt_state],
                    ).pvalue

            results_dict = dict()
            results_dict["Gene"] = token
            results_dict["Gene_name"] = name
            results_dict["Ensembl_ID"] = ensembl_id
            results_dict["Shift_to_goal_end"] = mean_goal_end
            results_dict["Goal_end_vs_random_pval"] = pval_goal_end
            if alt_end_state_exists is True:
                for alt_state in cell_states_to_model["alt_states"]:
                    results_dict[f"Shift_to_alt_end_{alt_state}"] = alt_end_state_dict[
                        f"{alt_state}_mean"
                    ]
                    results_dict[
                        f"Alt_end_vs_random_pval_{alt_state}"
                    ] = alt_end_state_dict[f"{alt_state}_pval"]

            cos_sims_df_i = pd.DataFrame(results_dict, index=[i])
            cos_sims_full_df = pd.concat([cos_sims_full_df, cos_sims_df_i])

        cos_sims_full_df["Goal_end_FDR"] = get_fdr(
            list(cos_sims_full_df["Goal_end_vs_random_pval"])
        )
        if alt_end_state_exists is True:
            for alt_state in cell_states_to_model["alt_states"]:
                cos_sims_full_df[f"Alt_end_FDR_{alt_state}"] = get_fdr(
                    list(cos_sims_full_df[f"Alt_end_vs_random_pval_{alt_state}"])
                )

        # quantify number of detections of each gene
        cos_sims_full_df["N_Detections"] = [
            n_detections_dict[token] for token in cos_sims_full_df["Gene"]
        ]

        # sort by shift to desired state
        cos_sims_full_df["Sig"] = [
            1 if fdr < 0.05 else 0 for fdr in cos_sims_full_df["Goal_end_FDR"]
        ]
        cos_sims_full_df = cos_sims_full_df.sort_values(
            by=["Sig", "Shift_to_goal_end", "Goal_end_FDR"],
            ascending=[False, False, True],
        )

        return cos_sims_full_df


# stats comparing cos sim shifts of test perturbations vs null distribution
def isp_stats_vs_null(cos_sims_df, dict_list, null_dict_list):
    cos_sims_full_df = cos_sims_df.copy()

    cos_sims_full_df["Test_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Null_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_vs_null_avg_shift"] = np.zeros(
        cos_sims_df.shape[0], dtype=float
    )
    cos_sims_full_df["Test_vs_null_pval"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_vs_null_FDR"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["N_Detections_test"] = np.zeros(
        cos_sims_df.shape[0], dtype="uint32"
    )
    cos_sims_full_df["N_Detections_null"] = np.zeros(
        cos_sims_df.shape[0], dtype="uint32"
    )

    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        test_shifts = []
        null_shifts = []

        for dict_i in dict_list:
            test_shifts += dict_i.get((token, "cell_emb"), [])

        for dict_i in null_dict_list:
            null_shifts += dict_i.get((token, "cell_emb"), [])

        cos_sims_full_df.loc[i, "Test_avg_shift"] = np.mean(test_shifts)
        cos_sims_full_df.loc[i, "Null_avg_shift"] = np.mean(null_shifts)
        cos_sims_full_df.loc[i, "Test_vs_null_avg_shift"] = np.mean(
            test_shifts
        ) - np.mean(null_shifts)
        cos_sims_full_df.loc[i, "Test_vs_null_pval"] = ranksums(
            test_shifts, null_shifts, nan_policy="omit"
        ).pvalue
        # remove nan values
        cos_sims_full_df.Test_vs_null_pval = np.where(
            np.isnan(cos_sims_full_df.Test_vs_null_pval),
            1,
            cos_sims_full_df.Test_vs_null_pval,
        )
        cos_sims_full_df.loc[i, "N_Detections_test"] = len(test_shifts)
        cos_sims_full_df.loc[i, "N_Detections_null"] = len(null_shifts)

    cos_sims_full_df["Test_vs_null_FDR"] = get_fdr(
        cos_sims_full_df["Test_vs_null_pval"]
    )

    cos_sims_full_df["Sig"] = [
        1 if fdr < 0.05 else 0 for fdr in cos_sims_full_df["Test_vs_null_FDR"]
    ]
    cos_sims_full_df = cos_sims_full_df.sort_values(
        by=["Sig", "Test_vs_null_avg_shift", "Test_vs_null_FDR"],
        ascending=[False, False, True],
    )
    return cos_sims_full_df


# stats for identifying perturbations with largest effect within a given set of cells
# fits a mixture model to 2 components (impact vs. non-impact) and
# reports the most likely component for each test perturbation
# Note: because assumes given perturbation has a consistent effect in the cells tested,
# we recommend only using the mixture model strategy with uniform cell populations
def isp_stats_mixture_model(cos_sims_df, dict_list, combos, anchor_token):
    names = ["Gene", "Gene_name", "Ensembl_ID"]

    if combos == 0:
        names += ["Test_avg_shift"]
    elif combos == 1:
        names += [
            "Anchor_shift",
            "Test_token_shift",
            "Sum_of_indiv_shifts",
            "Combo_shift",
            "Combo_minus_sum_shift",
        ]

    names += ["Impact_component", "Impact_component_percent"]

    cos_sims_full_df = pd.DataFrame(columns=names)
    avg_values = []
    gene_names = []

    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        name = cos_sims_df["Gene_name"][i]
        ensembl_id = cos_sims_df["Ensembl_ID"][i]
        cos_shift_data = []

        for dict_i in dict_list:
            if (combos == 0) and (anchor_token is not None):
                cos_shift_data += dict_i.get((anchor_token, token), [])
            else:
                cos_shift_data += dict_i.get((token, "cell_emb"), [])

        # Extract values for current gene
        if combos == 0:
            test_values = cos_shift_data
        elif combos == 1:
            test_values = []
            for tup in cos_shift_data:
                test_values.append(tup[2])

        if len(test_values) > 0:
            avg_value = np.mean(test_values)
            avg_values.append(avg_value)
            gene_names.append(name)

    # fit Gaussian mixture model to dataset of mean for each gene
    avg_values_to_fit = np.array(avg_values).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(avg_values_to_fit)

    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        name = cos_sims_df["Gene_name"][i]
        ensembl_id = cos_sims_df["Ensembl_ID"][i]
        cos_shift_data = []

        for dict_i in dict_list:
            if (combos == 0) and (anchor_token is not None):
                cos_shift_data += dict_i.get((anchor_token, token), [])
            else:
                cos_shift_data += dict_i.get((token, "cell_emb"), [])

        if combos == 0:
            mean_test = np.mean(cos_shift_data)
            impact_components = [
                get_impact_component(value, gm) for value in cos_shift_data
            ]
        elif combos == 1:
            anchor_cos_sim_megalist = [
                anchor for anchor, token, combo in cos_shift_data
            ]
            token_cos_sim_megalist = [token for anchor, token, combo in cos_shift_data]
            anchor_plus_token_cos_sim_megalist = [
                1 - ((1 - anchor) + (1 - token))
                for anchor, token, combo in cos_shift_data
            ]
            combo_anchor_token_cos_sim_megalist = [
                combo for anchor, token, combo in cos_shift_data
            ]
            combo_minus_sum_cos_sim_megalist = [
                combo - (1 - ((1 - anchor) + (1 - token)))
                for anchor, token, combo in cos_shift_data
            ]

            mean_anchor = np.mean(anchor_cos_sim_megalist)
            mean_token = np.mean(token_cos_sim_megalist)
            mean_sum = np.mean(anchor_plus_token_cos_sim_megalist)
            mean_test = np.mean(combo_anchor_token_cos_sim_megalist)
            mean_combo_minus_sum = np.mean(combo_minus_sum_cos_sim_megalist)

            impact_components = [
                get_impact_component(value, gm)
                for value in combo_anchor_token_cos_sim_megalist
            ]

        impact_component = get_impact_component(mean_test, gm)
        impact_component_percent = np.mean(impact_components) * 100

        data_i = [token, name, ensembl_id]
        if combos == 0:
            data_i += [mean_test]
        elif combos == 1:
            data_i += [
                mean_anchor,
                mean_token,
                mean_sum,
                mean_test,
                mean_combo_minus_sum,
            ]
        data_i += [impact_component, impact_component_percent]

        cos_sims_df_i = pd.DataFrame(dict(zip(names, data_i)), index=[i])
        cos_sims_full_df = pd.concat([cos_sims_full_df, cos_sims_df_i])

    # quantify number of detections of each gene
    if anchor_token is None:
        cos_sims_full_df["N_Detections"] = [
            n_detections(i, dict_list, "cell", anchor_token)
            for i in cos_sims_full_df["Gene"]
        ]
    else:
        cos_sims_full_df["N_Detections"] = [
            n_detections(i, dict_list, "gene", anchor_token)
            for i in cos_sims_full_df["Gene"]
        ]

    if combos == 0:
        cos_sims_full_df = cos_sims_full_df.sort_values(
            by=["Impact_component", "Test_avg_shift"], ascending=[False, True]
        )
    elif combos == 1:
        cos_sims_full_df = cos_sims_full_df.sort_values(
            by=["Impact_component", "Combo_minus_sum_shift"], ascending=[False, True]
        )
    return cos_sims_full_df


class InSilicoPerturberStats:
    valid_option_dict = {
        "mode": {
            "goal_state_shift",
            "vs_null",
            "mixture_model",
            "aggregate_data",
            "aggregate_gene_shifts",
        },
        "genes_perturbed": {"all", list},
        "combos": {0, 1},
        "anchor_gene": {None, str},
        "cell_states_to_model": {None, dict},
        "pickle_suffix": {None, str},
        "model_version": {"V1", "V2"},
    }

    def __init__(
        self,
        mode="mixture_model",
        genes_perturbed="all",
        combos=0,
        anchor_gene=None,
        cell_states_to_model=None,
        pickle_suffix="_raw.pickle",
        model_version="V2",
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_name_id_dictionary_file=ENSEMBL_DICTIONARY_FILE,
    ):
        """
        Initialize in silico perturber stats generator.

        **Parameters:**

        mode : {"goal_state_shift", "vs_null", "mixture_model", "aggregate_data", "aggregate_gene_shifts"}
            | Type of stats.
            | "goal_state_shift": perturbation vs. random for desired cell state shift
            | "vs_null": perturbation vs. null from provided null distribution dataset
            | "mixture_model": perturbation in impact vs. no impact component of mixture model (no goal direction)
            | "aggregate_data": aggregates cosine shifts for single perturbation in multiple cells
            | "aggregate_gene_shifts": aggregates cosine shifts of genes in response to perturbation(s)
        genes_perturbed : "all", list
            | Genes perturbed in isp experiment.
            | Default is assuming genes_to_perturb in isp experiment was "all" (each gene in each cell).
            | Otherwise, may provide a list of ENSEMBL IDs of genes perturbed as a group all together.
        combos : {0,1,2}
            | Whether genex perturbed in isp experiment were perturbed individually (0), in pairs (1), or in triplets (2).
        anchor_gene : None, str
            | ENSEMBL ID of gene to use as anchor in combination perturbations or in testing effect on downstream genes.
            | For example, if combos=1 and anchor_gene="ENSG00000136574":
            |    analyzes data for anchor gene perturbed in combination with each other gene.
            | However, if combos=0 and anchor_gene="ENSG00000136574":
            |    analyzes data for the effect of anchor gene's perturbation on the embedding of each other gene.
        cell_states_to_model : None, dict
            | Cell states to model if testing perturbations that achieve goal state change.
            | Four-item dictionary with keys: state_key, start_state, goal_state, and alt_states
            | state_key: key specifying name of column in .dataset that defines the start/goal states
            | start_state: value in the state_key column that specifies the start state
            | goal_state: value in the state_key column taht specifies the goal end state
            | alt_states: list of values in the state_key column that specify the alternate end states
            | For example: {"state_key": "disease",
            |               "start_state": "dcm",
            |               "goal_state": "nf",
            |               "alt_states": ["hcm", "other1", "other2"]}
        model_version : str
            | To auto-select settings for model version other than current default.
            | Current options: V1: models pretrained on ~30M cells, V2: models pretrained on ~104M cells
        token_dictionary_file : Path
            | Path to pickle file containing token dictionary (Ensembl ID:token).
        gene_name_id_dictionary_file : Path
            | Path to pickle file containing gene name to ID dictionary (gene name:Ensembl ID).
        """

        self.mode = mode
        self.genes_perturbed = genes_perturbed
        self.combos = combos
        self.anchor_gene = anchor_gene
        self.cell_states_to_model = cell_states_to_model
        self.pickle_suffix = pickle_suffix
        self.model_version = model_version

        self.validate_options()

        if self.model_version == "V1":
            from . import ENSEMBL_DICTIONARY_FILE_30M, TOKEN_DICTIONARY_FILE_30M
            token_dictionary_file=TOKEN_DICTIONARY_FILE_30M
            gene_name_id_dictionary_file=ENSEMBL_DICTIONARY_FILE_30M

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # load gene name dictionary (gene name:Ensembl ID)
        with open(gene_name_id_dictionary_file, "rb") as f:
            self.gene_name_id_dict = pickle.load(f)

        if anchor_gene is None:
            self.anchor_token = None
        else:
            self.anchor_token = self.gene_token_dict[self.anchor_gene]

    def validate_options(self):
        for attr_name, valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_name in {"anchor_gene"}:
                    continue
                elif attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [str, int, list, dict]) and isinstance(
                    attr_value, option
                ):
                    valid_type = True
                    break
            if not valid_type:
                logger.error(
                    f"Invalid option for {attr_name}. "
                    f"Valid options for {attr_name}: {valid_options}"
                )
                raise

        if self.cell_states_to_model is not None:
            if len(self.cell_states_to_model.items()) == 1:
                logger.warning(
                    "The single value dictionary for cell_states_to_model will be "
                    "replaced with a dictionary with named keys for start, goal, and alternate states. "
                    "Please specify state_key, start_state, goal_state, and alt_states "
                    "in the cell_states_to_model dictionary for future use. "
                    "For example, cell_states_to_model={"
                    "'state_key': 'disease', "
                    "'start_state': 'dcm', "
                    "'goal_state': 'nf', "
                    "'alt_states': ['hcm', 'other1', 'other2']}"
                )
                for key, value in self.cell_states_to_model.items():
                    if (len(value) == 3) and isinstance(value, tuple):
                        if (
                            isinstance(value[0], list)
                            and isinstance(value[1], list)
                            and isinstance(value[2], list)
                        ):
                            if len(value[0]) == 1 and len(value[1]) == 1:
                                all_values = value[0] + value[1] + value[2]
                                if len(all_values) == len(set(all_values)):
                                    continue
                # reformat to the new named key format
                state_values = flatten_list(list(self.cell_states_to_model.values()))
                self.cell_states_to_model = {
                    "state_key": list(self.cell_states_to_model.keys())[0],
                    "start_state": state_values[0][0],
                    "goal_state": state_values[1][0],
                    "alt_states": state_values[2:][0],
                }
            elif set(self.cell_states_to_model.keys()) == {
                "state_key",
                "start_state",
                "goal_state",
                "alt_states",
            }:
                if (
                    (self.cell_states_to_model["state_key"] is None)
                    or (self.cell_states_to_model["start_state"] is None)
                    or (self.cell_states_to_model["goal_state"] is None)
                ):
                    logger.error(
                        "Please specify 'state_key', 'start_state', and 'goal_state' in cell_states_to_model."
                    )
                    raise

                if (
                    self.cell_states_to_model["start_state"]
                    == self.cell_states_to_model["goal_state"]
                ):
                    logger.error("All states must be unique.")
                    raise

                if self.cell_states_to_model["alt_states"] is not None:
                    if not isinstance(self.cell_states_to_model["alt_states"], list):
                        logger.error(
                            "self.cell_states_to_model['alt_states'] must be a list (even if it is one element)."
                        )
                        raise
                    if len(self.cell_states_to_model["alt_states"]) != len(
                        set(self.cell_states_to_model["alt_states"])
                    ):
                        logger.error("All states must be unique.")
                        raise

            elif set(self.cell_states_to_model.keys()) == {
                "state_key",
                "start_state",
                "goal_state",
            }:
                self.cell_states_to_model["alt_states"] = []
            else:
                logger.error(
                    "cell_states_to_model must only have the following four keys: "
                    "'state_key', 'start_state', 'goal_state', 'alt_states'."
                    "For example, cell_states_to_model={"
                    "'state_key': 'disease', "
                    "'start_state': 'dcm', "
                    "'goal_state': 'nf', "
                    "'alt_states': ['hcm', 'other1', 'other2']}"
                )
                raise

            if self.anchor_gene is not None:
                self.anchor_gene = None
                logger.warning(
                    "anchor_gene set to None. "
                    "Currently, anchor gene not available "
                    "when modeling multiple cell states."
                )

        if self.combos > 0:
            if self.anchor_gene is None:
                logger.error(
                    "Currently, stats are only supported for combination "
                    "in silico perturbation run with anchor gene. Please add "
                    "anchor gene when using with combos > 0. "
                )
                raise

        if (self.mode == "mixture_model") and (self.genes_perturbed != "all"):
            logger.error(
                "Mixture model mode requires multiple gene perturbations to fit model "
                "so is incompatible with a single grouped perturbation."
            )
            raise
        if (self.mode == "aggregate_data") and (self.genes_perturbed == "all"):
            logger.error(
                "Simple data aggregation mode is for single perturbation in multiple cells "
                "so is incompatible with a genes_perturbed being 'all'."
            )
            raise

    def get_stats(
        self,
        input_data_directory,
        null_dist_data_directory,
        output_directory,
        output_prefix,
        null_dict_list=None,
    ):
        """
        Get stats for in silico perturbation data and save as results in output_directory.

        **Parameters:**

        input_data_directory : Path
            | Path to directory containing cos_sim dictionary inputs
        null_dist_data_directory : Path
            | Path to directory containing null distribution cos_sim dictionary inputs
        output_directory : Path
            | Path to directory where perturbation data will be saved as .csv
        output_prefix : str
            | Prefix for output .csv
        null_dict_list: list[dict]
            | List of loaded null distribution dictionary if more than one comparison vs. the null is to be performed

        **Outputs:**

        Definition of possible columns in .csv output file.

        | Of note, not all columns will be present in all output files.
        | Some columns are specific to particular perturbation modes.

        | "Gene": gene token
        | "Gene_name": gene name
        | "Ensembl_ID": gene Ensembl ID
        | "N_Detections": number of cells in which each gene or gene combination was detected in the input dataset
        | "Sig": 1 if FDR<0.05, otherwise 0

        | "Shift_to_goal_end": cosine shift from start state towards goal end state in response to given perturbation
        | "Shift_to_alt_end": cosine shift from start state towards alternate end state in response to given perturbation
        | "Goal_end_vs_random_pval": pvalue of cosine shift from start state towards goal end state by Wilcoxon
        |     pvalue compares shift caused by perturbing given gene compared to random genes
        | "Alt_end_vs_random_pval": pvalue of cosine shift from start state towards alternate end state by Wilcoxon
        |     pvalue compares shift caused by perturbing given gene compared to random genes
        | "Goal_end_FDR": Benjamini-Hochberg correction of "Goal_end_vs_random_pval"
        | "Alt_end_FDR": Benjamini-Hochberg correction of "Alt_end_vs_random_pval"

        | "Test_avg_shift": cosine shift in response to given perturbation in cells from test distribution
        | "Null_avg_shift": cosine shift in response to given perturbation in cells from null distribution (e.g. random cells)
        | "Test_vs_null_avg_shift": difference in cosine shift in cells from test vs. null distribution
        |     (i.e. "Test_avg_shift" minus "Null_avg_shift")
        | "Test_vs_null_pval": pvalue of cosine shift in test vs. null distribution
        | "Test_vs_null_FDR": Benjamini-Hochberg correction of "Test_vs_null_pval"
        | "N_Detections_test": "N_Detections" in cells from test distribution
        | "N_Detections_null": "N_Detections" in cells from null distribution

        | "Anchor_shift": cosine shift in response to given perturbation of anchor gene
        | "Test_token_shift": cosine shift in response to given perturbation of test gene
        | "Sum_of_indiv_shifts": sum of cosine shifts in response to individually perturbing test and anchor genes
        | "Combo_shift": cosine shift in response to given perturbation of both anchor and test gene(s) in combination
        | "Combo_minus_sum_shift": difference of cosine shifts in response combo perturbation vs. sum of individual perturbations
        |     (i.e. "Combo_shift" minus "Sum_of_indiv_shifts")
        | "Impact_component": whether the given perturbation was modeled to be within the impact component by the mixture model
        |     1: within impact component; 0: not within impact component
        | "Impact_component_percent": percent of cells in which given perturbation was modeled to be within impact component

        | In case of aggregating data / gene shifts:
        | "Perturbed": ID(s) of gene(s) being perturbed
        | "Affected": ID of affected gene or "cell_emb" indicating the impact on the cell embedding as a whole
        | "Cosine_sim_mean": mean of cosine similarity of cell or affected gene in original vs. perturbed
        | "Cosine_sim_stdev": standard deviation of cosine similarity of cell or affected gene in original vs. perturbed
        """

        if self.mode not in [
            "goal_state_shift",
            "vs_null",
            "mixture_model",
            "aggregate_data",
            "aggregate_gene_shifts",
        ]:
            logger.error(
                "Currently, only modes available are stats for goal_state_shift, "
                "vs_null (comparing to null distribution), "
                "mixture_model (fitting mixture model for perturbations with or without impact), "
                "and aggregating data for single perturbations or for gene embedding shifts."
            )
            raise

        self.gene_token_id_dict = invert_dict(self.gene_token_dict)
        self.gene_id_name_dict = invert_dict(self.gene_name_id_dict)

        # obtain total gene list
        if (self.combos == 0) and (self.anchor_token is not None):
            # cos sim data for effect of gene perturbation on the embedding of each other gene
            dict_list = read_dictionaries(
                input_data_directory,
                "gene",
                self.anchor_token,
                self.cell_states_to_model,
                self.pickle_suffix,
            )
            gene_list = get_gene_list(dict_list, "gene")
        elif (
            (self.combos == 0)
            and (self.anchor_token is None)
            and (self.mode == "aggregate_gene_shifts")
        ):
            dict_list = read_dictionaries(
                input_data_directory,
                "gene",
                self.anchor_token,
                self.cell_states_to_model,
                self.pickle_suffix,
            )
            gene_list = get_gene_list(dict_list, "cell")
        else:
            # cos sim data for effect of gene perturbation on the embedding of each cell
            dict_list = read_dictionaries(
                input_data_directory,
                "cell",
                self.anchor_token,
                self.cell_states_to_model,
                self.pickle_suffix,
            )
            gene_list = get_gene_list(dict_list, "cell")

        # initiate results dataframe
        cos_sims_df_initial = pd.DataFrame(
            {
                "Gene": gene_list,
                "Gene_name": [self.token_to_gene_name(item) for item in gene_list],
                "Ensembl_ID": [
                    token_tuple_to_ensembl_ids(genes, self.gene_token_id_dict)
                    if self.genes_perturbed != "all"
                    else self.gene_token_id_dict[genes[1]]
                    if isinstance(genes, tuple)
                    else self.gene_token_id_dict[genes]
                    for genes in gene_list
                ],
            },
            index=[i for i in range(len(gene_list))],
        )

        if self.mode == "goal_state_shift":
            cos_sims_df = isp_stats_to_goal_state(
                cos_sims_df_initial,
                dict_list,
                self.cell_states_to_model,
                self.genes_perturbed,
            )

        elif self.mode == "vs_null":
            if null_dict_list is None:
                null_dict_list = read_dictionaries(
                    null_dist_data_directory,
                    "cell",
                    self.anchor_token,
                    self.cell_states_to_model,
                    self.pickle_suffix,
                )
            cos_sims_df = isp_stats_vs_null(
                cos_sims_df_initial, dict_list, null_dict_list
            )

        elif self.mode == "mixture_model":
            cos_sims_df = isp_stats_mixture_model(
                cos_sims_df_initial, dict_list, self.combos, self.anchor_token
            )

        elif self.mode == "aggregate_data":
            cos_sims_df = isp_aggregate_grouped_perturb(
                cos_sims_df_initial, dict_list, self.genes_perturbed
            )

        elif self.mode == "aggregate_gene_shifts":
            if (self.genes_perturbed == "all") and (self.combos == 0):
                tuple_types = [
                    True if isinstance(genes, tuple) else False for genes in gene_list
                ]
                if all(tuple_types):
                    token_dtype = "tuple"
                elif not any(tuple_types):
                    token_dtype = "nontuple"
                else:
                    token_dtype = "mix"
            else:
                token_dtype = "mix"

            cos_sims_df = isp_aggregate_gene_shifts(
                cos_sims_df_initial,
                dict_list,
                self.gene_token_id_dict,
                self.gene_id_name_dict,
                token_dtype,
            )

        # save perturbation stats to output_path
        output_path = (Path(output_directory) / output_prefix).with_suffix(".csv")
        cos_sims_df.to_csv(output_path)

    def token_to_gene_name(self, item):
        if np.issubdtype(type(item), np.integer):
            return self.gene_id_name_dict.get(
                self.gene_token_id_dict.get(item, np.nan), np.nan
            )
        if isinstance(item, tuple):
            return tuple(
                [
                    self.gene_id_name_dict.get(
                        self.gene_token_id_dict.get(i, np.nan), np.nan
                    )
                    for i in item
                ]
            )
