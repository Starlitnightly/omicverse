"""
Geneformer in silico perturber stats generator.

Usage:
  from geneformer import InSilicoPerturberStats
  ispstats = InSilicoPerturberStats(mode="goal_state_shift",
                                    combos=0,
                                    anchor_gene=None,
                                    cell_states_to_model={"state_key": "disease", 
                                                          "start_state": "dcm", 
                                                          "goal_state": "nf", 
                                                          "alt_states": ["hcm", "other1", "other2"]})
  ispstats.get_stats("path/to/input_data",
                     None,
                     "path/to/output_directory",
                     "output_prefix")
"""


import os
import logging
import numpy as np
import pandas as pd
import pickle
import random
import statsmodels.stats.multitest as smt
from pathlib import Path
from scipy.stats import ranksums
from sklearn.mixture import GaussianMixture
from tqdm.notebook import trange, tqdm

from .in_silico_perturber import flatten_list

from .tokenizer import TOKEN_DICTIONARY_FILE

GENE_NAME_ID_DICTIONARY_FILE = Path(__file__).parent / "gene_name_id_dict.pkl"

logger = logging.getLogger(__name__)

# invert dictionary keys/values
def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}

# read raw dictionary files
def read_dictionaries(input_data_directory, cell_or_gene_emb, anchor_token):
    file_found = 0
    file_path_list = []
    dict_list = []
    for file in os.listdir(input_data_directory):
        # process only _raw.pickle files
        if file.endswith("_raw.pickle"):
            file_found = 1
            file_path_list += [f"{input_data_directory}/{file}"]
    for file_path in tqdm(file_path_list):
        with open(file_path, "rb") as fp:
            cos_sims_dict = pickle.load(fp)
            if cell_or_gene_emb == "cell":
                cell_emb_dict = {k: v for k,
                                v in cos_sims_dict.items() if v and "cell_emb" in k}
                dict_list += [cell_emb_dict]
            elif cell_or_gene_emb == "gene":
                gene_emb_dict = {k: v for k,
                                v in cos_sims_dict.items() if v and anchor_token == k[0]}  
                dict_list += [gene_emb_dict]
    if file_found == 0:
        logger.error(
                    "No raw data for processing found within provided directory. " \
                    "Please ensure data files end with '_raw.pickle'.")
        raise
    return dict_list

# get complete gene list
def get_gene_list(dict_list,mode):
    if mode == "cell":
        position = 0
    elif mode == "gene":
        position = 1
    gene_set = set()
    for dict_i in dict_list:
        gene_set.update([k[position] for k, v in dict_i.items() if v])
    gene_list = list(gene_set)
    if mode == "gene":
        gene_list.remove("cell_emb")
    gene_list.sort()
    return gene_list

def token_tuple_to_ensembl_ids(token_tuple, gene_token_id_dict):
    return tuple([gene_token_id_dict.get(i, np.nan) for i in token_tuple])

def n_detections(token, dict_list, mode, anchor_token):
    cos_sim_megalist = []
    for dict_i in dict_list:
        if mode == "cell":
            cos_sim_megalist += dict_i.get((token, "cell_emb"),[])
        elif mode == "gene":
            cos_sim_megalist += dict_i.get((anchor_token, token),[])
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
def isp_aggregate_grouped_perturb(cos_sims_df, dict_list):  
    names=["Cosine_shift"]
    cos_sims_full_df = pd.DataFrame(columns=names)

    cos_shift_data = []
    token = cos_sims_df["Gene"][0]
    for dict_i in dict_list:
        cos_shift_data += dict_i.get((token, "cell_emb"),[])
    cos_sims_full_df["Cosine_shift"] = cos_shift_data
    return cos_sims_full_df 

# stats comparing cos sim shifts towards goal state of test perturbations vs random perturbations
def isp_stats_to_goal_state(cos_sims_df, dict_list, cell_states_to_model, genes_perturbed):
    cell_state_key = cell_states_to_model["start_state"]
    if ("alt_states" not in cell_states_to_model.keys()) \
        or (len(cell_states_to_model["alt_states"]) == 0) \
        or (cell_states_to_model["alt_states"] == [None]):
        alt_end_state_exists = False
    elif (len(cell_states_to_model["alt_states"]) > 0) and (cell_states_to_model["alt_states"] != [None]):
        alt_end_state_exists = True
    
    # for single perturbation in multiple cells, there are no random perturbations to compare to
    if genes_perturbed != "all":
        names=["Shift_to_goal_end",
               "Shift_to_alt_end"]
        if alt_end_state_exists == False:
            names.remove("Shift_to_alt_end")
        cos_sims_full_df = pd.DataFrame(columns=names)
        
        cos_shift_data = []
        token = cos_sims_df["Gene"][0]
        for dict_i in dict_list:
            cos_shift_data += dict_i.get((token, "cell_emb"),[])
        if alt_end_state_exists == False:
            cos_sims_full_df["Shift_to_goal_end"] = [goal_end for start_state,goal_end in cos_shift_data] 
        if alt_end_state_exists == True:
            cos_sims_full_df["Shift_to_goal_end"] = [goal_end for start_state,goal_end,alt_end in cos_shift_data] 
            cos_sims_full_df["Shift_to_alt_end"] = [alt_end for start_state,goal_end,alt_end in cos_shift_data]
        
        # sort by shift to desired state
        cos_sims_full_df = cos_sims_full_df.sort_values(by=["Shift_to_goal_end"],
                                                            ascending=[False])
        return cos_sims_full_df     
            
    elif genes_perturbed == "all":
        random_tuples = []
        for i in trange(cos_sims_df.shape[0]):
            token = cos_sims_df["Gene"][i]
            for dict_i in dict_list:
                random_tuples += dict_i.get((token, "cell_emb"),[])

        if alt_end_state_exists == False:
            goal_end_random_megalist = [goal_end for start_state,goal_end in random_tuples]
        elif alt_end_state_exists == True:
            goal_end_random_megalist = [goal_end for start_state,goal_end,alt_end in random_tuples]
            alt_end_random_megalist = [alt_end for start_state,goal_end,alt_end in random_tuples]

        # downsample to improve speed of ranksums
        if len(goal_end_random_megalist) > 100_000:
            random.seed(42)
            goal_end_random_megalist = random.sample(goal_end_random_megalist, k=100_000)
        if alt_end_state_exists == True:
            if len(alt_end_random_megalist) > 100_000:
                random.seed(42)
                alt_end_random_megalist = random.sample(alt_end_random_megalist, k=100_000)

        names=["Gene",
               "Gene_name",
               "Ensembl_ID",
               "Shift_to_goal_end",
               "Shift_to_alt_end",
               "Goal_end_vs_random_pval",
               "Alt_end_vs_random_pval"]
        if alt_end_state_exists == False:
            names.remove("Shift_to_alt_end")
            names.remove("Alt_end_vs_random_pval")
        cos_sims_full_df = pd.DataFrame(columns=names)

        for i in trange(cos_sims_df.shape[0]):
            token = cos_sims_df["Gene"][i]
            name = cos_sims_df["Gene_name"][i]
            ensembl_id = cos_sims_df["Ensembl_ID"][i]
            cos_shift_data = []

            for dict_i in dict_list:
                cos_shift_data += dict_i.get((token, "cell_emb"),[])

            if alt_end_state_exists == False:
                goal_end_cos_sim_megalist = [goal_end for start_state,goal_end in cos_shift_data]    
            elif alt_end_state_exists == True:
                goal_end_cos_sim_megalist = [goal_end for start_state,goal_end,alt_end in cos_shift_data]
                alt_end_cos_sim_megalist = [alt_end for start_state,goal_end,alt_end in cos_shift_data]
                mean_alt_end = np.mean(alt_end_cos_sim_megalist)
                pval_alt_end = ranksums(alt_end_random_megalist,alt_end_cos_sim_megalist).pvalue

            mean_goal_end = np.mean(goal_end_cos_sim_megalist)
            pval_goal_end = ranksums(goal_end_random_megalist,goal_end_cos_sim_megalist).pvalue

            if alt_end_state_exists == False:
                data_i = [token, 
                          name,
                          ensembl_id,
                          mean_goal_end, 
                          pval_goal_end]
            elif alt_end_state_exists == True:
                data_i = [token, 
                          name,
                          ensembl_id,
                          mean_goal_end, 
                          mean_alt_end,
                          pval_goal_end,
                          pval_alt_end]

            cos_sims_df_i = pd.DataFrame(dict(zip(names,data_i)),index=[i])
            cos_sims_full_df = pd.concat([cos_sims_full_df,cos_sims_df_i])

        cos_sims_full_df["Goal_end_FDR"] = get_fdr(list(cos_sims_full_df["Goal_end_vs_random_pval"]))
        if alt_end_state_exists == True:
            cos_sims_full_df["Alt_end_FDR"] = get_fdr(list(cos_sims_full_df["Alt_end_vs_random_pval"]))

        # quantify number of detections of each gene
        cos_sims_full_df["N_Detections"] = [n_detections(i, dict_list, "cell", None) for i in cos_sims_full_df["Gene"]]

        # sort by shift to desired state\
        cos_sims_full_df["Sig"] = [1 if fdr<0.05 else 0 for fdr in cos_sims_full_df["Goal_end_FDR"]]
        cos_sims_full_df = cos_sims_full_df.sort_values(by=["Sig",
                                                            "Shift_to_goal_end",
                                                            "Goal_end_FDR"],
                                                            ascending=[False,False,True])
    
        return cos_sims_full_df

# stats comparing cos sim shifts of test perturbations vs null distribution
def isp_stats_vs_null(cos_sims_df, dict_list, null_dict_list):
    cos_sims_full_df = cos_sims_df.copy()

    cos_sims_full_df["Test_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Null_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_vs_null_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_vs_null_pval"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_vs_null_FDR"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["N_Detections_test"] = np.zeros(cos_sims_df.shape[0], dtype="uint32")
    cos_sims_full_df["N_Detections_null"] = np.zeros(cos_sims_df.shape[0], dtype="uint32")
    
    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        test_shifts = []
        null_shifts = []
        
        for dict_i in dict_list:
            test_shifts += dict_i.get((token, "cell_emb"),[])

        for dict_i in null_dict_list:
            null_shifts += dict_i.get((token, "cell_emb"),[])
        
        cos_sims_full_df.loc[i, "Test_avg_shift"] = np.mean(test_shifts)
        cos_sims_full_df.loc[i, "Null_avg_shift"] = np.mean(null_shifts)
        cos_sims_full_df.loc[i, "Test_vs_null_avg_shift"] = np.mean(test_shifts)-np.mean(null_shifts)       
        cos_sims_full_df.loc[i, "Test_vs_null_pval"] = ranksums(test_shifts,
            null_shifts, nan_policy="omit").pvalue

        cos_sims_full_df.loc[i, "N_Detections_test"] = len(test_shifts)
        cos_sims_full_df.loc[i, "N_Detections_null"] = len(null_shifts)

    cos_sims_full_df["Test_vs_null_FDR"] = get_fdr(cos_sims_full_df["Test_vs_null_pval"])
    
    cos_sims_full_df["Sig"] = [1 if fdr<0.05 else 0 for fdr in cos_sims_full_df["Test_vs_null_FDR"]]  
    cos_sims_full_df = cos_sims_full_df.sort_values(by=["Sig",
                                                        "Test_vs_null_avg_shift",
                                                        "Test_vs_null_FDR"],
                                                        ascending=[False,False,True])
    return cos_sims_full_df

# stats for identifying perturbations with largest effect within a given set of cells
# fits a mixture model to 2 components (impact vs. non-impact) and
# reports the most likely component for each test perturbation
# Note: because assumes given perturbation has a consistent effect in the cells tested,
# we recommend only using the mixture model strategy with uniform cell populations
def isp_stats_mixture_model(cos_sims_df, dict_list, combos, anchor_token):
    
    names=["Gene",
           "Gene_name",
           "Ensembl_ID"]
    
    if combos == 0:
        names += ["Test_avg_shift"]
    elif combos == 1:
        names += ["Anchor_shift",
                  "Test_token_shift",
                  "Sum_of_indiv_shifts",
                  "Combo_shift",
                  "Combo_minus_sum_shift"]
        
    names += ["Impact_component",
              "Impact_component_percent"]

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
                cos_shift_data += dict_i.get((anchor_token, token),[])
            else:
                cos_shift_data += dict_i.get((token, "cell_emb"),[])
            
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
                cos_shift_data += dict_i.get((anchor_token, token),[])
            else:
                cos_shift_data += dict_i.get((token, "cell_emb"),[])
        
        if combos == 0:
            mean_test = np.mean(cos_shift_data)
            impact_components = [get_impact_component(value,gm) for value in cos_shift_data]
        elif combos == 1:
            anchor_cos_sim_megalist = [anchor for anchor,token,combo in cos_shift_data]
            token_cos_sim_megalist = [token for anchor,token,combo in cos_shift_data]
            anchor_plus_token_cos_sim_megalist = [1-((1-anchor)+(1-token)) for anchor,token,combo in cos_shift_data]
            combo_anchor_token_cos_sim_megalist = [combo for anchor,token,combo in cos_shift_data]
            combo_minus_sum_cos_sim_megalist = [combo-(1-((1-anchor)+(1-token))) for anchor,token,combo in cos_shift_data]

            mean_anchor = np.mean(anchor_cos_sim_megalist)
            mean_token = np.mean(token_cos_sim_megalist)
            mean_sum = np.mean(anchor_plus_token_cos_sim_megalist)
            mean_test = np.mean(combo_anchor_token_cos_sim_megalist)
            mean_combo_minus_sum = np.mean(combo_minus_sum_cos_sim_megalist)
            
            impact_components = [get_impact_component(value,gm) for value in combo_anchor_token_cos_sim_megalist]
        
        impact_component = get_impact_component(mean_test,gm)
        impact_component_percent = np.mean(impact_components)*100
            
        data_i = [token, 
                  name, 
                  ensembl_id]
        if combos == 0:
            data_i += [mean_test]
        elif combos == 1:
            data_i += [mean_anchor, 
                       mean_token, 
                       mean_sum, 
                       mean_test,
                       mean_combo_minus_sum]
        data_i += [impact_component,
                   impact_component_percent]
        
        cos_sims_df_i = pd.DataFrame(dict(zip(names,data_i)),index=[i])
        cos_sims_full_df = pd.concat([cos_sims_full_df,cos_sims_df_i])
        
    # quantify number of detections of each gene
    cos_sims_full_df["N_Detections"] = [n_detections(i, 
                                                     dict_list, 
                                                     "gene", 
                                                     anchor_token) for i in cos_sims_full_df["Gene"]]
    
    if combos == 0:
        cos_sims_full_df = cos_sims_full_df.sort_values(by=["Impact_component",
                                                            "Test_avg_shift"],
                                                            ascending=[False,True])    
    elif combos == 1:
        cos_sims_full_df = cos_sims_full_df.sort_values(by=["Impact_component",
                                                            "Combo_minus_sum_shift"],
                                                            ascending=[False,True])
    return cos_sims_full_df

class InSilicoPerturberStats:
    valid_option_dict = {
        "mode": {"goal_state_shift","vs_null","mixture_model","aggregate_data"},
        "combos": {0,1},
        "anchor_gene": {None, str},
        "cell_states_to_model": {None, dict},
    }
    def __init__(
        self,
        mode="mixture_model",
        genes_perturbed="all",
        combos=0,
        anchor_gene=None,
        cell_states_to_model=None,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_name_id_dictionary_file=GENE_NAME_ID_DICTIONARY_FILE,
    ):
        """
        Initialize in silico perturber stats generator.

        Parameters
        ----------
        mode : {"goal_state_shift","vs_null","mixture_model","aggregate_data"}
            Type of stats.
            "goal_state_shift": perturbation vs. random for desired cell state shift
            "vs_null": perturbation vs. null from provided null distribution dataset
            "mixture_model": perturbation in impact vs. no impact component of mixture model (no goal direction)
            "aggregate_data": aggregates cosine shifts for single perturbation in multiple cells
        genes_perturbed : "all", list
            Genes perturbed in isp experiment.
            Default is assuming genes_to_perturb in isp experiment was "all" (each gene in each cell).
            Otherwise, may provide a list of ENSEMBL IDs of genes perturbed as a group all together.
        combos : {0,1,2}
            Whether to perturb genes individually (0), in pairs (1), or in triplets (2).
        anchor_gene : None, str
            ENSEMBL ID of gene to use as anchor in combination perturbations or in testing effect on downstream genes.
            For example, if combos=1 and anchor_gene="ENSG00000136574":
                analyzes data for anchor gene perturbed in combination with each other gene.
            However, if combos=0 and anchor_gene="ENSG00000136574":
                analyzes data for the effect of anchor gene's perturbation on the embedding of each other gene.
        cell_states_to_model: None, dict
            Cell states to model if testing perturbations that achieve goal state change.
            Four-item dictionary with keys: state_key, start_state, goal_state, and alt_states
            state_key: key specifying name of column in .dataset that defines the start/goal states
            start_state: value in the state_key column that specifies the start state
            goal_state: value in the state_key column taht specifies the goal end state
            alt_states: list of values in the state_key column that specify the alternate end states
            For example: {"state_key": "disease",
                          "start_state": "dcm",
                          "goal_state": "nf",
                          "alt_states": ["hcm", "other1", "other2"]}
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl ID:token).
        gene_name_id_dictionary_file : Path
            Path to pickle file containing gene name to ID dictionary (gene name:Ensembl ID).
        """

        self.mode = mode
        self.genes_perturbed = genes_perturbed
        self.combos = combos
        self.anchor_gene = anchor_gene
        self.cell_states_to_model = cell_states_to_model
        
        self.validate_options()

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
        for attr_name,valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_name in {"anchor_gene"}:
                    continue
                elif attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int,list,dict]) and isinstance(attr_value, option):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. " \
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise
        
        if self.cell_states_to_model is not None:
            if len(self.cell_states_to_model.items()) == 1:
                logger.warning(
                    "The single value dictionary for cell_states_to_model will be " \
                    "replaced with a dictionary with named keys for start, goal, and alternate states. " \
                    "Please specify state_key, start_state, goal_state, and alt_states " \
                    "in the cell_states_to_model dictionary for future use. " \
                    "For example, cell_states_to_model={" \
                            "'state_key': 'disease', " \
                            "'start_state': 'dcm', " \
                            "'goal_state': 'nf', " \
                            "'alt_states': ['hcm', 'other1', 'other2']}"
                )
                for key,value in self.cell_states_to_model.items():
                    if (len(value) == 3) and isinstance(value, tuple):
                        if isinstance(value[0],list) and isinstance(value[1],list) and isinstance(value[2],list):
                            if len(value[0]) == 1 and len(value[1]) == 1:
                                all_values = value[0]+value[1]+value[2]
                                if len(all_values) == len(set(all_values)):
                                    continue
                # reformat to the new named key format
                state_values = flatten_list(list(self.cell_states_to_model.values()))
                self.cell_states_to_model = {
                    "state_key": list(self.cell_states_to_model.keys())[0],
                    "start_state": state_values[0][0],
                    "goal_state": state_values[1][0],
                    "alt_states": state_values[2:][0]
                }
            elif set(self.cell_states_to_model.keys()) == {"state_key", "start_state", "goal_state", "alt_states"}:
                if (self.cell_states_to_model["state_key"] is None) \
                    or (self.cell_states_to_model["start_state"] is None) \
                    or (self.cell_states_to_model["goal_state"] is None):
                    logger.error(
                        "Please specify 'state_key', 'start_state', and 'goal_state' in cell_states_to_model.")
                    raise
                
                if self.cell_states_to_model["start_state"] == self.cell_states_to_model["goal_state"]:
                    logger.error(
                        "All states must be unique.")
                    raise

                if self.cell_states_to_model["alt_states"] is not None:
                    if type(self.cell_states_to_model["alt_states"]) is not list:
                        logger.error(
                            "self.cell_states_to_model['alt_states'] must be a list (even if it is one element)."
                        )
                        raise
                    if len(self.cell_states_to_model["alt_states"])!= len(set(self.cell_states_to_model["alt_states"])):
                        logger.error(
                            "All states must be unique.")
                        raise

            else:
                logger.error(
                    "cell_states_to_model must only have the following four keys: " \
                    "'state_key', 'start_state', 'goal_state', 'alt_states'." \
                    "For example, cell_states_to_model={" \
                            "'state_key': 'disease', " \
                            "'start_state': 'dcm', " \
                            "'goal_state': 'nf', " \
                            "'alt_states': ['hcm', 'other1', 'other2']}"
                )
                raise

            if self.anchor_gene is not None:
                self.anchor_gene = None
                logger.warning(
                    "anchor_gene set to None. " \
                    "Currently, anchor gene not available " \
                    "when modeling multiple cell states.")
                
        if self.combos > 0:
            if self.anchor_gene is None:
                logger.error(
                    "Currently, stats are only supported for combination " \
                    "in silico perturbation run with anchor gene. Please add " \
                    "anchor gene when using with combos > 0. ")
                raise
        
        if (self.mode == "mixture_model") and (self.genes_perturbed != "all"):
            logger.error(
                    "Mixture model mode requires multiple gene perturbations to fit model " \
                    "so is incompatible with a single grouped perturbation.")
            raise
        if (self.mode == "aggregate_data") and (self.genes_perturbed == "all"):
            logger.error(
                    "Simple data aggregation mode is for single perturbation in multiple cells " \
                    "so is incompatible with a genes_perturbed being 'all'.")
            raise            

    def get_stats(self,
                  input_data_directory,
                  null_dist_data_directory,
                  output_directory,
                  output_prefix):
        """
        Get stats for in silico perturbation data and save as results in output_directory.

        Parameters
        ----------
        input_data_directory : Path
            Path to directory containing cos_sim dictionary inputs
        null_dist_data_directory : Path
            Path to directory containing null distribution cos_sim dictionary inputs
        output_directory : Path
            Path to directory where perturbation data will be saved as .csv
        output_prefix : str
            Prefix for output .csv
            
        Outputs
        ----------
        Definition of possible columns in .csv output file.
        
        Of note, not all columns will be present in all output files.
        Some columns are specific to particular perturbation modes.
        
        "Gene": gene token
        "Gene_name": gene name
        "Ensembl_ID": gene Ensembl ID
        "N_Detections": number of cells in which each gene or gene combination was detected in the input dataset
        "Sig": 1 if FDR<0.05, otherwise 0
        
        "Shift_to_goal_end": cosine shift from start state towards goal end state in response to given perturbation
        "Shift_to_alt_end": cosine shift from start state towards alternate end state in response to given perturbation
        "Goal_end_vs_random_pval": pvalue of cosine shift from start state towards goal end state by Wilcoxon
            pvalue compares shift caused by perturbing given gene compared to random genes
        "Alt_end_vs_random_pval": pvalue of cosine shift from start state towards alternate end state by Wilcoxon
            pvalue compares shift caused by perturbing given gene compared to random genes
        "Goal_end_FDR": Benjamini-Hochberg correction of "Goal_end_vs_random_pval"
        "Alt_end_FDR": Benjamini-Hochberg correction of "Alt_end_vs_random_pval"
        
        "Test_avg_shift": cosine shift in response to given perturbation in cells from test distribution
        "Null_avg_shift": cosine shift in response to given perturbation in cells from null distribution (e.g. random cells)
        "Test_vs_null_avg_shift": difference in cosine shift in cells from test vs. null distribution
            (i.e. "Test_avg_shift" minus "Null_avg_shift")
        "Test_vs_null_pval": pvalue of cosine shift in test vs. null distribution
        "Test_vs_null_FDR": Benjamini-Hochberg correction of "Test_vs_null_pval"
        "N_Detections_test": "N_Detections" in cells from test distribution
        "N_Detections_null": "N_Detections" in cells from null distribution
        
        "Anchor_shift": cosine shift in response to given perturbation of anchor gene
        "Test_token_shift": cosine shift in response to given perturbation of test gene
        "Sum_of_indiv_shifts": sum of cosine shifts in response to individually perturbing test and anchor genes
        "Combo_shift": cosine shift in response to given perturbation of both anchor and test gene(s) in combination
        "Combo_minus_sum_shift": difference of cosine shifts in response combo perturbation vs. sum of individual perturbations
            (i.e. "Combo_shift" minus "Sum_of_indiv_shifts")
        "Impact_component": whether the given perturbation was modeled to be within the impact component by the mixture model
            1: within impact component; 0: not within impact component
        "Impact_component_percent": percent of cells in which given perturbation was modeled to be within impact component
        """

        if self.mode not in ["goal_state_shift", "vs_null", "mixture_model","aggregate_data"]:
            logger.error(
                "Currently, only modes available are stats for goal_state_shift, " \
                "vs_null (comparing to null distribution), and " \
                "mixture_model (fitting mixture model for perturbations with or without impact.")
            raise

        self.gene_token_id_dict = invert_dict(self.gene_token_dict)
        self.gene_id_name_dict = invert_dict(self.gene_name_id_dict)

        # obtain total gene list
        if (self.combos == 0) and (self.anchor_token is not None):
            # cos sim data for effect of gene perturbation on the embedding of each other gene
            dict_list = read_dictionaries(input_data_directory, "gene", self.anchor_token)
            gene_list = get_gene_list(dict_list, "gene")
        else:
            # cos sim data for effect of gene perturbation on the embedding of each cell
            dict_list = read_dictionaries(input_data_directory, "cell", self.anchor_token)
            gene_list = get_gene_list(dict_list, "cell")
        
        # initiate results dataframe
        cos_sims_df_initial = pd.DataFrame({"Gene": gene_list, 
                                            "Gene_name": [self.token_to_gene_name(item) \
                                                          for item in gene_list], \
                                            "Ensembl_ID": [token_tuple_to_ensembl_ids(genes, self.gene_token_id_dict) \
                                                           if self.genes_perturbed != "all" else \
                                                           self.gene_token_id_dict[genes[1]] \
                                                           if isinstance(genes,tuple) else \
                                                           self.gene_token_id_dict[genes] \
                                                           for genes in gene_list]}, \
                                             index=[i for i in range(len(gene_list))])

        if self.mode == "goal_state_shift":
            cos_sims_df = isp_stats_to_goal_state(cos_sims_df_initial, dict_list, self.cell_states_to_model, self.genes_perturbed)
            
        elif self.mode == "vs_null":
            null_dict_list = read_dictionaries(null_dist_data_directory, "cell", self.anchor_token)
            cos_sims_df = isp_stats_vs_null(cos_sims_df_initial, dict_list, null_dict_list)

        elif self.mode == "mixture_model":
            cos_sims_df = isp_stats_mixture_model(cos_sims_df_initial, dict_list, self.combos, self.anchor_token)
            
        elif self.mode == "aggregate_data":
            cos_sims_df = isp_aggregate_grouped_perturb(cos_sims_df_initial, dict_list)

        # save perturbation stats to output_path
        output_path = (Path(output_directory) / output_prefix).with_suffix(".csv")
        cos_sims_df.to_csv(output_path)

    def token_to_gene_name(self, item):
        if isinstance(item,int):
            return self.gene_id_name_dict.get(self.gene_token_id_dict.get(item, np.nan), np.nan)
        if isinstance(item,tuple):
            return tuple([self.gene_id_name_dict.get(self.gene_token_id_dict.get(i, np.nan), np.nan) for i in item])
