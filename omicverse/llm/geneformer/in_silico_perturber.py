"""
Geneformer in silico perturber.

**Usage:**

.. code-block :: python

    >>> from geneformer import InSilicoPerturber
    >>> isp = InSilicoPerturber(perturb_type="delete",
    ...                         perturb_rank_shift=None,
    ...                         genes_to_perturb="all",
    ...                         model_type="CellClassifier",
    ...                         num_classes=0,
    ...                         emb_mode="cell",
    ...                         filter_data={"cell_type":["cardiomyocyte"]},
    ...                         cell_states_to_model={"state_key": "disease", "start_state": "dcm", "goal_state": "nf", "alt_states": ["hcm", "other1", "other2"]},
    ...                         state_embs_dict ={"nf": emb_nf, "hcm": emb_hcm, "dcm": emb_dcm, "other1": emb_other1, "other2": emb_other2},
    ...                         max_ncells=None,
    ...                         emb_layer=0,
    ...                         forward_batch_size=100,
    ...                         nproc=16)
    >>> isp.perturb_data("path/to/model",
    ...                  "path/to/input_data",
    ...                  "path/to/output_directory",
    ...                  "output_prefix")

**Description:**

| Performs in silico perturbation (e.g. deletion or overexpression) of defined set of genes or all genes in sample of cells.
| Outputs impact of perturbation on cell or gene embeddings.
| Output files are analyzed with ``in_silico_perturber_stats``.

"""

import logging

# imports
import os
import pickle
from collections import defaultdict

import torch
from datasets import Dataset
from multiprocess import set_start_method
from tqdm.auto import trange

from . import TOKEN_DICTIONARY_FILE
from . import perturber_utils as pu
from .emb_extractor import get_embs

import datasets
datasets.logging.disable_progress_bar()


logger = logging.getLogger(__name__)


class InSilicoPerturber:
    valid_option_dict = {
        "perturb_type": {"delete", "overexpress", "inhibit", "activate"},
        "perturb_rank_shift": {None, 1, 2, 3},
        "genes_to_perturb": {"all", list},
        "combos": {0, 1},
        "anchor_gene": {None, str},
        "model_type": {"Pretrained", "GeneClassifier", "CellClassifier", "MTLCellClassifier", "Pretrained-Quantized", "MTLCellClassifier-Quantized"},
        "num_classes": {int},
        "emb_mode": {"cls", "cell", "cls_and_gene", "cell_and_gene"},
        "cell_emb_style": {"mean_pool"},
        "filter_data": {None, dict},
        "cell_states_to_model": {None, dict},
        "state_embs_dict": {None, dict},
        "max_ncells": {None, int},
        "cell_inds_to_perturb": {"all", dict},
        "emb_layer": {-1, 0},
        "model_version": {"V1", "V2"},
        "token_dictionary_file": {None, str},
        "forward_batch_size": {int},
        "nproc": {int},
    }

    def __init__(
        self,
        perturb_type="delete",
        perturb_rank_shift=None,
        genes_to_perturb="all",
        combos=0,
        anchor_gene=None,
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        cell_emb_style="mean_pool",
        filter_data=None,
        cell_states_to_model=None,
        state_embs_dict=None,
        max_ncells=None,
        cell_inds_to_perturb="all",
        emb_layer=-1,
        forward_batch_size=100,
        nproc=4,
        model_version="V2",
        token_dictionary_file=None,
        clear_mem_ncells=1000,
    ):
        """
        Initialize in silico perturber.

        **Parameters:**

        perturb_type : {"delete", "overexpress", "inhibit", "activate"}
            | Type of perturbation.
            | "delete": delete gene from rank value encoding
            | "overexpress": move gene to front of rank value encoding
            | *(TBA)* "inhibit": move gene to lower quartile of rank value encoding
            | *(TBA)* "activate": move gene to higher quartile of rank value encoding
        *(TBA)* perturb_rank_shift : None, {1,2,3}
            | Number of quartiles by which to shift rank of gene.
            | For example, if perturb_type="activate" and perturb_rank_shift=1:
            |     genes in 4th quartile will move to middle of 3rd quartile.
            |     genes in 3rd quartile will move to middle of 2nd quartile.
            |     genes in 2nd quartile will move to middle of 1st quartile.
            |     genes in 1st quartile will move to front of rank value encoding.
            | For example, if perturb_type="inhibit" and perturb_rank_shift=2:
            |     genes in 1st quartile will move to middle of 3rd quartile.
            |     genes in 2nd quartile will move to middle of 4th quartile.
            |     genes in 3rd or 4th quartile will move to bottom of rank value encoding.
        genes_to_perturb : "all", list
            | Default is perturbing each gene detected in each cell in the dataset.
            | Otherwise, may provide a list of ENSEMBL IDs of genes to perturb.
            | If gene list is provided, then perturber will only test perturbing them all together
            | (rather than testing each possible combination of the provided genes).
        combos : {0,1}
            | Whether to perturb genes individually (0) or in pairs (1).
        anchor_gene : None, str
            | ENSEMBL ID of gene to use as anchor in combination perturbations.
            | For example, if combos=1 and anchor_gene="ENSG00000148400":
            |     anchor gene will be perturbed in combination with each other gene.
        model_type : {"Pretrained", "GeneClassifier", "CellClassifier", "MTLCellClassifier", "Pretrained-Quantized", "MTLCellClassifier-Quantized"}
            | Whether model is the pretrained Geneformer or a fine-tuned gene, cell, or multitask cell classifier (+/- 8bit quantization).
        num_classes : int
            | If model is a gene or cell classifier, specify number of classes it was trained to classify.
            | For the pretrained Geneformer model, number of classes is 0 as it is not a classifier.
        emb_mode : {"cls", "cell", "cls_and_gene","cell_and_gene"}
            | Whether to output impact of perturbation on CLS token, cell, and/or gene embeddings.
            | Gene embedding shifts only available as compared to original cell, not comparing to goal state.
        cell_emb_style : "mean_pool"
            | Method for summarizing cell embeddings if not using CLS token.
            | Currently only option is mean pooling of gene embeddings for given cell.
        filter_data : None, dict
            | Default is to use all input data for in silico perturbation study.
            | Otherwise, dictionary specifying .dataset column name and list of values to filter by.
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
        state_embs_dict : None, dict
            | Embedding positions of each cell state to model shifts from/towards (e.g. mean or median).
            | Dictionary with keys specifying each possible cell state to model.
            | Values are target embedding positions as torch.tensor.
            | For example: {"nf": emb_nf,
            |               "hcm": emb_hcm,
            |               "dcm": emb_dcm,
            |               "other1": emb_other1,
            |               "other2": emb_other2}
        max_ncells : None, int
            | Maximum number of cells to test.
            | If None, will test all cells.
        cell_inds_to_perturb : "all", list
            | Default is perturbing each cell in the dataset.
            | Otherwise, may provide a dict of indices of cells to perturb with keys start_ind and end_ind.
            | start_ind: the first index to perturb.
            | end_ind: the last index to perturb (exclusive).
            | Indices will be selected *after* the filter_data criteria and sorting.
            | Useful for splitting extremely large datasets across separate GPUs.
        emb_layer : {-1, 0}
            | Embedding layer to use for quantification.
            | 0: last layer (recommended for questions closely tied to model's training objective)
            | -1: 2nd to last layer (recommended for questions requiring more general representations)
        forward_batch_size : int
            | Batch size for forward pass.
        nproc : int
            | Number of CPU processes to use.
        model_version : str
            | To auto-select settings for model version other than current default.
            | Current options: V1: models pretrained on ~30M cells, V2: models pretrained on ~104M cells
        token_dictionary_file : Path
            | Path to pickle file containing token dictionary (Ensembl ID:token).
        clear_mem_ncells : int
            | Clear memory every n cells.
        """
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        self.perturb_type = perturb_type
        self.perturb_rank_shift = perturb_rank_shift
        self.genes_to_perturb = genes_to_perturb
        self.combos = combos
        self.anchor_gene = anchor_gene
        if self.genes_to_perturb == "all":
            self.perturb_group = False
        else:
            self.perturb_group = True
            if (self.anchor_gene is not None) or (self.combos != 0):
                self.anchor_gene = None
                self.combos = 0
                logger.warning(
                    "anchor_gene set to None and combos set to 0. "
                    "If providing list of genes to perturb, "
                    "list of genes_to_perturb will be perturbed together, "
                    "without anchor gene or combinations."
                )
        self.model_type = model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.cell_emb_style = cell_emb_style
        self.filter_data = filter_data
        self.cell_states_to_model = cell_states_to_model
        self.state_embs_dict = state_embs_dict
        self.max_ncells = max_ncells
        self.cell_inds_to_perturb = cell_inds_to_perturb
        self.emb_layer = emb_layer
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.model_version = model_version
        self.token_dictionary_file = token_dictionary_file
        self.clear_mem_ncells = clear_mem_ncells             

        self.validate_options()

        if self.model_version == "V1":
            from . import TOKEN_DICTIONARY_FILE_30M
            self.token_dictionary_file = TOKEN_DICTIONARY_FILE_30M
            if self.emb_mode == "cls":
                self.emb_mode = "cell"
                logger.warning(
                    "model_version selected as V1 so changing emb_mode from 'cls' to 'cell' as V1 models do not have a <cls> token."
                )                
            if self.emb_mode == "cls_and_gene":
                self.emb_mode = "cell_and_gene"
                logger.warning(
                    "model_version selected as V1 so changing emb_mode from 'cls_and_gene' to 'cell_and_gene' as V1 models do not have a <cls> token."
                )
        
        # load token dictionary (Ensembl IDs:token)
        if self.token_dictionary_file is None:
            self.token_dictionary_file = TOKEN_DICTIONARY_FILE
        with open(self.token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)
        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}

        self.pad_token_id = self.gene_token_dict.get("<pad>")
        self.cls_token_id = self.gene_token_dict.get("<cls>")
        self.eos_token_id = self.gene_token_dict.get("<eos>")

        # Identify if special token is present in the token dictionary
        if (self.cls_token_id is not None) and (self.eos_token_id is not None):
            self.special_token = True
        else:
            if "cls" in self.emb_mode:
                logger.error(
                    f"emb_mode set to {self.emb_mode} but <cls> or <eos> token not in token dictionary."
                )
                raise
            self.special_token = False

        if self.anchor_gene is None:
            self.anchor_token = None
        else:
            try:
                self.anchor_token = [self.gene_token_dict[self.anchor_gene]]
            except KeyError:
                logger.error(f"Anchor gene {self.anchor_gene} not in token dictionary.")
                raise

        if self.genes_to_perturb == "all":
            self.tokens_to_perturb = "all"
        else:
            missing_genes = [
                gene
                for gene in self.genes_to_perturb
                if gene not in self.gene_token_dict.keys()
            ]
            if len(missing_genes) == len(self.genes_to_perturb):
                logger.error(
                    "None of the provided genes to perturb are in token dictionary."
                )
                raise
            elif len(missing_genes) > 0:
                logger.warning(
                    f"Genes to perturb {missing_genes} are not in token dictionary."
                )
            self.tokens_to_perturb = [
                self.gene_token_dict.get(gene) for gene in self.genes_to_perturb
            ]

    def validate_options(self):
        # first disallow options under development
        if self.perturb_type in ["inhibit", "activate"]:
            logger.error(
                "In silico inhibition and activation currently under development. "
                "Current valid options for 'perturb_type': 'delete' or 'overexpress'"
            )
            raise
        if (self.combos > 0) and (self.anchor_gene is None):
            logger.error(
                "Combination perturbation without anchor gene is currently under development. "
                "Currently, must provide anchor gene for combination perturbation."
            )
            raise

        # confirm arguments are within valid options and compatible with each other
        for attr_name, valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_value in valid_options:
                    continue
                if attr_name in ["anchor_gene"]:
                    if type(attr_name) in {str}:
                        continue
            valid_type = False
            for option in valid_options:
                if (option in [bool, int, list, dict, str]) and isinstance(
                    attr_value, option
                ):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. "
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise

        if self.perturb_type in ["delete", "overexpress"]:
            if self.perturb_rank_shift is not None:
                if self.perturb_type == "delete":
                    logger.warning(
                        "perturb_rank_shift set to None. "
                        "If perturb type is delete then gene is deleted entirely "
                        "rather than shifted by quartile"
                    )
                elif self.perturb_type == "overexpress":
                    logger.warning(
                        "perturb_rank_shift set to None. "
                        "If perturb type is overexpress then gene is moved to front "
                        "of rank value encoding rather than shifted by quartile"
                    )
            self.perturb_rank_shift = None

        if (self.anchor_gene is not None) and (self.emb_mode == "cell_and_gene"):
            self.emb_mode = "cell"
            logger.warning(
                "emb_mode set to 'cell'. "
                "Currently, analysis with anchor gene "
                "only outputs effect on cell embeddings."
            )

        if self.cell_states_to_model is not None:
            pu.validate_cell_states_to_model(self.cell_states_to_model)

            if self.anchor_gene is not None:
                self.anchor_gene = None
                logger.warning(
                    "anchor_gene set to None. "
                    "Currently, anchor gene not available "
                    "when modeling multiple cell states."
                )

            if self.state_embs_dict is None:
                logger.error(
                    "state_embs_dict must be provided for mode with cell_states_to_model. "
                    "Format is dictionary with keys specifying each possible cell state to model. "
                    "Values are target embedding positions as torch.tensor."
                )
                raise

            for state_emb in self.state_embs_dict.values():
                if not torch.is_tensor(state_emb):
                    logger.error(
                        "state_embs_dict must be dictionary with values being torch.tensor."
                    )
                    raise

            keys_absent = []
            for k, v in self.cell_states_to_model.items():
                if (k == "start_state") or (k == "goal_state"):
                    if v not in self.state_embs_dict.keys():
                        keys_absent.append(v)
                if k == "alt_states":
                    for state in v:
                        if state not in self.state_embs_dict.keys():
                            keys_absent.append(state)
            if len(keys_absent) > 0:
                logger.error(
                    "Each start_state, goal_state, and alt_states in cell_states_to_model "
                    "must be a key in state_embs_dict with the value being "
                    "the state's embedding position as torch.tensor. "
                    f"Missing keys: {keys_absent}"
                )
                raise

        if self.perturb_type in ["inhibit", "activate"]:
            if self.perturb_rank_shift is None:
                logger.error(
                    "If perturb_type is inhibit or activate then "
                    "quartile to shift by must be specified."
                )
                raise

        if self.filter_data is not None:
            for key, value in self.filter_data.items():
                if not isinstance(value, list):
                    self.filter_data[key] = [value]
                    logger.warning(
                        "Values in filter_data dict must be lists. "
                        f"Changing {key} value to list ([{value}])."
                    )

        if self.cell_inds_to_perturb != "all":
            if set(self.cell_inds_to_perturb.keys()) != {"start", "end"}:
                logger.error(
                    "If cell_inds_to_perturb is a dictionary, keys must be 'start' and 'end'."
                )
                raise
            if (
                self.cell_inds_to_perturb["start"] < 0
                or self.cell_inds_to_perturb["end"] < 0
            ):
                logger.error("cell_inds_to_perturb must be positive.")
                raise

    def perturb_data(
        self, model_directory, input_data_file, output_directory, output_prefix
    ):
        """
        Perturb genes in input data and save as results in output_directory.

        **Parameters:**

        model_directory : Path
            | Path to directory containing model
        input_data_file : Path
            | Path to directory containing .dataset inputs
        output_directory : Path
            | Path to directory where perturbation data will be saved as batched pickle files
        output_prefix : str
            | Prefix for output files
        """

        ### format output path ###
        output_path_prefix = os.path.join(
            output_directory, f"in_silico_{self.perturb_type}_{output_prefix}"
        )

        ### load model and define parameters ###
        model = pu.load_model(
            self.model_type, self.num_classes, model_directory, mode="eval"
        )
        self.max_len = pu.get_model_input_size(model)
        layer_to_quant = pu.quant_layers(model) + self.emb_layer

        ### filter input data ###
        # general filtering of input data based on filter_data argument
        filtered_input_data = pu.load_and_filter(
            self.filter_data, self.nproc, input_data_file
        )

        # Ensure emb_mode is cls if first token of the filtered input data is cls token
        if self.special_token:
            if (filtered_input_data["input_ids"][0][0] == self.cls_token_id) and (
                "cls" not in self.emb_mode
            ):
                logger.error(
                    "Emb mode 'cls' or 'cls_and_gene' required when first token is <cls>."
                )
                raise
            if "cls" in self.emb_mode:
                if (filtered_input_data["input_ids"][0][0] != self.cls_token_id) or (
                    filtered_input_data["input_ids"][0][-1] != self.eos_token_id
                ):
                    logger.error(
                        "Emb mode 'cls' and 'cls_and_gene' require that first token is <cls> and last token is <eos>."
                    )
                    raise

        filtered_input_data = self.apply_additional_filters(filtered_input_data)

        if self.perturb_group is True:
            if (self.special_token) and ("cls" in self.emb_mode):
                self.isp_perturb_set_special(
                    model, filtered_input_data, layer_to_quant, output_path_prefix
                )
            else:
                self.isp_perturb_set(
                    model, filtered_input_data, layer_to_quant, output_path_prefix
                )
        else:
            if (self.special_token) and ("cls" in self.emb_mode):
                self.isp_perturb_all_special(
                    model, filtered_input_data, layer_to_quant, output_path_prefix
                )
            else:
                self.isp_perturb_all(
                    model, filtered_input_data, layer_to_quant, output_path_prefix
                )

    def apply_additional_filters(self, filtered_input_data):
        # additional filtering of input data dependent on isp mode
        if self.cell_states_to_model is not None:
            # filter for cells with start_state and log result
            filtered_input_data = pu.filter_data_by_start_state(
                filtered_input_data, self.cell_states_to_model, self.nproc
            )

        if (self.tokens_to_perturb != "all") and (self.perturb_type != "overexpress"):
            # filter for cells with tokens_to_perturb and log result
            filtered_input_data = pu.filter_data_by_tokens_and_log(
                filtered_input_data,
                self.tokens_to_perturb,
                self.nproc,
                "genes_to_perturb",
            )

        if self.anchor_token is not None:
            # filter for cells with anchor gene and log result
            filtered_input_data = pu.filter_data_by_tokens_and_log(
                filtered_input_data, self.anchor_token, self.nproc, "anchor_gene"
            )

        # downsample and sort largest to smallest to encounter memory constraints earlier
        filtered_input_data = pu.downsample_and_sort(
            filtered_input_data, self.max_ncells
        )

        # slice dataset if cells_inds_to_perturb is not "all"
        if self.cell_inds_to_perturb != "all":
            filtered_input_data = pu.slice_by_inds_to_perturb(
                filtered_input_data, self.cell_inds_to_perturb
            )

        return filtered_input_data

    def isp_perturb_set(
        self,
        model,
        filtered_input_data: Dataset,
        layer_to_quant: int,
        output_path_prefix: str,
    ):
        def make_group_perturbation_batch(example):
            example_input_ids = example["input_ids"]
            example["tokens_to_perturb"] = self.tokens_to_perturb
            indices_to_perturb = [
                example_input_ids.index(token) if token in example_input_ids else None
                for token in self.tokens_to_perturb
            ]
            indices_to_perturb = [
                item for item in indices_to_perturb if item is not None
            ]
            if len(indices_to_perturb) > 0:
                example["perturb_index"] = indices_to_perturb
            else:
                # -100 indicates tokens to overexpress are not present in rank value encoding
                example["perturb_index"] = [-100]
            if self.perturb_type == "delete":
                example = pu.delete_indices(example)
            elif self.perturb_type == "overexpress":
                example = pu.overexpress_tokens(
                    example, self.max_len, self.special_token
                )
                example["n_overflow"] = pu.calc_n_overflow(
                    self.max_len,
                    example["length"],
                    self.tokens_to_perturb,
                    indices_to_perturb,
                )
            return example

        total_batch_length = len(filtered_input_data)
        if self.cell_states_to_model is None:
            cos_sims_dict = defaultdict(list)
        else:
            cos_sims_dict = {
                state: defaultdict(list)
                for state in pu.get_possible_states(self.cell_states_to_model)
            }

        perturbed_data = filtered_input_data.map(
            make_group_perturbation_batch, num_proc=self.nproc
        )

        if self.perturb_type == "overexpress":
            filtered_input_data = filtered_input_data.add_column(
                "n_overflow", perturbed_data["n_overflow"]
            )
            # remove overflow genes from original data so that embeddings are comparable
            # i.e. if original cell has genes 0:2047 and you want to overexpress new gene 2048,
            # then the perturbed cell will be 2048+0:2046 so we compare it to an original cell 0:2046.
            # (otherwise we will be modeling the effect of both deleting 2047 and adding 2048,
            # rather than only adding 2048)
            filtered_input_data = filtered_input_data.map(
                pu.truncate_by_n_overflow, num_proc=self.nproc
            )

        if self.emb_mode == "cell_and_gene":
            stored_gene_embs_dict = defaultdict(list)

        # iterate through batches
        for i in trange(0, total_batch_length, self.forward_batch_size):
            max_range = min(i + self.forward_batch_size, total_batch_length)
            inds_select = [i for i in range(i, max_range)]

            minibatch = filtered_input_data.select(inds_select)
            perturbation_batch = perturbed_data.select(inds_select)

            if self.cell_emb_style == "mean_pool":
                full_original_emb = get_embs(
                    model,
                    minibatch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    token_gene_dict=self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )
                indices_to_perturb = perturbation_batch["perturb_index"]
                # remove indices that were perturbed
                original_emb = pu.remove_perturbed_indices_set(
                    full_original_emb,
                    self.perturb_type,
                    indices_to_perturb,
                    self.tokens_to_perturb,
                    minibatch["length"],
                )
                full_perturbation_emb = get_embs(
                    model,
                    perturbation_batch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    token_gene_dict=self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )

                # remove overexpressed genes
                if self.perturb_type == "overexpress":
                    perturbation_emb = full_perturbation_emb[
                        :, len(self.tokens_to_perturb) :, :
                    ]

                elif self.perturb_type == "delete":
                    perturbation_emb = full_perturbation_emb[
                        :, : max(perturbation_batch["length"]), :
                    ]

                n_perturbation_genes = perturbation_emb.size()[1]

                # if no goal states, the cosine similarties are the mean of gene cosine similarities
                if (
                    self.cell_states_to_model is None
                    or self.emb_mode == "cell_and_gene"
                ):
                    gene_cos_sims = pu.quant_cos_sims(
                        perturbation_emb,
                        original_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="gene",
                    )

                # if there are goal states, the cosine similarities are the cell cosine similarities
                if self.cell_states_to_model is not None:
                    original_cell_emb = pu.mean_nonpadding_embs(
                        full_original_emb,
                        torch.tensor(minibatch["length"], device="cuda"),
                        dim=1,
                    )
                    perturbation_cell_emb = pu.mean_nonpadding_embs(
                        full_perturbation_emb,
                        torch.tensor(perturbation_batch["length"], device="cuda"),
                        dim=1,
                    )
                    cell_cos_sims = pu.quant_cos_sims(
                        perturbation_cell_emb,
                        original_cell_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="cell",
                    )

                # get cosine similarities in gene embeddings
                # if getting gene embeddings, need gene names
                if self.emb_mode == "cell_and_gene":
                    gene_list = minibatch["input_ids"]
                    # need to truncate gene_list
                    gene_list = [
                        [g for g in genes if g not in self.tokens_to_perturb][
                            :n_perturbation_genes
                        ]
                        for genes in gene_list
                    ]

                    for cell_i, genes in enumerate(gene_list):
                        for gene_j, affected_gene in enumerate(genes):
                            if len(self.genes_to_perturb) > 1:
                                tokens_to_perturb = tuple(self.tokens_to_perturb)
                            else:
                                tokens_to_perturb = self.tokens_to_perturb[0]

                            # fill in the gene cosine similarities
                            try:
                                stored_gene_embs_dict[
                                    (tokens_to_perturb, affected_gene)
                                ].append(gene_cos_sims[cell_i, gene_j].item())
                            except KeyError:
                                stored_gene_embs_dict[
                                    (tokens_to_perturb, affected_gene)
                                ] = gene_cos_sims[cell_i, gene_j].item()
                else:
                    gene_list = None

            if self.cell_states_to_model is None:
                # calculate the mean of the gene cosine similarities for cell shift
                # tensor of nonpadding lengths for each cell
                if self.perturb_type == "overexpress":
                    # subtract number of genes that were overexpressed
                    # since they are removed before getting cos sims
                    n_overexpressed = len(self.tokens_to_perturb)
                    nonpadding_lens = [
                        x - n_overexpressed for x in perturbation_batch["length"]
                    ]
                else:
                    nonpadding_lens = perturbation_batch["length"]
                cos_sims_data = pu.mean_nonpadding_embs(
                    gene_cos_sims, torch.tensor(nonpadding_lens, device="cuda")
                )
                cos_sims_dict = self.update_perturbation_dictionary(
                    cos_sims_dict,
                    cos_sims_data,
                    gene_list,
                )
            else:
                cos_sims_data = cell_cos_sims
                for state in cos_sims_dict.keys():
                    cos_sims_dict[state] = self.update_perturbation_dictionary(
                        cos_sims_dict[state],
                        cos_sims_data[state],
                        gene_list,
                    )
            del minibatch
            del perturbation_batch
            del original_emb
            del perturbation_emb
            del cos_sims_data

            torch.cuda.empty_cache()

        pu.write_perturbation_dictionary(
            cos_sims_dict,
            f"{output_path_prefix}_cell_embs_dict_{self.tokens_to_perturb}",
        )

        if self.emb_mode == "cell_and_gene":
            pu.write_perturbation_dictionary(
                stored_gene_embs_dict,
                f"{output_path_prefix}_gene_embs_dict_{self.tokens_to_perturb}",
            )

    def isp_perturb_set_special(
        self,
        model,
        filtered_input_data: Dataset,
        layer_to_quant: int,
        output_path_prefix: str,
    ):
        def make_group_perturbation_batch(example):
            example_input_ids = example["input_ids"]
            example["tokens_to_perturb"] = self.tokens_to_perturb
            indices_to_perturb = [
                example_input_ids.index(token) if token in example_input_ids else None
                for token in self.tokens_to_perturb
            ]
            indices_to_perturb = [
                item for item in indices_to_perturb if item is not None
            ]
            if len(indices_to_perturb) > 0:
                example["perturb_index"] = indices_to_perturb
            else:
                # -100 indicates tokens to overexpress are not present in rank value encoding
                example["perturb_index"] = [-100]
            if self.perturb_type == "delete":
                example = pu.delete_indices(example)
            elif self.perturb_type == "overexpress":
                example = pu.overexpress_tokens(
                    example, self.max_len, self.special_token
                )
                example["n_overflow"] = pu.calc_n_overflow(
                    self.max_len,
                    example["length"],
                    self.tokens_to_perturb,
                    indices_to_perturb,
                )
            return example

        total_batch_length = len(filtered_input_data)

    
        if self.cell_states_to_model is None:
            cos_sims_dict = defaultdict(list)
        else:
            cos_sims_dict = {
                state: defaultdict(list)
                for state in pu.get_possible_states(self.cell_states_to_model)
            }

        perturbed_data = filtered_input_data.map(
            make_group_perturbation_batch, num_proc=self.nproc
        )

        if self.perturb_type == "overexpress":
            filtered_input_data = filtered_input_data.add_column(
                "n_overflow", perturbed_data["n_overflow"]
            )
            filtered_input_data = filtered_input_data.map(
                pu.truncate_by_n_overflow_special, num_proc=self.nproc
            )

        if self.emb_mode == "cls_and_gene":
            stored_gene_embs_dict = defaultdict(list)

        # iterate through batches
        for i in trange(0, total_batch_length, self.forward_batch_size):
            max_range = min(i + self.forward_batch_size, total_batch_length)
            inds_select = [i for i in range(i, max_range)]

            minibatch = filtered_input_data.select(inds_select)
            perturbation_batch = perturbed_data.select(inds_select)

            ##### CLS Embedding Mode #####
            if self.emb_mode == "cls":
                indices_to_perturb = perturbation_batch["perturb_index"]

                original_cls_emb = get_embs(
                    model,
                    minibatch,
                    "cls",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    token_gene_dict=self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )

                perturbation_cls_emb = get_embs(
                    model,
                    perturbation_batch,
                    "cls",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    token_gene_dict=self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )

                # Calculate the cosine similarities
                cls_cos_sims = pu.quant_cos_sims(
                    perturbation_cls_emb,
                    original_cls_emb,
                    self.cell_states_to_model,
                    self.state_embs_dict,
                    emb_mode="cell",
                )

                # Update perturbation dictionary
                if self.cell_states_to_model is None:
                    cos_sims_dict = self.update_perturbation_dictionary(
                        cos_sims_dict,
                        cls_cos_sims,
                        gene_list=None,
                    )
                else:
                    for state in cos_sims_dict.keys():
                        cos_sims_dict[state] = self.update_perturbation_dictionary(
                            cos_sims_dict[state],
                            cls_cos_sims[state],
                            gene_list=None,
                        )

            ##### CLS and Gene Embedding Mode #####
            elif self.emb_mode == "cls_and_gene":              
                full_original_emb = get_embs(
                    model,
                    minibatch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )
                indices_to_perturb = perturbation_batch["perturb_index"]

                # remove indices that were perturbed
                original_emb = pu.remove_perturbed_indices_set(
                    full_original_emb,
                    self.perturb_type,
                    indices_to_perturb,
                    self.tokens_to_perturb,
                    minibatch["length"],
                )

                full_perturbation_emb = get_embs(
                    model,
                    perturbation_batch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )
    
                # remove special tokens and padding
                original_emb = original_emb[:, 1:-1, :]
                if self.perturb_type == "overexpress":
                    perturbation_emb = full_perturbation_emb[
                        :, 1 + len(self.tokens_to_perturb) : -1, :
                    ]
                elif self.perturb_type == "delete":
                    perturbation_emb = full_perturbation_emb[
                        :, 1 : max(perturbation_batch["length"]) - 1, :
                    ]
    
                n_perturbation_genes = perturbation_emb.size()[1]

                # truncate the original embedding as necessary
                if self.perturb_type == "overexpress":
                    def calc_perturbation_length(ids):
                        if ids == [-100]:
                            return 0
                        else:
                            return len(ids)

                    max_tensor_size = max([length - calc_perturbation_length(ids) - 2 for length, ids in zip(minibatch["length"], indices_to_perturb)])

                    max_n_overflow = max(minibatch["n_overflow"])
                    if max_n_overflow > 0 and perturbation_emb.size()[1] < original_emb.size()[1]:
                        original_emb = original_emb[:, 0 : perturbation_emb.size()[1], :]
                    elif perturbation_emb.size()[1] < original_emb.size()[1]:
                        original_emb = original_emb[:, 0:max_tensor_size, :]
    
                gene_cos_sims = pu.quant_cos_sims(
                    perturbation_emb,
                    original_emb,
                    self.cell_states_to_model,
                    self.state_embs_dict,
                    emb_mode="gene",
                )

                # get cls emb
                original_cls_emb = full_original_emb[:, 0, :]
                perturbation_cls_emb = full_perturbation_emb[:, 0, :]

                cls_cos_sims = pu.quant_cos_sims(
                    perturbation_cls_emb,
                    original_cls_emb,
                    self.cell_states_to_model,
                    self.state_embs_dict,
                    emb_mode="cell",
                )

                # get cosine similarities in gene embeddings
                # since getting gene embeddings, need gene names

                gene_list = minibatch["input_ids"]
                # need to truncate gene_list
                genes_to_exclude = self.tokens_to_perturb + [
                    self.cls_token_id,
                    self.eos_token_id,
                ]
                gene_list = [
                    [g for g in genes if g not in genes_to_exclude][
                        :n_perturbation_genes
                    ]
                    for genes in gene_list
                ]

                for cell_i, genes in enumerate(gene_list):
                    for gene_j, affected_gene in enumerate(genes):
                        if len(self.genes_to_perturb) > 1:
                            tokens_to_perturb = tuple(self.tokens_to_perturb)
                        else:
                            tokens_to_perturb = self.tokens_to_perturb[0]

                        # fill in the gene cosine similarities
                        try:
                            stored_gene_embs_dict[
                                (tokens_to_perturb, affected_gene)
                            ].append(gene_cos_sims[cell_i, gene_j].item())
                        except KeyError:
                            stored_gene_embs_dict[
                                (tokens_to_perturb, affected_gene)
                            ] = gene_cos_sims[cell_i, gene_j].item()

                if self.cell_states_to_model is None:
                    cos_sims_dict = self.update_perturbation_dictionary(
                        cos_sims_dict,
                        cls_cos_sims,
                        gene_list=None,
                    )
                else:
                    for state in cos_sims_dict.keys():
                        cos_sims_dict[state] = self.update_perturbation_dictionary(
                            cos_sims_dict[state],
                            cls_cos_sims[state],
                            gene_list=None,
                        )
                del full_original_emb
                del original_emb
                del full_perturbation_emb
                del perturbation_emb
                del gene_cos_sims

            del original_cls_emb
            del perturbation_cls_emb
            del cls_cos_sims
            del minibatch
            del perturbation_batch

            torch.cuda.empty_cache()

        pu.write_perturbation_dictionary(
            cos_sims_dict,
            f"{output_path_prefix}_cell_embs_dict_{self.tokens_to_perturb}",
        )

        if self.emb_mode == "cls_and_gene":
            pu.write_perturbation_dictionary(
                stored_gene_embs_dict,
                f"{output_path_prefix}_gene_embs_dict_{self.tokens_to_perturb}",
            )

    def isp_perturb_all(
        self,
        model,
        filtered_input_data: Dataset,
        layer_to_quant: int,
        output_path_prefix: str,
    ):
        pickle_batch = -1
        if self.cell_states_to_model is None:
            cos_sims_dict = defaultdict(list)
        else:
            cos_sims_dict = {
                state: defaultdict(list)
                for state in pu.get_possible_states(self.cell_states_to_model)
            }

        if self.emb_mode == "cell_and_gene":
            stored_gene_embs_dict = defaultdict(list)

        num_inds_perturbed = 1 + self.combos
        for h in trange(len(filtered_input_data)):
            example_cell = filtered_input_data.select([h])
            full_original_emb = get_embs(
                model,
                example_cell,
                "gene",
                layer_to_quant,
                self.pad_token_id,
                self.forward_batch_size,
                self.token_gene_dict,
                summary_stat=None,
                silent=True,
            )

            if self.cell_states_to_model is not None:
                original_cell_emb = pu.compute_nonpadded_cell_embedding(
                    full_original_emb, "mean_pool"
                )

            # gene_list is used to assign cos sims back to genes
            gene_list = example_cell["input_ids"][0][:]
            # need to remove the anchor gene
            if self.anchor_token is not None:
                for token in self.anchor_token:
                    gene_list.remove(token)
            # index 0 is not overexpressed so remove
            if self.perturb_type == "overexpress":
                gene_list = gene_list[num_inds_perturbed:]
            # remove perturbed index for gene list dict
            perturbed_gene_dict = {
                gene: gene_list[:i] + gene_list[i + 1 :]
                for i, gene in enumerate(gene_list)
            }

            perturbation_batch, indices_to_perturb = pu.make_perturbation_batch(
                example_cell,
                self.perturb_type,
                self.tokens_to_perturb,
                self.anchor_token,
                self.combos,
                self.nproc,
            )

            ispall_total_batch_length = len(perturbation_batch)
            for i in trange(
                0, ispall_total_batch_length, self.forward_batch_size, leave=False
            ):
                ispall_max_range = min(
                    i + self.forward_batch_size, ispall_total_batch_length
                )
                perturbation_minibatch = perturbation_batch.select(
                    [i for i in range(i, ispall_max_range)]
                )
                indices_to_perturb_mini = indices_to_perturb[i:ispall_max_range]
                gene_list_mini = gene_list[
                    i:ispall_max_range
                ]  # only perturbed genes from this minibatch

                full_perturbation_emb = get_embs(
                    model,
                    perturbation_minibatch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )

                del perturbation_minibatch

                # need to remove overexpressed gene to quantify cosine shifts
                if self.perturb_type == "overexpress":
                    perturbation_emb = full_perturbation_emb[:, num_inds_perturbed:, :]

                elif self.perturb_type == "delete":
                    perturbation_emb = full_perturbation_emb

                if (
                    self.cell_states_to_model is None
                    or self.emb_mode == "cell_and_gene"
                ):
                    original_emb_minibatch = pu.make_comparison_batch(
                        full_original_emb, indices_to_perturb_mini, perturb_group=False
                    )
                    gene_cos_sims = pu.quant_cos_sims(
                        perturbation_emb,
                        original_emb_minibatch,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="gene",
                    )
                    del original_emb_minibatch

                if self.cell_states_to_model is not None:
                    perturbation_cell_emb = pu.compute_nonpadded_cell_embedding(
                        full_perturbation_emb, "mean_pool"
                    )

                    cell_cos_sims = pu.quant_cos_sims(
                        perturbation_cell_emb,
                        original_cell_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="cell",
                    )
                    del perturbation_cell_emb

                if self.emb_mode == "cell_and_gene":
                    for perturbation_i, perturbed_gene in enumerate(gene_list_mini):
                        for gene_j, affected_gene in enumerate(
                            perturbed_gene_dict[perturbed_gene]
                        ):
                            try:
                                stored_gene_embs_dict[
                                    (perturbed_gene, affected_gene)
                                ].append(gene_cos_sims[perturbation_i, gene_j].item())
                            except KeyError:
                                stored_gene_embs_dict[
                                    (perturbed_gene, affected_gene)
                                ] = gene_cos_sims[perturbation_i, gene_j].item()

                del full_perturbation_emb

                if self.cell_states_to_model is None:
                    cos_sims_data = torch.mean(gene_cos_sims, dim=1)
                    cos_sims_dict = self.update_perturbation_dictionary(
                        cos_sims_dict,
                        cos_sims_data,
                        gene_list_mini,
                    )
                else:
                    cos_sims_data = cell_cos_sims
                    for state in cos_sims_dict.keys():
                        cos_sims_dict[state] = self.update_perturbation_dictionary(
                            cos_sims_dict[state],
                            cos_sims_data[state],
                            gene_list_mini,
                        )

                # save dict to disk every self.clear_mem_ncells/10 (default 100) simulated cells
                if i % self.clear_mem_ncells / 10 == 0:
                    pu.write_perturbation_dictionary(
                        cos_sims_dict,
                        f"{output_path_prefix}_dict_cell_embs_{h}batch{pickle_batch}",
                    )
                    if self.emb_mode == "cell_and_gene":
                        pu.write_perturbation_dictionary(
                            stored_gene_embs_dict,
                            f"{output_path_prefix}_dict_gene_embs_{h}batch{pickle_batch}",
                        )

                # reset and clear memory every self.clear_mem_ncells (default 1000) simulated cells or at the end of the example cell
                if i % self.clear_mem_ncells == 0:
                    pickle_batch += 1
                    if self.cell_states_to_model is None:
                        cos_sims_dict = defaultdict(list)
                    else:
                        cos_sims_dict = {
                            state: defaultdict(list)
                            for state in pu.get_possible_states(
                                self.cell_states_to_model
                            )
                        }

                    if self.emb_mode == "cell_and_gene":
                        stored_gene_embs_dict = defaultdict(list)

                    torch.cuda.empty_cache()

            pu.write_perturbation_dictionary(
                cos_sims_dict,
                f"{output_path_prefix}_dict_cell_embs_{h}batch{pickle_batch}",
            )

            if self.emb_mode == "cell_and_gene":
                pu.write_perturbation_dictionary(
                    stored_gene_embs_dict,
                    f"{output_path_prefix}_dict_gene_embs_{h}batch{pickle_batch}",
                )

            pickle_batch = -1
            if self.cell_states_to_model is None:
                cos_sims_dict = defaultdict(list)
            else:
                cos_sims_dict = {
                    state: defaultdict(list)
                    for state in pu.get_possible_states(self.cell_states_to_model)
                }

            if self.emb_mode == "cell_and_gene":
                stored_gene_embs_dict = defaultdict(list)

            # clear memory between cells
            del perturbation_batch
            del full_original_emb
            if self.cell_states_to_model is not None:
                del original_cell_emb
            torch.cuda.empty_cache()

    def isp_perturb_all_special(
        self,
        model,
        filtered_input_data: Dataset,
        layer_to_quant: int,
        output_path_prefix: str,
    ):
        pickle_batch = -1
        if self.cell_states_to_model is None:
            cos_sims_dict = defaultdict(list)
        else:
            cos_sims_dict = {
                state: defaultdict(list)
                for state in pu.get_possible_states(self.cell_states_to_model)
            }

        if self.emb_mode == "cls_and_gene":
            stored_gene_embs_dict = defaultdict(list)

        num_inds_perturbed = 1 + self.combos
        for h in trange(len(filtered_input_data)):
            example_cell = filtered_input_data.select([h])

            # get original example cell cls and/or gene embs for comparison
            if self.emb_mode == "cls":
                original_cls_emb = get_embs(
                    model,
                    example_cell,
                    "cls",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )
            elif self.emb_mode == "cls_and_gene":
                full_original_emb = get_embs(
                    model,
                    example_cell,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )
                original_cls_emb = full_original_emb[:, 0, :].clone().detach()

            # gene_list is used to assign cos sims back to genes
            gene_list = example_cell["input_ids"][0][:]

            # need to remove special tokens
            for token in [self.cls_token_id, self.eos_token_id]:
                gene_list.remove(token)
            # need to remove the anchor gene
            if self.anchor_token is not None:
                for token in self.anchor_token:
                    gene_list.remove(token)
            # index 0 is not overexpressed so remove
            if self.perturb_type == "overexpress":
                gene_list = gene_list[num_inds_perturbed:]
            # remove perturbed index for gene list dict
            perturbed_gene_dict = {
                gene: gene_list[:i] + gene_list[i + 1 :]
                for i, gene in enumerate(gene_list)
            }

            perturbation_batch, indices_to_perturb = pu.make_perturbation_batch_special(
                example_cell,
                self.perturb_type,
                self.tokens_to_perturb,
                self.anchor_token,
                self.combos,
                self.nproc,
            )

            ispall_total_batch_length = len(perturbation_batch)
            for i in trange(
                0, ispall_total_batch_length, self.forward_batch_size, leave=False
            ):
                ispall_max_range = min(
                    i + self.forward_batch_size, ispall_total_batch_length
                )
                perturbation_minibatch = perturbation_batch.select(
                    [i for i in range(i, ispall_max_range)]
                )
                indices_to_perturb_mini = indices_to_perturb[i:ispall_max_range]
                gene_list_mini = gene_list[
                    i:ispall_max_range
                ]  # only perturbed genes from this minibatch

                ##### CLS Embedding Mode #####
                if self.emb_mode == "cls":
                    # Extract cls embeddings from perturbed cells
                    perturbation_cls_emb = get_embs(
                        model,
                        perturbation_minibatch,
                        "cls",
                        layer_to_quant,
                        self.pad_token_id,
                        self.forward_batch_size,
                        self.token_gene_dict,
                        summary_stat=None,
                        silent=True,
                    )

                    # Calculate cosine similarities
                    cls_cos_sims = pu.quant_cos_sims(
                        perturbation_cls_emb,
                        original_cls_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="cell",
                    )

                    if self.cell_states_to_model is None:
                        cos_sims_dict = self.update_perturbation_dictionary(
                            cos_sims_dict,
                            cls_cos_sims,
                            gene_list_mini,
                        )
                    else:
                        for state in cos_sims_dict.keys():
                            cos_sims_dict[state] = self.update_perturbation_dictionary(
                                cos_sims_dict[state],
                                cls_cos_sims[state],
                                gene_list_mini,
                            )

                    del perturbation_minibatch
                    del perturbation_cls_emb
                    del cls_cos_sims

                ##### CLS and Gene Embedding Mode #####
                elif self.emb_mode == "cls_and_gene":
                    full_perturbation_emb = get_embs(
                        model,
                        perturbation_minibatch,
                        "gene",
                        layer_to_quant,
                        self.pad_token_id,
                        self.forward_batch_size,
                        self.token_gene_dict,
                        summary_stat=None,
                        silent=True,
                    )

                    # need to remove overexpressed gene and cls/eos to quantify cosine shifts
                    if self.perturb_type == "overexpress":
                        perturbation_emb = (
                            full_perturbation_emb[:, 1 + num_inds_perturbed : -1, :]
                            .clone()
                            .detach()
                        )
                    elif self.perturb_type == "delete":
                        perturbation_emb = (
                            full_perturbation_emb[:, 1:-1, :].clone().detach()
                        )

                    original_emb_minibatch = pu.make_comparison_batch(
                        full_original_emb, indices_to_perturb_mini, perturb_group=False
                    )

                    original_emb_minibatch = (
                        original_emb_minibatch[:, 1:-1, :].clone().detach()
                    )
                    gene_cos_sims = pu.quant_cos_sims(
                        perturbation_emb,
                        original_emb_minibatch,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="gene",
                    )

                    for perturbation_i, perturbed_gene in enumerate(gene_list_mini):
                        for gene_j, affected_gene in enumerate(
                            perturbed_gene_dict[perturbed_gene]
                        ):
                            try:
                                stored_gene_embs_dict[
                                    (perturbed_gene, affected_gene)
                                ].append(gene_cos_sims[perturbation_i, gene_j].item())
                            except KeyError:
                                stored_gene_embs_dict[
                                    (perturbed_gene, affected_gene)
                                ] = gene_cos_sims[perturbation_i, gene_j].item()

                    # get cls emb
                    perturbation_cls_emb = (
                        full_perturbation_emb[:, 0, :].clone().detach()
                    )

                    cls_cos_sims = pu.quant_cos_sims(
                        perturbation_cls_emb,
                        original_cls_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="cell",
                    )

                    if self.cell_states_to_model is None:
                        cos_sims_dict = self.update_perturbation_dictionary(
                            cos_sims_dict,
                            cls_cos_sims,
                            gene_list_mini,
                        )
                    else:
                        for state in cos_sims_dict.keys():
                            cos_sims_dict[state] = self.update_perturbation_dictionary(
                                cos_sims_dict[state],
                                cls_cos_sims[state],
                                gene_list_mini,
                            )

                    del perturbation_minibatch
                    del original_emb_minibatch
                    del full_perturbation_emb
                    del perturbation_emb
                    del perturbation_cls_emb
                    del cls_cos_sims
                    del gene_cos_sims

                # save dict to disk every self.clear_mem_ncells/10 (default 100) simulated cells
                if i % max(1, self.clear_mem_ncells / 10) == 0:
                    pu.write_perturbation_dictionary(
                        cos_sims_dict,
                        f"{output_path_prefix}_dict_cell_embs_{h}batch{pickle_batch}",
                    )
                    if self.emb_mode == "cls_and_gene":
                        pu.write_perturbation_dictionary(
                            stored_gene_embs_dict,
                            f"{output_path_prefix}_dict_gene_embs_{h}batch{pickle_batch}",
                        )

                # reset and clear memory every self.clear_mem_ncells (default 1000) simulated cells or at the end of the example cell
                if i % self.clear_mem_ncells == 0:
                    pickle_batch += 1
                    if self.cell_states_to_model is None:
                        cos_sims_dict = defaultdict(list)
                    else:
                        cos_sims_dict = {
                            state: defaultdict(list)
                            for state in pu.get_possible_states(
                                self.cell_states_to_model
                            )
                        }

                    if self.emb_mode == "cls_and_gene":
                        stored_gene_embs_dict = defaultdict(list)

                    torch.cuda.empty_cache()

            pu.write_perturbation_dictionary(
                cos_sims_dict,
                f"{output_path_prefix}_dict_cell_embs_{h}batch{pickle_batch}",
            )

            if self.emb_mode == "cls_and_gene":
                pu.write_perturbation_dictionary(
                    stored_gene_embs_dict,
                    f"{output_path_prefix}_dict_gene_embs_{h}batch{pickle_batch}",
                )

            pickle_batch = -1
            if self.cell_states_to_model is None:
                cos_sims_dict = defaultdict(list)
            else:
                cos_sims_dict = {
                    state: defaultdict(list)
                    for state in pu.get_possible_states(self.cell_states_to_model)
                }

            if self.emb_mode == "cls_and_gene":
                stored_gene_embs_dict = defaultdict(list)

            # clear memory between cells
            del perturbation_batch
            del original_cls_emb
            if self.emb_mode == "cls_and_gene":
                del full_original_emb
            torch.cuda.empty_cache()

    def update_perturbation_dictionary(
        self,
        cos_sims_dict: defaultdict,
        cos_sims_data: torch.Tensor,
        gene_list=None,
    ):
        if gene_list is not None and cos_sims_data.shape[0] != len(gene_list):
            logger.error(
                f"len(cos_sims_data.shape[0]) != len(gene_list). \n \
                            {cos_sims_data.shape[0]=}.\n \
                            {len(gene_list)=}."
            )
            raise

        if self.perturb_group is True:
            if len(self.tokens_to_perturb) > 1:
                perturbed_genes = tuple(self.tokens_to_perturb)
            else:
                perturbed_genes = self.tokens_to_perturb[0]

            # if cell embeddings, can just append
            # shape will be (batch size, 1)
            cos_sims_data = torch.squeeze(cos_sims_data).tolist()

            # handle case of single cell left
            if not isinstance(cos_sims_data, list):
                cos_sims_data = [cos_sims_data]

            cos_sims_dict[(perturbed_genes, "cell_emb")] += cos_sims_data

        else:
            for i, cos in enumerate(cos_sims_data.tolist()):
                cos_sims_dict[(gene_list[i], "cell_emb")].append(cos)

        return cos_sims_dict
