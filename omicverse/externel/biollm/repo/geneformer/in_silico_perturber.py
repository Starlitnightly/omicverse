"""
Geneformer in silico perturber.

Usage:
  from geneformer import InSilicoPerturber
  isp = InSilicoPerturber(perturb_type="delete",
                          perturb_rank_shift=None,
                          genes_to_perturb="all",
                          combos=0,
                          anchor_gene=None,
                          model_type="Pretrained",
                          num_classes=0,
                          emb_mode="cell",
                          cell_emb_style="mean_pool",
                          filter_data={"cell_type":["cardiomyocyte"]},
                          cell_states_to_model={"state_key": "disease", "start_state": "dcm", "goal_state": "nf", "alt_states": ["hcm", "other1", "other2"]},
                          max_ncells=None,
                          emb_layer=-1,
                          forward_batch_size=100,
                          nproc=4)
  isp.perturb_data("path/to/model",
                   "path/to/input_data",
                   "path/to/output_directory",
                   "output_prefix")
"""

# imports
import itertools as it
import logging
import numpy as np
import pickle
import re
import seaborn as sns; sns.set()
import torch
from collections import defaultdict
from datasets import Dataset, load_from_disk
from tqdm.notebook import trange
from transformers import BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification

from .tokenizer import TOKEN_DICTIONARY_FILE

logger = logging.getLogger(__name__)


# load data and filter by defined criteria
def load_and_filter(filter_data, nproc, input_data_file):
    data = load_from_disk(input_data_file)
    if filter_data is not None:
        for key,value in filter_data.items():
            def filter_data_by_criteria(example):
                return example[key] in value
            data = data.filter(filter_data_by_criteria, num_proc=nproc)
        if len(data) == 0:
            logger.error(
                    "No cells remain after filtering. Check filtering criteria.")
            raise
    data_shuffled = data.shuffle(seed=42)
    return data_shuffled

# load model to GPU
def load_model(model_type, num_classes, model_directory):
    if model_type == "Pretrained":
        model = BertForMaskedLM.from_pretrained(model_directory, 
                                                output_hidden_states=True, 
                                                output_attentions=False)
    elif model_type == "GeneClassifier":
        model = BertForTokenClassification.from_pretrained(model_directory,
                                                num_labels=num_classes,
                                                output_hidden_states=True, 
                                                output_attentions=False)
    elif model_type == "CellClassifier":
        model = BertForSequenceClassification.from_pretrained(model_directory, 
                                                num_labels=num_classes,
                                                output_hidden_states=True, 
                                                output_attentions=False)
    # put the model in eval mode for fwd pass
    model.eval()
    model = model.to("cuda:0")
    return model

def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums))+1

def get_model_input_size(model):
    return int(re.split("\(|,",str(model.bert.embeddings.position_embeddings))[1])

def flatten_list(megalist):
    return [item for sublist in megalist for item in sublist]

def measure_length(example):
    example["length"] = len(example["input_ids"])
    return example

def downsample_and_sort(data_shuffled, max_ncells):
    num_cells = len(data_shuffled)
    # if max number of cells is defined, then subsample to this max number
    if max_ncells != None:
        num_cells = min(max_ncells,num_cells)
    data_subset = data_shuffled.select([i for i in range(num_cells)])
    # sort dataset with largest cell first to encounter any memory errors earlier
    data_sorted = data_subset.sort("length",reverse=True)
    return data_sorted

def get_possible_states(cell_states_to_model):
    possible_states = []
    for key in ["start_state","goal_state"]:
        possible_states += [cell_states_to_model[key]]
    possible_states += cell_states_to_model.get("alt_states",[])
    return possible_states

def forward_pass_single_cell(model, example_cell, layer_to_quant):
    example_cell.set_format(type="torch")
    input_data = example_cell["input_ids"]
    with torch.no_grad():
        outputs = model(
            input_ids = input_data.to("cuda")
        )
    emb = torch.squeeze(outputs.hidden_states[layer_to_quant])
    del outputs
    return emb

def perturb_emb_by_index(emb, indices): 
    mask = torch.ones(emb.numel(), dtype=torch.bool)    
    mask[indices] = False   
    return emb[mask]

def delete_indices(example):    
    indices = example["perturb_index"]
    if any(isinstance(el, list) for el in indices):
        indices = flatten_list(indices) 
    for index in sorted(indices, reverse=True): 
        del example["input_ids"][index] 
    return example

# for genes_to_perturb = "all" where only genes within cell are overexpressed
def overexpress_indices(example):
    indices = example["perturb_index"]
    if any(isinstance(el, list) for el in indices):
        indices = flatten_list(indices)
    for index in sorted(indices, reverse=True):
        example["input_ids"].insert(0, example["input_ids"].pop(index))
    return example

# for genes_to_perturb = list of genes to overexpress that are not necessarily expressed in cell
def overexpress_tokens(example):
    # -100 indicates tokens to overexpress are not present in rank value encoding
    if example["perturb_index"] != [-100]:
        example = delete_indices(example)
    [example["input_ids"].insert(0, token) for token in example["tokens_to_perturb"][::-1]]
    return example

def remove_indices_from_emb(emb, indices_to_remove, gene_dim):
    # indices_to_remove is list of indices to remove
    indices_to_keep = [i for i in range(emb.size()[gene_dim]) if i not in indices_to_remove]
    num_dims = emb.dim()
    emb_slice = [slice(None) if dim != gene_dim else indices_to_keep for dim in range(num_dims)]
    sliced_emb = emb[emb_slice]
    return sliced_emb

def remove_indices_from_emb_batch(emb_batch, list_of_indices_to_remove, gene_dim):
    output_batch = torch.stack([
                    remove_indices_from_emb(emb_batch[i, :, :], idx, gene_dim-1) for 
                    i, idx in enumerate(list_of_indices_to_remove)
                    ])
    return output_batch

def make_perturbation_batch(example_cell, 
                            perturb_type, 
                            tokens_to_perturb, 
                            anchor_token, 
                            combo_lvl,
                            num_proc):
    if tokens_to_perturb == "all":
        if perturb_type in ["overexpress","activate"]:
            range_start = 1
        elif perturb_type in ["delete","inhibit"]:
            range_start = 0
        indices_to_perturb = [[i] for i in range(range_start,example_cell["length"][0])]
    elif combo_lvl>0 and (anchor_token is not None):    
        example_input_ids = example_cell["input_ids "][0]   
        anchor_index = example_input_ids.index(anchor_token[0]) 
        indices_to_perturb = [sorted([anchor_index,i]) if i!=anchor_index else None for i in range(example_cell["length"][0])]  
        indices_to_perturb = [item for item in indices_to_perturb if item is not None]
    else:
        example_input_ids = example_cell["input_ids"][0]
        indices_to_perturb = [[example_input_ids.index(token)] if token in example_input_ids else None for token in tokens_to_perturb]
        indices_to_perturb = [item for item in indices_to_perturb if item is not None]
    
    # create all permutations of combo_lvl of modifiers from tokens_to_perturb
    if combo_lvl>0 and (anchor_token is None):
        if tokens_to_perturb != "all":
            if len(tokens_to_perturb) == combo_lvl+1:
                indices_to_perturb = [list(x) for x in it.combinations(indices_to_perturb, combo_lvl+1)]
        else:
            all_indices = [[i] for i in range(example_cell["length"][0])]
            all_indices = [index for index in all_indices if index not in indices_to_perturb]
            indices_to_perturb = [[[j for i in indices_to_perturb for j in i], x] for x in all_indices]
    length = len(indices_to_perturb)
    perturbation_dataset = Dataset.from_dict({"input_ids": example_cell["input_ids"]*length, 
                                              "perturb_index": indices_to_perturb})
    if length<400:
        num_proc_i = 1
    else:
        num_proc_i = num_proc
    if perturb_type == "delete":
        perturbation_dataset = perturbation_dataset.map(delete_indices, num_proc=num_proc_i)
    elif perturb_type == "overexpress":
        perturbation_dataset = perturbation_dataset.map(overexpress_indices, num_proc=num_proc_i)
    return perturbation_dataset, indices_to_perturb

# perturbed cell emb removing the activated/overexpressed/inhibited gene emb
# so that only non-perturbed gene embeddings are compared to each other
# in original or perturbed context
def make_comparison_batch(original_emb_batch, indices_to_perturb, perturb_group):
    all_embs_list = []

    # if making comparison batch for multiple perturbations in single cell
    if perturb_group == False:
        original_emb_list = [original_emb_batch]*len(indices_to_perturb)
    # if making comparison batch for single perturbation in multiple cells
    elif perturb_group == True:
        original_emb_list = original_emb_batch
        
    
    for i in range(len(original_emb_list)):
        original_emb = original_emb_list[i]
        indices = indices_to_perturb[i]
        if indices == [-100]:
            all_embs_list += [original_emb[:]]
            continue
        emb_list = []
        start = 0
        if any(isinstance(el, list) for el in indices):
            indices = flatten_list(indices)
        for i in sorted(indices):
            emb_list += [original_emb[start:i]]
            start = i+1
        emb_list += [original_emb[start:]]
        all_embs_list += [torch.cat(emb_list)]
    len_set = set([emb.size()[0] for emb in all_embs_list])
    if len(len_set) > 1:
        max_len = max(len_set)
        all_embs_list = [pad_2d_tensor(emb, None, max_len, 0) for emb in all_embs_list]
    return torch.stack(all_embs_list)

# average embedding position of goal cell states
def get_cell_state_avg_embs(model,
                            filtered_input_data,
                            cell_states_to_model,
                            layer_to_quant,
                            pad_token_id,
                            forward_batch_size,
                            num_proc):
    
    model_input_size = get_model_input_size(model)
    possible_states = get_possible_states(cell_states_to_model)
    state_embs_dict = dict()
    for possible_state in possible_states:
        state_embs_list = []
        original_lens = []
        
        def filter_states(example):
            state_key = cell_states_to_model["state_key"]
            return example[state_key] in [possible_state]
        filtered_input_data_state = filtered_input_data.filter(filter_states, num_proc=num_proc)
        total_batch_length = len(filtered_input_data_state)
        if ((total_batch_length-1)/forward_batch_size).is_integer():
            forward_batch_size = forward_batch_size-1
        max_len = max(filtered_input_data_state["length"])
        for i in range(0, total_batch_length, forward_batch_size):
            max_range = min(i+forward_batch_size, total_batch_length)
                
            state_minibatch = filtered_input_data_state.select([i for i in range(i, max_range)])
            state_minibatch.set_format(type="torch")
            
            input_data_minibatch = state_minibatch["input_ids"]
            original_lens += state_minibatch["length"]
            input_data_minibatch = pad_tensor_list(input_data_minibatch, 
                                                   max_len, 
                                                   pad_token_id, 
                                                   model_input_size)
            attention_mask = gen_attention_mask(state_minibatch, max_len)

            with torch.no_grad():
                outputs = model(
                    input_ids = input_data_minibatch.to("cuda"),
                    attention_mask = attention_mask
                )
            
            state_embs_i = outputs.hidden_states[layer_to_quant]
            state_embs_list += [state_embs_i]
            del outputs
            del state_minibatch
            del input_data_minibatch
            del attention_mask
            del state_embs_i
            torch.cuda.empty_cache()

        state_embs = torch.cat(state_embs_list)
        avg_state_emb = mean_nonpadding_embs(state_embs, torch.Tensor(original_lens).to("cuda"))
        avg_state_emb = torch.mean(avg_state_emb, dim=0, keepdim=True)
        state_embs_dict[possible_state] = avg_state_emb
    return state_embs_dict

# quantify cosine similarity of perturbed vs original or alternate states
def quant_cos_sims(model, 
                   perturb_type,
                   perturbation_batch, 
                   forward_batch_size, 
                   layer_to_quant, 
                   original_emb,
                   tokens_to_perturb,
                   indices_to_perturb,
                   perturb_group,
                   cell_states_to_model,
                   state_embs_dict,
                   pad_token_id,
                   model_input_size,
                   nproc):
    cos = torch.nn.CosineSimilarity(dim=2)
    total_batch_length = len(perturbation_batch)
    if ((total_batch_length-1)/forward_batch_size).is_integer():
        forward_batch_size = forward_batch_size-1
    if cell_states_to_model is None:
        if perturb_group == False: # (if perturb_group is True, original_emb is filtered_input_data)
            comparison_batch = make_comparison_batch(original_emb, indices_to_perturb, perturb_group)
        cos_sims = []
    else:
        possible_states = get_possible_states(cell_states_to_model)
        cos_sims_vs_alt_dict = dict(zip(possible_states,[[] for i in range(len(possible_states))]))
    
    # measure length of each element in perturbation_batch
    perturbation_batch = perturbation_batch.map(
            measure_length, num_proc=nproc
        )
    
    for i in range(0, total_batch_length, forward_batch_size):
        max_range = min(i+forward_batch_size, total_batch_length)
            
        perturbation_minibatch = perturbation_batch.select([i for i in range(i, max_range)])
        # determine if need to pad or truncate batch
        minibatch_length_set = set(perturbation_minibatch["length"])
        minibatch_lengths = perturbation_minibatch["length"]
        if (len(minibatch_length_set) > 1) or (max(minibatch_length_set) > model_input_size):
            needs_pad_or_trunc = True
        else:
            needs_pad_or_trunc = False
            max_len = max(minibatch_length_set)

        if needs_pad_or_trunc == True: 
            max_len = min(max(minibatch_length_set),model_input_size)
            def pad_or_trunc_example(example):
                example["input_ids"] = pad_or_truncate_encoding(example["input_ids"], 
                                                               pad_token_id, 
                                                               max_len)
                return example
            perturbation_minibatch = perturbation_minibatch.map(pad_or_trunc_example, num_proc=nproc)
            
        perturbation_minibatch.set_format(type="torch")
        
        input_data_minibatch = perturbation_minibatch["input_ids"]
        attention_mask = gen_attention_mask(perturbation_minibatch, max_len)
        
        # extract embeddings for perturbation minibatch
        with torch.no_grad():
            outputs = model(
                input_ids = input_data_minibatch.to("cuda"),
                attention_mask = attention_mask
            )
        del input_data_minibatch
        del perturbation_minibatch
        del attention_mask
        
        if len(indices_to_perturb)>1:
            minibatch_emb = torch.squeeze(outputs.hidden_states[layer_to_quant])
        else:
            minibatch_emb = outputs.hidden_states[layer_to_quant]
        
        if perturb_type == "overexpress":
            # remove overexpressed genes to quantify effect on remaining genes
            if perturb_group == False:
                overexpressed_to_remove = 1
            if perturb_group == True:
                overexpressed_to_remove = len(tokens_to_perturb)
            minibatch_emb = minibatch_emb[:,overexpressed_to_remove:,:]

        # if quantifying single perturbation in multiple different cells, pad original batch and extract embs
        if perturb_group == True:
            # pad minibatch of original batch to extract embeddings
            # truncate to the (model input size - # tokens to overexpress) to ensure comparability
            # since max input size of perturb batch will be reduced by # tokens to overexpress 
            original_minibatch = original_emb.select([i for i in range(i, max_range)])
            original_minibatch_lengths = original_minibatch["length"]
            original_minibatch_length_set = set(original_minibatch["length"])

            indices_to_perturb_minibatch = indices_to_perturb[i:i+forward_batch_size]
            
            if perturb_type == "overexpress":
                new_max_len = model_input_size - len(tokens_to_perturb)
            else:
                new_max_len = model_input_size
            if (len(original_minibatch_length_set) > 1) or (max(original_minibatch_length_set) > new_max_len):
                new_max_len = min(max(original_minibatch_length_set),new_max_len)
                def pad_or_trunc_example(example):
                    example["input_ids"] = pad_or_truncate_encoding(example["input_ids"], pad_token_id, new_max_len)
                    return example
                original_minibatch = original_minibatch.map(pad_or_trunc_example, num_proc=nproc)
            original_minibatch.set_format(type="torch")
            original_input_data_minibatch = original_minibatch["input_ids"]
            attention_mask = gen_attention_mask(original_minibatch, new_max_len)
            # extract embeddings for original minibatch
            with torch.no_grad():
                original_outputs = model(
                    input_ids = original_input_data_minibatch.to("cuda"),
                    attention_mask = attention_mask
                )
            del original_input_data_minibatch
            del original_minibatch
            del attention_mask

            if len(indices_to_perturb)>1:
                original_minibatch_emb = torch.squeeze(original_outputs.hidden_states[layer_to_quant])
            else:
                original_minibatch_emb = original_outputs.hidden_states[layer_to_quant]

            # embedding dimension of the genes
            gene_dim = 1
            # exclude overexpression due to case when genes are not expressed but being overexpressed
            if perturb_type != "overexpress":
                original_minibatch_emb = remove_indices_from_emb_batch(original_minibatch_emb, 
                                                                       indices_to_perturb_minibatch, 
                                                                       gene_dim)

        # cosine similarity between original emb and batch items
        if cell_states_to_model is None:
            if perturb_group == False:
                minibatch_comparison = comparison_batch[i:max_range]
            elif perturb_group == True:
                minibatch_comparison = original_minibatch_emb

            cos_sims += [cos(minibatch_emb, minibatch_comparison).to("cpu")]
        elif cell_states_to_model is not None:
            for state in possible_states:
                if perturb_group == False:
                    cos_sims_vs_alt_dict[state] += cos_sim_shift(original_emb, 
                                                                minibatch_emb, 
                                                                state_embs_dict[state],
                                                                perturb_group)
                elif perturb_group == True:
                    cos_sims_vs_alt_dict[state] += cos_sim_shift(original_minibatch_emb, 
                                                                minibatch_emb, 
                                                                state_embs_dict[state],
                                                                perturb_group,
                                                                torch.tensor(original_minibatch_lengths, device="cuda"),
                                                                torch.tensor(minibatch_lengths, device="cuda"))                    
        del outputs
        del minibatch_emb
        if cell_states_to_model is None:
            del minibatch_comparison
        torch.cuda.empty_cache()
    if cell_states_to_model is None:
        cos_sims_stack = torch.cat(cos_sims)
        return cos_sims_stack
    else:
        for state in possible_states:
            cos_sims_vs_alt_dict[state] = torch.cat(cos_sims_vs_alt_dict[state])
        return cos_sims_vs_alt_dict

# calculate cos sim shift of perturbation with respect to origin and alternative cell
def cos_sim_shift(original_emb, 
                  minibatch_emb, 
                  end_emb, 
                  perturb_group, 
                  original_minibatch_lengths = None, 
                  minibatch_lengths = None):
    cos = torch.nn.CosineSimilarity(dim=2)
    if not perturb_group:
        original_emb = torch.mean(original_emb,dim=0,keepdim=True)
        original_emb = original_emb[None, :]
        origin_v_end = torch.squeeze(cos(original_emb, end_emb)) #test
    else:
        if original_emb.size() != minibatch_emb.size():
            logger.error(
                f"Embeddings are not the same dimensions. " \
                f"original_emb is {original_emb.size()}. " \
                f"minibatch_emb is {minibatch_emb.size()}. "
            )
            raise

        if original_minibatch_lengths is not None:
            original_emb = mean_nonpadding_embs(original_emb, original_minibatch_lengths)
        # else:
        #     original_emb = torch.mean(original_emb,dim=1,keepdim=True)

        end_emb = torch.unsqueeze(end_emb, 1)
        origin_v_end = cos(original_emb, end_emb)
        origin_v_end = torch.squeeze(origin_v_end)
    if minibatch_lengths is not None:
        perturb_emb = mean_nonpadding_embs(minibatch_emb, minibatch_lengths)
    else:
        perturb_emb = torch.mean(minibatch_emb,dim=1,keepdim=True)

    perturb_v_end = cos(perturb_emb, end_emb)
    perturb_v_end = torch.squeeze(perturb_v_end)
    return [(perturb_v_end-origin_v_end).to("cpu")]

def pad_list(input_ids, pad_token_id, max_len):
    input_ids = np.pad(input_ids, 
                       (0, max_len-len(input_ids)), 
                       mode='constant', constant_values=pad_token_id)
    return input_ids

def pad_tensor(tensor, pad_token_id, max_len):
    tensor = torch.nn.functional.pad(tensor, pad=(0, 
                                     max_len - tensor.numel()), 
                                     mode='constant', 
                                     value=pad_token_id)
    return tensor

def pad_2d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 1:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(tensor, pad=pad, 
                                     mode='constant', 
                                     value=pad_token_id)
    return tensor

def pad_or_truncate_encoding(encoding, pad_token_id, max_len):
    if isinstance(encoding, torch.Tensor):
        encoding_len = tensor.size()[0]
    elif isinstance(encoding, list):
        encoding_len = len(encoding)
    if encoding_len > max_len:
        encoding = encoding[0:max_len]
    elif encoding_len < max_len:
        if isinstance(encoding, torch.Tensor):
            encoding = pad_tensor(encoding, pad_token_id, max_len)
        elif isinstance(encoding, list):
            encoding = pad_list(encoding, pad_token_id, max_len)
    return encoding

# pad list of tensors and convert to tensor
def pad_tensor_list(tensor_list, dynamic_or_constant, pad_token_id, model_input_size):
    
    # Determine maximum tensor length
    if dynamic_or_constant == "dynamic":
        max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    elif type(dynamic_or_constant) == int:
        max_len = dynamic_or_constant
    else:
        max_len = model_input_size
        logger.warning(
                    "If padding style is constant, must provide integer value. " \
                    f"Setting padding to max input size {model_input_size}.")

    # pad all tensors to maximum length
    tensor_list = [pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list]

    # return stacked tensors
    return torch.stack(tensor_list)

def gen_attention_mask(minibatch_encoding, max_len = None):
    if max_len == None:
        max_len = max(minibatch_encoding["length"])
    original_lens = minibatch_encoding["length"]
    attention_mask = [[1]*original_len
                      +[0]*(max_len - original_len)
                      if original_len <= max_len
                      else [1]*max_len
                      for original_len in original_lens]
    return torch.tensor(attention_mask).to("cuda")

# get cell embeddings excluding padding
def mean_nonpadding_embs(embs, original_lens, device):
    # mask based on padding lengths
    mask = torch.arange(embs.size(1)).unsqueeze(0).to(device) < original_lens.unsqueeze(1)

    # extend mask dimensions to match the embeddings tensor
    mask = mask.unsqueeze(2).expand_as(embs)

    # use the mask to zero out the embeddings in padded areas
    masked_embs = embs * mask.float()

    # sum and divide by the lengths to get the mean of non-padding embs
    mean_embs = masked_embs.sum(1) / original_lens.view(-1, 1).float()
    return mean_embs

class InSilicoPerturber:
    valid_option_dict = {
        "perturb_type": {"delete","overexpress","inhibit","activate"},
        "perturb_rank_shift": {None, 1, 2, 3},
        "genes_to_perturb": {"all", list},
        "combos": {0, 1},
        "anchor_gene": {None, str},
        "model_type": {"Pretrained","GeneClassifier","CellClassifier"},
        "num_classes": {int},
        "emb_mode": {"cell","cell_and_gene"},
        "cell_emb_style": {"mean_pool"},
        "filter_data": {None, dict},
        "cell_states_to_model": {None, dict},
        "max_ncells": {None, int},
        "cell_inds_to_perturb": {"all", dict},
        "emb_layer": {-1, 0},
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
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data=None,
        cell_states_to_model=None,
        max_ncells=None,
        cell_inds_to_perturb="all",
        emb_layer=-1,
        forward_batch_size=100,
        nproc=4,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize in silico perturber.

        Parameters
        ----------
        perturb_type : {"delete","overexpress","inhibit","activate"}
            Type of perturbation.
            "delete": delete gene from rank value encoding
            "overexpress": move gene to front of rank value encoding
            "inhibit": move gene to lower quartile of rank value encoding
            "activate": move gene to higher quartile of rank value encoding
        perturb_rank_shift : None, {1,2,3}
            Number of quartiles by which to shift rank of gene.
            For example, if perturb_type="activate" and perturb_rank_shift=1:
                genes in 4th quartile will move to middle of 3rd quartile.
                genes in 3rd quartile will move to middle of 2nd quartile.
                genes in 2nd quartile will move to middle of 1st quartile.
                genes in 1st quartile will move to front of rank value encoding.
            For example, if perturb_type="inhibit" and perturb_rank_shift=2:
                genes in 1st quartile will move to middle of 3rd quartile.
                genes in 2nd quartile will move to middle of 4th quartile.
                genes in 3rd or 4th quartile will move to bottom of rank value encoding.
        genes_to_perturb : "all", list
            Default is perturbing each gene detected in each cell in the dataset.
            Otherwise, may provide a list of ENSEMBL IDs of genes to perturb.
            If gene list is provided, then perturber will only test perturbing them all together
            (rather than testing each possible combination of the provided genes).
        combos : {0,1}
            Whether to perturb genes individually (0) or in pairs (1).
        anchor_gene : None, str
            ENSEMBL ID of gene to use as anchor in combination perturbations.
            For example, if combos=1 and anchor_gene="ENSG00000148400":
                anchor gene will be perturbed in combination with each other gene.
        model_type : {"Pretrained","GeneClassifier","CellClassifier"}
            Whether model is the pretrained Geneformer or a fine-tuned gene or cell classifier.
        num_classes : int
            If model is a gene or cell classifier, specify number of classes it was trained to classify.
            For the pretrained Geneformer model, number of classes is 0 as it is not a classifier.
        emb_mode : {"cell","cell_and_gene"}
            Whether to output impact of perturbation on cell and/or gene embeddings.
        cell_emb_style : "mean_pool"
            Method for summarizing cell embeddings.
            Currently only option is mean pooling of gene embeddings for given cell.
        filter_data : None, dict
            Default is to use all input data for in silico perturbation study.
            Otherwise, dictionary specifying .dataset column name and list of values to filter by.
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
        max_ncells : None, int
            Maximum number of cells to test.
            If None, will test all cells.
        cell_inds_to_perturb : "all", list
            Default is perturbing each cell in the dataset.
            Otherwise, may provide a dict of indices of cells to perturb with keys start_ind and end_ind.
            start_ind: the first index to perturb.
            end_ind: the last index to perturb (exclusive).
            Indices will be selected *after* the filter_data criteria and sorting.
            Useful for splitting extremely large datasets across separate GPUs.
        emb_layer : {-1, 0}
            Embedding layer to use for quantification.
            -1: 2nd to last layer (recommended for pretrained Geneformer)
            0: last layer (recommended for cell classifier fine-tuned for disease state)
        forward_batch_size : int
            Batch size for forward pass.
        nproc : int
            Number of CPU processes to use.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl ID:token).
        """

        self.perturb_type = perturb_type
        self.perturb_rank_shift = perturb_rank_shift
        self.genes_to_perturb = genes_to_perturb
        self.combos = combos
        self.anchor_gene = anchor_gene
        if self.genes_to_perturb == "all":
            self.perturb_group = False  
        else:
            self.perturb_group = True
            if (self.anchor_gene != None) or (self.combos != 0):
                self.anchor_gene = None
                self.combos = 0
                logger.warning(
                    "anchor_gene set to None and combos set to 0. " \
                    "If providing list of genes to perturb, " \
                    "list of genes_to_perturb will be perturbed together, "\
                    "without anchor gene or combinations.")
        self.model_type = model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.cell_emb_style = cell_emb_style
        self.filter_data = filter_data
        self.cell_states_to_model = cell_states_to_model
        self.max_ncells = max_ncells
        self.cell_inds_to_perturb = cell_inds_to_perturb
        self.emb_layer = emb_layer
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc

        self.validate_options()

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get("<pad>")

        if self.anchor_gene is None:
            self.anchor_token = None
        else:
            try:
                self.anchor_token = [self.gene_token_dict[self.anchor_gene]]
            except KeyError:
                logger.error(
                    f"Anchor gene {self.anchor_gene} not in token dictionary."
                )
                raise

        if self.genes_to_perturb == "all":
            self.tokens_to_perturb = "all"
        else:
            missing_genes = [gene for gene in self.genes_to_perturb if gene not in self.gene_token_dict.keys()]
            if len(missing_genes) == len(self.genes_to_perturb):
                logger.error(
                    "None of the provided genes to perturb are in token dictionary."
                )
                raise
            elif len(missing_genes)>0:
                logger.warning(
                    f"Genes to perturb {missing_genes} are not in token dictionary.")
            self.tokens_to_perturb = [self.gene_token_dict.get(gene) for gene in self.genes_to_perturb]

    def validate_options(self):
        # first disallow options under development
        if self.perturb_type in ["inhibit", "activate"]:
            logger.error(
                "In silico inhibition and activation currently under development. " \
                "Current valid options for 'perturb_type': 'delete' or 'overexpress'"
            )
            raise
        
        # confirm arguments are within valid options and compatible with each other
        for attr_name,valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_value in valid_options:
                    continue
                if attr_name in ["anchor_gene"]:
                    if type(attr_name) in {str}:
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
        
        if self.perturb_type in ["delete","overexpress"]:
            if self.perturb_rank_shift is not None:
                if self.perturb_type == "delete":
                    logger.warning(
                        "perturb_rank_shift set to None. " \
                        "If perturb type is delete then gene is deleted entirely " \
                        "rather than shifted by quartile")
                elif self.perturb_type == "overexpress":
                    logger.warning(
                        "perturb_rank_shift set to None. " \
                        "If perturb type is overexpress then gene is moved to front " \
                        "of rank value encoding rather than shifted by quartile")
            self.perturb_rank_shift = None
        
        if (self.anchor_gene is not None) and (self.emb_mode == "cell_and_gene"):
            self.emb_mode = "cell"
            logger.warning(
                "emb_mode set to 'cell'. " \
                "Currently, analysis with anchor gene " \
                "only outputs effect on cell embeddings.")
        
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
        
        if self.perturb_type in ["inhibit","activate"]:
            if self.perturb_rank_shift is None:
                logger.error(
                    "If perturb_type is inhibit or activate then " \
                    "quartile to shift by must be specified.")
                raise
        
        if self.filter_data is not None:
            for key,value in self.filter_data.items():
                if type(value) != list:
                    self.filter_data[key] = [value]
                    logger.warning(
                        "Values in filter_data dict must be lists. " \
                        f"Changing {key} value to list ([{value}]).")
                    
        if self.cell_inds_to_perturb != "all":
            if set(self.cell_inds_to_perturb.keys()) != {"start", "end"}:
                logger.error(
                    "If cell_inds_to_perturb is a dictionary, keys must be 'start' and 'end'."
                )
                raise
            if self.cell_inds_to_perturb["start"] < 0 or self.cell_inds_to_perturb["end"] < 0:
                logger.error(
                    'cell_inds_to_perturb must be positive.'
                )
                raise

    def perturb_data(self, 
                     model_directory,
                     input_data_file,
                     output_directory,
                     output_prefix):
        """
        Perturb genes in input data and save as results in output_directory.

        Parameters
        ----------
        model_directory : Path
            Path to directory containing model
        input_data_file : Path
            Path to directory containing .dataset inputs
        output_directory : Path
            Path to directory where perturbation data will be saved as batched pickle files
        output_prefix : str
            Prefix for output files
        """

        filtered_input_data = load_and_filter(self.filter_data, self.nproc, input_data_file)
        model = load_model(self.model_type, self.num_classes, model_directory)
        layer_to_quant = quant_layers(model)+self.emb_layer
        
        if self.cell_states_to_model is None:
            state_embs_dict = None
        else:
            # confirm that all states are valid to prevent futile filtering
            state_name = self.cell_states_to_model["state_key"]
            state_values = filtered_input_data[state_name]
            for value in get_possible_states(self.cell_states_to_model):
                if value not in state_values:
                    logger.error(
                        f"{value} is not present in the dataset's {state_name} attribute.")
                    raise
            # get dictionary of average cell state embeddings for comparison
            downsampled_data = downsample_and_sort(filtered_input_data, self.max_ncells)
            state_embs_dict = get_cell_state_avg_embs(model,
                                                      downsampled_data,
                                                      self.cell_states_to_model,
                                                      layer_to_quant,
                                                      self.pad_token_id,
                                                      self.forward_batch_size,
                                                      self.nproc)
            # filter for start state cells
            start_state = self.cell_states_to_model["start_state"]
            def filter_for_origin(example):
                return example[state_name] in [start_state]
            
            filtered_input_data = filtered_input_data.filter(filter_for_origin, num_proc=self.nproc)
            
        self.in_silico_perturb(model,
                              filtered_input_data,
                              layer_to_quant,
                              state_embs_dict,
                              output_directory,
                              output_prefix)
    
    # determine effect of perturbation on other genes
    def in_silico_perturb(self,
                          model,
                          filtered_input_data,
                          layer_to_quant,
                          state_embs_dict,
                          output_directory,
                          output_prefix):
        
        output_path_prefix = f"{output_directory}in_silico_{self.perturb_type}_{output_prefix}_dict_1Kbatch"
        model_input_size = get_model_input_size(model)
        
        # filter dataset for cells that have tokens to be perturbed
        if self.anchor_token is not None:
            def if_has_tokens_to_perturb(example):
                return (len(set(example["input_ids"]).intersection(self.anchor_token))==len(self.anchor_token))
            filtered_input_data = filtered_input_data.filter(if_has_tokens_to_perturb, num_proc=self.nproc) 
            if len(filtered_input_data) == 0:
                logger.error(
                        "No cells in dataset contain anchor gene.")
                raise
            else:
                logger.info(f"# cells with anchor gene: {len(filtered_input_data)}")
            
        if (self.tokens_to_perturb != "all") and (self.perturb_type != "overexpress"):
            # minimum # genes needed for perturbation test
            min_genes = len(self.tokens_to_perturb)
            
            def if_has_tokens_to_perturb(example):
                return (len(set(example["input_ids"]).intersection(self.tokens_to_perturb))>=min_genes)
            filtered_input_data = filtered_input_data.filter(if_has_tokens_to_perturb, num_proc=self.nproc)
            if len(filtered_input_data) == 0:
                logger.error(
                        "No cells in dataset contain all genes to perturb as a group.")
                raise
            
        cos_sims_dict = defaultdict(list)
        pickle_batch = -1
        filtered_input_data = downsample_and_sort(filtered_input_data, self.max_ncells)
        if self.cell_inds_to_perturb != "all":
            if self.cell_inds_to_perturb["start"] >= len(filtered_input_data):
                logger.error("cell_inds_to_perturb['start'] is larger than the filtered dataset.")
                raise
            if self.cell_inds_to_perturb["end"] > len(filtered_input_data):
                logger.warning("cell_inds_to_perturb['end'] is larger than the filtered dataset. \
                               Setting to the end of the filtered dataset.")
                self.cell_inds_to_perturb["end"] = len(filtered_input_data)
            filtered_input_data = filtered_input_data.select([i for i in range(self.cell_inds_to_perturb["start"], self.cell_inds_to_perturb["end"])])
        
        # make perturbation batch w/ single perturbation in multiple cells
        if self.perturb_group == True:
                
            def make_group_perturbation_batch(example):
                example_input_ids = example["input_ids"]
                example["tokens_to_perturb"] = self.tokens_to_perturb
                indices_to_perturb = [example_input_ids.index(token) if token in example_input_ids else None for token in self.tokens_to_perturb]
                indices_to_perturb = [item for item in indices_to_perturb if item is not None]
                if len(indices_to_perturb) > 0:
                    example["perturb_index"] = indices_to_perturb
                else:
                    # -100 indicates tokens to overexpress are not present in rank value encoding
                    example["perturb_index"] = [-100]
                if self.perturb_type == "delete":
                    example = delete_indices(example)
                elif self.perturb_type == "overexpress":
                    example = overexpress_tokens(example)
                return example
            
            perturbation_batch = filtered_input_data.map(make_group_perturbation_batch, num_proc=self.nproc)
            indices_to_perturb = perturbation_batch["perturb_index"]

            cos_sims_data = quant_cos_sims(model, 
                                           self.perturb_type,
                                           perturbation_batch, 
                                           self.forward_batch_size, 
                                           layer_to_quant, 
                                           filtered_input_data,
                                           self.tokens_to_perturb,
                                           indices_to_perturb,
                                           self.perturb_group,
                                           self.cell_states_to_model,
                                           state_embs_dict,
                                           self.pad_token_id,
                                           model_input_size,
                                           self.nproc)

            perturbed_genes = tuple(self.tokens_to_perturb)
            original_lengths = filtered_input_data["length"]
            if self.cell_states_to_model is None:
                # update cos sims dict
                # key is tuple of (perturbed_gene, affected_gene)
                # or (perturbed_genes, "cell_emb") for avg cell emb change
                cos_sims_data = cos_sims_data.to("cuda")
                max_padded_len = cos_sims_data.shape[1]
                for j in range(cos_sims_data.shape[0]):
                    # remove padding before mean pooling cell embedding
                    original_length = original_lengths[j]
                    gene_list = filtered_input_data[j]["input_ids"]
                    indices_removed = indices_to_perturb[j]
                    padding_to_remove = max_padded_len - (original_length \
                                                          - len(self.tokens_to_perturb) \
                                                          - len(indices_removed))
                    nonpadding_cos_sims_data = cos_sims_data[j][:-padding_to_remove]
                    cell_cos_sim = torch.mean(nonpadding_cos_sims_data).item()
                    cos_sims_dict[(perturbed_genes, "cell_emb")] += [cell_cos_sim]

                    if self.emb_mode == "cell_and_gene":
                        for k in range(cos_sims_data.shape[1]):
                            cos_sim_value = nonpadding_cos_sims_data[k]
                            affected_gene = gene_list[k].item()
                            cos_sims_dict[(perturbed_genes, affected_gene)] += [cos_sim_value.item()]
            else:
                # update cos sims dict
                # key is tuple of (perturbed_genes, "cell_emb")
                # value is list of tuples of cos sims for cell_states_to_model
                origin_state_key = self.cell_states_to_model["start_state"]
                cos_sims_origin = cos_sims_data[origin_state_key]
                for j in range(cos_sims_origin.shape[0]):
                    data_list = []
                    for data in list(cos_sims_data.values()):
                        data_item = data.to("cuda")
                        data_list += [data_item[j].item()]
                    cos_sims_dict[(perturbed_genes, "cell_emb")] += [tuple(data_list)]
            
            with open(f"{output_path_prefix}_raw.pickle", "wb") as fp:
                pickle.dump(cos_sims_dict, fp)

        # make perturbation batch w/ multiple perturbations in single cell
        if self.perturb_group == False:

            for i in trange(len(filtered_input_data)):
                example_cell = filtered_input_data.select([i])
                original_emb = forward_pass_single_cell(model, example_cell, layer_to_quant)
                gene_list = torch.squeeze(example_cell["input_ids"])

                # reset to original type to prevent downstream issues due to forward_pass_single_cell modifying as torch format in place
                example_cell = filtered_input_data.select([i])

                if self.anchor_token is None:
                    for combo_lvl in range(self.combos+1):
                        perturbation_batch, indices_to_perturb = make_perturbation_batch(example_cell, 
                                                                                        self.perturb_type,
                                                                                        self.tokens_to_perturb,
                                                                                        self.anchor_token,
                                                                                        combo_lvl,
                                                                                        self.nproc)
                        cos_sims_data = quant_cos_sims(model,
                                                       self.perturb_type,
                                                       perturbation_batch, 
                                                       self.forward_batch_size, 
                                                       layer_to_quant, 
                                                       original_emb, 
                                                       self.tokens_to_perturb,
                                                       indices_to_perturb,
                                                       self.perturb_group,
                                                       self.cell_states_to_model,
                                                       state_embs_dict,
                                                       self.pad_token_id,
                                                       model_input_size,
                                                       self.nproc)

                        if self.cell_states_to_model is None:
                            # update cos sims dict
                            # key is tuple of (perturbed_gene, affected_gene)
                            # or (perturbed_gene, "cell_emb") for avg cell emb change
                            cos_sims_data = cos_sims_data.to("cuda")
                            for j in range(cos_sims_data.shape[0]):
                                if self.tokens_to_perturb != "all":
                                    j_index = torch.tensor(indices_to_perturb[j])
                                    if j_index.shape[0]>1:
                                        j_index = torch.squeeze(j_index)
                                else:
                                    j_index = torch.tensor([j])
                                perturbed_gene = torch.index_select(gene_list, 0, j_index)

                                if perturbed_gene.shape[0]==1:
                                    perturbed_gene = perturbed_gene.item()
                                elif perturbed_gene.shape[0]>1:
                                    perturbed_gene = tuple(perturbed_gene.tolist())

                                cell_cos_sim = torch.mean(cos_sims_data[j]).item()
                                cos_sims_dict[(perturbed_gene, "cell_emb")] += [cell_cos_sim]

                                # not_j_index = list(set(i for i in range(gene_list.shape[0])).difference(j_index))
                                # gene_list_j = torch.index_select(gene_list, 0, j_index)
                                if self.emb_mode == "cell_and_gene":
                                    for k in range(cos_sims_data.shape[1]):
                                        cos_sim_value = cos_sims_data[j][k]
                                        affected_gene = gene_list[k].item()
                                        cos_sims_dict[(perturbed_gene, affected_gene)] += [cos_sim_value.item()]
                        else:
                            # update cos sims dict
                            # key is tuple of (perturbed_gene, "cell_emb")
                            # value is list of tuples of cos sims for cell_states_to_model
                            origin_state_key = self.cell_states_to_model["start_state"]
                            cos_sims_origin = cos_sims_data[origin_state_key]

                            for j in range(cos_sims_origin.shape[0]):
                                if (self.tokens_to_perturb != "all") or (combo_lvl>0):
                                    j_index = torch.tensor(indices_to_perturb[j])
                                    if j_index.shape[0]>1:
                                        j_index = torch.squeeze(j_index)
                                else:
                                    j_index = torch.tensor([j])
                                perturbed_gene = torch.index_select(gene_list, 0, j_index)

                                if perturbed_gene.shape[0]==1:
                                    perturbed_gene = perturbed_gene.item()
                                elif perturbed_gene.shape[0]>1:
                                    perturbed_gene = tuple(perturbed_gene.tolist())

                                data_list = []
                                for data in list(cos_sims_data.values()):
                                    data_item = data.to("cuda")
                                    cell_data = torch.mean(data_item[j]).item()
                                    data_list += [cell_data]
                                cos_sims_dict[(perturbed_gene, "cell_emb")] += [tuple(data_list)]

                elif self.anchor_token is not None:
                    perturbation_batch, indices_to_perturb = make_perturbation_batch(example_cell, 
                                                                                     self.perturb_type,
                                                                                     self.tokens_to_perturb,
                                                                                     None,  # first run without anchor token to test individual gene perturbations
                                                                                     0,
                                                                                     self.nproc)
                    cos_sims_data = quant_cos_sims(model,
                                                   self.perturb_type,
                                                   perturbation_batch,
                                                   self.forward_batch_size,
                                                   layer_to_quant,
                                                   original_emb,
                                                   self.tokens_to_perturb,
                                                   indices_to_perturb,
                                                   self.perturb_group,
                                                   self.cell_states_to_model,
                                                   state_embs_dict,
                                                   self.pad_token_id,
                                                   model_input_size,
                                                   self.nproc)
                    cos_sims_data = cos_sims_data.to("cuda")

                    combo_perturbation_batch, combo_indices_to_perturb = make_perturbation_batch(example_cell, 
                                                                                                 self.perturb_type,
                                                                                                 self.tokens_to_perturb,
                                                                                                 self.anchor_token,
                                                                                                 1,
                                                                                                 self.nproc)
                    combo_cos_sims_data = quant_cos_sims(model,
                                                         self.perturb_type,
                                                         combo_perturbation_batch,
                                                         self.forward_batch_size,
                                                         layer_to_quant,
                                                         original_emb,
                                                         self.tokens_to_perturb,
                                                         combo_indices_to_perturb,
                                                         self.perturb_group,
                                                         self.cell_states_to_model,
                                                         state_embs_dict,
                                                         self.pad_token_id,
                                                         model_input_size,
                                                         self.nproc)
                    combo_cos_sims_data = combo_cos_sims_data.to("cuda")

                    # update cos sims dict
                    # key is tuple of (perturbed_gene, "cell_emb") for avg cell emb change
                    anchor_index = example_cell["input_ids"][0].index(self.anchor_token[0])
                    anchor_cell_cos_sim = torch.mean(cos_sims_data[anchor_index]).item()
                    non_anchor_indices = [k for k in range(cos_sims_data.shape[0]) if k != anchor_index]
                    cos_sims_data = cos_sims_data[non_anchor_indices,:]

                    for j in range(cos_sims_data.shape[0]):

                        if j<anchor_index:
                            j_index = torch.tensor([j])
                        else:
                            j_index = torch.tensor([j+1])

                        perturbed_gene = torch.index_select(gene_list, 0, j_index)
                        perturbed_gene = perturbed_gene.item()

                        cell_cos_sim = torch.mean(cos_sims_data[j]).item()
                        combo_cos_sim = torch.mean(combo_cos_sims_data[j]).item()
                        cos_sims_dict[(perturbed_gene, "cell_emb")] += [(anchor_cell_cos_sim, # cos sim anchor gene alone
                                                                         cell_cos_sim, # cos sim deleted gene alone
                                                                         combo_cos_sim)] # cos sim anchor gene + deleted gene

                # save dict to disk every 100 cells
                if (i/100).is_integer():
                    with open(f"{output_path_prefix}{pickle_batch}_raw.pickle", "wb") as fp:
                        pickle.dump(cos_sims_dict, fp)
                # reset and clear memory every 1000 cells
                if (i/1000).is_integer():
                    pickle_batch = pickle_batch+1
                    # clear memory
                    del perturbed_gene
                    del cos_sims_data
                    if self.cell_states_to_model is None:
                        del cell_cos_sim
                    if self.cell_states_to_model is not None:
                        del cell_data
                        del data_list
                    elif self.anchor_token is None:
                        if self.emb_mode == "cell_and_gene":
                            del affected_gene
                            del cos_sim_value
                    else:
                        del combo_cos_sim
                        del combo_cos_sims_data
                    # reset dict
                    del cos_sims_dict
                    cos_sims_dict = defaultdict(list)
                    torch.cuda.empty_cache()

            # save remainder cells
            with open(f"{output_path_prefix}{pickle_batch}_raw.pickle", "wb") as fp:
                pickle.dump(cos_sims_dict, fp)