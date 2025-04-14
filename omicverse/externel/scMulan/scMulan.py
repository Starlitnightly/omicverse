import torch
import os
import sys
from .model.model import MulanConfig, scMulanModel
from .model.model_kvcache import scMulanModel_kv
import torch.nn.functional as F
from .utils.hf_tokenizer import scMulanTokenizer
import scipy.sparse
import numpy as np
from tqdm import tqdm
from anndata import AnnData
from typing import Optional
import pandas as pd
import io
import multiprocessing
multiprocessing.set_start_method('spawn',force=True)

class scMulan:
    def __init__(self, model, adata, meta_info, tokenizer, n_express_level, **kwargs):
        self.model = model
        self.meta_info = meta_info
        self.tokenizer = tokenizer
        self.mulan_gene_set = self.meta_info['gene_set']
        self.check_adata(adata,**kwargs)
        self.n_express_level = n_express_level
        self.mulan_cell_type_entities = list(self.meta_info['cell_type'] | self.meta_info['MCT'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def data_preprocess(self,):

        # sparse check
        # self.adata_sparse = False #scipy.sparse.issparse(self.adata.X) # TODO use sparsity
        # # get COO matrix for analysis
        # if self.adata_sparse:
        #     self.adata_matrix = self.adata.X.tocoo()
        # else:pass
            #print('adata is not sparse, use dense matrix and dataframe')
            # self.adata_matrix = self.adata.X.toarray()
        cellDFHVG = pd.DataFrame(self.adata.X.toarray(), columns = self.mulan_gene_set)
        cellDFHVG.index = list(self.adata.obs.index)
        self.adata_matrix = cellDFHVG

        


    def get_gene_expression_dict(self, i, matrix):
        genes_series = matrix.loc[i]
        expressed_genes = genes_series[genes_series > 0].index.tolist()
        expr_values = genes_series[expressed_genes].values
        cell_expression_dict = {gene: expr_value for gene, expr_value in zip(expressed_genes, expr_values)}
        return cell_expression_dict
    
    def prepare_gene_expression_codings(self, i, matrix):

        cell_expression_dict = self.get_gene_expression_dict(i, matrix)
        expressed_genes = list(cell_expression_dict.keys())[::-1]
        expression_values = list(cell_expression_dict.values())[::-1]
        if len(expression_values) == 0:  # Check if the array is empty
            max_expression = 0  # Set a default value or handle accordingly
        else:
            max_expression = np.max(expression_values)
        #max_expression = np.max(expression_values)
        bins = np.linspace(0, max_expression, self.n_express_level+1)
        binned_expr = np.digitize(expression_values, bins, right=True)

        return expressed_genes, binned_expr
    
    def make_encoded_annotation_prompt_one_cell(self, expressed_genes, binned_expr, annotation_task_token = '<PCT>'):

        prefix = expressed_genes + [annotation_task_token] # add pre-defined task token to guide model generate cell type annotations
        ec_binned_expr = np.append(binned_expr,[0]*(len([annotation_task_token]))) # add a zero for task token
        ec_prefix = self.tokenizer.encode(prefix) 
        prefix_len_with_task_token = len(ec_prefix) # length with task token

        return (ec_prefix, ec_binned_expr, prefix_len_with_task_token)
    

    def get_cell_type(self, i, matrix, **kwargs):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
            ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
            prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0).to(device)
            prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0).to(device)
            generated_tokens = self.model.generate_cellGenesis(prompt_entities,prompt_values, max_new_tokens= prefix_len_with_task_token + 3, top_k=1, **kwargs)[0].cpu().tolist()
            pred_names = self.tokenizer.convert_ids_to_tokens(generated_tokens[0][-3:-1])
            coarse_cell_type = pred_names[-2] if self.is_cell_type_entity(pred_names[-2]) else 'Unclassified'
            fine_cell_type = pred_names[-1] if self.is_cell_type_entity(pred_names[-1]) else 'Unclassified'

        return coarse_cell_type, fine_cell_type
    
    def cell_type_pred_process_subdata(self, idx_subset, device_id, save_path = None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')

        self.model.to(device)
        fine_cell_type_pred = []
        # print(f'using device {device_id}, on processing {os.getpid()}')
        for idx in tqdm(idx_subset, desc=f"⏳Generating cell type labels for each cell on device {device_id}"):
            coarse_cell_type, fine_cell_type = self.get_cell_type(idx, self.adata_matrix, **kwargs)
            fine_cell_type_pred.append(fine_cell_type)
        torch.cuda.empty_cache()
        
        return fine_cell_type_pred

    def get_cell_types_for_adata(self, parallel = False, n_process = None,  **kwargs):

        self.data_preprocess()
        if parallel:
            assert n_process is not None, print('n_process must be set if using parallel')
            print(f'⚡ Speed up by multiprocessing with {n_process} processes and {torch.cuda.device_count()} GPUs...')
            fine_cell_type_pred = self.process_data_in_parallel(self.cell_type_pred_process_subdata, n_process)
            self.adata.obs['cell_type_from_scMulan'] = fine_cell_type_pred

        fine_cell_type_pred = []
        for i in tqdm(range(self.adata.n_obs), desc="⏳Generating cell type labels for each cell"):
            _, fine_cell_type = self.get_cell_type(i, self.adata_matrix, **kwargs)
            fine_cell_type_pred.append(fine_cell_type)
        self.adata.obs['cell_type_from_scMulan'] = fine_cell_type_pred

    
    def get_cell_embedding(self, i, matrix, **kwargs):

        with torch.no_grad():
            expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
            ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
            prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0).cuda()
            prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0).cuda()
            _,_,hidden = self.model.generate_cellGenesis(prompt_entities,prompt_values,
                                                                    max_new_tokens= prefix_len_with_task_token + 3,
                                                                    top_k=1, return_hidden=True,**kwargs) # +3 is passing CT1, CT2,<#E#>
            hidden = hidden[-1][0,-2,:].cpu().numpy() #TODO custom choose embedding

        return hidden
    
    def embedding_process_subdata(self, idx_subset, device_id, save_path = None, **kwargs):

        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        self.model.to(device)

        hidden_states = np.zeros((len(idx_subset), self.model.hidden_dim))

        for j,idx in enumerate(tqdm(idx_subset, desc=f"⏳ Collecting cell embeddings for each cell on device {device_id}")):
            hidden = self.get_cell_embedding(idx, self.adata_matrix, **kwargs)
            hidden_states[j] = hidden

        torch.cuda.empty_cache()
        if save_path:
            torch.save(hidden_states, save_path)

        return hidden_states
    
    def get_cell_embeddings_for_adata(self, parallel = False, n_process = None, save_dir = None, **kwargs):

        self.data_preprocess()
        if parallel:
            assert n_process is not None, print('n_process must be set if using parallel')
            print(f'⚡ Speed up by multiprocessing with {n_process} processes and {torch.cuda.device_count()} GPUs...')
            # hidden_states = self.process_data_in_parallel(self.embedding_process_subdata, n_process, save_dir)
            hidden_states = self.process_data_in_parallel(self.embedding_process_subdata, n_process, save_dir)
        else:
            hidden_states = []
            for i in tqdm(range(self.adata.n_obs), desc="⏳Collecting cell embeddings for each cell"):
                hidden = self.get_cell_embedding(i, self.adata_matrix, **kwargs)
                hidden_states.append(hidden)
        self.adata.obsm['X_scMulan'] = np.array(hidden_states)
    

    def get_cell_type_and_embd(self, i, matrix, **kwargs):

        with torch.no_grad():
            expressed_genes, binned_expr = self.prepare_gene_expression_codings(i, matrix)
            ec_prefix, ec_binned_expr, prefix_len_with_task_token = self.make_encoded_annotation_prompt_one_cell(expressed_genes, binned_expr)
            prompt_entities = torch.tensor(ec_prefix[:prefix_len_with_task_token]).unsqueeze(0)
            prompt_entities = prompt_entities.cuda() if torch.cuda.is_available() else prompt_entities
            prompt_values = torch.tensor(ec_binned_expr[:prefix_len_with_task_token]).unsqueeze(0)
            prompt_values = prompt_values.cuda() if torch.cuda.is_available() else prompt_values
            generated_entities, generated_values, hidden = self.model.generate_cellGenesis(prompt_entities,prompt_values, 
                                                                                            max_new_tokens= prefix_len_with_task_token + 3,
                                                                                            top_k=1, return_hidden=True, **kwargs) # +3 is passing CT1, CT2,<#E#>
            pred_names = self.tokenizer.convert_ids_to_tokens(generated_entities[0].cpu().tolist()[-3:-1])
            # coarse_cell_type = pred_names[-2] if self.is_cell_type_entity(pred_names[-2]) else 'Unclassified'
            fine_cell_type = pred_names[-1] if self.is_cell_type_entity(pred_names[-1]) else 'Unclassified'
            hidden = hidden[-1][0,-2,:].cpu().numpy()

        return fine_cell_type, hidden



    def cell_type_and_embd_process_subdata(self, idx_subset, device_id, save_path = None, **kwargs):        
        torch.cuda.set_device(device_id)
        self.model.to(device_id)
        pred_embd_list = []

        for idx in tqdm(idx_subset, desc=f"⏳ Generating cell type labels and embds for each cell on device {device_id}"):
            # fine_cell_type, hidden = self.get_cell_type_and_embd(idx, self.adata_matrix,**kwargs)
            fine_cell_type, hidden = self.get_cell_type_and_embd(idx, self.adata_matrix, **kwargs)
            pred_embd_list.append([fine_cell_type, hidden])

        torch.cuda.empty_cache()
        if save_path:
            torch.save(pred_embd_list, save_path)

        return pred_embd_list
    
    
    def get_cell_types_and_embds_for_adata(self, parallel = False, n_process = None, save_dir = None, **kwargs):

        self.data_preprocess()
        if parallel:
            assert n_process is not None, print('n_process must be set if using parallel')
            print(f'⚡ Speed up by multiprocessing with {n_process} processes and {torch.cuda.device_count()} GPUs...')
            results = self.process_data_in_parallel(self.cell_type_and_embd_process_subdata, n_process, save_dir)
        else:
            results = []
            for idx in tqdm(self.adata.obs_names, desc="⏳ Collecting cell embeddings for each cell"):
                ct, hidden = self.get_cell_type_and_embd(idx, self.adata_matrix, **kwargs)
                results.append([ct, hidden])

        cell_types = [pair[0] for pair in results]
        hidden_embds = [pair[1] for pair in results]
        self.adata.obs['cell_type_from_scMulan'] = cell_types
        self.adata.obsm['X_scMulan'] = np.array(hidden_embds)
        
    def is_cell_type_entity(self, token_entity):
        return token_entity in self.mulan_cell_type_entities
    
    def cuda_count(self,):
        print(f'scMulan is currently available to {torch.cuda.device_count()} GPUs.')
        return torch.cuda.device_count()

    def check_adata(self, adata, force=False): # set force as True to pass check adata anyway.

        if force:
            print('✅ forcing pass check')
            print("👸 scMulan is ready")
            self.adata = adata.copy()
            return True
        # check normalize and log1p
        adata_max = adata.X.max()
        assert adata_max < 10, f'🚫 Please make sure adata is processed with normalization (sum = 1e4) and log1p, your adata max is {adata_max}.'
        # check gene symbol uniform
        adata_var = set(adata.var_names.tolist())
        mulan_geneset = set(self.meta_info['gene_set'])
        count = len(adata_var.intersection(mulan_geneset))
        assert count == len(self.meta_info['gene_set']), f'🚫 Please make sure adata is processed with uniformed gene symbol, your gene set has {count} overlap with scMulan.'
        # use mulan gene set
        self.adata = adata[:,self.mulan_gene_set].copy()
        print('✅ adata passed check')
        print("👸 scMulan is ready")
        

    def process_data_in_parallel(self, func, n_process, save_dir = None):

        # idxs = np.array_split(np.arange(self.adata.n_obs), n_process)
        idxs = np.array_split(self.adata.obs_names, n_process)
        
        devices = [i % torch.cuda.device_count() for i in range(n_process)]
        args = []
        for idx_subset, device_id, proc_id in zip(idxs, devices, range(n_process)):
            if save_dir:
                save_path = os.path.join(save_dir, f"process_{proc_id}.pt")
            else:
                save_path = None
            args.append((idx_subset, device_id, save_path))

        with multiprocessing.Pool(n_process) as pool:
            results = pool.starmap(func, args)
        combined_results = [item for sublist in results for item in sublist]

        return combined_results
    


def model_inference(ckp_path: str,
                    adata: AnnData,
                    meta_info_path: str = os.path.join(os.path.dirname(__file__), 'utils', 'meta_info.pt'),
                    kv_cache: Optional[bool] = False,
                    **kwargs,
                    ):
    
    ckp = torch.load(ckp_path, map_location='cpu')
    gptconf = MulanConfig(**ckp['model_args'])
    if kv_cache:
        model = scMulanModel_kv(gptconf)
    else:
        model = scMulanModel(gptconf)
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(ckp['model'])
    model.eval()
    model.hidden_dim = ckp['model_args']['n_embd']
    # model.half()
    meta_info = torch.load(meta_info_path)
    tokenizer = scMulanTokenizer(meta_info['token_set'])
    n_express_level = ckp['model_args']['expression_level']

    scml = scMulan(model,adata,meta_info,tokenizer,n_express_level,**kwargs)

    return scml


    


        

    



    

        