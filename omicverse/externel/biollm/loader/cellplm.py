#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: cellplm.py
@time: 2025/4/15 17:49
"""
#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: cellplm.py
@time: 2025/4/3 11:13
"""
from .loader_base import LoaderBase
from ..repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from ..repo.CellPLM.model import OmicsFormer
import torch
import numpy as np
from ..data_preprocess.cellplm_handler import CellplmHandler
import json
from ..repo.CellPLM.utils.data import XDict


class CellPLM(LoaderBase):
    """
    Load and manage the scGPT model for gene expression analysis.

    This class provides methods to load the vocabulary, model parameters,
    and perform preprocessing steps on the single-cell data.

    Args:
        args (Namespace): Command line arguments containing model configuration.

    Attributes:
        vocab (GeneVocab): Vocabulary object for gene representation.
        model (TransformerModel): Loaded Transformer model.
    """
    def __init__(self, args):
        """
        Initializes the LoadScgpt class.

        Args:
            args (Namespace): Command line arguments containing model configuration.
        """
        super(CellPLM, self).__init__(args)
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.init_model()
        self.model = self.model.to(self.args.device)
        self.data_handler = CellplmHandler(self.args.vocab_file)

    def load_model(self):
        """
        Loads the Transformer model with specified parameters.

        Returns:
            TransformerModel: The initialized Transformer model.
        """
        config_path = self.args.model_param_file
        with open(config_path, "r") as openfile:
            config = json.load(openfile)
        config['head_type'] = self.args['head_type']
        model = OmicsFormer(**config)
        return model

    def load_vocab(self):
        """
        Loads the gene vocabulary from a file and adds special tokens.

        Returns:
            GeneVocab: The loaded gene vocabulary.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        return vocab

    def get_dataloader(self,
                       adata,
                       var_key,
                       obs_key,
                       n_hvg,
                       batch_size,
                       ddp_train,
                       shuffle,
                       drop_last,
                       order_required,
                       num_workers=0,
                       obs_id_output=None):
        adata = self.data_handler.preprocess(adata, var_key, obs_key, obs_id_output, n_hvg)
        dataset = self.data_handler.make_dataset(adata, obs_key, order_required)
        dataloader = self.data_handler.make_dataloader(dataset, batch_size, ddp_train, shuffle, drop_last, num_workers)
        return dataloader

    def get_gene_embedding(self, gene_ids):
        """
        Gets gene embeddings for specified gene IDs.

        Args:
            gene_ids (list): List of gene IDs.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        gene_embeddings = self.model.embedder.feat_enc(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info(f'finished get gene embedding!')
        idx2gene = self.get_idx2gene()
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2gene[i] for i in gene_ids]
        return {'gene_names': gene_names, 'gene_emb': gene_embeddings.detach().cpu().numpy()}

    def get_cell_embedding(self, adata, pool_type='cls'):
        """
        Generates cell embeddings from the given AnnData object.

        This method retrieves cell embeddings using a data loader and the model,
        processing the data in batches. It returns a NumPy array of embeddings
        for each cell in the input AnnData.

        Args:
            adata: AnnData object containing the data to process.


        Returns:
            np.ndarray: A 2D NumPy array of shape (n_cells, embsize),
            where n_cells is the number of cells and embsize is the embedding size.

        Raises:
            RuntimeError: If any issues arise during model inference or data loading.
        """
        self.logger.info('start to get cell embedding!')

        data_loader = self.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=None, n_hvg=self.args.n_hvg,
                                          batch_size=None, ddp_train=False,
                                          shuffle=False, drop_last=False, order_required=True,)
        self.logger.info('get dataloader Done!')
        order_list = []
        batch_size = adata.shape[0]
        with torch.no_grad():
            self.model.eval()
            pred = []
            for i, data_dict in enumerate(data_loader):
                idx = torch.arange(data_dict['x_seq'].shape[0])

                for j in range(0, len(idx), batch_size):
                    print(j)
                    if len(idx) - j < batch_size:
                        cur = idx[j:]
                    else:
                        cur = idx[j:j + batch_size]
                    input_dict = {}
                    for k in data_dict:
                        if k == 'x_seq':
                            input_dict[k] = data_dict[k].index_select(0, cur).to(self.args.device)
                        elif k not in ['gene_list', 'split']:
                            input_dict[k] = data_dict[k][cur].to(self.args.device)
                    x_dict = XDict(input_dict)
                    out_dict, _ = self.model(x_dict, data_dict['gene_list'])
                    order_list.append(input_dict['order_list'])
                    pred.append(out_dict['pred'])  # [input_dict['order_list']])
            order = torch.cat(order_list)
            order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
            pred = torch.cat(pred)
            pred = pred[order]
            cell_embeddings = pred.detach().cpu().numpy()
            self.logger.info("get cell embedding Done!")
            return cell_embeddings

    def get_gene_expression_embedding(self, adata):
        pass

    def get_embedding(self, emb_type, adata=None, gene_ids=None):
        """
        Retrieves embeddings based on the specified type.

        This method calls the appropriate embedding method based on the
        provided embedding type, which can be 'gene', 'cell', or
        'gene-expression'.

        Args:
            emb_type: (str) Type of embedding to retrieve. Must be one of
                'gene', 'cell', or 'gene-expression'.
            adata: Optional; AnnData object for cell or gene-expression
                embeddings.
            gene_ids: Optional; List of gene IDs for gene embeddings.
                Required if emb_type is 'gene'.

        Returns:
            Either:
                - np.ndarray: Gene embeddings if emb_type is 'gene'.
                - np.ndarray: Cell embeddings if emb_type is 'cell'.
                - Tuple[np.ndarray, List[str]]: Gene expression embeddings
                  and names if emb_type is 'gene-expression'.

        Raises:
            ValueError: If emb_type is invalid or required arguments are
                missing.
        """
        self.model = self.model.eval()
        assert emb_type in ['gene', 'cell',
                            'gene-expression'], 'Invalid embedding type, must be gene, cell or gene-expression'
        if emb_type == 'gene' and gene_ids is None:
            raise ValueError('gene_ids must not be None if emb_type is gene!')
        if emb_type != 'gene' and adata is None:
            raise ValueError('adata must not be None if emb_type is cell or gene-expression!')
        if emb_type == 'gene':
            return self.get_gene_embedding(gene_ids)
        elif emb_type == 'cell':
            return self.get_cell_embedding(adata)
        else:
            raise ValueError(f"output out range, the {emb_type} emb type is not suported!")
            # return self.get_gene_expression_embedding(adata)
