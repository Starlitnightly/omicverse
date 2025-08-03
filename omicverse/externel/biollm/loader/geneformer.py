#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :geneformer.py
# @Time      :2025/4/15 17:40
# @Author    :Qiansqian Chen

import os
import numpy as np
import pickle

from .loader_base import LoaderBase
from transformers import BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification
from ..repo.geneformer.emb_extractor import get_embs
from ..repo.geneformer.tokenizer import TranscriptomeTokenizer
import torch
from scipy.sparse import issparse, csr_matrix
from ..data_preprocess.geneformer_handler import GeneformerHandler


class Geneformer(LoaderBase):
    """
    The LoadGeneformer class provides a specific implementation for loading and utilizing
    the Geneformer model, which can be used in various single-cell and gene expression analysis tasks.
    This class supports loading pre-trained loader, generating embeddings, and creating tokenized datasets
    from input data.

    Attributes:
        vocab (GeneVocab): Vocabulary object containing gene-to-index mappings.
        model (torch.nn.Module): Initialized model based on the specified model type and configuration.
    """
    def __init__(self, args=None, cfs_file=None, data_path=None, labels_num = 0):
        """
        Initializes the LoadGeneformer class, setting up configurations, loading the model,
        and placing it on the specified device.

        Args:
            args (Namespace, optional): Configuration arguments for the model and task.
            cfs_file (str, optional): Path to a configuration file for loading settings.
            data_path (str, optional): Path to the input data file.
        """
        super(Geneformer, self).__init__(args, cfs_file)
        self.vocab = self.load_vocab()
        if data_path is not None:
            self.args.input_file = data_path
        if os.path.exists(f'{self.args.output_dir}/label_dict.pk'):
            with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
                label_list = pickle.load(fp)
                self.num_classes = len(label_list)
        else:
            self.num_classes = labels_num
        self.model = self.load_model()
        self.model = self.model.to(self.args.device)
        self.data_handler = GeneformerHandler(vocab_path = self.args.vocab_file, gene_median_file = self.args.gene_median_file)

    def load_model(self):
        """
        Loads the specified model type based on the arguments provided.
        Supports loading different model types, such as pretrained, gene classifiers, and cell classifiers.

        Returns:
            torch.nn.Module: The initialized model based on specified type and parameters.
        """

        if self.args.model_type == "Pretrained":
            model = BertForMaskedLM.from_pretrained(self.args.model_file,
                                                    output_hidden_states=True,
                                                    output_attentions=False)

        else:
            # if os.path.exists(f'{self.args.output_dir}/label_dict.pk'):
            #     with open(f'{self.args.output_dir}/label_dict.pk', 'rb') as fp:
            #         label_list = pkl.load(fp)
            #     num_classes = len(label_list)
            # else:
            #     adata = sc.read_h5ad(self.args.input_file)
            #     num_classes = len(adata.obs[self.args.label_key].unique())
            # print(self.args.output_dir, self.args.input_file, num_classes)
            if self.args.model_type == "GeneClassifier":
                model = BertForTokenClassification.from_pretrained(self.args.model_file,
                                                                   num_labels=self.num_classes,
                                                                   output_hidden_states=True,
                                                                   output_attentions=False)

            elif self.args.model_type == "CellClassifier":
                model = BertForSequenceClassification.from_pretrained(self.args.model_file,
                                                                      num_labels=self.num_classes,
                                                                      output_hidden_states=True,
                                                                      output_attentions=False)

        return model

    def get_gene_embedding(self, gene_ids):
        """
        Gets gene embeddings for specified gene IDs.

        Args:
            gene_ids (list): List of gene IDs.

        Returns:
            np.ndarray: Gene embeddings as a NumPy array.
        """
        self.logger.info('start to get gene embedding!')
        idx2genes = self.get_idx2gene()
        if not isinstance(gene_ids, torch.Tensor):
            gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(self.args.device)
        emb = self.model.bert.embeddings.word_embeddings(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
            self.wandb.log({'GPU': total_gpu})
        gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2genes[i] for i in gene_ids]
        return {'gene_names': gene_names, 'gene_emb': emb.detach().cpu().numpy()}

    def get_cell_embedding(self,adata):
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
        self.logger.info("start to get cell embedding!")
        dataset = self.data_handler.load_data(adata=adata)

        emb = get_embs(
            self.model,
            dataset=dataset,
            emb_mode=self.args.emb_type,
            pad_token_id=self.vocab["<pad>"],
            forward_batch_size=self.args.batch_size, device=self.args.device)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
            self.wandb.log({'GPU': total_gpu})
        emb = emb.detach().cpu().numpy()
        self.logger.info("get cell embedding Done!")
        return emb

    def get_gene_expression_embedding(self, adata):
        """
                Computes gene expression embeddings for the provided AnnData object.

                This method processes the data in batches and retrieves embeddings
                for each gene. It averages embeddings across batches and returns
                both the embeddings and corresponding gene names.

                Args:
                    adata: AnnData object containing the gene expression data.

                Returns:
                    Tuple[np.ndarray, List[str]]: A tuple containing:
                        - A 2D NumPy array of shape (n_genes, embsize) with gene
                          embeddings.
                        - A list of gene names corresponding to the embeddings.

                Raises:
                    RuntimeError: If any issues arise during model inference or data processing.
                """
        self.logger.info("start to get gene expression!")
        dataset = self.data_handler.load_data(adata=adata)
        emb, gene_ids = get_embs(
            self.model,
            dataset=dataset,
            emb_mode=self.args.emb_type,
            pad_token_id=self.vocab["<pad>"],
            forward_batch_size=self.args.batch_size, device=self.args.device)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
            self.wandb.log({'GPU': total_gpu})
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        idx2genes = self.get_idx2gene()
        gene_names = [idx2genes[i] for i in gene_ids]
        self.logger.info("get gene expression Done!")
        return {'gene_names': gene_names, 'gene_emb': emb}


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
            return self.get_gene_expression_embedding(adata)

    def get_dataloader(self,
                       adata,
                       data_path,
                       cell_type_key,
                       nproc,
                       add_length
                       ):
        tokenized_dataset = self.data_handler.make_dataset(adata,data_path,cell_type_key,nproc,add_length)
        if isinstance(tokenized_dataset, tuple):
            tokenized_dataset = tokenized_dataset[0]
        # dataloader = self.data_handler.make_dataloader(dataset, batch_size, ddp_train, shuffle, drop_last, num_workers)
        return tokenized_dataset

if __name__ == "__main__":
    from biollm.utils.utils import load_config
    import pickle as pkl
    import os
    import scanpy as sc

    config_file = '../../tutorials_bak/zero-shot/configs/geneformer_gene-expression_emb.toml'
    configs = load_config(config_file)

    obj = Geneformer(configs)
    print(obj.args)
    adata = sc.read_h5ad(configs.input_file)

    obj.model = obj.model.to(configs.device)
    print(obj.model.device)
    emb = obj.get_embedding(obj.args.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir, exist_ok=True)
    with open(obj.args.output_dir + f'/geneformer_{obj.args.emb_type}_emb.pk', 'wb') as w:
        res = {'gene_names': list(obj.get_gene2idx().keys()), 'gene_emb': emb}
        pkl.dump(emb, w)