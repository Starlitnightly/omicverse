#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: scgpt.py
@time: 2024/3/3 15:02
"""
from .loader_base import LoaderBase
from ..repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from ..repo.scgpt.model import TransformerModel
from ..algorithm.perturbation import ScgptPerturbation
import torch
from tqdm import tqdm
import numpy as np
from ..data_preprocess.scgpt_handler import ScgptHandler


class Scgpt(LoaderBase):
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
        super(Scgpt, self).__init__(args)
        self.vocab = self.load_vocab()
        self.model = self.load_model()
        self.init_model()
        self.model = self.model.to(self.args.device)
        self.data_handler = ScgptHandler(self.args.vocab_file)

    def load_model(self):
        """
        Loads the Transformer model with specified parameters.

        Returns:
            TransformerModel: The initialized Transformer model.
        """
        ntokens = len(self.vocab)
        model_param = {
            'ntoken': ntokens,
            'd_model': self.args.embsize,
            'nhead': self.args.nheads,
            'd_hid': self.args.d_hid,
            'nlayers': self.args.nlayers,
            'nlayers_cls': self.args.nlayers_cls if 'nlayers_cls' in self.args else 3,
            'n_cls': self.args.n_cls if 'n_cls' in self.args else 1,
            'dropout': 0.5,
            'pad_token': "<pad>",
            'do_mvc': self.args.do_mvc,
            'do_dab': self.args.do_dab,
            'use_batch_labels': False,
            'num_batch_labels': None,
            'domain_spec_batchnorm': False,
            'input_emb_style': "continuous",
            'cell_emb_style': "cls",
            'mvc_decoder_style': "inner product",
            'ecs_threshold': 0.3,
            'explicit_zero_prob': False,
            'fast_transformer_backend': "flash",
            'pre_norm': False,
            'vocab': self.vocab,
            'pad_value': self.args.pad_value,
            'n_input_bins': self.args.n_bins,
            'use_fast_transformer': True,
        }
        for i in model_param:
            if i in self.args:
                model_param[i] = self.args[i]
        print(model_param)
        model = TransformerModel(**model_param)
        return model

    def load_pert_model(self):
        """
        Loads the perturbation model with specified parameters.

        Returns:
            ScgptPerturbation: The initialized perturbation model.
        """
        ntokens = len(self.vocab)
        pert_model = ScgptPerturbation(
            ntokens,
            self.args.embsize,
            self.args.nheads,
            self.args.d_hid,
            self.args.nlayers,
            nlayers_cls=self.args.nlayers_cls,
            vocab=self.vocab,
            n_cls=1,
            dropout=self.args.dropout,
            pad_token=self.args.pad_token,
            pad_value=self.args.pad_value,
            pert_pad_id=self.args.pert_pad_id,
            do_mvc=self.args.MVC,
            cell_emb_style=self.args.cell_emb_style,
            mvc_decoder_style=self.args.mvc_decoder_style,
            use_fast_transformer=self.args.use_fast_transformer,
        )
        return pert_model

    def load_vocab(self):
        """
        Loads the gene vocabulary from a file and adds special tokens.

        Returns:
            GeneVocab: The loaded gene vocabulary.
        """
        vocab = GeneVocab.from_file(self.args.vocab_file)
        special_tokens = ['<pad>', '<cls>', '<eoc>']
        for token in special_tokens:
            if token not in vocab:
                vocab.append_token(token)
        vocab.set_default_index(vocab['<pad>'])
        return vocab

    def freezon_model(self, keep_layers=[-2]):
        """
        Freezes model parameters except for specified layers.

        Args:
            keep_layers (list): List of layers to keep trainable.
        """
        model_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        for name, param in self.model.named_parameters():
            if 'encoder' in name and "transformer_encoder" not in name:
                param.requires_grad = False
                print(name)
        ft_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        self.logger.info(f"Total pretrain-model Encoder Params {model_param_count}")
        self.logger.info(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")

    def get_dataloader(self,
                       adata,
                       var_key,
                       obs_key,
                       n_hvg,
                       bin_num,
                       batch_size,
                       ddp_train,
                       shuffle,
                       drop_last,
                       num_workers=1,
                       obs_id_output=None,
                       sort_seq_batch=None,
                       batch_id_key=None):
        adata = self.data_handler.preprocess(adata, var_key, obs_key, obs_id_output, n_hvg, bin_num)
        if obs_key:
            dataset = self.data_handler.make_dataset(adata, self.vocab, self.args, sort_seq_batch,
                                                     obs_id_key=f'{obs_key}_id', batch_id_key=batch_id_key)
        else:
            dataset = self.data_handler.make_dataset(adata, self.vocab, self.args, sort_seq_batch,
                                                     obs_id_key=None, batch_id_key=batch_id_key)
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
        gene_embeddings = self.model.encoder(gene_ids)
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        self.logger.info(f'finished get gene embedding!')
        idx2gene = self.get_idx2gene()
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2gene[i] for i in gene_ids]
        return {'gene_names': gene_names, 'gene_emb': gene_embeddings.detach().cpu().numpy()}

    def encoder(self, batch_data):
        """
        Encodes the batch data to obtain gene embeddings.

        Args:
            batch_data (dict): A dictionary containing gene IDs and values.

        Returns:
            Tensor: Encoded embeddings of the input batch.
        """
        input_gene_ids = batch_data["gene_ids"].to(self.args.device)
        input_values = batch_data["values"].to(self.args.device)

        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
        embeddings = self.model._encode(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )  # batch_size * max_seq_len * dim
        return embeddings

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
                                          bin_num=self.args.n_bins, batch_size=self.args.batch_size, ddp_train=False,
                                          shuffle=False, drop_last=False)
        self.logger.info('get dataloader Done!')
        cell_embeddings = np.zeros((adata.shape[0], self.args.embsize), dtype=np.float32)
        # celltypes = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for batch_data in tqdm(data_loader, desc='Cell embedding'):
                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                # celltypes.extend(list(batch_data["celltype_labels"].detach().cpu().numpy()))
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
                output = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                embeddings = output['cell_emb']  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count: count + len(embeddings)] = embeddings
                count += len(embeddings)
            # cell_embeddings = cell_embeddings / np.linalg.norm(
            #     cell_embeddings, axis=1, keepdims=True
            # )
            if self.wandb:
                total_gpu = torch.cuda.memory_allocated() / (1024 ** 3)
                self.wandb.log({'GPU': total_gpu})
            self.logger.info("get cell embedding Done!")
            return cell_embeddings

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
        data_loader = self.get_dataloader(adata=adata, var_key=self.args.var_key, obs_key=None, n_hvg=self.args.n_hvg,
                                          bin_num=self.args.n_bins, batch_size=self.args.batch_size, ddp_train=False,
                                          shuffle=False, drop_last=False)
        gene_embs = {}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):

            for batch_data in tqdm(data_loader, desc='Gene expression embedding'):

                input_gene_ids = batch_data["gene_ids"].to(self.args.device)
                input_values = batch_data["values"].to(self.args.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.args.pad_token])
                embeddings = self.model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )  # batch_size * max_seq_len * dim
                embeddings = embeddings.detach().cpu().numpy()
                for m in range(len(embeddings)):
                    for n in range(len(embeddings[m])):
                        if input_gene_ids[m][n].item() not in gene_embs.keys():
                            gene_embs.update({input_gene_ids[m][n].item(): [embeddings[m][n]]})
                        else:
                            gene_embs[input_gene_ids[m][n].item()].append(embeddings[m][n])
        if self.wandb:
            total_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.wandb.log({'GPU': total_gpu})
        if 60694 in gene_embs:
            del gene_embs[60694]

        for k, v in gene_embs.items():
            gene_embs[k] = np.mean(np.stack(v), axis=0)

        gene_ids = list(gene_embs.keys())
        gene_embs = np.stack(list(gene_embs.values()))

        idx2gene = self.get_idx2gene()
        if isinstance(gene_ids, torch.Tensor):
            gene_ids = gene_ids.detach().cpu().numpy()
        gene_names = [idx2gene[i] for i in gene_ids]
        self.logger.info("get gene expression Done!")

        return {'gene_names': gene_names, 'gene_embs': gene_embs}

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
