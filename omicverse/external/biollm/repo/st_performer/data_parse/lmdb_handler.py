#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: lmdb_handler.py
@time: 2024/1/11 13:49

change log:
2024/1/19 11:13  write code of write h5ad3lmdb to lmdb_handler.py
"""
import os
import lmdb
import scanpy as sc
import pandas as pd
from st_performer.tokenizer import GeneTokenizer, OraganTokenizer, DiseaseTokenizer, Tokennizer
import json
from scipy.sparse import issparse
from st_performer.data_parse.data_utils import normalize
from st_performer.utils import get_logger
import numpy as np
from traceback import print_exc


class LmdbHelper(object):
    def __init__(self, lmdb_path, vocab_dir, write_frequency=50000, map_size=int(5000e9), is_write=False, data_source='stereo',
                 sample_path=None, celltype_curated_path=None):
        self.lmdb_path = lmdb_path
        self.vocab_dir = vocab_dir
        self.write_frequency = write_frequency
        self.db = None
        self.txn = None
        self.gene_tokenizer = GeneTokenizer(os.path.join(vocab_dir, 'gene_vocab.json'))
        self.organ_tokenizer = OraganTokenizer(os.path.join(vocab_dir, 'organ_vocab.json'))
        self.disease_tokenizer = DiseaseTokenizer(os.path.join(vocab_dir, 'disease_vocab.json'))
        self.celltype_tokenizer = Tokennizer(os.path.join(vocab_dir, 'celltype_vocab.json'))
        self.sequence_tokenizer = Tokennizer(os.path.join(vocab_dir, 'sequence_vocab.json'))
        self.data_source = data_source
        self.sample_df = pd.read_csv(sample_path, header=0) if sample_path else None
        self.curated_celltype = self.celltype_map(celltype_curated_path) if celltype_curated_path else {}
        self.logger = get_logger()
        self.file_index = {}
        # self.init_db(map_size, is_write)
        self.logger.info('Gene vocab size: {}'.format(self.gene_tokenizer.vocab_size))

    @staticmethod
    def celltype_map(cell_path):
        df = pd.read_csv(cell_path, header=0)[['celltype', 'ontology_name']]
        df.drop_duplicates(inplace=True)
        res = dict(zip(df['celltype'], df['ontology_name']))
        return res

    @staticmethod
    def get_cells_number(adata_path):
        adata = sc.read_h5ad(adata_path, backed='r')
        cells = adata.obs.shape[0]
        adata.file.close()
        return cells

    def update_file_index(self, file_path):
        self.file_index[file_path] = len(self.file_index)

    def get_txn(self, write=True):
        return self.db.begin(write=write)

    def init_db(self, map_size, is_write):
        self.db = lmdb.open(self.lmdb_path, map_size=map_size)
        self.txn = self.get_txn(is_write)

    def get_length(self):
        res = self.txn.get(b'__len__')
        length = int(res.decode("utf-8")) if res else 0
        return length

    def h5ad2lmdb(self, h5ad_file, db_length=0, source='stereo'):
        cells = self.get_cells_number(h5ad_file)
        self.logger.info('cell number before qc: {}'.format(cells))
        self.update_file_index(h5ad_file)
        gene_index = self.file_index[h5ad_file]
        if cells > 100000:
            adata = sc.read_h5ad(h5ad_file, backed='r')
            for j in range(0, cells, 30000):
                sub_adata = adata[j: j + 30000, :].to_memory()
                if source == 'stereo':
                    db_length = self.parse_stereo_miner_data(sub_adata, self.sample_df, gene_index, db_length)
                if source == 'cellxgene':
                    db_length = self.parse_cxg_data(sub_adata, gene_index, db_length)
                if source == 'panglao':
                    db_length = self.parse_panglao_data(sub_adata, gene_index, db_length)
        else:
            adata = sc.read_h5ad(h5ad_file)
            if source == 'stereo':
                db_length = self.parse_stereo_miner_data(adata, self.sample_df, gene_index, db_length)
            if source == 'cellxgene':
                db_length = self.parse_cxg_data(adata, gene_index, db_length)
            if source == 'panglao':
                db_length = self.parse_panglao_data(adata, gene_index, db_length)
        return db_length

    def folder2lmdb(self, h5ad_folder, db_length=0):
        files = os.listdir(h5ad_folder)
        if os.path.exists(self.lmdb_path + '/deal_files.txt'):
            with open(self.lmdb_path + '/deal_files.txt', 'r') as tmpf:
                deal_file_list = [i.strip('\n') for i in tmpf]
                for i in deal_file_list:
                    if i not in self.file_index:
                        self.update_file_index(i)
        else:
            deal_file_list = []
        fd = open(self.lmdb_path + '/deal_files.txt', 'a')
        for f in files:
            if len(deal_file_list) > 0 and os.path.join(h5ad_folder, f) in deal_file_list:
                continue
            try:
                self.logger.info("db_length: {}".format(db_length))
                if f.endswith('.h5ad'):
                    db_length = self.h5ad2lmdb(os.path.join(h5ad_folder, f), db_length, self.data_source)
                fd.write('{}\n'.format(os.path.join(h5ad_folder, f)))
                fd.flush()
                self.logger.info("deal file: {}".format(os.path.join(h5ad_folder, f)))
            except Exception as e:
                self.logger.error("error: {}, file: {}".format(e, os.path.join(h5ad_folder, f)))
                print_exc()
        # self.gene_tokenizer.vocab_to_json()
        self.disease_tokenizer.vocab_to_json()
        self.celltype_tokenizer.vocab_to_json()
        self.sequence_tokenizer.vocab_to_json()
        self.organ_tokenizer.vocab_to_json()
        self.txn.commit()
        self.txn = self.get_txn(write=True)
        self.logger.info('end to load data {}, db_length: {}'.format(h5ad_folder, db_length))
        self.db.sync()
        self.db.close()
        return db_length

    def files2lmdb(self, file_path, db_length=0):
        with open(file_path, 'r') as fd:
            files = [f.strip() for f in fd.readlines()]
        self.logger.info("files len: {}".format(len(files)))
        if os.path.exists(self.lmdb_path + '/deal_files.txt'):
            with open(self.lmdb_path + '/deal_files.txt', 'r') as tmpf:
                deal_file_list = [i.strip('\n') for i in tmpf]
                for i in deal_file_list:
                    if i not in self.file_index:
                        self.update_file_index(i)
        else:
            deal_file_list = []
        fd = open(self.lmdb_path + '/deal_files.txt', 'a')
        self.logger.info('start to load the error file: {}'.format(file_path))

        for f in files:
            if len(deal_file_list) > 0 and f in deal_file_list:
                continue
            try:
                self.logger.info("db_length: {}".format(db_length))
                self.logger.info("start to parse the file: {}".format(f))
                if f.endswith('.h5ad'):
                    db_length = self.h5ad2lmdb(f, db_length, self.data_source)
                fd.write('{}\n'.format(f))
                fd.flush()
                self.logger.info("deal file: {}".format(f))
            except Exception as e:
                self.logger.error("error: {}, file: {}".format(e, f))
                print_exc()
        # self.gene_tokenizer.vocab_to_json()
        self.disease_tokenizer.vocab_to_json()
        self.celltype_tokenizer.vocab_to_json()
        self.sequence_tokenizer.vocab_to_json()
        self.organ_tokenizer.vocab_to_json()
        self.txn.commit()
        self.txn = self.get_txn(write=True)
        self.logger.info('end to load data {}, db_length: {}'.format(file_path, db_length))
        self.db.sync()
        self.db.close()
        return db_length


    @staticmethod
    def get_token_ids(tokenizer, tokens_list):
        token_ids = []
        use_index = []
        for k, i in enumerate(tokens_list):
            if i in tokenizer.vocab:
                use_index.append(k)
                token_ids.append(tokenizer.vocab[i])
        return token_ids, use_index


    def parse_stereo_miner_data(self, adata, sample_df, gene_index, db_length=0):
        adata = normalize(adata)
        adata.obs = (pd.merge(adata.obs, sample_df, on=['sample_id'], how='left')).set_index(adata.obs.index)
        sparse_flag = issparse(adata.X)
        result = {}
        gene_id, use_index = self.get_token_ids(self.gene_tokenizer, adata.var['Symbol'].values)
        adata.X = adata.X.astype(float)
        adata.obs.fillna('<unk>', inplace=True)
        # print(adata.obs['disease'].value_counts())
        for i in range(adata.obs.shape[0]):
            express_x = adata.X[i].A[use_index].tolist() if sparse_flag else list(adata.X[i][use_index])
            result['express_x'] = express_x
            organ = adata.uns['organ'].replace("'", "")
            result['organ'] = self.get_token_ids(self.organ_tokenizer, [organ])[0]
            result['sequence'] = self.get_token_ids(self.sequence_tokenizer, [adata.uns['platform']])[0]
            result['disease'] = self.get_token_ids(self.disease_tokenizer, [adata.obs['disease'][i]])[0]
            result['celltype'] = self.get_token_ids(self.celltype_tokenizer,
                                                    [self.curated_celltype.get(adata.obs['cell_type'][i],
                                                                               adata.obs['cell_type'][i])])[0]
            result['gene_index'] = 'g{}'.format(gene_index)
            self.write_lmdb(result, db_length)
            db_length += 1
            if db_length % self.write_frequency == 0:
                self.txn.commit()
        self.txn.put(b'__len__', str(db_length).encode())
        self.txn.put('g{}'.format(gene_index).encode(), np.array(gene_id))
        self.txn.commit()
        return db_length

    def parse_cxg_data(self, adata, gene_index, db_length=0):
        _, unique_index = np.unique(adata.var['feature_name'].values, return_index=True)
        adata = adata[:, unique_index]
        sc.pp.calculate_qc_metrics(adata)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata = normalize(adata)
        sparse_flag = issparse(adata.X)
        result = {}
        gene_id, use_index = self.get_token_ids(self.gene_tokenizer, adata.var['feature_name'].values)
        adata.X = adata.X.astype(float)
        self.logger.info('adata cell number: {}'.format(adata.obs.shape[0]))
        self.logger.info('gene index: {} '.format(gene_index))
        for i in range(adata.obs.shape[0]):
            express_x = adata.X[i].A.reshape(-1)[use_index].tolist() if sparse_flag else list(adata.X[i][use_index])
            if len(express_x) != len(gene_id):
                self.logger.info('data Error: the length of genes is not equal to the length of express_x.')
                continue
            result['express_x'] = express_x
            organ = adata.obs['general_tissue'].iloc[i]
            result['organ'] = organ
            self.organ_tokenizer.update_vocab(organ)
            result['sequence'] = adata.obs['assay'].iloc[i]
            self.sequence_tokenizer.update_vocab(result['sequence'])
            result['disease'] = adata.obs['disease'].iloc[i]
            self.disease_tokenizer.update_vocab(result['disease'])
            celltype = self.curated_celltype.get(adata.obs['cell_type'].iloc[i], adata.obs['cell_type'].iloc[i])
            result['celltype'] = celltype
            self.celltype_tokenizer.update_vocab(celltype)
            result['gene_index'] = 'g{}'.format(gene_index)
            self.write_lmdb(result, db_length)
            db_length += 1
            if db_length % self.write_frequency == 0:
                self.txn.commit()
                self.txn = self.get_txn(write=True)
                self.logger.info('write to lmbd: {}'.format(db_length))
        self.txn.put(b'__len__', str(db_length).encode())
        self.txn.put('g{}'.format(gene_index).encode(), np.array(gene_id))
        # self.txn.commit()
        return db_length

    def parse_panglao_data(self, adata, gene_index, db_length=0):
        adata = normalize(adata)
        sparse_flag = issparse(adata.X)
        result = {}
        gene_id, use_index = self.get_token_ids(self.gene_tokenizer, adata.var['Symbol'].values)
        adata.X = adata.X.astype(float)
        self.logger.info('adata cell number: {}'.format(adata.obs.shape[0]))
        self.logger.info('gene index: {} '.format(gene_index))
        for i in range(adata.obs.shape[0]):
            express_x = adata.X[i].A.reshape(-1)[use_index].tolist() if sparse_flag else list(adata.X[i][use_index])
            result['express_x'] = express_x
            organ = adata.uns['organ']
            result['organ'] = organ
            self.organ_tokenizer.update_vocab(organ)
            result['sequence'] = adata.uns['platform']
            self.sequence_tokenizer.update_vocab(result['sequence'])
            result['disease'] = '<unk>'
            self.disease_tokenizer.update_vocab(result['disease'])
            celltype = '<unk>'
            result['celltype'] = celltype
            self.celltype_tokenizer.update_vocab(celltype)
            result['gene_index'] = 'g{}'.format(gene_index)
            self.write_lmdb(result, db_length)
            db_length += 1
            if db_length % self.write_frequency == 0:
                self.txn.commit()
                self.txn = self.get_txn(write=True)
                self.logger.info('write to lmbd: {}'.format(db_length))
        self.txn.put(b'__len__', str(db_length).encode())
        self.txn.put('g{}'.format(gene_index).encode(), np.array(gene_id))
        # self.txn.commit()
        return db_length

    def write_lmdb(self, record, db_length):
        index = db_length
        self.txn.put(str(index).encode(), json.dumps(record).encode())
        return index

    def read_lmdb(self, index):
        res = json.loads(self.txn.get(str(index).encode()))
        gene_id = np.frombuffer(self.txn.get(res['gene_index']), dtype=np.int64)
        res['gene_id'] = gene_id
        return res


if __name__ == '__main__':
    import sys
    is_parse_all = 0
    is_parse_organ = 0
    # is_parse_all = int(sys.argv[1])
    cellxgene_path = "/home/share/huadjyin/home/s_huluni/cellxgene/curated/RNA/Homo_sapiens"
    lmdb_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/cellxgene_6w/'
    vocab_dir = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/cellxgene_6w/vocab.2024.03.06/'
    data_source = 'cellxgene'
    # cellxgene_path = "/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/panglao/human_qc"
    # lmdb_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/panglao/gene2vec'
    # vocab_dir = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/panglao/gene2vec/vocab/'
    # data_source = 'panglao'
    sample_path = None
    if is_parse_all == 1:
        organs = ["adipose_tissue", "adrenal_gland", "axilla", "bladder_organ", "bone_marrow", "breast",
                  "central_nervous_system", "colon", "digestive_system", "embryo", "endocrine_gland",
                  "esophagogastric_junction", "esophagus", "exocrine_gland", "eye", "fallopian_tube", "gallbladder",
                  "heart", "immune_system", "intestine", "kidney", "lamina_propria", "large_intestine", "liver", "lung",
                  "lymph_node", "mucosa", "musculature", "nose", "omentum", "ovary", "pancreas", "placenta", "pleura",
                  "pleural_fluid", "prostate_gland", "reproductive_system", "respiratory_system", "saliva",
                  "skeletal_system", "skin_of_body", "small_intestine", "spinal_cord", "spleen", "stomach", "testis",
                  "tongue", "ureter", "uterus", "vasculature", "yolk_sac", "blood", "brain"]

        map_size = int(3000e9)
        # map_size = int(10000e9)
        obj = LmdbHelper(lmdb_path + 'all.db', vocab_dir, write_frequency=10000, is_write=True, data_source=data_source,
                         sample_path=sample_path, map_size=map_size)
        obj.init_db(map_size, is_write=True)
        if os.path.exists(lmdb_path + 'all.db'):
            init_num = obj.get_length()
        else:
            init_num = 0
        obj.txn.commit()
        obj.db.close()
        obj.logger.info('lmbd length is: {}'.format(init_num))
        if data_source == 'cellxgene':
            for i in organs:
                obj.init_db(map_size, is_write=True)
                db_len = obj.folder2lmdb(os.path.join(cellxgene_path, i), init_num)
                init_num = db_len
        if data_source == 'panglao':
            obj.init_db(map_size, is_write=True)
            db_len = obj.folder2lmdb(cellxgene_path, init_num)
    if is_parse_organ == 1:
        organs = ["kidney", "heart", "lung", "brain", "blood"]
        map_size = int(5000e9)
        for i in organs:
            if not os.path.exists(os.path.join(vocab_dir, i)):
                os.mkdir(os.path.join(vocab_dir, i))
            obj = LmdbHelper(lmdb_path + i + '.db', os.path.join(vocab_dir, i), write_frequency=10000, is_write=True,
                             data_source=data_source, sample_path=sample_path, map_size=map_size)
            obj.init_db(map_size, is_write=True)
            if os.path.exists(lmdb_path + i + '.db'):
                init_num = obj.get_length()
            else:
                init_num = 0
            obj.logger.info('lmbd length is: {}'.format(init_num))
            obj.folder2lmdb(os.path.join(cellxgene_path, i), init_num)

    is_parse_error = 1
    if is_parse_error:
        error_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/st_performer/logs/load_error_h5ad.txt'
        # map_size = int(3000e9)
        map_size = int(10000e9)
        obj = LmdbHelper(lmdb_path + 'all.db.2024.03.06', vocab_dir, write_frequency=10000, is_write=True, data_source=data_source,
                         sample_path=sample_path, map_size=map_size)
        obj.init_db(map_size, is_write=True)
        if os.path.exists(lmdb_path + 'all.db.2024.03.06'):
            init_num = obj.get_length()
        else:
            init_num = 0
        obj.txn.commit()
        obj.db.close()
        obj.logger.info('lmbd length is: {}'.format(init_num))
        obj.init_db(map_size, is_write=True)
        db_len = obj.files2lmdb(error_path, init_num)
