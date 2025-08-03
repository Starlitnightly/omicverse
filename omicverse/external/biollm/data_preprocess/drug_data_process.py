import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
import random
# import deepchem as dc
# from rdkit import Chem
from biollm.utils.utils import load_config


class DrugDataProcess:
    def __init__(self, config_file):
        self.args = load_config(config_file)
        self.device = self.args.device
        self.TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                          "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                          "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                          "STAD", "THCA", 'COAD/READ']

    # def process_drug(self):
    #     pubchemid2smile = {item.split('\t')[0]: item.split('\t')[1].strip() for item in
    #                        open(self.args.drug_smiles_file).readlines()}
    #
    #     if not os.path.exists(self.args.save_path):
    #         os.makedirs(f'{self.args.save_path}/drug_graph_feat')
    #         os.makedirs(f'{self.args.save_path}/data/leave_drug')
    #         os.makedirs(f'{self.args.save_path}/data/random_test')
    #         os.makedirs(f'{self.args.save_path}/data/test_data')
    #         os.makedirs(f'{self.args.save_path}/modelSave/no')
    #         os.makedirs(f'{self.args.save_path}/modelSave/scbert')
    #         os.makedirs(f'{self.args.save_path}/modelSave/scgpt')
    #         os.makedirs(f'{self.args.save_path}/modelSave/geneformer')
    #         os.makedirs(f'{self.args.save_path}/modelSave/scfoundation')
    #         os.makedirs(f'{self.args.save_path}/modelSave/scmamba')
    #     molecules = []
    #
    #     for each in pubchemid2smile.keys():
    #         print(each)
    #         molecules = []  # 分子结构列表
    #         molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
    #         featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    #         mol_object = featurizer.featurize(molecules)
    #         features = mol_object[0].atom_features
    #         degree_list = mol_object[0].deg_list
    #         adj_list = mol_object[0].canon_adj_list
    #         torch.save((features, adj_list, degree_list), '%s/drug_graph_feat/%s.pth' % (self.args.save_path, each))
    #         print(f'Drug gragh feature saved to ', '%s/drug_graph_feat/%s.pth' % (self.args.save_path, each))

    def MetadataGenerate(self, Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Drug_feature_file,
                         Gene_expression_file, Methylation_file, filtered):
        # drug_id --> pubchem_id
        with open(Drug_info_file, 'r') as f:
            reader = csv.reader(f)
            rows = [item for item in reader]
            drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

        # map cellline --> cancer type
        cellline2cancertype = {}
        with open(Cell_line_info_file) as f:
            for line in f.readlines()[1:]:
                cellline_id = line.split('\t')[1]
                TCGA_label = line.strip().split('\t')[-1]
                # if TCGA_label in TCGA_label_set:
                cellline2cancertype[cellline_id] = TCGA_label

        # load demap cell lines genomic mutation features
        mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
        cell_line_id_set = list(mutation_feature.index)

        # load drug features
        drug_pubchem_id_set = []
        drug_feature = {}
        for each in os.listdir(Drug_feature_file):
            drug_pubchem_id_set.append(each.split('.')[0])
            feat_mat, adj_list, degree_list = torch.load('%s/%s' % (Drug_feature_file, each))
            drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
        assert len(drug_pubchem_id_set) == len(drug_feature.values())

        # load gene expression features
        gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])

        # only keep overlapped cell lines
        mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]

        # load methylation
        methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
        assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
        experiment_data = pd.read_csv(self.args.cancer_response_exp_file, sep=',', header=0, index_col=[0])
        # filter experiment data
        drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
        experiment_data_filtered = experiment_data.loc[drug_match_list]

        data_idx = []
        for each_drug in experiment_data_filtered.index:
            for each_cellline in experiment_data_filtered.columns:
                pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[
                                        each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                        ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                        data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
        nb_celllines = len(set([item[0] for item in data_idx]))
        nb_drugs = len(set([item[1] for item in data_idx]))
        torch.save([mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx], f'{self.args.save_path}/data/meta.pth')
        print(
            '%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
        return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx

    def DataSplit(self, data_idx, ratio=0.95):
        data_train_idx, data_test_idx = [], []
        for each_type in self.TCGA_label_set:
            data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
            train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
            test_list = [item for item in data_subtype_idx if item not in train_list]
            data_train_idx += train_list
            data_test_idx += test_list
        print('Data split.')
        return data_train_idx, data_test_idx

    def DrugSplit(self, data_idx, all_drug_ids, leave_drug):
        test_drug_ids = all_drug_ids[leave_drug]
        train_drug_ids = [item for item in all_drug_ids if item not in test_drug_ids]
        data_test_idx = [item for item in data_idx if item[1] in test_drug_ids]
        data_train_idx = [item for item in data_idx if item[1] in train_drug_ids]
        print(f'Drug split. Fold:{leave_drug}. Drug: {all_drug_ids[leave_drug]}.')
        return data_train_idx, data_test_idx

    def NormalizeAdj(self, adj):
        adj = adj + torch.eye(adj.shape[0], device=self.device)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        a_norm = adj.mm(d).t().mm(d)
        return a_norm

    def random_adjacency_matrix(self, n):
        matrix = torch.randint(0, 2, (n, n), device=self.device)
        # No vertex connects to itself
        matrix.fill_diagonal_(0)
        # If i is connected to j, j is connected to i
        matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
        return matrix

    # calculate feature matrix and adjacency matrix
    def CalculateGraphFeat(self, feat_mat, adj_list):
        assert feat_mat.shape[0] == len(adj_list)
        feat = torch.zeros((self.args.max_atoms, feat_mat.shape[-1]), dtype=torch.float32, device=self.device)
        adj_mat = torch.zeros((self.args.max_atoms, self.args.max_atoms), dtype=torch.float32, device=self.device)
        if self.args.israndom:
            feat = torch.rand(self.args.max_atoms, feat_mat.shape[-1], device=self.device)
            adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = self.random_adjacency_matrix(self.args.max_atoms - feat_mat.shape[0])
        feat[:feat_mat.shape[0], :] = torch.from_numpy(feat_mat).to(self.device)
        for i in range(len(adj_list)):
            nodes = adj_list[i]
            for each in nodes:
                adj_mat[i, int(each)] = 1
        assert torch.allclose(adj_mat, adj_mat.T)
        adj_ = adj_mat[:len(adj_list), :len(adj_list)]
        adj_2 = adj_mat[len(adj_list):, len(adj_list):]
        norm_adj_ = self.NormalizeAdj(adj_)
        norm_adj_2 = self.NormalizeAdj(adj_2)
        adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
        adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2
        return [feat, adj_mat]

    def FeatureExtract(self, data_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature):
        cancer_type_list = []
        nb_instance = len(data_idx)
        nb_mutation_feature = mutation_feature.shape[1]
        nb_gexpr_features = gexpr_feature.shape[1]
        nb_methylation_features = methylation_feature.shape[1]

        # data initialization
        drug_data = [[] for item in range(nb_instance)]
        mutation_data = torch.zeros((nb_instance, 1, nb_mutation_feature, 1), dtype=torch.float32, device=self.device)
        gexpr_data = torch.zeros((nb_instance, nb_gexpr_features), dtype=torch.float32, device=self.device)
        methylation_data = torch.zeros((nb_instance, nb_methylation_features), dtype=torch.float32, device=self.device)
        target = torch.zeros(nb_instance, dtype=torch.float32, device=self.device)

        print('Feature Extracting...')
        for idx in tqdm(range(nb_instance)):
            cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
            # modify
            feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
            # fill drug data,padding to the same size with zeros
            drug_data[idx] = self.CalculateGraphFeat(feat_mat, adj_list)
            # randomlize X A
            mutation_data[idx, 0, :, 0] = torch.from_numpy(mutation_feature.loc[cell_line_id].values).float().to(self.device)
            gexpr_data[idx, :] = torch.from_numpy(gexpr_feature.loc[cell_line_id].values).float().to(self.device)
            methylation_data[idx, :] = torch.from_numpy(methylation_feature.loc[cell_line_id].values).float().to(self.device)
            target[idx] = ln_IC50
            cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
        return drug_data, mutation_data, gexpr_data, methylation_data, target, cancer_type_list

