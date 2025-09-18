import torch as th
import numpy as np

class CustomBatch:
    """
    Custom collate function for returning formatted batches
    """
    def __init__(self, data):
        
        transposed_data = list(zip(*data))
        S = th.stack(transposed_data[0], 0)[:,0]
        U = th.stack(transposed_data[1], 0)[:,0]
        normedS = th.stack(transposed_data[2], 0)[:,0]
        normedU = th.stack(transposed_data[3], 0)[:,0]
        maskS = th.stack(transposed_data[4], 0)[:,0]
        maskU = th.stack(transposed_data[5], 0)[:,0]
        spliced_size_factor = th.stack(transposed_data[6], 0)[:,0]
        unspliced_size_factor = th.stack(transposed_data[7], 0)[:,0]
        root = th.stack(transposed_data[8], 0)[:,0]
        velo_genes_mask = th.stack(transposed_data[10], 0)[:,0]
        batch_onehot = th.stack(transposed_data[11], 0)[:,0]
        batch_id = th.stack(transposed_data[12], 0)[:,0]
        exp_time = th.stack(transposed_data[13], 0)[:,0]
        celltype_id = th.stack(transposed_data[14], 0)[:,0]
        
        self.batch = {'S': S,
                'U': U,
                'normedS': normedS,
                'normedU': normedU,
                'maskS': maskS,
                'maskU': maskU,
                'spliced_size_factor': spliced_size_factor,
                'unspliced_size_factor': unspliced_size_factor,
                'root': root,
                'velo_genes_mask': velo_genes_mask,
                'batch_onehot': batch_onehot,
                'batch_id': batch_id,
                'exp_time': exp_time,
                      'celltype_id': celltype_id}

        
    def __getbatch__(self):
        return self.batch

def collate(batch):
    return CustomBatch(batch).__getbatch__()


class CustomBatchANVI:
    """
    Custom collate function for returning formatted batches
    """
    def __init__(self, data):
        
        transposed_data = list(zip(*data))
        S = th.stack(transposed_data[0], 0)[:,0]
        U = th.stack(transposed_data[1], 0)[:,0]
        normedS = th.stack(transposed_data[2], 0)[:,0]
        normedU = th.stack(transposed_data[3], 0)[:,0]
        maskS = th.stack(transposed_data[4], 0)[:,0]
        maskU = th.stack(transposed_data[5], 0)[:,0]
        spliced_size_factor = th.stack(transposed_data[6], 0)[:,0]
        unspliced_size_factor = th.stack(transposed_data[7], 0)[:,0]
        root = th.stack(transposed_data[8], 0)[:,0]
        velo_genes_mask = th.stack(transposed_data[10], 0)[:,0]
        batch_onehot = th.stack(transposed_data[11], 0)[:,0]
        batch_id = th.stack(transposed_data[12], 0)[:,0]
        celltype = th.stack(transposed_data[13], 0)[:,0]
        exp_time = th.stack(transposed_data[14], 0)[:,0]
        celltype_id = th.stack(transposed_data[15], 0)[:,0]
        
        self.batch = {'S': S,
                'U': U,
                'normedS': normedS,
                'normedU': normedU,
                'maskS': maskS,
                'maskU': maskU,
                'spliced_size_factor': spliced_size_factor,
                'unspliced_size_factor': unspliced_size_factor,
                'root': root,
                'velo_genes_mask': velo_genes_mask,
                'batch_onehot': batch_onehot,
                'batch_id': batch_id,
                'celltype': celltype,
                'exp_time': exp_time,
                      'celltype_id': celltype_id}

        
    def __getbatch__(self):
        return self.batch

def collate_anvi(batch):
    return CustomBatchANVI(batch).__getbatch__()
