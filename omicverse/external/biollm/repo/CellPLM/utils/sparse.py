import torch
import math

def sparse_diag(x):
    indices = torch.arange(len(x), device=x.device).unsqueeze(0).repeat(2, 1)
    values = x
    return torch.sparse_coo_tensor(indices, values, (len(x), len(x)), device=x.device)

def sparse_normalize(x):
    size_factor = sparse_diag(1. / (torch.sparse.sum(x, dim=1).to_dense() + 1e-8))
    res = torch.sparse.mm(size_factor, x)
    return res

def sparse_tpm(x):
    x = sparse_normalize(x) * 1e4
    x = torch.log1p(x)
    return x

def create_sparse_tensor(x, i):
    # x is a list of 4 tensors
    return torch.sparse_csr_tensor(x[0][i], x[1][i],
                                            x[2][i],
                                            x[3][i].tolist()).to_sparse().float().coalesce()

def mask_with_renormalize(x, mask, keep_nodes, mask_feature_rate):
    masked_x_seq = torch.sparse.FloatTensor(x.indices(),
                                            torch.where(mask[x.indices()[0],
                                                             x.indices()[1]],
                                                        torch.where(torch.isin(x.indices()[0],
                                                            torch.from_numpy(keep_nodes).to(x.device)),
                                                        x.values(),
                                                        x.values() + math.log(1) - math.log(1-mask_feature_rate))),
                                            x.shape)
    return masked_x_seq

def simple_mask(x, mask):
    masked_x_seq = torch.sparse.FloatTensor(x.indices(),
                                            torch.where(mask[x.indices()[0],
                                                             x.indices()[1]],
                                                        0.,
                                                        x.values()),
                                            x.shape)
    return masked_x_seq