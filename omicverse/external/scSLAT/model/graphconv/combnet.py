from typing import List, Optional, Union, Any

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

def sym_norm(edge_index:torch.Tensor,
             num_nodes:int,
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved:Optional[bool]=False,
             dtype:Optional[Any]=None
    )-> List:
    r"""
    Replace `GCNConv.norm` from https://github.com/mengliu1998/DeeperGNN/issues/2
    """
    from torch_scatter import scatter_add
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class CombUnweighted(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)
    
    Parameters
    ----------
    K
        K-hop neighbor to propagate
    """
    def __init__(self, K:Optional[int]=1,
                 cached:Optional[bool]=False,
                 bias:Optional[bool]=True,
                 **kwargs):
        super(CombUnweighted, self).__init__(aggr='add', **kwargs)
        self.K = K
        
    def forward(self, x:torch.Tensor,
                edge_index:torch.Tensor,
                edge_weight:Union[torch.Tensor,None]=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
        #                                 dtype=x.dtype)
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
                                        dtype=x.dtype)

        xs = [x]
        for k in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        return torch.cat(xs, dim = 1)
        # return torch.stack(xs, dim=0).mean(dim=0)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                        #  self.in_channels, self.out_channels,
                                         self.K)