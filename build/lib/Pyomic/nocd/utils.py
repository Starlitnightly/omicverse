"""Various utility functions."""
import numpy as np
import scipy.sparse as sp
import torch

from typing import Union


def l2_reg_loss(model, scale=1e-5):
    """Get L2 loss for model weights."""
    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale


def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     cuda: bool = True,
                     ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:
    """Convert a scipy sparse matrix to a torch sparse tensor.

    Args:
        matrix: Sparse matrix to convert.
        cuda: Whether to move the resulting tensor to GPU.

    Returns:
        sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

    """
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor.coalesce()


def coms_list_to_matrix(communities_list, num_nodes=None):
    """Convert a communities list of len [C] to an [N, C] communities matrix.

    Parameters
    ----------
    communities_list : list
        List of lists of nodes belonging to respective community.
    num_nodes : int, optional
        Total number of nodes. This needs to be here in case
        some nodes are not in any communities, but the resulting
        matrix must have the correct shape [num_nodes, num_coms].

    Returns
    -------
    communities_matrix : np.array, shape [num_nodes, num_coms]
        Binary matrix of community assignments.
    """
    num_coms = len(communities_list)
    if num_nodes is None:
        num_nodes = max(max(cmty) for cmty in communities_list) + 1
    communities_matrix = np.zeros([num_nodes, num_coms], dtype=np.float32)
    for cmty_idx, nodes in enumerate(communities_list):
        communities_matrix[nodes, cmty_idx] = 1
    return communities_matrix


def coms_matrix_to_list(communities_matrix):
    """Convert an [N, C] communities matrix to a communities list of len [C].

    Parameters
    ----------
    communities_matrix : np.ndarray or sp.spmatrix, shape [num_nodes, num_coms]
        Binary matrix of community assignments.

    Returns
    -------
    communities_list : list
        List of lists of nodes belonging to respective community.

    """
    num_nodes, num_coms = communities_matrix.shape
    communities_list = [[] for _ in range(num_coms)]
    nodes, communities = communities_matrix.nonzero()
    for node, cmty in zip(nodes, communities):
        communities_list[cmty].append(node)
    return communities_list


def plot_sparse_clustered_adjacency(A, num_coms, z, o, ax=None, markersize=0.25):
    import seaborn as sns
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    colors = sns.color_palette('hls', num_coms)
    sns.set_style('white')

    crt = 0
    for idx in np.where(np.diff(z[o]))[0].tolist() + [z.shape[0]]:
        ax.axhline(y=idx, linewidth=0.5, color='black', linestyle='--')
        ax.axvline(x=idx, linewidth=0.5, color='black', linestyle='--')
        crt = idx + 1

    ax.spy(A[o][:, o], markersize=markersize)
    ax.tick_params(axis='both', which='both', labelbottom='off', labelleft='off', labeltop='off')


def adjacency_split_naive(A, p_val, neg_mul=1, max_num_val=None):
    edges = np.column_stack(sp.tril(A).nonzero())
    num_edges = edges.shape[0]
    num_val_edges = int(num_edges * p_val)
    if max_num_val is not None:
        num_val_edges = min(num_val_edges, max_num_val)

    shuffled = np.random.permutation(num_edges)
    which_val = shuffled[:num_val_edges]
    which_train = shuffled[num_val_edges:]
    train_ones = edges[which_train]
    val_ones = edges[which_val]
    A_train = sp.coo_matrix((np.ones_like(train_ones.T[0]), (train_ones.T[0], train_ones.T[1])),
                            shape=A.shape).tocsr()
    A_train = A_train.maximum(A_train.T)

    num_nodes = A.shape[0]
    num_val_nonedges = neg_mul * num_val_edges
    candidate_zeros = np.random.choice(np.arange(num_nodes, dtype=np.int32),
                                       size=(2 * num_val_nonedges, 2), replace=True)
    cne1, cne2 = candidate_zeros[:, 0], candidate_zeros[:, 1]
    to_keep = (1 - A[cne1, cne2]).astype(np.bool).A1
    val_zeros = candidate_zeros[to_keep][:num_val_nonedges]
    if to_keep.sum() < num_val_nonedges:
        raise ValueError("Couldn't produce enough non-edges")

    return A_train, val_ones, val_zeros
