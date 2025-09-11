from sklearn.neighbors import NearestNeighbors
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from .init_pca import init_pca
from .pca import pca
from .mds import mds
import scipy.sparse.linalg as sp_linalg
import numpy as np
import math
from tqdm import tqdm

from scipy.sparse.linalg import eigsh, LinearOperator


def learning_l(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch):
    """
    This function returns representation of the landmarks in the lower-dimensional space and the number of nearest
    neighbors of landmarks. It computes the gradient using probability matrix P and Q of data blocks.

    """
    # Obtain size and dimension of landmarks
    N, dim = X_samp.shape

    # Compute the number of nearest neighbors of landmarks adaptively
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Compute high-dimensional probability matrix P
    if k1 > 0:
        row = []
        col = []
        Pval = []
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in range(N):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            snn = np.zeros((1, N))
            snn[:, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            mod_dis = (1 - snn / max(np.max(snn), np.finfo(float).tiny)) ** agg_coef * cdist(X_samp[i:i + 1, :], X_samp)
            sort_dis = np.sort(mod_dis, axis=1)
            idx = np.argsort(mod_dis, axis=1)
            mean_samp_dis_squared = np.square(np.mean(sort_dis[0, :k2]))
            Pval.extend(
                np.exp(-0.5 * np.square(sort_dis[0, :k2]) / np.maximum(mean_samp_dis_squared, np.finfo(float).tiny)))
            row.extend((i * np.ones((k2, 1))).flatten().tolist())
            col.extend(idx[0, :k2])
        P = csr_matrix((Pval, (row, col)), shape=(N, N))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_samp, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_samp).kneighbors(X_samp)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis],
                                                              np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N))
    # Symmetrize matrix P
    P = (P + P.transpose()) / 2

    # Initialize embedding Y of landmarks
    if initialize == 'le':
        # 归一化图拉普拉斯：L = I - D^{-1/2} P D^{-1/2}
        # 注：若 P 对称，axis=0/1 求和等价；此处沿用你原实现的 axis=0。
        deg = np.asarray(P.sum(axis=0)).ravel()

        # 避免度为0导致的除零
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

        n = P.shape[0]

        # 通过 LinearOperator 隐式实现 L 的 matvec，避免显式构造大矩阵
        def matvec(x):
            # y = D^{-1/2} P D^{-1/2} x
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y  # (I - D^{-1/2} P D^{-1/2}) x

        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)

        # 取最小的 k 个特征对（含零特征值），丢掉首个零特征向量
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')

        # 保险起见按特征值从小到大排序后丢第一列
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]])
    elif initialize == 'pca':
        Y = pca(X_samp, no_dims)
    elif initialize == 'mds':
        Y = mds(X_samp, no_dims)

    # Normalize matrix P
    P = P / (np.sum(P) - N)

    # Compute the start and end markers of each data block
    no_blocks = math.ceil(N / 3000)
    mark = np.zeros((no_blocks, 2))
    for i in range(no_blocks):
        mark[i, :] = [i * math.ceil(N / no_blocks), min((i + 1) * math.ceil(N / no_blocks) - 1, N - 1)]

    # Initialization
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad = np.zeros((N, no_dims))
    
    # Use tqdm for epoch progress tracking
    for epoch in tqdm(range(1, T_epoch + 1), desc="Training epochs", unit="epoch"):
        # Update learning rate
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        Pgrad = np.zeros((N, no_dims))
        Qgrad = np.zeros((N, no_dims))
        sumQ = 0
        # Compute gradient

        for i in range(no_blocks):
            idx = [j for j in range(int(mark[i, 0]), int(mark[i, 1]) + 1)]
            D = cdist(Y[idx], Y) ** 2
            Q1 = 1 / (1 + np.log(1 + D))
            QQ1 = 1 / (1 + D)
            del D
            Pmat = -4 * P[idx, :].multiply(Q1).multiply(QQ1).toarray()
            Qmat = -4 * Q1 ** 2 * QQ1
            del QQ1
            len_blk = len(idx)
            idPQ = np.column_stack((np.array(range(len_blk)), idx[0] + np.array(range(len_blk))))
            Pmat[idPQ[:, 0], idPQ[:, 1]] = Pmat[idPQ[:, 0], idPQ[:, 1]] - np.sum(Pmat, axis=1)
            Qmat[idPQ[:, 0], idPQ[:, 1]] = Qmat[idPQ[:, 0], idPQ[:, 1]] - np.sum(Qmat, axis=1)
            Pgrad[idx] = Pmat @ Y
            Qgrad[idx] = Qmat @ Y
            del Pmat, Qmat
            sumQ = sumQ + np.sum(Q1)
        # Update embedding Y
        Y = Y - alpha * (Pgrad - Qgrad / (sumQ - N) + (epoch - 1) / (epoch + 2) * preGrad)
        preGrad = Pgrad - Qgrad / (sumQ - N)

    #print(str(epoch - 1) + ' epochs have been computed!')
    return Y, k2