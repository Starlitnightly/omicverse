import numpy as np
from scipy import sparse
from sys import getsizeof

from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import igraph
import networkx as nx

def pairwise_scc(X1, X2):
    X1 = X1.argsort(axis=1).argsort(axis=1)
    X2 = X2.argsort(axis=1).argsort(axis=1)
    X1 = (X1-X1.mean(axis=1, keepdims=True))/X1.std(axis=1, keepdims=True)
    X2 = (X2-X2.mean(axis=1, keepdims=True))/X2.std(axis=1, keepdims=True)
    sccmat = np.empty([X1.shape[0], X2.shape[0]], float)
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            c = np.dot( X1[i,:], X2[j,:]) / float(X1.shape[1])
            sccmat[i,j] = c
    return sccmat

def standardize(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = np.empty_like(X)
    for i in range(X.shape[1]):
        if X_std[i] > 0:
            X_standard[:,i] = (X[:,i] - X_mean[i]) / X_std[i]
        else:
            X_standard[:,i] = 0.0
    return X_standard

def partial_corr(x, y, cov, method="spearman", aggregate=True):
    # Standardize data
    x = standardize(x)
    y = standardize(y)
    cov = standardize(cov)
    if aggregate:
        rs = []
        for i in range(cov.shape[1]):
            beta_x = np.linalg.lstsq(cov[:,i].reshape(-1,1), x, rcond=None)[0]
            beta_y = np.linalg.lstsq(cov[:,i].reshape(-1,1), y, rcond=None)[0]
            res_x = x - cov[:,i].reshape(-1,1) @ beta_x
            res_y = y - cov[:,i].reshape(-1,1) @ beta_y
            if method == "spearman":
                r,p = spearmanr(res_x, res_y)
            elif method == "pearson":
                r,p = pearsonr(res_x, res_y)
            rs.append(r)
        r = np.mean(rs); p = -1
    else:
        beta_x = np.linalg.lstsq(cov, x, rcond=None)[0]
        beta_y = np.linalg.lstsq(cov, y, rcond=None)[0]
        res_x = x - cov @ beta_x
        res_y = y - cov @ beta_y
        if method == "spearman":
            r,p = spearmanr(res_x, res_y)
        elif method == "pearson":
            r,p = pearsonr(res_x, res_y)
    return r, p

def semipartial_corr(x, y, xcov=None, ycov=None, method="spearman"):
    # Standardize data
    x = standardize(x)
    y = standardize(y)
    if not xcov is None:
        xcov = standardize(xcov)
    if not ycov is None:
        ycov = standardize(ycov)
    res_x = np.copy(x); res_y = np.copy(y)
    if not xcov is None:
        beta_x = np.linalg.lstsq(xcov, x, rcond=None)[0]
        res_x = x - xcov @ beta_x
    if not ycov is None:
        beta_y = np.linalg.lstsq(ycov, y, rcond=None)[0]
        res_y = y - ycov @ beta_y
    if method == "spearman":
        r,p = spearmanr(res_x, res_y)
    elif method == "pearson":
        r,p = pearsonr(res_x, res_y)
    return r, p

def treebased_score(x,
    y,
    cov,
    method="rf",
    n_trees=100,
    n_repeat=10,
    max_depth=5,
    max_features="sqrt",
    learning_rate=0.1,
    subsample=1.0
):
    random_seeds = np.random.randint(100000, size=n_repeat)
    X_train = np.concatenate((x.reshape(-1,1), cov), axis=1)
    ranks = []
    for i in range(n_repeat):
        if method == "rf":
            model = RandomForestRegressor(n_estimators=n_trees, 
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_seeds[i],
                n_jobs=-1
            )
        elif method == "gbt":
            model = GradientBoostingRegressor(n_estimators=n_trees,
                max_depth=max_depth,
                max_features=max_features,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_seeds[i]
            )
        model.fit(X_train, y)
        importance = model.feature_importances_
        rank = np.where(np.argsort(-importance)==0)[0][0]
        ranks.append(float(cov.shape[1]-rank)/float(cov.shape[1]))
    return np.mean(ranks)

def treebased_score_multifeature(
    X,
    y,
    cov,
    method='rf',
    n_trees=100,
    n_repeat=10,
    max_depth=5,
    max_features='sqrt',
    learning_rate=0.1,
    subsample=1.0
):
    random_seeds = np.random.randint(100000, size=n_repeat)
    X_train = np.concatenate((X, cov), axis=1)
    ranks = np.empty([X.shape[1], n_repeat], float)
    for i in range(n_repeat):
        if method == 'rf':
            model = RandomForestRegressor(n_estimators=n_trees, 
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_seeds[i],
                n_jobs=-1
            )
        elif method == 'gbt':
            model = GradientBoostingRegressor(n_estimators=n_trees,
                max_depth=max_depth,
                max_features=max_features,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_seeds[i]
            )
        model.fit(X_train, y)
        importance = model.feature_importances_
        sorted_idx = np.argsort(-importance)
        for j in range(X.shape[1]):
            rank = np.where(sorted_idx==j)[0][0]
            ranks[j,i] = float(X_train.shape[1] - rank - 1) / float(X_train.shape[1]-1)
    return np.mean(ranks, axis=1)
        

def vf_diff(vf_1, vf_2):
    norm_vf_1 = np.linalg.norm(vf_1, axis=1)
    norm_vf_2 = np.linalg.norm(vf_2, axis=1)
    dot_vf = (vf_1 * vf_2).sum(axis=1)
    
    s_cosine = dot_vf / (norm_vf_1 * norm_vf_2)
    s_cosine[np.isnan(s_cosine)] = 0.0

    s_coop = dot_vf

    return s_cosine, s_coop

def d_graph_local_jaccard(A1, A2):
    a = float( np.sum( (A1 != 0) * (A2 != 0) ) )
    b = float( np.sum( (A1 != 0) + (A2 != 0) ) )
    if b == 0:
        d = 1
    else:
        d = a / b
    return 1 - d

def d_graph_local_jaccard_weighted(A1, A2):
    a = float( np.sum( np.minimum(A1, A2) ) )
    b = float( np.sum( np.maximum(A1, A2) ) )
    if b == 0:
        d = 1
    else:
        d = a / b
    return 1 - d

# %%
def d_graph_mesoscale_heat(A1, A2, tau, p=2):
    A1_lap = sparse.csgraph.laplacian(A1, normed = True)
    w,v = np.linalg.eigh(A1_lap.todense())
    # tmp_idx = np.argsort(w)
    # w = w[tmp_idx]
    # v = v[:,tmp_idx]
    W = np.diag(w)
    R1 = v.real @ np.exp( - tau * W ) @ v.real.T

    A2_lap = sparse.csgraph.laplacian(A2, normed = True)
    w,v = np.linalg.eigh(A2_lap.todense())
    # tmp_idx = np.argsort(w)
    # w = w[tmp_idx]
    # v = v[:,tmp_idx]
    W = np.diag(w)
    R2 = v.real @ np.exp( - tau * W ) @ v.real.T

    delta = R1 - R2

    N = A1.shape[0]

    d = np.trace(delta.T @ delta) / float(N)

    # d = np.linalg.norm(R1-R2) ** 2 / float(N)
    
    return d

# %%
def d_graph_global_structure(A1, A2, w1=0.45, w2=0.45, w3=0.1):
    # Ref: Schieber, Tiago A., et al. "Quantification of network 
    # structural dissimilarities." Nature communications 8.1 (2017): 1-10.
    A1 = A1.astype(bool).astype(int)
    A2 = A2.astype(bool).astype(int)
    g1 = igraph.Graph(directed=True).Adjacency( (A1>0).tolist() )
    g2 = igraph.Graph(directed=True).Adjacency( (A2>0).tolist() )
    n = g1.vcount()
    m = g2.vcount()
    pg1 = nnd(g1)
    pg2 = nnd(g2)
    pm = np.zeros([max(m,n)], float)
    pm[:n-1] = pg1[:n-1]
    pm[-1] = pg1[n-1]
    pm[:m-1] = pm[:m-1] + pg2[:m-1]
    pm[-1] = pm[-1]+pg2[m-1]
    pm = pm / 2
    first = np.sqrt( max( (entropia(pm)-(entropia(pg1[:n])+entropia(pg2[:m]))/2)/np.log(2), 0) )
    second = np.abs(np.sqrt(pg1[n]) - np.sqrt(pg2[m]))

    pg1 = alpha(A1)
    pg2 = alpha(A2)
    m = max(len(pg1), len(pg2))
    Pg1 = np.zeros([m], float)
    Pg2 = np.zeros([m], float)
    Pg1[m-len(pg1):] = pg1[:]
    Pg2[m-len(pg2):] = pg2[:]
    third = np.sqrt( (entropia((Pg1+Pg2)/2)-(entropia(pg1)+entropia(pg2))/2 )/np.log(2))/2
    A1_compl = np.ones_like(A1) - A1
    for i in range(A1.shape[0]): A1_compl[i,i] = 1
    A2_compl = np.ones_like(A2) - A2
    for i in range(A2.shape[0]): A2_compl[i,i] = 1
    pg1 = alpha(A1_compl)
    pg2 = alpha(A2_compl)
    m = max(len(pg1), len(pg2))
    Pg1 = np.zeros([m], float)
    Pg2 = np.zeros([m], float)
    Pg1[m-len(pg1):] = pg1[:]
    Pg2[m-len(pg2):] = pg2[:]
    third = third + np.sqrt( (entropia((Pg1+Pg2)/2)-(entropia(pg1)+entropia(pg2))/2 )/np.log(2))/2

    return w1*first + w2*second + w3*third


def alpha(A):
    n = A.shape[0]
    g_nx = nx.from_numpy_array(A, create_using=nx.DiGraph)
    g_degree = dict( g_nx.degree() )
    for key in g_degree.keys():
        g_degree[key] = g_degree[key] / (n-1)
    g_nx.remove_edges_from(nx.selfloop_edges(g_nx))
    centrality = nx.katz_centrality(g_nx, beta=g_degree, alpha=1/float(n), normalized=False)
    centrality_vec = []
    for i in range(n):
        centrality_vec.append(centrality[i])
    centrality_vec = np.array(centrality_vec)
    r = np.sort(centrality_vec) / n**2
    rr = np.zeros([n+1],float)
    rr[:n] = r[:]
    rr[-1] = max(0,1-r.sum())
    return rr

def entropia(a):
    a = a[np.where(a>0)]
    return -np.sum(a*np.log(a))

def node_distance(g):
    n = g.vcount()
    a = np.zeros([n,n], float)
    m = np.array( g.shortest_paths(mode="ALL"), float )
    m[np.isinf(m)] = n
    quem = np.unique(m)
    quem = quem[np.where(quem!=0)]
    for mm in quem:
        tmp_mask = (m == mm)
        a[:,int(mm)-1] = tmp_mask.astype(float).sum(axis=1)
    return a / (n-1)

def nnd(g):
    n = g.vcount()
    nd = node_distance(g)
    pdfm = nd.mean(axis=0)
    norm = np.log( max(2, len(np.where(pdfm[:-1]>0)[0])+1) )
    y = max(0, entropia(pdfm)-entropia(nd)/n)/norm
    out = np.zeros([len(pdfm)+1],float)
    out[:n] = pdfm[:]; out[n] = y
    return out

# %%
def spatial_weight(
    X,
    bandwidth = None,
    k = 5,
    function = 'gaussian',
    row_standardize = False,
    zero_diagonal = False
):
    import libpysal
    kw = libpysal.weights.Kernel(X,
        bandwidth = bandwidth, k = k, function = function)
    if row_standardize:
        kw.transform = 'r'
    I = []; J = []; W = []
    for i in range(X.shape[0]):
        nnb = len(kw.neighbors[i])
        for j in range(nnb):
            if zero_diagonal and i == kw.neighbors[i][j]:
                continue
            I.append(i)
            J.append(kw.neighbors[i][j])
            W.append(kw.weights[i][j])
    I = np.array(I, int)
    J = np.array(J, int)
    W = np.array(W, float)
    
    return I, J, W

# %%
def tmp_moranI_vector_global(
    u,
    v,
    I,
    J,
    W
):
    """
    Liu, Yu, Daoqin Tong, and Xi Liu. "Measuring spatial autocorrelation
    of vectors." Geographical Analysis 47.3 (2015): 300-319.
    """
    n = float( len(u) )
    a = np.sum( W * (u[I]*u[J]+v[I]*v[J]) )
    b = np.sum( u**2 + v**2 )
    c = np.sum( W )
    moranI = ( n * a ) / ( b * c )
    return moranI

def tmp_moranI_vector_global_3d(
    u,
    v,
    w,
    I,
    J,
    W
):
    """
    Liu, Yu, Daoqin Tong, and Xi Liu. "Measuring spatial autocorrelation
    of vectors." Geographical Analysis 47.3 (2015): 300-319.
    """
    n = float( len(u) )
    a = np.sum( W * (u[I]*u[J]+v[I]*v[J] + w[I]*w[J]) )
    b = np.sum( u**2 + v**2 + w**2)
    c = np.sum( W )
    moranI = ( n * a ) / ( b * c )
    return moranI

def tmp_moranI_vector_local(
    u,
    v,
    I,
    J,
    W
):
    n = float( len(u) )
    m2 = np.sum( u**2 + v**2 ) / n
    print(m2)
    moranI = np.zeros([len(u)])
    for i in range(len(I)):
        moranI[I[i]] += u[I[i]] * W[i] * u[J[i]] + v[I[i]] * W[i] * v[J[i]]
    moranI = moranI / m2
    return moranI

def moranI_vector_global(
    X,
    V,
    weight_bandwidth = None,
    weight_k = 5,
    weight_function = 'triangular',
    weight_row_standardize = False,
    n_permutations = 999
):
    z_V = V - V.mean(axis=0)
    I, J, W = spatial_weight(X,
        bandwidth = weight_bandwidth, k = weight_k,
        function = weight_function, row_standardize = weight_row_standardize,
        zero_diagonal = True)
    if V.shape[1] == 2: 
        moranI = tmp_moranI_vector_global(z_V[:,0], z_V[:,1], I, J, W)
    elif V.shape[1] == 3:
        moranI = tmp_moranI_vector_global_3d(z_V[:,0], z_V[:,1], z_V[:,2], I, J, W)
    permute_moranI = []
    idx = np.arange(X.shape[0])
    for i in range(n_permutations):
        permute_idx = np.random.permutation(idx)
        if V.shape[1] == 2:
            tmp_I = tmp_moranI_vector_global(z_V[permute_idx,0], z_V[permute_idx,1], I, J, W)
        elif V.shape[1] == 3:
            tmp_I = tmp_moranI_vector_global_3d(z_V[permute_idx,0], z_V[permute_idx,1], z_V[permute_idx,2], I, J, W)
        permute_moranI.append(tmp_I)
    permute_moranI = np.abs( np.array(permute_moranI, float) )
    p_value = ( permute_moranI > np.abs(moranI) ).astype(float).sum() / float(n_permutations)
    
    return moranI, p_value

def moranI_vector_local(
    X,
    V,
    weight_bandwidth = None,
    weight_k = 5,
    weight_function = 'triangular',
    weight_row_standardize = False
):
    z_V = V - V.mean(axis=0)
    I, J, W = spatial_weight(X,
        bandwidth = weight_bandwidth, k = weight_k,
        function = weight_function, row_standardize = weight_row_standardize,
        zero_diagonal = True)
    moranI = tmp_moranI_vector_local(z_V[:,0], z_V[:,1], I, J, W)

    return moranI




# %%

def preprocess_vector_field(
    X,
    V,
    knn_smoothing = -1,
    normalize_vf = 'quantile',
    quantile = 0.99
):
    if knn_smoothing >= 2:
        nbrs = NearestNeighbors(n_neighbors=knn_smoothing, algorithm='ball_tree').fit(X)
        _, idx = nbrs.kneighbors(X)
        V_smoothed = np.zeros_like(V)
        for i in range(knn_smoothing):
            V_smoothed = V_smoothed + V[idx[:,i],:]
        V_smoothed = V_smoothed / float(knn_smoothing)
        V = V_smoothed
    V_norm = np.linalg.norm(V, axis=1)
    if normalize_vf == 'quantile':
        scale = np.quantile(V_norm, quantile)
        if scale > 0:
            V = V / scale
    elif normalize_vf == 'unit_norm':
        V = normalize(V)
    return V

# def d_vf_cosine_similarity(
#     X,
#     V1,
#     V2,
#     knn_smoothing = -1,
#     normalize = "quantile",
#     quantile = 0.99
# ):
#     if knn_smoothing >= 2:
#         nbrs = NearestNeighbors(n_neighbors=knn_smoothing, algorithm='ball_tree').fit(X)
#         _, idx = nbrs.kneighbors(X)
#         V_smoothed = np.zero_like(V)