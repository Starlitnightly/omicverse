# %%
import igraph
import numpy as np
from sklearn.neighbors import kneighbors_graph

def knn_graph(D,k):
    """Construct a k-nearest-neighbor graph as igraph object.
    :param D: a distance matrix for constructing the knn graph
    :type D: class:`numpy.ndarray`
    :param k: number of nearest neighbors
    :type k: int
    :return: a knn graph object
    :rtype: class:`igraph.Graph`
    """

    n = D.shape[0]
    G = igraph.Graph()
    G.add_vertices(n)
    edges = []
    weights = []
    for i in range(n):
        sorted_ind = np.argsort(D[i,:])
        for j in range(1,1+k):
            # if i < sorted_ind[j]:
            edges.append( (i, sorted_ind[j]) )
            weights.append(D[i, sorted_ind[j]])
                # weights.append(1.)
    G.add_edges(edges)
    G.es['weight'] = weights
    return G

def knn_graph_embedding(X, k):
    A = kneighbors_graph(X, k, include_self=False, mode='distance')
    G = igraph.Graph.Weighted_Adjacency(A)
    return (G)

def leiden_clustering(
    D,
    k = 5,
    resolution = 1.0,
    random_seed = 1,
    n_iterations = -1,
    input = 'distance'
):
    import leidenalg
    if input == 'distance':
        G = knn_graph(D, k)
    elif input == 'embedding':
        G = knn_graph_embedding(D, k)
    partition_kwargs = {'resolution_parameter':resolution,
        'seed': random_seed,
        'n_iterations': n_iterations}
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, 
        **partition_kwargs)
    return np.array( part.membership, int )
# %%
