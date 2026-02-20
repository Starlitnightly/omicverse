from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch

# Try to import PyG KNN
try:
    from ...pp.pyg_knn_implementation import pyg_knn_search
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return -torch.log1p(a * distances ** (2 * b))

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (torch.nn.functional.logsigmoid(probabilities_distance)-probabilities_distance)
        * repulsion_strength)

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(
        distance_embedding, _a, _b
    )
    # set true probabilities based on negative sampling
    # Use the same device as the embeddings
    device = embedding_to.device
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)), dim=0,
    ).to(device)

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph,
        probabilities_distance,
    )
    loss = torch.mean(ce_loss)
    return loss

def get_umap_graph(X, n_neighbors=10, metric="cosine", random_state=None, use_pyg='auto'):
    """
    Build UMAP graph with optional PyG KNN acceleration.

    Arguments:
        X: Input data (numpy array or torch tensor)
        n_neighbors: Number of neighbors
        metric: Distance metric (euclidean, cosine, etc.)
        random_state: Random seed
        use_pyg: Use PyTorch Geometric KNN
                 'auto' - use PyG for euclidean metric if available (default)
                 True - force PyG (fallback to PyNNDescent if fails)
                 False - always use PyNNDescent

    Returns:
        umap_graph: Fuzzy simplicial set (sparse matrix)
    """
    random_state = check_random_state(None) if random_state == None else random_state

    # Convert to numpy if it's a torch tensor
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X

    # Determine whether to use PyG KNN
    should_use_pyg = False
    if use_pyg == 'auto':
        # Auto: use PyG for euclidean metric if available
        should_use_pyg = PYG_AVAILABLE and metric == 'euclidean'
    elif use_pyg == True:
        should_use_pyg = PYG_AVAILABLE

    # Try PyG KNN first if enabled
    if should_use_pyg:
        try:
            print(f"   🚀 Using PyTorch Geometric KNN (faster)")

            # Flatten input for KNN
            X_flat = X_np.reshape((len(X_np), np.prod(np.shape(X_np)[1:])))
            X_torch = torch.from_numpy(X_flat).float()

            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Get KNN using PyG
            knn_indices, knn_dists = pyg_knn_search(
                X_torch,
                k=n_neighbors,
                device=device
            )

            # Build fuzzy simplicial set
            umap_graph, sigmas, rhos = fuzzy_simplicial_set(
                X=X_np,
                n_neighbors=n_neighbors,
                metric=metric,
                random_state=random_state,
                knn_indices=knn_indices,
                knn_dists=knn_dists,
            )

            return umap_graph

        except Exception as e:
            print(f"   ⚠️  PyG KNN failed ({str(e)}), falling back to PyNNDescent")

    # Fallback to PyNNDescent
    print(f"   📊 Using PyNNDescent KNN")

    # number of trees in random projection forest
    n_trees = 5 + int(round((X_np.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X_np.shape[0]))))

    # get nearest neighbors
    nnd = NNDescent(
        X_np.reshape((len(X_np), np.prod(np.shape(X_np)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X_np,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph