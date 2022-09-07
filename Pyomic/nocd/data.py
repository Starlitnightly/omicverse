import numpy as np
import scipy.sparse as sp


def load_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'Z' : The community labels in sparse matrix format
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                           loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])

        if 'attr_matrix.data' in loader.keys():
            X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
        else:
            X = None

        Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                           loader['labels.indptr']), shape=loader['labels.shape'])

        # Remove self-loops
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()

        # Convert label matrix to numpy
        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)

        graph = {
            'A': A,
            'X': X,
            'Z': Z
        }

        node_names = loader.get('node_names')
        if node_names is not None:
            node_names = node_names.tolist()
            graph['node_names'] = node_names

        attr_names = loader.get('attr_names')
        if attr_names is not None:
            attr_names = attr_names.tolist()
            graph['attr_names'] = attr_names

        class_names = loader.get('class_names')
        if class_names is not None:
            class_names = class_names.tolist()
            graph['class_names'] = class_names

        return graph
