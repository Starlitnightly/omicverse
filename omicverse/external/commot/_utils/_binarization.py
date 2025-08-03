import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# %%
def binarize_sparse_matrix(
    A,
    method = None, 
    cutoff = 0,
    random_state = 1,
    append_zeros = 'full'
):
    # A is a scipy sparse csr_matrix
    # Anything smaller than or equal to the cutoff is set to 0.
    A_binary = A.copy()
    n = A.shape[0]
    nnz = len(A.data)
    tmp_data = A.data
    if append_zeros == 'full':
        tmp_zeros = np.zeros(n*n-nnz)
    elif append_zeros == 'match':
        tmp_zeros = np.zeros_like(tmp_data)
    tmp_X = np.concatenate((tmp_data, tmp_zeros), axis=0)
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=random_state).fit(tmp_X.reshape(-1,1))
        cl_centers = kmeans.cluster_centers_.reshape(-1)
        if cl_centers[1] < cl_centers[0]:
            new_data = 1 - kmeans.labels_[:nnz]
        else:
            new_data = kmeans.labels_[:nnz]
    elif method == 'gaussian_mixture':
        gm = GaussianMixture(n_components=2, random_state=random_state).fit(tmp_X.reshape(-1,1))
        cl_centers = gm.means_.reshape(-1)
        labels = gm.predict(tmp_data.reshape(-1,1))
        if cl_centers[1] < cl_centers[0]:
            labels = 1 - labels
        new_data = labels
    new_data[np.where(tmp_data <= cutoff)] = 0
    A_binary.data[:] = new_data[:]
    A_binary.eliminate_zeros()

    return A_binary
# %%
