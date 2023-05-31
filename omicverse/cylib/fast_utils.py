import numpy as np

def calc_mean_and_var_sparse(M, N, data, indices, indptr, axis):
    i, j=0,0

    size=0
    value=0.0

    size = N if axis == 0 else M
    mean = np.zeros(size, dtype = np.float64)
    var = np.zeros(size, dtype = np.float64)

    mean_view = mean
    var_view = var

    for i in range(M):
        for j in range(indptr[i], indptr[i + 1]):
            value = data[j]
            if axis == 0:
                mean_view[indices[j]] += value
                var_view[indices[j]] += value * value
            else:
                mean_view[i] += value
                var_view[i] += value * value

    size = M if axis == 0 else N
    for i in range(mean_view.size):
        mean_view[i] /= size
        var_view[i] = (var_view[i] - size * mean_view[i] * mean_view[i]) / (size - 1)

    return mean, var

def calc_stat_per_batch_sparse(M, N, data, indices, indptr,nbatch, codes):
    i, j=0,0

    col=0
    code=0
    value=0.0

    ncells = np.zeros(nbatch, dtype = np.int32)
    means = np.zeros((N, nbatch), dtype = np.float64)
    partial_sum = np.zeros((N, nbatch), dtype = np.float64)

    ncells_view = ncells
    means_view = means
    ps_view = partial_sum

    for i in range(M):
        code = codes[i]
        ncells_view[code] += 1
        for j in range(indptr[i], indptr[i + 1]):
            col = indices[j]
            value = data[j]
            means_view[col, code] += value
            ps_view[col, code] += value * value

    for j in range(nbatch):
        if ncells_view[j] > 1:
            for i in range(N):
                means_view[i, j] /= ncells_view[j]
                ps_view[i, j] = ps_view[i, j] - ncells_view[j] * means_view[i, j] * means_view[i, j]

    return ncells, means, partial_sum

def calc_mean_and_var_dense(M, N, X, axis):
    i, j=0,0

    size=0
    value=0.0

    size = N if axis == 0 else M
    mean = np.zeros(size, dtype = np.float64)
    var = np.zeros(size, dtype = np.float64)

    mean_view = mean
    var_view = var

    for i in range(M):
        for j in range(N):
            value = X[i, j]
            if axis == 0:
                mean_view[j] += value
                var_view[j] += value * value
            else:
                mean_view[i] += value
                var_view[i] += value * value

    size = M if axis == 0 else N
    for i in range(mean_view.size):
        mean_view[i] /= size
        var_view[i] = (var_view[i] - size * mean_view[i] * mean_view[i]) / (size - 1)

    return mean, var


def calc_stat_per_batch_dense(M, N, X, nbatch, codes):
    i, j=0,0

    code, col=0,0
    value=0.0

    ncells = np.zeros(nbatch, dtype = np.int32)
    means = np.zeros((N, nbatch), dtype = np.float64)
    partial_sum = np.zeros((N, nbatch), dtype = np.float64)

    ncells_view = ncells
    means_view = means
    ps_view = partial_sum

    for i in range(M):
        code = codes[i]
        ncells_view[code] += 1
        for j in range(N):
            value = X[i, j]
            means_view[j, code] += value
            ps_view[j, code] += value * value

    for j in range(nbatch):
        if ncells_view[j] > 1:
            for i in range(N):
                means_view[i, j] /= ncells_view[j]
                ps_view[i, j] = ps_view[i, j] - ncells_view[j] * means_view[i, j] * means_view[i, j]

    return ncells, means, partial_sum