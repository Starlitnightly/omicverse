import numpy as np


def pps(knn, rnn, order):
    """
    Plum Pudding Sampling (PPS)
    This function returns the ID of landmarks.

    Parameters are:

    'knn'      - A N by k matrix. Each row represents the KNN of each point.
    'rnn'      - A vector of length N. Each row represents the RNN of each point.
    'order'    - A positive integer specifying the order of KNN. Once a landmark is selected, its KNN will be removed
                 from the queue. PPS supports the removal of multi-order KNN.

    """
    id_samp = []
    id_sort = sorted(range(len(rnn)), key=lambda k: rnn[k], reverse=True)
    while len(id_sort) > 0:
        id_samp.append(id_sort[0])
        rm_pts = [id_sort[0]]
        for _ in range(order):
            rm_pts.extend(knn[rm_pts].flatten().tolist())
        rm_pts = set(rm_pts)
        rm_id = np.where(np.isin(id_sort, list(rm_pts)))[0]
        id_sort = [id_sort[i] for i in range(len(id_sort)) if i not in rm_id]
    return id_samp
