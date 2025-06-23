"""
Utilities for the chord diagram.
"""

from collections import defaultdict
import numpy as np


def dist(points):
    '''
    Compute the distance between two points.

    Parameters
    ----------
    points : array of length 4
        The coordinates of the two points, P1 = (x1, y1) and P2 = (x2, y2)
        in the order [x1, y1, x2, y2].
    '''
    x1, y1 = points[0]
    x2, y2 = points[1]

    return np.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))


def polar2xy(r, theta):
    '''
    Convert the coordinates of a point P from polar (r, theta) to cartesian
    (x, y).
    '''
    return np.array([r*np.cos(theta), r*np.sin(theta)])


def compute_positions(mat, deg, in_deg, out_deg, start_at, is_sparse, sort,
                      directed, extent, pad, arc, rotation, nodePos, pos):
    '''
    Compute all arcs and chords start/end positions.

    Parameters
    ----------
    mat : matrix
        Original matrix.
    deg : array
        Out (if undirected) or total (if directed) degree to compute the
        starting positions.
    in_deg : array
        In-degree to compute the end positions (if directed).
    out_deg : array
        Out-degree to compute the end positions (if directed).
    y : array
        Used to compute the arcs' ends.
    start_at : float
        Start of the first arc.
    is_sparse : bool
        Whether the matrix is sparse.
    sort : bool
        Sorting method.
    directed : bool
        Whether the chords are directed.
    extent : float in ]0, 360]
        Angular aperture of the diagram.
    pad : float
        Gap between entries.
    arc : list
        Used to store the arcs start and endpoints.
    rotation : list
        Store the rotation booleans for the names.
    nodePos : list
        Store the name positions.
    pos : dict
        Store the start and end positions for each arc under the form:
        (start1, end1, start2, end2), where (start1, end1) are the limits of the
        chords starting point, and (start2, end2) are the limits of the chord's
        end point.
    '''
    num_nodes = len(deg)

    # find position for each start and end
    y = deg / np.sum(deg).astype(float) * (extent - pad * num_nodes)

    if directed:
        y_out = out_deg / np.sum(deg).astype(float) * (extent - pad * num_nodes)

    starts = [start_at] + (
        start_at + np.cumsum(y + pad*np.ones(num_nodes))).tolist()

    out_ends = [s + d for s, d in zip(starts, (y_out if directed else y))]


    # relative positions within an arc
    zmat = [
        _get_normed_line(mat, i, out_deg if directed else deg, starts[i],
                         out_ends[i], is_sparse)
        for i in range(num_nodes)
    ]

    zin_mat = zmat

    if directed:
        zin_mat = [
            _get_normed_line(mat.T, i, in_deg, out_ends[i], starts[i + 1] - pad,
                             is_sparse)
            for i in range(num_nodes)
        ]

    # sort
    mat_ids = _get_sorted_ids(sort, zmat, num_nodes, directed)

    # compute positions
    for i in range(num_nodes):
        # arcs
        start = starts[i]
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start + end)

        if -30 <= (angle % 360) <= 180:
            angle -= 90
            rotation.append(False)
        else:
            angle -= 270
            rotation.append(True)

        nodePos.append(
            tuple(polar2xy(1.05, 0.5*(start + end)*np.pi/180.)) + (angle,))

        # sort chords
        z = zmat[i]
        z0 = start

        for j in mat_ids[i]:
            # compute the arrival points
            zj = zin_mat[j]
            startj = out_ends[j] if directed else starts[j]

            jids = mat_ids[j]

            distsort = (sort == "distance")

            if directed and not distsort:
                jids = jids[::-1]

            stop = np.where(np.equal(jids, i))[0][0]

            startji = startj + zj[jids[:stop]].sum()

            if distsort and directed:
                # we want j first for target positions
                startji += zj[j]

            if distsort and directed and i == j:
                pos[(i, j)] = (z0, z0 + z[j], startj, startj + zj[j])
            else:
                pos[(i, j)] = (z0, z0 + z[j], startji, startji + zj[jids[stop]])

            z0 += z[j]


# In-file functions

def _get_normed_line(mat, i, x, start, end, is_sparse):
    if is_sparse:
        row = mat.getrow(i).todense().A1
        return (row / x[i]) * (end - start)

    return (mat[i, :] / x[i]) * (end - start)


def _get_sorted_ids(sort, zmat, num_nodes, directed):
    mat_ids = defaultdict(lambda: list(range(num_nodes)))

    if sort == "size":
        mat_ids = [np.argsort(z) for z in zmat]
    elif sort == "distance":
        mat_ids = []
        for i in range(num_nodes):
            remainder = 0 if num_nodes % 2 else -1

            ids  = list(range(i - int(0.5*num_nodes), i))[::-1]

            if not directed:
                ids += [i]

            ids += list(range(i + int(0.5*num_nodes) + remainder, i, -1))

            if directed:
                ids += [i]

            # put them back into [0, num_nodes[
            ids = np.array(ids)
            ids[ids < 0] += num_nodes
            ids[ids >= num_nodes] -= num_nodes

            mat_ids.append(ids)
    elif sort is not None:
        raise ValueError("Invalid `sort`: '{}'".format(sort))

    return mat_ids