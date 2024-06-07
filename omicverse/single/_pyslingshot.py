import numpy as np


def mahalanobis(X1, X2, S1, S2):
    S_inv = np.linalg.inv(S1 + S2)
    diff = (X1 - X2).reshape(-1, 1)
    return np.matmul(np.matmul(diff.T, S_inv), diff)

def isint(x):
    if isinstance(x, int):
        return True
    if isinstance(x, str):
        return False
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def isstr(x):
    return isinstance(x, str)

def scale_to_range(x, a=0, b=1):
    return ((x - x.min()) / (x.max() - x.min())) * (b - a) + a

class Lineage:
    def __init__(self, clusters):
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __repr__(self):
        return 'Lineage' + str(self.clusters)

    def __iter__(self):
        for c in self.clusters:
            yield c

import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        step_size = 0.8
        if lower_half != 0:
            step_size /= lower_half

        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, step_size)        

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

class SlingshotPlotter:
    def __init__(self, sling):
        self.sling = sling

    def clusters(self, ax, labels=None, s=8, alpha=1., color_mode='clusters'):
        fig = plt.gcf()
        sling = self.sling
        if labels is None:
            labels = np.arange(sling.num_clusters)

        # Plot clusters and start cluster
        ax.scatter(
            sling.cluster_centres[sling.start_node][0],
            sling.cluster_centres[sling.start_node][1], c='red')

        if color_mode == 'clusters':
            colors = np.array(sns.color_palette('cubehelix', n_colors=sling.num_clusters))
            colors = generate_colormap(sling.num_clusters)

            handles = [
                Patch(color=colors.colors[k], label=labels[k]) for k in range(sling.num_clusters)
            ]
            ax.legend(handles=handles)
            colors = colors.colors[sling.cluster_label_indices]
        elif color_mode == 'pseudotime':
            colors = np.zeros_like(self.sling.curves[0].pseudotimes_interp)
            for l_idx, lineage in enumerate(sling.lineages):
                curve = self.sling.curves[l_idx]
                cell_mask = np.logical_or.reduce(
                    np.array([sling.cluster_label_indices == k for k in lineage]))
                colors[cell_mask] = curve.pseudotimes_interp[cell_mask]
        elif type(color_mode) is np.array:
            colors = color_mode
        else:
            colors = 'black'

        main_scatter = ax.scatter(sling.data[:, 0], sling.data[:, 1],
                   c=colors,
                   s=s,
                   alpha=alpha)

        if color_mode == 'pseudotime':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(main_scatter, cax=cax, orientation='vertical')

    def curves(self, ax, curves):
        for l_idx, curve in enumerate(curves):
            s_interp, p_interp, order = curve.unpack_params()
            ax.plot(
                p_interp[order, 0],
                p_interp[order, 1],
                label=f'Lineage {l_idx}',
                alpha=1)
            ax.legend()

    def network(self, cluster_to_label, figsize=(8, 10)):
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        plt.figure(figsize=figsize)
        G = nx.DiGraph(scale=0.02)
        lineages = self.sling.lineages
        root = cluster_to_label[lineages[0].clusters[0]]

        for lineage in lineages:
            parent = root
            for l in lineage:
                node = cluster_to_label[l]
                G.add_node(node)
                G.add_edge(parent, node)
                parent = node

        plt.title('Lineages')
        pos = graphviz_layout(G, prog='dot')
        label_options = dict(
            ec="k", fc='b', alpha=0.9,
            boxstyle='round,pad=0.2'
        )

        nx.draw(
            G, pos,
            arrows=True,
            node_size=[len(v) * 100 for v in G.nodes()]
        )
        nx.draw_networkx_labels(G, pos, font_size=14, font_color='w', bbox=label_options)


from typing import Union

import numpy as np
from anndata import AnnData


from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from collections import deque
from tqdm.autonotebook import tqdm



class Slingshot:
    def __init__(
            self,
            data: Union[AnnData, np.ndarray],
            cluster_labels_onehot=None,
            celltype_key=None,
            obsm_key='X_umap',
            start_node=0,
            end_nodes=None,
            debug_level=None
    ):
        """
        Constructs a new `Slingshot` object.
        Args:
            data: either an AnnData object or a numpy array containing the dimensionality-reduced data of shape (num_cells, 2)
            cluster_labels: cluster assignments of shape (num_cells). Only required if `data` is not an AnnData object.
            celltype_key: key into AnnData.obs indicating cell type. Only required if `data` is an AnnData object.
            obsm_key: key into AnnData.obsm indicating the dimensionality-reduced data. Only required if `data` is an AnnData object.
            start_node: the starting node of the minimum spanning tree
            end_nodes: any terminal nodes
            debug_level:
        """
        if isinstance(data, AnnData):
            assert celltype_key is not None, "Must provide celltype key if data is an AnnData object"
            cluster_labels = data.obs[celltype_key]

            if isint(cluster_labels[0]):
                cluster_max = cluster_labels.max()
                self.cluster_label_indices = cluster_labels
            elif isstr(cluster_labels[0]):
                cluster_max = len(np.unique(cluster_labels))
                # Convert list of str labels into a list of int indices
                self.cluster_label_indices = np.array([np.where(np.unique(cluster_labels) == label)[0][0] for label in cluster_labels])
            else:
                raise ValueError("Unexpected cluster label dtype.")
            self.cluster_dict=dict(zip(cluster_labels,self.cluster_label_indices))
            cluster_labels_onehot = np.zeros((cluster_labels.shape[0], cluster_max + 1))
            cluster_labels_onehot[np.arange(cluster_labels.shape[0]), self.cluster_label_indices] = 1

            data = data.obsm[obsm_key]
        else:
            assert cluster_labels_onehot is not None, "Must provide cluster labels if data is not an AnnData object"
            cluster_labels = self.cluster_labels_onehot.argmax(axis=1)
        self.data = data
        self.cluster_labels_onehot = cluster_labels_onehot
        self.cluster_labels = cluster_labels
        self.num_clusters = self.cluster_label_indices.max() + 1
        self.start_node = self.cluster_dict[start_node]
        #self.end_nodes = [] if end_nodes is None else end_nodes
        self.end_nodes = [] if end_nodes is None else [self.cluster_dict[end_node] for end_node in end_nodes]
        cluster_centres = [data[self.cluster_label_indices == k].mean(axis=0) for k in range(self.num_clusters)]
        self.cluster_centres = np.stack(cluster_centres)
        self.lineages = None      # list of Lineages
        self.cluster_lineages = None # lineages belonging to each cluster
        self.curves = None   # list of principle curves len = #lineages
        self.cell_weights = None  # weights indicating cluster assignments
        self.distances = None
        self.branch_clusters = None
        self._tree = None

        # Plotting and printing
        debug_level = 0 if debug_level is None else dict(verbose=1)[debug_level]
        self.debug_level = debug_level
        self._set_debug_axes(None)
        self.plotter = SlingshotPlotter(self)

        # Construct smoothing kernel for the shrinking step
        self.kernel_x = np.linspace(-3, 3, 512)
        kde = KernelDensity(bandwidth=1., kernel='gaussian')
        kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
        self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    @property
    def tree(self):
        if self._tree is None:
            self.construct_mst(self.start_node)
        return self._tree

    def load_params(self, filepath):
        if self.curves is None:
            self.get_lineages()
        params = np.load(filepath, allow_pickle=True).item()
        self.curves = params['curves']   # list of principle curves len = #lineages
        self.cell_weights = params['cell_weights']  # weights indicating cluster assignments
        self.distances = params['distances']

    def save_params(self, filepath):
        params = dict(
            curves=self.curves,
            cell_weights=self.cell_weights,
            distances=self.distances
        )
        np.save(filepath, params)

    def _set_debug_axes(self, axes):
        self.debug_axes = axes
        self.debug_plot_mst = axes is not None
        self.debug_plot_lineages = axes is not None
        self.debug_plot_avg = axes is not None

    def construct_mst(self, start_node):
        """
        Arguments:
            start_node: the starting node of the minimum spanning tree
        Returns:
            children: a dictionary mapping clusters to the children of each cluster
        """
        # Calculate empirical covariance of clusters
        emp_covs = np.stack([np.cov(self.data[self.cluster_label_indices == i].T) for i in range(self.num_clusters)])
        dists = np.zeros((self.num_clusters, self.num_clusters))
        for i in range(self.num_clusters):
            for j in range(i, self.num_clusters):
                dist = mahalanobis(
                    self.cluster_centres[i],
                    self.cluster_centres[j],
                    emp_covs[i],
                    emp_covs[j]
                )
                dists[i, j] = dist
                dists[j, i] = dist

        # Find minimum spanning tree excluding end nodes
        mst_dists = np.delete(np.delete(dists, self.end_nodes, axis=0), self.end_nodes, axis=1)  # Delete end nodes
        tree = minimum_spanning_tree(mst_dists)
        # On the left: indices with ends removed; on the right: index into an array where the ends are skipped
        index_mapping = np.array([c for c in range(self.num_clusters - len(self.end_nodes))])
        for i, end_node in enumerate(self.end_nodes):
            index_mapping[end_node - i:] += 1

        connections = {k: list() for k in range(self.num_clusters)}
        cx = tree.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            i = index_mapping[i]
            j = index_mapping[j]
            connections[i].append(j)
            connections[j].append(i)

        for end in self.end_nodes:
            i = np.argmin(np.delete(dists[end], self.end_nodes))
            connections[i].append(end)
            connections[end].append(i)

        # for i,j,v in zip(cx.row, cx.col, cx.data):
        visited = [False for _ in range(self.num_clusters)]
        queue = list()
        queue.append(start_node)
        children = {k: list() for k in range(self.num_clusters)}
        while len(queue) > 0: # BFS to construct children dict
            current_node = queue.pop()
            visited[current_node] = True
            for child in connections[current_node]:
                if not visited[child]:
                    children[current_node].append(child)
                    queue.append(child)

        # Plot clusters and MST
        if self.debug_plot_mst:
            self.plotter.clusters(self.debug_axes[0, 0], alpha=0.5)
            for root, kids in children.items():
                for child in kids:
                    start = [self.cluster_centres[root][0], self.cluster_centres[child][0]]
                    end = [self.cluster_centres[root][1], self.cluster_centres[child][1]]
                    self.debug_axes[0, 0].plot(start, end, c='black')
            self.debug_plot_mst = False

        self._tree = children
        return children

    def fit(self, num_epochs=10, debug_axes=None):
        self._set_debug_axes(debug_axes)
        if self.curves is None:  # Initial curves and pseudotimes:
            self.get_lineages()
            self.construct_initial_curves()
            self.cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                                 for l in range(len(self.lineages))]
            self.cell_weights = np.stack(self.cell_weights, axis=1)

        for epoch in tqdm(range(num_epochs)):
            # Calculate cell weights
            # cell weight is a matrix #cells x #lineages indicating cell-lineage assignment
            self.calculate_cell_weights()

            # Fit principal curve for all lineages using existing curves
            self.fit_lineage_curves()

            # Ensure starts at 0
            for l_idx, lineage in enumerate(self.lineages):
                curve = self.curves[l_idx]
                min_time = np.min(curve.pseudotimes_interp[self.cell_weights[:, l_idx] > 0])
                curve.pseudotimes_interp -= min_time

            # Determine average curves
            shrinkage_percentages, cluster_children, cluster_avg_curves = \
                self.avg_curves()

            # Shrink towards average curves in areas of cells common to all branch lineages
            self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)

            self.debug_plot_lineages = False
            self.debug_plot_avg = False

            if self.debug_axes is not None and epoch == num_epochs - 1:  # plot curves
                self.plotter.clusters(self.debug_axes[1, 1], s=2, alpha=0.5)
                self.plotter.curves(self.debug_axes[1, 1], self.curves)

    def construct_initial_curves(self):
        """Constructs lineage principal curves using piecewise linear initialisation"""
        from pcurvepy2 import PrincipalCurve
        piecewise_linear = list()
        distances = list()

        for l_idx, lineage in enumerate(self.lineages):
            # Calculate piecewise linear path
            p = np.stack(self.cluster_centres[lineage.clusters])
            s = np.zeros(p.shape[0])  # TODO

            cell_mask = np.logical_or.reduce(
                np.array([self.cluster_label_indices == k for k in lineage]))
            cells_involved = self.data[cell_mask]

            curve = PrincipalCurve(k=3)
            curve.project_to_curve(cells_involved, points=p)
            d_sq, dist = curve.project_to_curve(self.data, points=curve.points_interp[curve.order])
            distances.append(d_sq)

            # piecewise_linear.append(PrincipalCurve.from_params(s, p))
            piecewise_linear.append(curve)

        self.curves = piecewise_linear
        self.distances = distances

    def get_lineages(self):
        tree = self.construct_mst(self.start_node)

        # Determine lineages by parsing the MST
        branch_clusters = deque()
        def recurse_branches(path, v):
            num_children = len(tree[v])
            if num_children == 0:  # at leaf, add a None token
                return path + [v, None]
            elif num_children == 1:
                return recurse_branches(path + [v], tree[v][0])
            else:  # at branch
                branch_clusters.append(v)
                return [recurse_branches(path + [v], tree[v][i]) for i in range(num_children)]

        def flatten(li):
            if li[-1] is None:  # special None token indicates a leaf
                yield Lineage(li[:-1])
            else:  # otherwise yield from children
                for l in li:
                    yield from flatten(l)

        lineages = recurse_branches([], self.start_node)
        lineages = list(flatten(lineages))
        self.lineages = lineages
        self.branch_clusters = branch_clusters

        self.cluster_lineages = {k: list() for k in range(self.num_clusters)}
        for l_idx, lineage in enumerate(self.lineages):
            for k in lineage:
                self.cluster_lineages[k].append(l_idx)

        if self.debug_level > 0:
            print('Lineages:', lineages)

    def fit_lineage_curves(self):
        """Updates curve using a cubic spline and projection of data"""
        assert self.lineages is not None
        assert self.curves is not None
        distances = list()

        # Calculate principal curves
        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]

            # Fit principal curve through data
            # Weights are important as they effectively silence points
            # that are not associated with the lineage.
            curve.fit(
                self.data,
                max_iter=1,
                w=self.cell_weights[:, l_idx]
            )

            if self.debug_plot_lineages:
                cell_mask = np.logical_or.reduce(
                    np.array([self.cluster_label_indices == k for k in lineage]))
                cells_involved = self.data[cell_mask]
                self.debug_axes[0, 1].scatter(cells_involved[:, 0], cells_involved[:, 1], s=2, alpha=0.5)
                alphas = curve.pseudotimes_interp
                alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
                for i in np.random.permutation(self.data.shape[0])[:50]:
                    path_from = (self.data[i][0], curve.points_interp[i][0])
                    path_to = (self.data[i][1], curve.points_interp[i][1])
                    self.debug_axes[0, 1].plot(path_from, path_to, c='black', alpha=alphas[i])
                self.debug_axes[0, 1].plot(curve.points_interp[curve.order, 0],
                                           curve.points_interp[curve.order, 1], label=str(lineage))

            d_sq, dist = curve.project_to_curve(self.data, curve.points_interp[curve.order])
            distances.append(d_sq)
        self.distances = distances
        if self.debug_plot_lineages:
            self.debug_axes[0, 1].legend()

    def calculate_cell_weights(self):
        """TODO: annotate, this is a translation from R"""
        cell_weights = [self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1)
                        for l in range(len(self.lineages))]
        cell_weights = np.stack(cell_weights, axis=1)

        d_sq = np.stack(self.distances, axis=1)
        d_ord = np.argsort(d_sq, axis=None)
        w_prob = cell_weights/cell_weights.sum(axis=1, keepdims=True)  # shape (cells, lineages)
        w_rnk_d = np.cumsum(w_prob.reshape(-1)[d_ord]) / w_prob.sum()

        z = d_sq
        z_shape = z.shape
        z = z.reshape(-1)
        z[d_ord] = w_rnk_d
        z = z.reshape(z_shape)
        z_prime = 1 - z ** 2
        z_prime[cell_weights == 0] = np.nan
        w0 = cell_weights.copy()
        cell_weights = z_prime / np.nanmax(z_prime, axis=1, keepdims=True) #rowMins(D) / D
        np.nan_to_num(cell_weights, nan=1, copy=False) # handle 0/0
        # cell_weights[is.na(cell_weights)] <- 0
        cell_weights[cell_weights > 1] = 1
        cell_weights[cell_weights < 0] = 0
        cell_weights[w0 == 0] = 0

        reassign = True
        if reassign:
            # add if z < .5
            cell_weights[z < .5] = 1 #(rowMins(D) / D)[idx]

            # drop if z > .9 and cell_weights < .1
            ridx = (z.max(axis=1) > .9) & (cell_weights.min(axis=1) < .1)
            w0 = cell_weights[ridx]
            z0 = z[ridx]
            w0[(z0 > .9) & (w0 < .1)] = 0 # !is.na(Z0) & Z0 > .9 & W0 < .1
            cell_weights[ridx] = w0

        self.cell_weights = cell_weights

    def avg_curves(self):
        """
        Starting at leaves, calculate average curves for each branch

        :return: shrinkage_percentages, cluster_children, cluster_avg_curves
        """
        cell_weights = self.cell_weights
        shrinkage_percentages = list()
        cluster_children = dict()  # maps cluster to children
        lineage_avg_curves = dict()
        cluster_avg_curves = dict()
        branch_clusters = self.branch_clusters.copy()
        if self.debug_level > 0:
            print('Reversing from leaf to root')
        if self.debug_plot_avg:
            self.plotter.clusters(self.debug_axes[1, 0], s=4, alpha=0.4)

        while len(branch_clusters) > 0:
            # Starting at leaves, find lineages involved in branch
            k = branch_clusters.pop()
            branch_lineages = self.cluster_lineages[k]
            cluster_children[k] = set()
            for l_idx in branch_lineages:  # loop all lineages through branch
                if l_idx in lineage_avg_curves:  # add avg curve
                    curve = lineage_avg_curves[l_idx]
                else:  # or add leaf curve
                    curve = self.curves[l_idx]
                cluster_children[k].add(curve)

            # Calculate the average curve for this branch
            branch_curves = list(cluster_children[k])
            if self.debug_level > 0:
                print(f'Averaging branch @{k} with lineages:', branch_lineages, branch_curves)

            avg_curve = self.avg_branch_curves(branch_curves)
            cluster_avg_curves[k] = avg_curve
            # avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))

            # Calculate shrinkage weights using areas where cells share lineages
            # note that this also captures cells in average curves, since the
            # lineages which are averaged are present in branch_lineages
            common = cell_weights[:, branch_lineages] > 0
            common_mask = common.mean(axis=1) == 1.
            shrinkage_percent = dict()
            for curve in branch_curves:
                shrinkage_percent[curve] = self.shrinkage_percent(curve, common_mask)
            shrinkage_percentages.append(shrinkage_percent)

            # Add avg_curve to lineage_avg_curve for cluster_children
            for l in branch_lineages:
                lineage_avg_curves[l] = avg_curve
            # # check for degenerate case (if one curve won't be
            # # shrunk, then the other curve shouldn't be,
            # # either)
            # new.avg.order <- avg.order
            # all.zero <- vapply(pct.shrink[[i]], function(pij){
            #     return(all(pij == 0))
            # }, TRUE)
            # if(any(all.zero)){
            #     if(allow.breaks){
            #         new.avg.order[[i]] <- NULL
            #         message('Curves for ', ns[1], ' and ',
            #             ns[2], ' appear to be going in opposite ',
            #             'directions. No longer forcing them to ',
            #             'share an initial point. To manually ',
            #             'override this, set allow.breaks = ',
            #             'FALSE.')
            #     }
            #     pct.shrink[[i]] <- lapply(pct.shrink[[i]],
            #         function(pij){
            #             pij[] <- 0
            #             return(pij)
            #         })
            # }
        if self.debug_plot_avg:
            self.debug_axes[1, 0].legend()
        return shrinkage_percentages, cluster_children, cluster_avg_curves

    def shrink_curves(self, cluster_children, shrinkage_percentages, cluster_avg_curves):
        """
        Starting at root, shrink curves for each branch

        Parameters:
            cluster_children:
            shrinkage_percentages:
            cluster_avg_curves:
        :return:
        """
        branch_clusters = self.branch_clusters.copy()
        while len(branch_clusters) > 0:
            # Starting at root, find lineages involves in branch
            k = branch_clusters.popleft()
            shrinkage_percent = shrinkage_percentages.pop()
            branch_curves = list(cluster_children[k])
            cluster_avg_curve = cluster_avg_curves[k]
            if self.debug_level > 0:
                print(f'Shrinking branch @{k} with curves:', branch_curves)

            # Specify the avg curve for this branch
            self.shrink_branch_curves(branch_curves, cluster_avg_curve, shrinkage_percent)

    def shrink_branch_curves(self, branch_curves, avg_curve, shrinkage_percent):
        """
        Shrinks curves through a branch to the average curve.
        
        Arguments:
            branch_curves: list of `PrincipalCurve`s associated with the branch.
            avg_curve: `PrincipalCurve` for average curve.
            shrinkage_percent: percentage shrinkage, in same order as curve.pseudotimes
       
        """
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        # Go through "child" lineages, shrinking the curves toward the above average
        for curve in branch_curves:  # curve might be an average curve or a leaf curve
            pct = shrinkage_percent[curve]

            s_interp, p_interp, order = curve.unpack_params()
            avg_s_interp, avg_p_interp, avg_order = avg_curve.unpack_params()
            shrunk_curve = np.zeros_like(p_interp)
            for j in range(num_dims_reduced):
                orig = p_interp[order, j]
                avg = np.interp(#interp1d(
                    s_interp[order],
                    avg_s_interp[avg_order],     # x
                    avg_p_interp[avg_order, j])#,  # y
                    # assume_sorted=True,
                    # bounds_error=False,
                    # fill_value='extrapolate',
                    # extrapolate_extrema=True)
                # avg = lin_interpolator#(s_interp[order])
                shrunk_curve[:, j] = (avg * pct + orig * (1 - pct))
            # w <- pcurve$w
            # pcurve = project_to_curve(X, as.matrix(s[pcurve$ord, ,drop = FALSE]), stretch = stretch)
            # pcurve$w <- w
            # self.debug_axes[1, 1].plot(
            #     shrunk_curve[:, 0],
            #     shrunk_curve[:, 1],
            #     label='shrunk', alpha=0.2, c='black')
            curve.project_to_curve(self.data, points=shrunk_curve)
            #     for(jj in seq_along(ns)){
            #         n <- ns[jj]
            #         if(grepl('Lineage',n)){
            #             l.ind <- as.numeric(gsub('Lineage','',n))
            #             pcurves[[l.ind]] <- shrunk[[jj]]
            #         }
            #         if(grepl('average',n)){
            #             a.ind <- as.numeric(gsub('average','',n))
            #             avg.lines[[a.ind]] <- shrunk[[jj]]
            #         }
            #     }
            # }
            # avg.order <- new.avg.order

    def shrinkage_percent(self, curve, common_ind):
        """Determines how much to shrink a curve"""
        # pst <- crv$lambda
        # pts2wt <- pst
        s_interp, order = curve.pseudotimes_interp, curve.order
        # Cosine kernel quartiles:
        x = self.kernel_x
        y = self.kernel_y
        y = (y.sum() - np.cumsum(y)) / sum(y)
        q1 = np.percentile(s_interp[common_ind], 25)
        q3 = np.percentile(s_interp[common_ind], 75)
        a = q1 - 1.5 * (q3 - q1)
        b = q3 + 1.5 * (q3 - q1)
        x = scale_to_range(x, a=a, b=b)
        if q1 == q3:
            pct_l = np.zeros(s_interp.shape[0])
        else:
            pct_l = np.interp(
                s_interp[order],
                x, y
            )

        return pct_l

    def avg_branch_curves(self, branch_curves):
        """branch_lineages is a list of lineages passing through branch"""
        from pcurvepy2 import PrincipalCurve
        # s_interps, p_interps, orders
        num_cells = branch_curves[0].points_interp.shape[0]
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        # 1. Interpolate all the lineages over the shared time domain
        branch_s_interps = np.stack([c.pseudotimes_interp for c in branch_curves], axis=1)
        max_shared_pseudotime = branch_s_interps.max(axis=0).min()  # take minimum of maximum pseudotimes for each lineage
        combined_pseudotime = np.linspace(0, max_shared_pseudotime, num_cells)
        curves_dense = list()
        for curve in branch_curves:
            lineage_curve = np.zeros((combined_pseudotime.shape[0], num_dims_reduced))
            order = curve.order
            # Linearly interpolate each dimension as a function of pseudotime
            for j in range(num_dims_reduced):
                lin_interpolator = interp1d(
                    curve.pseudotimes_interp[order], # x
                    curve.points_interp[order, j],   # y
                    assume_sorted=True
                )
                lineage_curve[:, j] = lin_interpolator(combined_pseudotime)
            curves_dense.append(lineage_curve)

        curves_dense = np.stack(curves_dense, axis=1)  # (n, L_b, J)

        # 2. Average over these curves and project the data onto the result
        avg = curves_dense.mean(axis=1)  # avg is already "sorted"
        avg_curve = PrincipalCurve()
        avg_curve.project_to_curve(self.data, points=avg)
        # avg_curve.pseudotimes_interp -= avg_curve.pseudotimes_interp.min()
        if self.debug_plot_avg:
            self.debug_axes[1, 0].plot(avg[:, 0], avg[:, 1], c='blue', linestyle='--', label='average', alpha=0.7)
            _, p_interp, order = avg_curve.unpack_params()
            self.debug_axes[1, 0].plot(p_interp[order, 0], p_interp[order, 1], c='red', label='data projected', alpha=0.7)

        # avg.curve$w <- rowSums(vapply(pcurves, function(p){ p$w }, rep(0,nrow(X))))
        return avg_curve

    @property
    def unified_pseudotime(self):
        pseudotime = np.zeros_like(self.curves[0].pseudotimes_interp)
        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]
            cell_mask = np.logical_or.reduce(
                np.array([self.cluster_label_indices == k for k in lineage]))
            pseudotime[cell_mask] = curve.pseudotimes_interp[cell_mask]
        return pseudotime

    def list_lineages(self, cluster_to_label):
        for lineage in self.lineages:
            print(', '.join([
                cluster_to_label[l] for l in lineage
            ]))