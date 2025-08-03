"""
Tools to draw a chord diagram in python
"""

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.patches as patches

from matplotlib.colors import ColorConverter, Colormap
from matplotlib.path import Path

import numpy as np
import scipy.sparse as ssp

from .gradient import gradient
from .utilities import _get_normed_line, compute_positions, dist, polar2xy


LW = 0.3


def chord_diagram(mat, names=None, order=None, sort="size", directed=False,
                  colors=None, cmap=None, use_gradient=False, chord_colors=None,
                  alpha=0.7, start_at=0, extent=360, width=0.1, pad=2., gap=0.03,
                  chordwidth=0.7, min_chord_width=0, fontsize=12.8,
                  fontcolor="k", rotate_names=False, ax=None, show=False):
    """
    Plot a chord diagram.

    Draws a representation of many-to-many interactions between elements, given
    by an interaction matrix.
    The elements are represented by arcs proportional to their degree and the
    interactions (or fluxes) are drawn as chords joining two arcs:

    * for undirected chords, the size of the arc is proportional to its
      out-degree (or simply its degree if the matrix is fully symmetrical), i.e.
      the sum of the element's row.
    * for directed chords, the size is proportional to the total-degree, i.e.
      the sum of the element's row and column.

    Parameters
    ----------
    mat : square matrix
        Flux data, ``mat[i, j]`` is the flux from i to j.
    names : list of str, optional (default: no names)
        Names of the nodes that will be displayed (must be ordered as the
        matrix entries).
    order : list, optional (default: order of the matrix entries)
        Order in which the arcs should be placed around the trigonometric
        circle.
    sort : str, optional (default: "size")
        Order in which the chords should be sorted: either None (unsorted),
        "size" (default, drawing largest chords first), or "distance"
        (drawing the chords of the two closest arcs at each end of the current
        arc, then progressing towards the connexions with the farthest arcs in
        both drections as we move towards the center of the current arc).
    directed : bool, optional (default: False)
        Whether the chords should be directed, like edges in a graph, with one
        part of each arc dedicated to outgoing chords and the other to incoming
        ones.
    colors : list, optional (default: from `cmap`)
        List of user defined colors or floats.
    cmap : str or colormap object (default: viridis)
        Colormap that will be used to color the arcs and chords by default.
        See `chord_colors` to use different colors for chords.
    use_gradient : bool, optional (default: False)
        Whether a gradient should be use so that chord extremities have the
        same color as the arc they belong to.
    chord_colors : str, or list of colors, optional (default: None)
        Specify color(s) to fill the chords differently from the arcs.
        When the keyword is not used, chord colors default to the colomap given
        by `colors`.
        Possible values for `chord_colors` are:

        * a single color (do not use an RGB tuple, use hex format instead),
          e.g. "red" or "#ff0000"; all chords will have this color
        * a list of colors, e.g. ``["red", "green", "blue"]``, one per node
          (in this case, RGB tuples are accepted as entries to the list).
          Each chord will get its color from its associated source node, or
          from both nodes if `use_gradient` is True.
    alpha : float in [0, 1], optional (default: 0.7)
        Opacity of the chord diagram.
    start_at : float, optional (default : 0)
        Location, in degrees, where the diagram should start on the unit circle.
        Default is to start at 0 degrees, i.e. (x, y) = (1, 0) or 3 o'clock),
        and move counter-clockwise
    extent : float, optional (default : 360)
        The angular aperture, in degrees, of the diagram.
        Default is to use the whole circle, i.e. 360 degrees, but in some cases
        it can be useful to use only a part of it.
    width : float, optional (default: 0.1)
        Width/thickness of the ideogram arc.
    pad : float, optional (default: 2)
        Distance between two neighboring ideogram arcs. Unit: degree.
    gap : float, optional (default: 0)
        Distance between the arc and the beginning of the cord.
    chordwidth : float, optional (default: 0.7)
        Position of the control points for the chords, controlling their shape.
    min_chord_width : float, optional (default: 0)
        Minimal chord width to replace small entries and zero reciprocals in
        the matrix.
    fontsize : float, optional (default: 12.8)
        Size of the fonts for the names.
    fontcolor : str or list, optional (default: black)
        Color of the fonts for the names.
    rotate_names : (list of) bool(s), optional (default: False)
        Whether to rotate all names (if single boolean) or some of them (if
        list) by 90°.
    ax : matplotlib axis, optional (default: new axis)
        Matplotlib axis where the plot should be drawn.
    show : bool, optional (default: False)
        Whether the plot should be displayed immediately via an automatic call
        to `plt.show()`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # copy matrix
    is_sparse = ssp.issparse(mat)

    if is_sparse:
        mat = ssp.csr_matrix(mat)
    else:
        mat = np.array(mat, copy=True)

    num_nodes = mat.shape[0]

    # don't use gradient with directed chords
    use_gradient *= not directed

    # set min entry size for small entries and zero reciprocals
    # mat[i, j]:  i -> j
    if is_sparse and min_chord_width:
        nnz = mat.nonzero()

        mat.data[mat.data < min_chord_width] = min_chord_width

        # check zero reciprocals
        for i, j in zip(*nnz):
            if mat[j, i] < min_chord_width:
                mat[j, i] = min_chord_width
    elif min_chord_width:
        nnz = mat > 0

        mat[nnz] = np.maximum(mat[nnz], min_chord_width)

        # check zero reciprocals
        for i, j in zip(*np.where(~nnz)):
            if mat[j, i]:
                mat[i, j] = min_chord_width

    # check name rotations
    if isinstance(rotate_names, Sequence):
        assert len(rotate_names) == num_nodes, \
            "Wrong number of entries in 'rotate_names'."
    else:
        rotate_names = [rotate_names]*num_nodes

    # check order
    if order is not None:
        mat = mat[order][:, order]

        rotate_names = [rotate_names[i] for i in order]

        if names is not None:
            names = [names[i] for i in order]

        if colors is not None:
            colors = [colors[i] for i in order]

    # configure colors
    if colors is None:
        colors = np.linspace(0, 1, num_nodes)

    if isinstance(fontcolor, str):
        fontcolor = [fontcolor]*num_nodes
    else:
        assert len(fontcolor) == num_nodes, \
            "One fontcolor per node is required."

    if cmap is None:
        cmap = mpl.colormaps["viridis"]
    elif not isinstance(cmap, Colormap):
        cmap = mpl.colormaps[cmap]

    if isinstance(colors, (list, tuple, np.ndarray)):
        assert len(colors) == num_nodes, "One color per node is required."

        # check color type
        first_color = colors[0]

        if isinstance(first_color, (int, float, np.integer)):
            colors = cmap(colors)[:, :3]
        else:
            colors = [ColorConverter.to_rgb(c) for c in colors]
    else:
        raise ValueError("`colors` should be a list.")

    if chord_colors is None:
       chord_colors = colors
    else:
        try:
            chord_colors = [ColorConverter.to_rgb(chord_colors)] * num_nodes
        except ValueError:
            assert len(chord_colors) == num_nodes, \
                "If `chord_colors` is a list of colors, it should include " \
                "one color per node (here {} colors).".format(num_nodes)

    # sum over rows
    out_deg = mat.sum(axis=1).A1 if is_sparse else mat.sum(axis=1)
    in_deg = None
    degree = out_deg.copy()

    if directed:
        # also sum over columns
        in_deg = mat.sum(axis=0).A1 if is_sparse else mat.sum(axis=0)
        degree += in_deg

    pos = {}
    pos_dir = {}
    arc = []
    nodePos = []
    rotation = []

    # compute all values and optionally apply sort
    compute_positions(mat, degree, in_deg, out_deg, start_at, is_sparse, sort,
                      directed, extent, pad, arc, rotation, nodePos, pos)

    # plot
    for i in range(num_nodes):
        color = colors[i]

        # plot the arcs
        start_at, end = arc[i]

        ideogram_arc(start=start_at, end=end, radius=1.0, color=color,
                     width=width, alpha=alpha, ax=ax)

        chord_color = chord_colors[i]

        # plot self-chords if directed is False
        if not directed and mat[i, i]:
            start1, end1, _, _ = pos[(i, i)]
            self_chord_arc(start1, end1, radius=1 - width - gap,
                           chordwidth=0.7*chordwidth, color=chord_color,
                           alpha=alpha, ax=ax)

        # plot all other chords
        targets = range(num_nodes) if directed else range(i)

        for j in targets:
            cend = chord_colors[j]

            start1, end1, start2, end2 = pos[(i, j)]

            if mat[i, j] > 0 or (not directed and mat[j, i] > 0):
                chord_arc(
                    start1, end1, start2, end2, radius=1 - width - gap, gap=gap,
                    chordwidth=chordwidth, color=chord_color, cend=cend,
                    alpha=alpha, ax=ax, use_gradient=use_gradient,
                    extent=extent, directed=directed)

    # add names if necessary
    if names is not None:
        assert len(names) == num_nodes, "One name per node is required."

        prop = {
            "fontsize": fontsize,
            "ha": "center",
            "va": "center",
            "rotation_mode": "anchor"
        }

        for i, (pos, name, r) in enumerate(zip(nodePos, names, rotation)):
            rotate = rotate_names[i]
            pp = prop.copy()
            pp["color"] = fontcolor[i]

            if rotate:
                angle  = np.average(arc[i])
                rotate = 90

                if 90 < angle < 180 or 270 < angle:
                    rotate = -90

                if 90 < angle < 270:
                    pp["ha"] = "right"
                else:
                    pp["ha"] = "left"
            elif r:
                pp["va"] = "top"
            else:
                pp["va"] = "bottom"

            ax.text(pos[0], pos[1], name, rotation=pos[2] + rotate, **pp)

    # configure axis
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.set_aspect(1)
    ax.axis('off')

    plt.tight_layout()

    if show:
        plt.show()

    return nodePos


# ------------ #
# Subfunctions #
# ------------ #

def initial_path(start, end, radius, width, factor=4/3):
    ''' First 16 vertices and 15 instructions are the same for everyone '''
    if start > end:
        start, end = end, start

    start *= np.pi/180.
    end   *= np.pi/180.

    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/
    # how-to-create-circle-with-b%C3%A9zier-curves
    # use 16-vertex curves (4 quadratic Beziers which accounts for worst case
    # scenario of 360 degrees)
    inner = radius*(1-width)
    opt   = factor * np.tan((end-start)/ 16.) * radius
    inter1 = start*(3./4.)+end*(1./4.)
    inter2 = start*(2./4.)+end*(2./4.)
    inter3 = start*(1./4.)+end*(3./4.)

    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, inter1) + polar2xy(opt, inter1-0.5*np.pi),
        polar2xy(radius, inter1),
        polar2xy(radius, inter1),
        polar2xy(radius, inter1) + polar2xy(opt, inter1+0.5*np.pi),
        polar2xy(radius, inter2) + polar2xy(opt, inter2-0.5*np.pi),
        polar2xy(radius, inter2),
        polar2xy(radius, inter2),
        polar2xy(radius, inter2) + polar2xy(opt, inter2+0.5*np.pi),
        polar2xy(radius, inter3) + polar2xy(opt, inter3-0.5*np.pi),
        polar2xy(radius, inter3),
        polar2xy(radius, inter3),
        polar2xy(radius, inter3) + polar2xy(opt, inter3+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    return start, end, verts, codes


def ideogram_arc(start, end, radius=1., width=0.2, color="r", alpha=0.7,
                 ax=None):
    '''
    Draw an arc symbolizing a region of the chord diagram.

    Parameters
    ----------
    start : float (degree in 0, 360)
        Starting degree.
    end : float (degree in 0, 360)
        Final degree.
    radius : float, optional (default: 1)
        External radius of the arc.
    width : float, optional (default: 0.2)
        Width of the arc.
    ax : matplotlib axis, optional (default: not plotted)
        Axis on which the arc should be plotted.
    color : valid matplotlib color, optional (default: "r")
        Color of the arc.

    Returns
    -------
    verts, codes : lists
        Vertices and path instructions to draw the shape.
    '''
    start, end, verts, codes = initial_path(start, end, radius, width)

    opt    = 4./3. * np.tan((end-start)/ 16.) * radius
    inner  = radius*(1-width)
    inter1 = start*(3./4.) + end*(1./4.)
    inter2 = start*(2./4.) + end*(2./4.)
    inter3 = start*(1./4.) + end*(3./4.)

    verts += [
        polar2xy(inner, end),
        polar2xy(inner, end) + polar2xy(opt*(1-width), end-0.5*np.pi),
        polar2xy(inner, inter3) + polar2xy(opt*(1-width), inter3+0.5*np.pi),
        polar2xy(inner, inter3),
        polar2xy(inner, inter3),
        polar2xy(inner, inter3) + polar2xy(opt*(1-width), inter3-0.5*np.pi),
        polar2xy(inner, inter2) + polar2xy(opt*(1-width), inter2+0.5*np.pi),
        polar2xy(inner, inter2),
        polar2xy(inner, inter2),
        polar2xy(inner, inter2) + polar2xy(opt*(1-width), inter2-0.5*np.pi),
        polar2xy(inner, inter1) + polar2xy(opt*(1-width), inter1+0.5*np.pi),
        polar2xy(inner, inter1),
        polar2xy(inner, inter1),
        polar2xy(inner, inter1) + polar2xy(opt*(1-width), inter1-0.5*np.pi),
        polar2xy(inner, start) + polar2xy(opt*(1-width), start+0.5*np.pi),
        polar2xy(inner, start),
        polar2xy(radius, start),
    ]

    codes += [
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]

    if ax is not None:
        path  = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, alpha=alpha,
                                  edgecolor=color, lw=LW)
        ax.add_patch(patch)

    return verts, codes


def chord_arc(start1, end1, start2, end2, radius=1.0, gap=0.03, pad=2,
              chordwidth=0.7, ax=None, color="r", cend="r", alpha=0.7,
              use_gradient=False, extent=360, directed=False):
    '''
    Draw a chord between two regions (arcs) of the chord diagram.

    Parameters
    ----------
    start1 : float (degree in 0, 360)
        Starting degree.
    end1 : float (degree in 0, 360)
        Final degree.
    start2 : float (degree in 0, 360)
        Starting degree.
    end2 : float (degree in 0, 360)
        Final degree.
    radius : float, optional (default: 1)
        External radius of the arc.
    gap : float, optional (default: 0)
        Distance between the arc and the beginning of the cord.
    chordwidth : float, optional (default: 0.2)
        Width of the chord.
    ax : matplotlib axis, optional (default: not plotted)
        Axis on which the chord should be plotted.
    color : valid matplotlib color, optional (default: "r")
        Color of the chord or of its beginning if `use_gradient` is True.
    cend : valid matplotlib color, optional (default: "r")
        Color of the end of the chord if `use_gradient` is True.
    alpha : float, optional (default: 0.7)
        Opacity of the chord.
    use_gradient : bool, optional (default: False)
        Whether a gradient should be use so that chord extremities have the
        same color as the arc they belong to.
    extent : float, optional (default : 360)
        The angular aperture, in degrees, of the diagram.
        Default is to use the whole circle, i.e. 360 degrees, but in some cases
        it can be useful to use only a part of it.
    directed : bool, optional (default: False)
        Whether the chords should be directed, ending in an arrow.

    Returns
    -------
    verts, codes : lists
        Vertices and path instructions to draw the shape.
    '''
    chordwidth2 = chordwidth

    dtheta1 = min((start1 - end2) % extent, (end2 - start1) % extent)
    dtheta2 = min((end1 - start2) % extent, (start2 - end1) % extent)

    start1, end1, verts, codes = initial_path(start1, end1, radius, chordwidth)

    if directed:
        if start2 > end2:
            start2, end2 = end2, start2

        start2 *= np.pi/180.
        end2   *= np.pi/180.

        tip = 0.5*(start2 + end2)
        asize = max(gap, 0.02)

        verts2 = [
            polar2xy(radius - asize, start2),
            polar2xy(radius, tip),
            polar2xy(radius - asize, end2)
        ]
    else:
        start2, end2, verts2, _ = initial_path(start2, end2, radius, chordwidth)

    chordwidth2 *= np.clip(0.4 + (dtheta1 - 2*pad) / (15*pad), 0.2, 1)

    chordwidth *= np.clip(0.4 + (dtheta2 - 2*pad) / (15*pad), 0.2, 1)

    rchord  = radius * (1-chordwidth)
    rchord2 = radius * (1-chordwidth2)

    verts += [polar2xy(rchord, end1), polar2xy(rchord, start2)] + verts2

    verts += [
        polar2xy(rchord2, end2),
        polar2xy(rchord2, start1),
        polar2xy(radius, start1),
    ]

    # update codes

    codes += [
        Path.CURVE4,
        Path.CURVE4,
    ]

    if directed:
        codes += [
            Path.CURVE4,
            Path.LINETO,
            Path.LINETO,
        ]
    else:
        codes += [
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

    codes += [
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax is not None:
        path = Path(verts, codes)

        if use_gradient:
            # find the start and end points of the gradient
            points, min_angle = None, None

            if dtheta1 < dtheta2:
                points = [
                    polar2xy(radius, start1),
                    polar2xy(radius, end2),
                ]

                min_angle = dtheta1
            else:
                points = [
                    polar2xy(radius, end1),
                    polar2xy(radius, start2),
                ]

                min_angle = dtheta1

            # make the patch
            patch = patches.PathPatch(path, facecolor="none",
                                      edgecolor="none", lw=LW)
            ax.add_patch(patch)  # this is required to clip the gradient

            # make the grid
            x = y = np.linspace(-1, 1, 100)
            meshgrid = np.meshgrid(x, y)

            gradient(points[0], points[1], min_angle, color, cend, meshgrid,
                     patch, ax, alpha)
        else:
            patch = patches.PathPatch(path, facecolor=color, alpha=alpha,
                                      edgecolor=color, lw=LW)

            idx = 16

            ax.add_patch(patch)

    return verts, codes


def self_chord_arc(start, end, radius=1.0, chordwidth=0.7, ax=None,
                   color=(1,0,0), alpha=0.7):
    start, end, verts, codes = initial_path(start, end, radius, chordwidth)

    rchord = radius * (1 - chordwidth)

    verts += [
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
    ]

    codes += [
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax is not None:
        path  = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, alpha=alpha,
                                  edgecolor=color, lw=LW)
        ax.add_patch(patch)

    return verts, codes