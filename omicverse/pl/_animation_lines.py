import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from .._registry import register_function


class Streamlines(object):
    """
    Copyright (c) 2011 Raymond Speth.
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    See: http://web.mit.edu/speth/Public/streamlines.py
    """
    def __init__(self, X, Y, U, V, res=0.125,
                 spacing=2, maxLen=2500, detectLoops=False):
        """
        Initialize streamline tracer from gridded velocity field.

        Parameters
        ----------
        X : array-like
            1D or 2D grid x coordinates.
        Y : array-like
            1D or 2D grid y coordinates.
        U : array-like
            Velocity x component on the grid.
        V : array-like
            Velocity y component on the grid.
        res : float
            Step resolution along streamline integration.
        spacing : int
            Minimum spacing (in grid cells) between seed points.
        maxLen : int
            Maximum integration length for a streamline.
        detectLoops : bool
            Whether to stop when loops or near-stationary cycles are detected.
        """
        self.spacing = spacing
        self.detectLoops = detectLoops
        self.maxLen = maxLen
        self.res = res
        xa = np.asanyarray(X)
        ya = np.asanyarray(Y)
        self.x = xa if xa.ndim == 1 else xa[0]
        self.y = ya if ya.ndim == 1 else ya[:, 0]
        self.u = U
        self.v = V
        self.dx = (self.x[-1] - self.x[0]) / (self.x.size - 1)  # assume a regular grid
        self.dy = (self.y[-1] - self.y[0]) / (self.y.size - 1)  # assume a regular grid
        self.dr = self.res * np.sqrt(self.dx * self.dy)
        # marker for which regions have contours
        self.used = np.zeros(self.u.shape, dtype=bool)
        self.used[0] = True
        self.used[-1] = True
        self.used[:, 0] = True
        self.used[:, -1] = True
        # Don't try to compute streamlines in regions where there is no velocity data
        for i in range(self.x.size):
            for j in range(self.y.size):
                if self.u[j, i] == 0.0 and self.v[j, i] == 0.0:
                    self.used[j, i] = True
        # Make the streamlines
        self.streamlines = []
        while not self.used.all():
            nz = np.transpose(np.logical_not(self.used).nonzero())
            # Make a streamline starting at the first unrepresented grid point
            self.streamlines.append(self._makeStreamline(self.x[nz[0][1]],
                                                         self.y[nz[0][0]]))

    def _interp(self, x, y):
        """
        Bilinearly interpolate velocity at coordinate ``(x, y)``.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        tuple
            Interpolated velocity components ``(u, v)``.
        """
        i = (x - self.x[0]) / self.dx
        ai = i % 1
        j = (y - self.y[0]) / self.dy
        aj = j % 1
        i, j = int(i), int(j)
        # Bilinear interpolation
        u = (self.u[j, i] * (1 - ai) * (1 - aj) +
             self.u[j, i + 1] * ai * (1 - aj) +
             self.u[j + 1, i] * (1 - ai) * aj +
             self.u[j + 1, i + 1] * ai * aj)
        v = (self.v[j, i] * (1 - ai) * (1 - aj) +
             self.v[j, i + 1] * ai * (1 - aj) +
             self.v[j + 1, i] * (1 - ai) * aj +
             self.v[j + 1, i + 1] * ai * aj)
        self.used[j:j + self.spacing, i:i + self.spacing] = True
        return u, v

    def _makeStreamline(self, x0, y0):
        """
        Integrate streamline forward and backward from a seed point.

        Parameters
        ----------
        x0 : float
            Seed x coordinate.
        y0 : float
            Seed y coordinate.

        Returns
        -------
        tuple
            Streamline coordinate lists ``(x_vals, y_vals)``.
        """
        sx, sy = self._makeHalfStreamline(x0, y0, 1)  # forwards
        rx, ry = self._makeHalfStreamline(x0, y0, -1)  # backwards
        rx.reverse()
        ry.reverse()
        return rx + [x0] + sx, ry + [y0] + sy

    def _makeHalfStreamline(self, x0, y0, sign):
        """
        Integrate one directional half-streamline from a seed point.

        Parameters
        ----------
        x0 : float
            Seed x coordinate.
        y0 : float
            Seed y coordinate.
        sign : int
            Integration direction sign (``+1`` forward, ``-1`` backward).

        Returns
        -------
        tuple
            Half-streamline coordinate lists ``(x_vals, y_vals)``.
        """
        xmin = self.x[0]
        xmax = self.x[-1]
        ymin = self.y[0]
        ymax = self.y[-1]
        sx = []
        sy = []
        x = x0
        y = y0
        i = 0
        while xmin < x < xmax and ymin < y < ymax:
            u, v = self._interp(x, y)
            theta = np.arctan2(v, u)
            x += sign * self.dr * np.cos(theta)
            y += sign * self.dr * np.sin(theta)
            sx.append(x)
            sy.append(y)
            i += 1
            if self.detectLoops and i % 10 == 0 and self._detectLoop(sx, sy):
                break
            if i > self.maxLen / 2:
                break
        return sx, sy

    def _detectLoop(self, xVals, yVals):
        """
        Detect whether streamline trajectory forms a near-loop.

        Parameters
        ----------
        xVals : list
            Streamline x coordinates.
        yVals : list
            Streamline y coordinates.

        Returns
        -------
        bool
            ``True`` if loop-like recurrence is detected.
        """
        x = xVals[-1]
        y = yVals[-1]
        D = np.array([np.hypot(x - xj, y - yj)
                      for xj, yj in zip(xVals[:-1], yVals[:-1])])
        return (D < 0.9 * self.dr).any()


def compute_velocity_on_grid(
        X_emb,
        V_emb,
        density=None,
        smooth=None,
        n_neighbors=None,
        min_mass=None,
        autoscale=True,
        adjust_for_stream=True,
        cutoff_perc=None,
):
    """
    Interpolate cell-level velocity vectors onto a regular grid.

    Parameters
    ----------
    X_emb : np.ndarray
        Cell embedding coordinates with shape ``(n_cells, 2)``.
    V_emb : np.ndarray
        Cell velocity vectors in embedding space with shape ``(n_cells, 2)``.
    density : float or None
        Grid density factor.
    smooth : float or None
        Gaussian smoothing scale.
    n_neighbors : int or None
        Number of neighbors used for local velocity interpolation.
    min_mass : float or None
        Minimum local mass threshold for valid velocity grid points.
    autoscale : bool
        Reserved compatibility parameter.
    adjust_for_stream : bool
        Whether to reshape output for direct use in ``streamplot``.
    cutoff_perc : float or None
        Percentile cutoff for pruning low-confidence vectors.

    Returns
    -------
    tuple
        ``(X_grid, V_grid)`` grid coordinates and interpolated vector field.
    """
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]
    
    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth
    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)
    
    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T
    
    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = min(int(n_obs / 50), 20)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)
    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    
    if min_mass is None:
        min_mass = 1
    if adjust_for_stream == True:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)
        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass
        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)
        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
    
    return X_grid, V_grid


def nan_helper(y):
    """
    Provide convenient NaN mask and index converter.

    Parameters
    ----------
    y : np.ndarray
        Input 1D array.

    Returns
    -------
    tuple
        ``(nan_mask, index_func)`` for NaN interpolation workflows.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def animate_streamplot(X_grid, V_grid, adata=None, 
                      # Animation parameters
                      n_frames=20, interval=40, fps=25,
                      # Visual parameters  
                      alpha_animate=0.7, cmap_stream='Blues', linewidth=0.5,
                      segment_length=1, figsize=(4, 4),
                      # Background plotting parameters
                      basis='X_umap', color='celltype', palette=None,
                      background_color='black',
                      # Save parameters
                      saveto='animation.gif', show_plot=True,
                      # Streamline parameters
                      streamline_res=0.125, streamline_spacing=2, streamline_maxLen=2500):
    """
    Create an animated streamplot visualization.
    
    Parameters
    ----------
    X_grid : array-like
        Grid coordinates ``[X, Y]`` where each item is a 2D mesh array.
    V_grid : array-like
        Velocity field ``[U, V]`` aligned with ``X_grid``.
    adata : AnnData or None
        Optional AnnData used to draw embedding points as background.
    n_frames : int
        Number of animation frames.
    interval : int
        Delay between frames in milliseconds.
    fps : int
        Frames per second used when saving GIF.
    alpha_animate : float
        Alpha transparency of animated streamlines.
    cmap_stream : str
        Colormap used to render streamlines.
    linewidth : float
        Base line width of streamline segments.
    segment_length : float
        Segment-length scaling factor for animation speed/visuals.
    figsize : tuple
        Figure size passed to matplotlib.
    basis : str
        Embedding key used when plotting ``adata`` background.
    color : str
        Observation/feature key used for background coloring.
    palette : dict or None
        Optional color mapping for categorical backgrounds.
    saveto : str or None
        Output GIF path. If ``None``, animation is not saved.
    show_plot : bool
        Whether to display animation figure.
    streamline_res : float
        Resolution parameter used by streamline tracing.
    streamline_spacing : int
        Minimum spacing between generated streamlines.
    streamline_maxLen : int
        Maximum streamline integration length.
        
    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for downstream display or saving.
    """
    
    print("Creating streamlines...")
    streamlines = Streamlines(X_grid[0], X_grid[1], V_grid[0], V_grid[1],
                             res=streamline_res, spacing=streamline_spacing, 
                             maxLen=streamline_maxLen)
    
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(background_color)
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor(background_color)
    
    # Add background scatter plot if data provided
    if adata is not None:
        from ._single import embedding

        embedding(
            adata,
            basis=basis,
            color=color,
            palette=palette,
            ax=ax,
            show=False,
            legend_loc=None,
            frameon=False,
            title=''
        )
    
    # Prepare animation data
    lengths = []
    colors = []
    lines = []
    linewidths = []
    
    print(f"Processing {len(streamlines.streamlines)} streamlines...")
    for i, (x, y) in enumerate(streamlines.streamlines):
        # Handle NaN values
        x_ = np.array(x)
        nans, func_ = nan_helper(x_)
        if np.any(nans) and np.any(~nans):
            x_[nans] = np.interp(func_(nans), func_(~nans), x_[~nans])
        
        y_ = np.array(y)
        nans, func_ = nan_helper(y_)
        if np.any(nans) and np.any(~nans):
            y_[nans] = np.interp(func_(nans), func_(~nans), y_[~nans])
        
        # Create line segments
        points = np.array([x_, y_]).T.reshape(-1, 1, 2)
        if len(points) < 2:
            continue
            
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n = len(segments)
        
        # Calculate segment lengths and cumulative distances
        D = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=-1)) / segment_length
        np.random.seed(i + 42)
        L = D.cumsum().reshape(n, 1) + np.random.uniform(0, 1)
        
        # Initialize colors
        C = np.zeros((n, 4))
        C[::-1] = (L * 1.5) % 1
        C[:, 3] = alpha_animate
        
        lw = L.flatten().tolist()
        
        # Create line collection
        line = LineCollection(segments, color=C, linewidth=linewidth)
        
        lengths.append(L)
        colors.append(C)
        linewidths.append(lw)
        lines.append(line)
        
        ax.add_collection(line)
    
    # Set axis limits
    ax.set_xlim(X_grid[0].min(), X_grid[0].max())
    ax.set_ylim(X_grid[1].min(), X_grid[1].max())
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    print(f"Created {len(lines)} animated streamlines")
    
    # Animation update function
    def update(frame_no):
        cmap = plt.cm.get_cmap(cmap_stream)
        if background_color == 'black':
            cmap_colors = cmap(np.linspace(0, 0.1, 100))  # Light portion of colormap
        else:
            cmap_colors = cmap(np.linspace(0, 1, 100))  # Light portion of colormap
        
        for i in range(len(lines)):
            if len(lengths[i]) == 0:
                continue
                
            lengths[i] -= 0.05
            
            # Update colors with clipping to avoid hard blacks
            colors[i][::-1] = np.clip(0.1 + (lengths[i] * 1.5) % 1, 0.2, 0.9)
            colors[i][:, 3] = alpha_animate
            
            # Update line widths
            temp = (lengths[i] * 1) % 2
            linewidths[i] = temp.flatten().tolist()
            
            # Apply colormap
            for row in range(colors[i].shape[0]):
                color_idx = int(colors[i][row, 0] * 99)  # Map to colormap index
                color_idx = np.clip(color_idx, 0, 99)
                colors[i][row, :3] = cmap_colors[color_idx][:3]
            
            colors[i][:, 3] = alpha_animate
            lines[i].set_linewidth(linewidths[i])
            lines[i].set_color(colors[i])
    
    # Create animation
    print(f"Creating animation with {n_frames} frames...")
    animation = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=True)
    
    # Save animation
    if saveto:
        try:
            animation.save(saveto, writer='pillow', fps=fps,
                          savefig_kwargs={'facecolor': 'black'})
            print(f"Animation saved to {saveto}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            print("You can still view the animation interactively")
    
    if show_plot:
        plt.show()
    
    return animation


# Example usage:
# animation = animate_streamplot(X_grid, V_grid, adata, 
#                               color='celltype', palette=color_dict,
#                               saveto='my_animation.gif')

@register_function(
    aliases=['添加流线图', 'add_streamplot', 'rna velocity streamplot'],
    category="pl",
    description="Overlay RNA-velocity streamlines on low-dimensional embeddings to visualize directionality of state transitions.",
    prerequisites={'optional_functions': ['pp.neighbors', 'pp.umap']},
    requires={'obsm': ['X_umap or selected basis'], 'obsp': ['connectivities'], 'layers': ['velocity (optional)']},
    produces={},
    auto_fix='none',
    examples=['ov.pl.add_streamplot(adata, basis="umap", velocity_key="velocity")'],
    related=['pl.embedding', 'single.Velo', 'utils.cal_paga']
)
def add_streamplot(
    adata,
    basis='X_umap',
    velocity_key='velocity_S',
    density=1,
    smooth=0.5,
    min_mass=1,
    autoscale=True,
    adjust_for_stream=True,
    ax=None,
    arrow_color="k",
    stream_kwargs=None,
):
    """
    Overlay velocity streamlines on a low-dimensional embedding.
    
    Parameters
    ----------
    adata : AnnData
        AnnData containing embedding coordinates and velocity projections.
    basis : str
        ``adata.obsm`` key for embedding coordinates.
    velocity_key : str
        ``adata.obsm`` key for velocity vectors in embedding space.
    density : float
        Grid density for streamline interpolation.
    smooth : float
        Gaussian smoothing applied on velocity grid.
    min_mass : float
        Minimum mass threshold used by grid construction.
    autoscale : bool
        Whether to autoscale vectors.
    adjust_for_stream : bool
        Whether to adjust vectors for streamline plotting.
    ax : matplotlib.axes.Axes or None
        Target axes. Uses current axes if ``None``.
    arrow_color : str
        Streamline color.
    stream_kwargs : dict or None
        Extra keyword arguments passed to ``Axes.streamplot``.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes with streamlines.
    
    Examples
    --------
    >>> ov.pl.embedding(adata, basis='X_umap')
    >>> ov.pl.add_streamplot(adata, basis='X_umap', velocity_key='velocity_S')
    """
    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=adata.obsm[basis],
        V_emb=adata.obsm[velocity_key],
        density=density,
        smooth=smooth,
        min_mass=min_mass,
        autoscale=autoscale,
        adjust_for_stream=adjust_for_stream,
    )
    if stream_kwargs is None:
        stream_kwargs = {
        "linewidth": 0.5,
        "density": 2,
        "zorder": 3,
        
        "arrowsize": 1,
        "arrowstyle": "-|>",
            "maxlength": 4,
        "integration_direction": "both",
    }
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color=arrow_color, **stream_kwargs)
    return ax
