import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import networkx as nx
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
try:
    import marsilea as ma
    import marsilea.plotter as mp
    from matplotlib.colors import Normalize
    MARSILEA_AVAILABLE = True
except ImportError:
    MARSILEA_AVAILABLE = False

from ._cpdbviz_plus import CellChatVizPlus

class CellChatViz(CellChatVizPlus):
    """
    CellChat-like visualization for CellPhoneDB AnnData
    """
    
    def __init__(self, adata, palette=None):
        """
        Initialize with CellPhoneDB AnnData object
        
        Args:
            adata: AnnData
                AnnData object with CellPhoneDB results
                - obs: 'sender', 'receiver'
                - var: interaction information including 'classification'
                - layers: 'pvalues', 'means'
            palette: dict or list, optional
                Color palette for cell types. Can be:
                - dict: mapping cell type names to colors
                - list: list of colors (will be mapped to cell types in alphabetical order)
                - None: use default color scheme
        """
        self.adata = adata
        self.cell_types = self._get_unique_cell_types()
        self.n_cell_types = len(self.cell_types)
        self.palette = self._validate_palette(palette)
        self._color_cache = None  # Cache for consistent colors across all methods
        
    def _get_unique_cell_types(self):
        """Get unique cell types from sender and receiver"""
        senders = self.adata.obs['sender'].unique()
        receivers = self.adata.obs['receiver'].unique()
        return sorted(list(set(list(senders) + list(receivers))))
    
    def _validate_palette(self, palette):
        """
        Validate and process the palette parameter
        
        Args:
            palette: dict, list, or None
                Color palette specification
            
        Returns:
            dict or None
            Validated palette as dict mapping cell types to colors, or None
        """
        if palette is None:
            return None
        
        if isinstance(palette, dict):
            # Validate that all cell types have colors
            missing_types = set(self.cell_types) - set(palette.keys())
            if missing_types:
                import warnings
                warnings.warn(f"Palette missing colors for cell types: {missing_types}. Will use default colors for missing types.")
            return palette
        
        elif isinstance(palette, (list, tuple)):
            # Map colors to cell types in alphabetical order
            if len(palette) < len(self.cell_types):
                import warnings
                warnings.warn(f"Palette has {len(palette)} colors but {len(self.cell_types)} cell types. Colors will be recycled.")
            
            # Create mapping
            palette_dict = {}
            for i, cell_type in enumerate(self.cell_types):
                palette_dict[cell_type] = palette[i % len(palette)]
            return palette_dict
        
        else:
            raise ValueError("Palette must be a dictionary, list, or None")
    
    def _get_cell_type_colors(self):
        """Get cell type color mapping, ensuring all methods use consistent colors"""
        # If already cached colors, return directly
        if self._color_cache is not None:
            return self._color_cache
        
        cell_type_colors = {}
        
        # Prioritize user-provided palette
        if self.palette is not None:
            cell_type_colors.update(self.palette)
        
        # If palette doesn't cover all cell types, try to get color info from adata.uns
        missing_types = set(self.cell_types) - set(cell_type_colors.keys())
        if missing_types:
            color_keys = [key for key in self.adata.uns.keys() if key.endswith('_colors')]
            
            # Look for possible cell type colors
            for key in color_keys:
                # Extract cell type name (remove '_colors' suffix)
                celltype_key = key.replace('_colors', '')
                if celltype_key in self.adata.obs.columns:
                    categories = self.adata.obs[celltype_key].cat.categories
                    colors = self.adata.uns[key]
                    # Only keep colors for cell types in our list
                    for i, cat in enumerate(categories):
                        if cat in missing_types and i < len(colors):
                            cell_type_colors[cat] = colors[i]
                    break
        
        # If still missing colors, use default color mapping
        missing_types = set(self.cell_types) - set(cell_type_colors.keys())
        if missing_types:
            # Use fixed color mapping to ensure consistency
            import matplotlib.cm as cm
            # Use tab20 colormap, but ensure stable color assignment
            tab20_colors = cm.tab20(np.linspace(0, 1, 20))
            from ..pl._palette import palette_56
            
            # Assign colors to missing cell types
            for i, ct in enumerate(sorted(missing_types)):
                cell_type_colors[ct] = palette_56[i]
        
        # Ensure color mapping stability: sort by cell type name
        sorted_colors = {}
        for ct in sorted(self.cell_types):
            sorted_colors[ct] = cell_type_colors[ct]
        
        # Cache color mapping
        self._color_cache = sorted_colors
        
        return sorted_colors
    
    def _create_custom_colormap(self, cell_color):
        """
        Create a custom colormap based on cell type color
        
        Args:
            cell_color: str
                Base color for the cell type
        
        Returns:
            cmap: matplotlib.colors.LinearSegmentedColormap
                Custom colormap
        """
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.colors as mcolors
        
        # Convert color to RGB if it's a hex string
        if isinstance(cell_color, str):
            base_rgb = mcolors.to_rgb(cell_color)
        else:
            base_rgb = cell_color[:3] if len(cell_color) >= 3 else cell_color
        
        # Create gradient from light to dark
        colors = [(1.0, 1.0, 1.0, 0.3), base_rgb + (1.0,)]  # White transparent to full color
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        return cmap
    
    def _draw_curved_arrow(self, ax, start_pos, end_pos, weight, max_weight, color, 
                          edge_width_max=10, curve_strength=0.3, arrowsize=10):
        """
        Draw curved arrows, mimicking CellChat's rotated blooming effect
        
        Args:
            ax: matplotlib.axes.Axes
                Matplotlib axes object
            start_pos: tuple
                Starting position (x, y)
            end_pos: tuple
                Ending position (x, y)
            weight: float
                Edge weight
            max_weight: float
                Maximum weight for normalization
            color: str or tuple
                Edge color
            edge_width_max: float
                Maximum edge width
            curve_strength: float
                Strength of the curve (0 = straight, higher = more curved)
            arrowsize: float
                Size of the arrow head
        """
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.patches import ConnectionPatch
        import matplotlib.patches as patches
        
        # Calculate arrow width
        width = (weight / max_weight) * edge_width_max
        
        # Calculate midpoint and normal vector to create curved effect
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate vector
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Calculate midpoint
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Calculate perpendicular vector (for curvature)
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Normalize perpendicular vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Add curvature offset
            curve_offset = curve_strength * length
            control_x = mid_x + perp_x * curve_offset
            control_y = mid_y + perp_y * curve_offset
            
            # Create curved path
            from matplotlib.path import Path
            import matplotlib.patches as patches
            
            # Define Bezier curve path
            verts = [
                (start_x, start_y),  # Start point
                (control_x, control_y),  # Control point
                (end_x, end_y),  # End point
            ]
            
            codes = [
                Path.MOVETO,  # Move to start point
                Path.CURVE3,  # Quadratic Bezier curve
                Path.CURVE3,  # Quadratic Bezier curve
            ]
            
            path = Path(verts, codes)
            
            # Draw curved line
            patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                                    linewidth=width, alpha=0.7)
            ax.add_patch(patch)
            
            # Add arrow head
            # Calculate arrow direction
            arrow_dx = end_x - control_x
            arrow_dy = end_y - control_y
            arrow_length = np.sqrt(arrow_dx**2 + arrow_dy**2)
            
            if arrow_length > 0:
                # Normalize direction vector
                arrow_dx /= arrow_length
                arrow_dy /= arrow_length
                
                # Arrow head size
                head_length = arrowsize * 0.01
                head_width = arrowsize * 0.008
                
                # Calculate three points of arrow head
                # Arrow tip
                tip_x = end_x
                tip_y = end_y
                
                # Two base points of arrow
                base_x = tip_x - arrow_dx * head_length
                base_y = tip_y - arrow_dy * head_length
                
                left_x = base_x - arrow_dy * head_width
                left_y = base_y + arrow_dx * head_width
                right_x = base_x + arrow_dy * head_width
                right_y = base_y - arrow_dx * head_width
                
                # Draw arrow head
                triangle = plt.Polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], 
                                     color=color, alpha=0.8)
                ax.add_patch(triangle)
    
    def _draw_self_loop(self, ax, pos, weight, max_weight, color, edge_width_max):
        """
        Draw self-loops (connections from cell type to itself)
        
        Args:
            ax: matplotlib.axes.Axes
                Matplotlib axes object
            pos: tuple
                Position (x, y)
            weight: float
                Edge weight
            max_weight: float
                Maximum weight for normalization
            color: str or tuple
                Edge color
            edge_width_max: float
                Maximum edge width
        """
        import matplotlib.patches as patches
        
        x, y = pos
        width = (weight / max_weight) * edge_width_max
        
        # Create a small circle as self-loop
        radius = 0.15
        circle = patches.Circle((x + radius, y), radius, fill=False, 
                              edgecolor=color, linewidth=width, alpha=0.7)
        ax.add_patch(circle)
        
        # Add small arrow
        arrow_x = x + radius + radius * 0.7
        arrow_y = y
        arrow = patches.FancyArrowPatch((arrow_x - 0.05, arrow_y), (arrow_x, arrow_y),
                                      arrowstyle='->', mutation_scale=10, 
                                      color=color, alpha=0.8)
        ax.add_patch(arrow)
    
    def compute_aggregated_network(self, pvalue_threshold=0.05, use_means=True):
        """
        Compute aggregated cell communication network
        
        Args:
            pvalue_threshold: float
                P-value threshold for significant interactions
            use_means: bool
                Whether to use mean expression values as weights
        
        Returns:
            count_matrix: np.array
                Number of interactions between cell types
            weight_matrix: np.array
                Sum of interaction strengths between cell types
        """
        # Initialize matrices
        count_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
        weight_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
        
        # Get significant interactions
        pvalues = self.adata.layers['pvalues']
        means = self.adata.layers['means']
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
            sender_idx = self.cell_types.index(sender)
            receiver_idx = self.cell_types.index(receiver)
            
            # Count significant interactions
            sig_mask = pvalues[i, :] < pvalue_threshold
            count_matrix[sender_idx, receiver_idx] += np.sum(sig_mask)
            
            # Sum interaction strengths
            if use_means:
                weight_matrix[sender_idx, receiver_idx] += np.sum(means[i, sig_mask])
        
        return count_matrix, weight_matrix
    
    def netVisual_circle(self, matrix, title="Cell-Cell Communication Network", 
                        edge_width_max=10, vertex_size_max=50, show_labels=True,
                        cmap='Blues', figsize=(10, 10), use_sender_colors=True,
                        use_curved_arrows=True, curve_strength=0.3, adjust_text=False):
        """
        Circular network visualization (similar to CellChat's circle plot)
        Uses sender cell type colors as edge gradient colors
        
        Args:
            matrix: np.array
                Interaction matrix (count or weight)
            title: str
                Plot title
            edge_width_max: float
                Maximum edge width
            vertex_size_max: float
                Maximum vertex size
            show_labels: bool
                Whether to show cell type labels
            cmap: str
                Colormap for edges (used when use_sender_colors=False)
            figsize: tuple
                Figure size
            use_sender_colors: bool
                Whether to use different colors for different sender cell types (default: True)
            use_curved_arrows: bool
                Whether to use curved arrows like CellChat (default: True)
            curve_strength: float
                Strength of the curve (0 = straight, higher = more curved)
            adjust_text: bool
                Whether to use adjust_text library to prevent label overlapping (default: False)
                If True, uses plt.text instead of nx.draw_networkx_labels
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create circular layout
        angles = np.linspace(0, 2*np.pi, self.n_cell_types, endpoint=False)
        pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_cell_types))
        
        # Add edges with weights
        max_weight = matrix.max()
        if max_weight == 0:
            max_weight = 1  # Avoid division by zero
            
        for i in range(self.n_cell_types):
            for j in range(self.n_cell_types):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j], sender_idx=i)
        
        # Draw nodes
        node_sizes = matrix.sum(axis=1) + matrix.sum(axis=0)
        if node_sizes.max() > 0:
            node_sizes = (node_sizes / node_sizes.max() * vertex_size_max * 100) + 200
        else:
            node_sizes = np.full(self.n_cell_types, 200)
        
        # Get cell type colors for nodes
        cell_colors = self._get_cell_type_colors()
        node_colors = [cell_colors.get(self.cell_types[i], '#1f77b4') for i in range(self.n_cell_types)]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, 
                              ax=ax, alpha=0.8, edgecolors='black', linewidths=1)
        
        # Draw edges with curved arrows
        if use_curved_arrows and len(G.edges()) > 0:
            if use_sender_colors:
                # Group edges by sender
                edges_by_sender = {}
                for u, v, data in G.edges(data=True):
                    sender_idx = data['sender_idx']
                    if sender_idx not in edges_by_sender:
                        edges_by_sender[sender_idx] = []
                    edges_by_sender[sender_idx].append((u, v, data['weight']))
                
                # Draw curved edges for each sender with its specific color
                for sender_idx, edges in edges_by_sender.items():
                    sender_cell_type = self.cell_types[sender_idx]
                    sender_color = cell_colors.get(sender_cell_type, '#1f77b4')
                    
                    for u, v, weight in edges:
                        start_pos = pos[u]
                        end_pos = pos[v]
                        
                        # Handle self-loops
                        if u == v:
                            # Draw self-loop
                            self._draw_self_loop(ax, start_pos, weight, max_weight, 
                                               sender_color, edge_width_max)
                        else:
                            # Draw curved arrow
                            self._draw_curved_arrow(ax, start_pos, end_pos, weight, max_weight, 
                                                  sender_color, edge_width_max, curve_strength)
            else:
                # Use traditional single colormap
                cmap_obj = plt.cm.get_cmap(cmap)
                for u, v, data in G.edges(data=True):
                    weight = data['weight']
                    color = cmap_obj(weight / max_weight)
                    
                    start_pos = pos[u]
                    end_pos = pos[v]
                    
                    if u == v:
                        self._draw_self_loop(ax, start_pos, weight, max_weight, 
                                           color, edge_width_max)
                    else:
                        self._draw_curved_arrow(ax, start_pos, end_pos, weight, max_weight, 
                                              color, edge_width_max, curve_strength)
        else:
            # Use traditional straight arrows
            if use_sender_colors and len(G.edges()) > 0:
                # Group edges by sender
                edges_by_sender = {}
                for u, v, data in G.edges(data=True):
                    sender_idx = data['sender_idx']
                    if sender_idx not in edges_by_sender:
                        edges_by_sender[sender_idx] = []
                    edges_by_sender[sender_idx].append((u, v, data['weight']))
                
                # Draw edges for each sender with its specific color
                for sender_idx, edges in edges_by_sender.items():
                    sender_cell_type = self.cell_types[sender_idx]
                    sender_color = cell_colors.get(sender_cell_type, '#1f77b4')
                    custom_cmap = self._create_custom_colormap(sender_color)
                    
                    edge_list = [(u, v) for u, v, w in edges]
                    weights = [w for u, v, w in edges]
                    edge_widths = [(w / max_weight) * edge_width_max for w in weights]
                    
                    # Normalize weights for color mapping
                    if len(weights) > 1 and max(weights) > min(weights):
                        norm_weights = [(w - min(weights))/(max(weights) - min(weights)) for w in weights]
                    else:
                        norm_weights = [0.5] * len(weights)  # Use middle color if all weights are same
                    
                    edge_colors = [custom_cmap(nw) for nw in norm_weights]
                    
                    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                          edge_color=edge_colors, alpha=0.7,
                                          arrows=True, arrowsize=10, ax=ax)
            else:
                # Use traditional single colormap
                edges = list(G.edges())
                if edges:
                    weights = [G[u][v]['weight'] for u, v in edges]
                    edge_widths = [(w / max_weight) * edge_width_max for w in weights]
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                                          edge_color=weights, edge_cmap=plt.cm.get_cmap(cmap),
                                          arrows=True, arrowsize=10, ax=ax)
        
        # Add labels
        if show_labels:
            label_pos = {i: (1.15*np.cos(angle), 1.15*np.sin(angle)) 
                        for i, angle in enumerate(angles)}
            labels = {i: self.cell_types[i] for i in range(self.n_cell_types)}
            
            if adjust_text:
                # Use plt.text with adjust_text to prevent overlapping
                try:
                    from adjustText import adjust_text
                    
                    texts = []
                    for i in range(self.n_cell_types):
                        x, y = label_pos[i]
                        text = ax.text(x, y, self.cell_types[i], 
                                     fontsize=10, ha='center', va='center',
                                     )
                        texts.append(text)
                    
                    # Adjust text positions to avoid overlapping
                    adjust_text(texts, ax=ax,
                              expand_points=(1.2, 1.2),
                              expand_text=(1.2, 1.2),
                              force_points=0.5,
                              force_text=0.5,
                              arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, lw=0.5))
                    
                except ImportError:
                    import warnings
                    warnings.warn("adjustText library not found. Using default nx.draw_networkx_labels instead.")
                    nx.draw_networkx_labels(G, label_pos, labels, font_size=10, ax=ax)
            else:
                # Use traditional networkx labels
                nx.draw_networkx_labels(G, label_pos, labels, font_size=10, ax=ax)
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        # Add legend for sender colors
        if use_sender_colors and len(G.edges()) > 0:
            # Create legend showing sender cell types and their colors
            legend_elements = []
            for i, cell_type in enumerate(self.cell_types):
                if np.any(matrix[i, :] > 0):  # Only show cell types that send signals
                    color = cell_colors.get(cell_type, '#1f77b4')
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, 
                                                    label=f'{cell_type} (sender)'))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='lower center', 
                         bbox_to_anchor=(1.05, 0), fontsize=8)
        else:
            # Add traditional colorbar for edge weights
            edges = list(G.edges())
            if edges:
                weights = [G[u][v]['weight'] for u, v in edges]
                sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), 
                                          norm=plt.Normalize(vmin=0, vmax=max_weight))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Interaction Strength', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig, ax
    
    def get_ligand_receptor_pairs(self, min_interactions=1, pvalue_threshold=0.05):
        """
        Get all significant ligand-receptor pair lists
        
        Args:
            min_interactions: int
                Minimum interaction count threshold
            pvalue_threshold: float
                P-value threshold for significance
        
        Returns:
            lr_pairs: list
                Significant ligand-receptor pair list
            lr_stats: dict
                Statistics for each ligand-receptor pair
        """
        # Determine ligand-receptor pair column name
        if 'gene_name' in self.adata.var.columns:
            lr_column = 'gene_name'
        elif 'interaction_name' in self.adata.var.columns:
            lr_column = 'interaction_name'
        else:
            lr_column = None
            print("Warning: No specific L-R pair column found, using variable names")
        
        if lr_column:
            lr_pairs = self.adata.var[lr_column].unique()
        else:
            lr_pairs = self.adata.var.index.unique()
        
        lr_stats = {}
        significant_pairs = []
        
        for lr_pair in lr_pairs:
            if lr_column:
                lr_mask = self.adata.var[lr_column] == lr_pair
            else:
                lr_mask = self.adata.var.index == lr_pair
            
            # Calculate total interactions for this ligand-receptor pair
            total_interactions = 0
            significant_interactions = 0
            
            for i in range(len(self.adata.obs)):
                pvals = self.adata.layers['pvalues'][i, lr_mask]
                means = self.adata.layers['means'][i, lr_mask]
                
                sig_mask = pvals < pvalue_threshold
                total_interactions += len(pvals)
                significant_interactions += np.sum(sig_mask)
            
            lr_stats[lr_pair] = {
                'total_interactions': total_interactions,
                'significant_interactions': significant_interactions,
                'significance_rate': significant_interactions / max(1, total_interactions)
            }
            
            if significant_interactions >= min_interactions:
                significant_pairs.append(lr_pair)
        
        return significant_pairs, lr_stats
    
    def mean(self, count_min=1):
        """
        Compute mean expression matrix for cell-cell interactions (like CellChat)
        
        Args:
            count_min: int
                Minimum count threshold to filter interactions (default: 1)
            
        Returns:
            mean_matrix: pd.DataFrame
                Mean expression matrix with senders as index and receivers as columns
        """
        # Initialize matrix
        mean_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
        
        # Get means data
        means = self.adata.layers['means']
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
            sender_idx = self.cell_types.index(sender)
            receiver_idx = self.cell_types.index(receiver)
            
            # Sum mean expression for this sender-receiver pair, applying count_min filter
            interaction_means = means[i, :]
            # Apply count_min threshold (interactions below threshold are set to 0)
            filtered_means = np.where(interaction_means >= count_min, interaction_means, 0)
            mean_matrix[sender_idx, receiver_idx] = np.sum(filtered_means)
        
        # Convert to DataFrame for easier handling
        mean_df = pd.DataFrame(mean_matrix, 
                              index=self.cell_types, 
                              columns=self.cell_types)
        
        return mean_df
    
    def pvalue(self, count_min=1):
        """
        Compute p-value matrix for cell-cell interactions (like CellChat)
        
        Args:
            count_min: int
                Minimum count threshold to filter interactions (default: 1)
            
        Returns:
            pvalue_matrix: pd.DataFrame
                Average p-value matrix with senders as index and receivers as columns
        """
        # Initialize matrix
        pvalue_matrix = np.ones((self.n_cell_types, self.n_cell_types))  # Default p=1
        
        # Get data
        pvalues = self.adata.layers['pvalues']
        means = self.adata.layers['means']
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
            sender_idx = self.cell_types.index(sender)
            receiver_idx = self.cell_types.index(receiver)
            
            # Get interaction data for this sender-receiver pair
            interaction_pvals = pvalues[i, :]
            interaction_means = means[i, :]
            
            # Apply count_min filter - only consider interactions above threshold
            valid_mask = interaction_means >= count_min
            
            if np.any(valid_mask):
                # Compute average p-value for valid interactions
                pvalue_matrix[sender_idx, receiver_idx] = np.mean(interaction_pvals[valid_mask])
            else:
                # No valid interactions, keep default p=1
                pvalue_matrix[sender_idx, receiver_idx] = 1.0
        
        # Convert to DataFrame for easier handling
        pvalue_df = pd.DataFrame(pvalue_matrix, 
                                index=self.cell_types, 
                                columns=self.cell_types)
        
        return pvalue_df
    
    def analyze_pathway_statistics(self, pathway_stats, show_details=True):
        """
        Analyze and display detailed pathway statistics
        
        Args:
            pathway_stats: dict
                Dictionary returned from get_signaling_pathways
            show_details: bool
                Whether to show detailed statistics for each pathway
        
        Returns:
            summary_df: pd.DataFrame
                Summary statistics for all pathways
        """
        if not pathway_stats:
            print("No pathway statistics available. Run get_signaling_pathways() first.")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_data = []
        for pathway, stats in pathway_stats.items():
            row = {
                'pathway': pathway,
                'n_lr_pairs': stats['n_lr_pairs'],
                'n_tests': stats['n_tests'],
                'n_significant': stats['n_significant_interactions'],
                'significance_rate': stats['significance_rate'],
                'combined_pvalue': stats['combined_pvalue'],
                'mean_expression': stats['mean_expression'],
                'max_expression': stats['max_expression'],
                'n_significant_cell_pairs': len(stats['significant_cell_pairs'])
            }
            
            # Add corrected p-value if available
            if 'corrected_pvalue' in stats:
                row['corrected_pvalue'] = stats['corrected_pvalue']
                row['is_significant'] = stats['is_significant_corrected']
            else:
                row['is_significant'] = stats['combined_pvalue'] < 0.05
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('combined_pvalue')
        
        if show_details:
            print("📊 Pathway Analysis Summary:")
            print("=" * 80)
            print(f"{'Pathway':<30} {'L-R':<4} {'Tests':<6} {'Sig':<4} {'Rate':<6} {'P-val':<8} {'Expr':<6}")
            print("-" * 80)
            
            for _, row in summary_df.head(20).iterrows():  # Show top 20
                significance_marker = "***" if row['is_significant'] else "   "
                print(f"{row['pathway'][:28]:<30} {row['n_lr_pairs']:<4} {row['n_tests']:<6} "
                      f"{row['n_significant']:<4} {row['significance_rate']:.2f}  "
                      f"{row['combined_pvalue']:.1e} {row['mean_expression']:.2f} {significance_marker}")
            
            if len(summary_df) > 20:
                print(f"... and {len(summary_df) - 20} more pathways")
        
        return summary_df
    
    def compute_pathway_communication(self, method='mean', min_lr_pairs=1, min_expression=0.1):
        """
        Calculate pathway-level cell communication strength (similar to CellChat methods)
        
        Args:
            method: str
                Aggregation method: 'mean', 'sum', 'max', 'median' (default: 'mean')
            min_lr_pairs: int
                Minimum L-R pair count in pathway (default: 1)  
            min_expression: float
                Minimum expression threshold (default: 0.1)
            
        Returns:
            pathway_communication: dict
                Contains communication matrix and statistics for each pathway
        """
        pathways = [p for p in self.adata.var['classification'].unique() if pd.notna(p)]
        pathway_communication = {}
        
        print(f"🔬 Calculating cell communication strength for {len(pathways)} pathways...")
        print(f"   - Aggregation method: {method}")
        print(f"   - Minimum expression threshold: {min_expression}")
        
        for pathway in pathways:
            pathway_mask = self.adata.var['classification'] == pathway
            pathway_lr_pairs = self.adata.var.loc[pathway_mask, 'interacting_pair'].tolist()
            
            if len(pathway_lr_pairs) < min_lr_pairs:
                continue
                
            # Initialize pathway communication matrix
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_pvalue_matrix = np.ones((self.n_cell_types, self.n_cell_types))
            valid_interactions_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # Get all L-R pair data for this pathway between this cell pair
                pathway_means = self.adata.layers['means'][i, pathway_mask]
                pathway_pvals = self.adata.layers['pvalues'][i, pathway_mask]
                
                # Filter low expression interactions
                valid_mask = pathway_means >= min_expression
                
                if np.any(valid_mask):
                    valid_means = pathway_means[valid_mask]
                    valid_pvals = pathway_pvals[valid_mask]
                    
                    # Calculate pathway-level communication strength
                    if method == 'mean':
                        pathway_strength = np.mean(valid_means)
                    elif method == 'sum':
                        pathway_strength = np.sum(valid_means)
                    elif method == 'max':
                        pathway_strength = np.max(valid_means)
                    elif method == 'median':
                        pathway_strength = np.median(valid_means)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    # Calculate pathway-level p-value (use minimum p-value as pathway significance)
                    pathway_pval = np.min(valid_pvals)
                    
                    pathway_matrix[sender_idx, receiver_idx] = pathway_strength
                    pathway_pvalue_matrix[sender_idx, receiver_idx] = pathway_pval
                    valid_interactions_matrix[sender_idx, receiver_idx] = len(valid_means)
                else:
                    # No valid interactions
                    pathway_matrix[sender_idx, receiver_idx] = 0
                    pathway_pvalue_matrix[sender_idx, receiver_idx] = 1.0
                    valid_interactions_matrix[sender_idx, receiver_idx] = 0
            
            # Store pathway communication results
            pathway_communication[pathway] = {
                'communication_matrix': pd.DataFrame(pathway_matrix, 
                                                   index=self.cell_types, 
                                                   columns=self.cell_types),
                'pvalue_matrix': pd.DataFrame(pathway_pvalue_matrix,
                                            index=self.cell_types,
                                            columns=self.cell_types),
                'n_valid_interactions': pd.DataFrame(valid_interactions_matrix,
                                                    index=self.cell_types,
                                                    columns=self.cell_types),
                'lr_pairs': pathway_lr_pairs,
                'total_strength': pathway_matrix.sum(),
                'max_strength': pathway_matrix.max(),
                'mean_strength': pathway_matrix[pathway_matrix > 0].mean() if (pathway_matrix > 0).any() else 0,
                'significant_pairs': np.sum(pathway_pvalue_matrix < 0.05),
                'aggregation_method': method
            }
        
        print(f"✅ Completed pathway communication strength calculation for {len(pathway_communication)} pathways")
        
        return pathway_communication
    
    def get_significant_pathways_v2(self, pathway_communication=None, 
                                   strength_threshold=0.1, pvalue_threshold=0.05, 
                                   min_significant_pairs=1):
        """
        Determine significant pathways based on pathway-level communication strength (more aligned with CellChat logic)
        
        Args:
            pathway_communication: dict or None
                Pathway communication results, if None then recalculate
            strength_threshold: float
                Pathway strength threshold (default: 0.1)
            pvalue_threshold: float  
                p-value threshold (default: 0.05)
            min_significant_pairs: int
                Minimum significant cell pair count (default: 1)
            
        Returns:
            significant_pathways: list
                Significant pathway list
            pathway_summary: pd.DataFrame
                Pathway statistics summary
        """
        if pathway_communication is None:
            pathway_communication = self.compute_pathway_communication()
        
        pathway_summary_data = []
        significant_pathways = []
        
        for pathway, data in pathway_communication.items():
            comm_matrix = data['communication_matrix']
            pval_matrix = data['pvalue_matrix']
            
            # Pathway-level statistics
            total_strength = data['total_strength']
            max_strength = data['max_strength']
            mean_strength = data['mean_strength']
            
            # Use .values to ensure returning numpy arrays instead of pandas Series
            pval_values = pval_matrix.values
            comm_values = comm_matrix.values
            
            n_significant_pairs = np.sum((pval_values < pvalue_threshold) & (comm_values >= strength_threshold))
            n_total_pairs = np.sum(comm_values > 0)
            
            # Determine if pathway is significant
            is_significant = (total_strength >= strength_threshold and 
                            n_significant_pairs >= min_significant_pairs)
            
            pathway_summary_data.append({
                'pathway': pathway,
                'total_strength': total_strength,
                'max_strength': max_strength,
                'mean_strength': mean_strength,
                'n_lr_pairs': len(data['lr_pairs']),
                'n_active_cell_pairs': n_total_pairs,
                'n_significant_pairs': n_significant_pairs,
                'significance_rate': n_significant_pairs / max(1, n_total_pairs),
                'is_significant': is_significant
            })
            
            if is_significant:
                significant_pathways.append(pathway)
        
        # Create summary DataFrame
        pathway_summary = pd.DataFrame(pathway_summary_data)
        pathway_summary = pathway_summary.sort_values('total_strength', ascending=False)
        
        print(f"📊 Pathway significance analysis results:")
        print(f"   - Total pathways: {len(pathway_summary)}")
        print(f"   - Significant pathways: {len(significant_pathways)}")
        print(f"   - Strength threshold: {strength_threshold}")
        print(f"   - p-value threshold: {pvalue_threshold}")
        
        # Show top pathways
        print(f"\n🏆 Top 10 pathways by total strength:")
        print("-" * 100)
        print(f"{'Pathway':<30} {'Total':<8} {'Max':<7} {'Mean':<7} {'L-R':<4} {'Active':<6} {'Sig':<4} {'Rate':<6} {'Status'}")
        print("-" * 100)
        
        for _, row in pathway_summary.head(10).iterrows():
            status = "***" if row['is_significant'] else "   "
            print(f"{row['pathway'][:28]:<30} {row['total_strength']:<8.2f} {row['max_strength']:<7.2f} "
                  f"{row['mean_strength']:<7.2f} {row['n_lr_pairs']:<4} {row['n_active_cell_pairs']:<6} "
                  f"{row['n_significant_pairs']:<4} {row['significance_rate']:<6.2f} {status}")
        
        return significant_pathways, pathway_summary
    
    def demo_curved_arrows(self, signaling_pathway=None, curve_strength=0.4, figsize=(12, 10)):
        """
        Demo function to show curved arrow effects
        
        Args:
            signaling_pathway: str or None
                Signaling pathway to visualize, if None use aggregated network
            curve_strength: float
                Arrow curvature strength (0-1), 0 for straight lines, higher values for more curvature
            figsize: tuple
                Figure size
        
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        print("🌸 Demonstrating CellChat-style curved arrow effects...")
        print(f"📏 Curvature strength: {curve_strength} (recommended range: 0.2-0.6)")
        
        if signaling_pathway is not None:
            # Visualize specific signaling pathway
            fig, ax = self.netVisual_aggregate(
                signaling=signaling_pathway,
                layout='circle',
                focused_view=True,
                use_curved_arrows=True,
                curve_strength=curve_strength,
                figsize=figsize,
                use_sender_colors=True
            )
            print(f"✨ Generated curved arrow network plot for signaling pathway '{signaling_pathway}'")
        else:
            # Visualize aggregated network
            _, weight_matrix = self.compute_aggregated_network()
            fig, ax = self.netVisual_circle_focused(
                matrix=weight_matrix,
                title="Cell-Cell Communication Network (Curved Arrows)",
                use_curved_arrows=True,
                curve_strength=curve_strength,
                figsize=figsize,
                use_sender_colors=True
            )
            print("✨ Generated curved arrow plot for aggregated network")
        
        print("💡 Tips:")
        print("  - curve_strength=0.2: Slight curvature")
        print("  - curve_strength=0.4: Medium curvature (recommended)") 
        print("  - curve_strength=0.6: Strong curvature")
        print("  - use_curved_arrows=False: Switch back to straight arrows")
        
        return fig, ax
    
    def netVisual_circle_focused(self, matrix, title="Cell-Cell Communication Network", 
                                edge_width_max=10, vertex_size_max=50, show_labels=True,
                                cmap='Blues', figsize=(10, 10), min_interaction_threshold=0,
                                use_sender_colors=True, use_curved_arrows=True, curve_strength=0.3):
        """
        Draw focused circular network diagram, showing only cell types with actual interactions
        
        Args:
            matrix: np.array
                Interaction matrix (count or weight)
            title: str
                Plot title
            edge_width_max: float
                Maximum edge width
            vertex_size_max: float
                Maximum vertex size
            show_labels: bool
                Whether to show cell type labels
            cmap: str
                Colormap for edges (used when use_sender_colors=False)
            figsize: tuple
                Figure size
            min_interaction_threshold: float
                Minimum interaction strength to include cell type
            use_sender_colors: bool
                Whether to use different colors for different sender cell types
            use_curved_arrows: bool
                Whether to use curved arrows like CellChat (default: True)
            curve_strength: float
                Strength of the curve (0 = straight, higher = more curved)
            
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        # Find cell types with actual interactions
        interaction_mask = (matrix.sum(axis=0) + matrix.sum(axis=1)) > min_interaction_threshold
        active_cell_indices = np.where(interaction_mask)[0]
        
        if len(active_cell_indices) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No interactions above threshold', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig, ax
        
        # Create filtered matrix and cell type list
        filtered_matrix = matrix[np.ix_(active_cell_indices, active_cell_indices)]
        active_cell_types = [self.cell_types[i] for i in active_cell_indices]
        n_active_cells = len(active_cell_types)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create circular layout for active cells only
        angles = np.linspace(0, 2*np.pi, n_active_cells, endpoint=False)
        pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n_active_cells))
        
        # Add edges with weights and original sender indices
        max_weight = filtered_matrix.max()
        if max_weight == 0:
            max_weight = 1  # Avoid division by zero
            
        for i in range(n_active_cells):
            for j in range(n_active_cells):
                if filtered_matrix[i, j] > 0:
                    # Store original sender index for color mapping
                    original_sender_idx = active_cell_indices[i]
                    G.add_edge(i, j, weight=filtered_matrix[i, j], 
                             original_sender_idx=original_sender_idx)
        
        # Draw nodes
        node_sizes = filtered_matrix.sum(axis=1) + filtered_matrix.sum(axis=0)
        if node_sizes.max() > 0:
            node_sizes = (node_sizes / node_sizes.max() * vertex_size_max * 100) + 200
        else:
            node_sizes = np.full(n_active_cells, 200)
        
        # Get cell type colors for nodes
        cell_colors = self._get_cell_type_colors()
        node_colors = [cell_colors.get(active_cell_types[i], '#1f77b4') for i in range(n_active_cells)]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, 
                              ax=ax, alpha=0.8, edgecolors='black', linewidths=1)
        
        # Draw edges with curved arrows
        if use_curved_arrows and len(G.edges()) > 0:
            if use_sender_colors:
                # Group edges by original sender
                edges_by_sender = {}
                for u, v, data in G.edges(data=True):
                    original_sender_idx = data['original_sender_idx']
                    if original_sender_idx not in edges_by_sender:
                        edges_by_sender[original_sender_idx] = []
                    edges_by_sender[original_sender_idx].append((u, v, data['weight']))
                
                # Draw curved edges for each sender with its specific color
                for original_sender_idx, edges in edges_by_sender.items():
                    sender_cell_type = self.cell_types[original_sender_idx]
                    sender_color = cell_colors.get(sender_cell_type, '#1f77b4')
                    
                    for u, v, weight in edges:
                        start_pos = pos[u]
                        end_pos = pos[v]
                        
                        # Handle self-loops
                        if u == v:
                            # Draw self-loop
                            self._draw_self_loop(ax, start_pos, weight, max_weight, 
                                               sender_color, edge_width_max)
                        else:
                            # Draw curved arrow
                            self._draw_curved_arrow(ax, start_pos, end_pos, weight, max_weight, 
                                                  sender_color, edge_width_max, curve_strength)
            else:
                # Use traditional single colormap
                cmap_obj = plt.cm.get_cmap(cmap)
                for u, v, data in G.edges(data=True):
                    weight = data['weight']
                    color = cmap_obj(weight / max_weight)
                    
                    start_pos = pos[u]
                    end_pos = pos[v]
                    
                    if u == v:
                        self._draw_self_loop(ax, start_pos, weight, max_weight, 
                                           color, edge_width_max)
                    else:
                        self._draw_curved_arrow(ax, start_pos, end_pos, weight, max_weight, 
                                              color, edge_width_max, curve_strength)
        else:
            # Use traditional straight arrows
            if use_sender_colors and max_weight > 0:
                # Group edges by original sender
                edges_by_sender = {}
                for u, v, data in G.edges(data=True):
                    original_sender_idx = data['original_sender_idx']
                    if original_sender_idx not in edges_by_sender:
                        edges_by_sender[original_sender_idx] = []
                    edges_by_sender[original_sender_idx].append((u, v, data['weight']))
                
                # Draw edges for each sender with its specific color
                for original_sender_idx, edges in edges_by_sender.items():
                    sender_cell_type = self.cell_types[original_sender_idx]
                    sender_color = cell_colors.get(sender_cell_type, '#1f77b4')
                    custom_cmap = self._create_custom_colormap(sender_color)
                    
                    edge_list = [(u, v) for u, v, w in edges]
                    weights = [w for u, v, w in edges]
                    edge_widths = [(w / max_weight) * edge_width_max for w in weights]
                    
                    # Normalize weights for color mapping
                    norm_weights = [(w - min(weights))/(max(weights) - min(weights)) if max(weights) > min(weights) else 0.5 for w in weights]
                    edge_colors = [custom_cmap(nw) for nw in norm_weights]
                    
                    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=edge_widths,
                                          edge_color=edge_colors, alpha=0.7,
                                          arrows=True, arrowsize=10, ax=ax)
            else:
                # Use traditional single colormap
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                if weights:
                    edge_widths = [(w / max_weight) * edge_width_max for w in weights]
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                                          edge_color=weights, edge_cmap=plt.cm.get_cmap(cmap),
                                          arrows=True, arrowsize=10, ax=ax)
        
        # Add labels
        if show_labels:
            label_pos = {i: (1.15*np.cos(angle), 1.15*np.sin(angle)) 
                        for i, angle in enumerate(angles)}
            labels = {i: active_cell_types[i] for i in range(n_active_cells)}
            nx.draw_networkx_labels(G, label_pos, labels, font_size=10, ax=ax)
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        # Add legend for sender colors
        if use_sender_colors and max_weight > 0:
            # Create legend showing sender cell types and their colors
            legend_elements = []
            active_senders = set()
            for u, v, data in G.edges(data=True):
                original_sender_idx = data['original_sender_idx']
                active_senders.add(original_sender_idx)
            
            for sender_idx in active_senders:
                cell_type = self.cell_types[sender_idx]
                color = cell_colors.get(cell_type, '#1f77b4')
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, 
                                                label=f'{cell_type} (sender)'))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='center left', 
                         bbox_to_anchor=(1, 0.5), fontsize=8)
        else:
            # Add traditional colorbar for edge weights
            if weights:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), 
                                          norm=plt.Normalize(vmin=0, vmax=max_weight))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Interaction Strength', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig, ax
    
    def netVisual_individual_circle(self, pvalue_threshold=0.05, vertex_size_max=50, 
                                   edge_width_max=10, show_labels=True, cmap='Blues', 
                                   figsize=(20, 15), ncols=4, use_sender_colors=True):
        """
        Draw individual circular network diagrams for each cell type, showing its outgoing signals
        Mimics CellChat functionality, using sender cell type colors as edge gradients
        
        Args:
            pvalue_threshold: float
                P-value threshold for significant interactions
            vertex_size_max: float
                Maximum vertex size
            edge_width_max: float
                Maximum edge width (consistent across all plots for comparison)
            show_labels: bool
                Whether to show cell type labels
            cmap: str
                Colormap for edges (used when use_sender_colors=False)
            figsize: tuple
                Figure size
            ncols: int
                Number of columns in subplot layout
            use_sender_colors: bool
                Whether to use sender cell type colors for edges (default: True)
        
        Returns:
            fig: matplotlib.figure.Figure
                Figure containing all subplots
        """
        # Compute weight matrix
        _, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
        
        # Get the maximum weight across the entire matrix for consistent scaling
        global_max_weight = weight_matrix.max()
        if global_max_weight == 0:
            global_max_weight = 1  # Avoid division by zero
        
        # Calculate subplot layout
        nrows = (self.n_cell_types + ncols - 1) // ncols
        
        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        # Create circular layout (same for all subplots)
        angles = np.linspace(0, 2*np.pi, self.n_cell_types, endpoint=False)
        pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
        
        # Calculate vertex sizes based on total communication strength
        vertex_weights = weight_matrix.sum(axis=1) + weight_matrix.sum(axis=0)
        if vertex_weights.max() > 0:
            vertex_sizes = (vertex_weights / vertex_weights.max() * vertex_size_max * 100) + 200
        else:
            vertex_sizes = np.full(self.n_cell_types, 200)
        
        # Get cell type colors for consistency
        cell_colors = self._get_cell_type_colors()
        
        for i in range(self.n_cell_types):
            # Calculate subplot position
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            # Create individual matrix (only outgoing from cell type i)
            individual_matrix = np.zeros_like(weight_matrix)
            individual_matrix[i, :] = weight_matrix[i, :]
            
            # Create graph for this cell type
            G = nx.DiGraph()
            G.add_nodes_from(range(self.n_cell_types))
            
            # Add edges
            for j in range(self.n_cell_types):
                if individual_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=individual_matrix[i, j], sender_idx=i)
            
            # Draw nodes with consistent vertex sizes and colors
            node_colors = [cell_colors.get(self.cell_types[j], '#1f77b4') for j in range(self.n_cell_types)]
            
            nx.draw_networkx_nodes(G, pos, node_size=vertex_sizes, 
                                  node_color=node_colors, 
                                  ax=ax, alpha=0.8, edgecolors='black', linewidths=1)
            
            # Draw edges with sender-specific colors
            edges = list(G.edges())
            if edges and use_sender_colors:
                # Use sender's color for all edges from this cell type
                sender_cell_type = self.cell_types[i]
                sender_color = cell_colors.get(sender_cell_type, '#1f77b4')
                custom_cmap = self._create_custom_colormap(sender_color)
                
                weights = [G[u][v]['weight'] for u, v in edges]
                edge_widths = [(w / global_max_weight) * edge_width_max for w in weights]
                
                # Normalize weights for color mapping
                if len(weights) > 1 and max(weights) > min(weights):
                    norm_weights = [(w - min(weights))/(max(weights) - min(weights)) for w in weights]
                else:
                    norm_weights = [0.5] * len(weights)  # Use middle color if all weights are same
                
                edge_colors = [custom_cmap(nw) for nw in norm_weights]
                
                nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                      edge_color=edge_colors, alpha=0.7,
                                      arrows=True, arrowsize=15, ax=ax)
            elif edges:
                # Use traditional single colormap
                weights = [G[u][v]['weight'] for u, v in edges]
                edge_widths = [(w / global_max_weight) * edge_width_max for w in weights]
                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                                      edge_color=weights, edge_cmap=plt.cm.get_cmap(cmap),
                                      arrows=True, arrowsize=15, ax=ax)
            
            # Add labels
            if show_labels:
                label_pos = {j: (1.2*np.cos(angle), 1.2*np.sin(angle)) 
                           for j, angle in enumerate(angles)}
                labels = {j: self.cell_types[j] for j in range(self.n_cell_types)}
                nx.draw_networkx_labels(G, label_pos, labels, font_size=8, ax=ax)
            
            # Set title with cell type name and use sender color
            title_color = cell_colors.get(self.cell_types[i], '#000000')
            ax.set_title(self.cell_types[i], fontsize=12, pad=10, weight='bold', color=title_color)
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(self.n_cell_types, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].axis('off')
        
        # Add a single colorbar for the entire figure
        if global_max_weight > 0:
            # Create colorbar on the right side
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            if use_sender_colors:
                # Use a neutral colormap for the colorbar when using sender colors
                sm = plt.cm.ScalarMappable(cmap='viridis', 
                                          norm=plt.Normalize(vmin=0, vmax=global_max_weight))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cbar_ax)
                cbar.set_label('Interaction Strength\n(Colors: Sender Cell Types)', rotation=270, labelpad=25)
            else:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), 
                                          norm=plt.Normalize(vmin=0, vmax=global_max_weight))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cbar_ax)
                cbar.set_label('Interaction Strength', rotation=270, labelpad=20)
        
        # Add main title
        fig.suptitle('Individual Cell Type Communication Networks\n(Outgoing Signals with Sender Colors)', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        
        return fig
    
    def netVisual_individual_circle_incoming(self, pvalue_threshold=0.05, vertex_size_max=50, 
                                           edge_width_max=10, show_labels=True, cmap='Reds', 
                                           figsize=(20, 15), ncols=4, use_sender_colors=True):
        """
        Draw individual circular network diagrams for each cell type, showing its incoming signals
        Uses sender cell type colors as edge gradients
        
        Args:
            pvalue_threshold: float
                P-value threshold for significant interactions
            vertex_size_max: float
                Maximum vertex size
            edge_width_max: float
                Maximum edge width
            show_labels: bool
                Whether to show cell type labels
            cmap: str
                Colormap for edges (used when use_sender_colors=False)
            figsize: tuple
                Figure size
            ncols: int
                Number of columns in subplot layout
            use_sender_colors: bool
                Whether to use sender cell type colors for edges (default: True)
        
        Returns:
            fig: matplotlib.figure.Figure
                Figure containing all subplots
        """
        # Compute weight matrix
        _, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
        
        # Get the maximum weight across the entire matrix for consistent scaling
        global_max_weight = weight_matrix.max()
        if global_max_weight == 0:
            global_max_weight = 1  # Avoid division by zero
        
        # Calculate subplot layout
        nrows = (self.n_cell_types + ncols - 1) // ncols
        
        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        # Create circular layout (same for all subplots)
        angles = np.linspace(0, 2*np.pi, self.n_cell_types, endpoint=False)
        pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
        
        # Calculate vertex sizes based on total communication strength
        vertex_weights = weight_matrix.sum(axis=1) + weight_matrix.sum(axis=0)
        if vertex_weights.max() > 0:
            vertex_sizes = (vertex_weights / vertex_weights.max() * vertex_size_max * 100) + 200
        else:
            vertex_sizes = np.full(self.n_cell_types, 200)
        
        # Get cell type colors for consistency
        cell_colors = self._get_cell_type_colors()
        
        for i in range(self.n_cell_types):
            # Calculate subplot position
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            # Create individual matrix (only incoming to cell type i)
            individual_matrix = np.zeros_like(weight_matrix)
            individual_matrix[:, i] = weight_matrix[:, i]
            
            # Create graph for this cell type
            G = nx.DiGraph()
            G.add_nodes_from(range(self.n_cell_types))
            
            # Add edges
            for j in range(self.n_cell_types):
                if individual_matrix[j, i] > 0:
                    G.add_edge(j, i, weight=individual_matrix[j, i], sender_idx=j)
            
            # Draw nodes with consistent vertex sizes and colors
            node_colors = [cell_colors.get(self.cell_types[j], '#1f77b4') for j in range(self.n_cell_types)]
            
            nx.draw_networkx_nodes(G, pos, node_size=vertex_sizes, 
                                  node_color=node_colors, 
                                  ax=ax, alpha=0.8, edgecolors='black', linewidths=1)
            
            # Draw edges with sender-specific colors
            edges = list(G.edges())
            if edges and use_sender_colors:
                # Group edges by sender for incoming connections
                edges_by_sender = {}
                for u, v, data in G.edges(data=True):
                    sender_idx = data['sender_idx']
                    if sender_idx not in edges_by_sender:
                        edges_by_sender[sender_idx] = []
                    edges_by_sender[sender_idx].append((u, v, data['weight']))
                
                # Draw edges for each sender with its specific color
                for sender_idx, sender_edges in edges_by_sender.items():
                    sender_cell_type = self.cell_types[sender_idx]
                    sender_color = cell_colors.get(sender_cell_type, '#1f77b4')
                    custom_cmap = self._create_custom_colormap(sender_color)
                    
                    edge_list = [(u, v) for u, v, w in sender_edges]
                    weights = [w for u, v, w in sender_edges]
                    edge_widths = [(w / global_max_weight) * edge_width_max for w in weights]
                    
                    # Normalize weights for color mapping
                    if len(weights) > 1 and max(weights) > min(weights):
                        norm_weights = [(w - min(weights))/(max(weights) - min(weights)) for w in weights]
                    else:
                        norm_weights = [0.5] * len(weights)  # Use middle color if all weights are same
                    
                    edge_colors = [custom_cmap(nw) for nw in norm_weights]
                    
                    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=edge_widths,
                                          edge_color=edge_colors, alpha=0.7,
                                          arrows=True, arrowsize=15, ax=ax)
            elif edges:
                # Use traditional single colormap
                weights = [G[u][v]['weight'] for u, v in edges]
                edge_widths = [(w / global_max_weight) * edge_width_max for w in weights]
                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                                      edge_color=weights, edge_cmap=plt.cm.get_cmap(cmap),
                                      arrows=True, arrowsize=15, ax=ax)
            
            # Add labels
            if show_labels:
                label_pos = {j: (1.2*np.cos(angle), 1.2*np.sin(angle)) 
                           for j, angle in enumerate(angles)}
                labels = {j: self.cell_types[j] for j in range(self.n_cell_types)}
                nx.draw_networkx_labels(G, label_pos, labels, font_size=8, ax=ax)
            
            # Set title with cell type name and use receiver color (target cell type)
            title_color = cell_colors.get(self.cell_types[i], '#000000')
            ax.set_title(self.cell_types[i], fontsize=12, pad=10, weight='bold', color=title_color)
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(self.n_cell_types, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].axis('off')
        
        # Add a single colorbar for the entire figure
        if global_max_weight > 0:
            # Create colorbar on the right side
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            if use_sender_colors:
                # Use a neutral colormap for the colorbar when using sender colors
                sm = plt.cm.ScalarMappable(cmap='viridis', 
                                          norm=plt.Normalize(vmin=0, vmax=global_max_weight))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cbar_ax)
                cbar.set_label('Interaction Strength\n(Colors: Sender Cell Types)', rotation=270, labelpad=25)
            else:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), 
                                          norm=plt.Normalize(vmin=0, vmax=global_max_weight))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cbar_ax)
                cbar.set_label('Interaction Strength', rotation=270, labelpad=20)
        
        # Add main title
        fig.suptitle('Individual Cell Type Communication Networks\n(Incoming Signals with Sender Colors)', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        
        return fig
    
    def netVisual_heatmap(self, matrix, title="Communication Heatmap", 
                         cmap='Reds', figsize=(10, 8), show_values=True):
        """
        Heatmap visualization of cell-cell communication
        
        Args:
            matrix: np.array
                Interaction matrix
            title: str
                Plot title
            cmap: str
                Colormap
            show_values: bool
                Whether to show values in cells
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(matrix, index=self.cell_types, columns=self.cell_types)
        
        # Plot heatmap
        sns.heatmap(df, annot=show_values, fmt='.0f', cmap=cmap, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Target (Receiver)', fontsize=12)
        ax.set_ylabel('Source (Sender)', fontsize=12)
        
        plt.tight_layout()
        return fig, ax
    
    def netVisual_heatmap_marsilea(self, signaling=None, pvalue_threshold=0.05, 
                                color_heatmap="Reds", add_dendrogram=True,
                                add_row_sum=True, add_col_sum=True, 
                                linewidth=0.5, figsize=(8, 6), title="Communication Heatmap"):
        """
        Use marsilea package to draw cell-cell communication heatmap (mimicking CellChat's netVisual_heatmap function)
        
        Args:
            signaling: str, list or None
                Specific signaling pathway names. If None, show aggregated results of all pathways
            pvalue_threshold: float
                P-value threshold for significant interactions
            color_heatmap: str
                Heatmap colormap
            add_dendrogram: bool
                Whether to add dendrogram
            add_row_sum: bool
                Whether to show row sums on the left
            add_col_sum: bool
                Whether to show column sums on top  
            linewidth: float
                Grid line width
            figsize: tuple
                Figure size
            title: str
                Heatmap title
            
        Returns:
            h: marsilea heatmap object
        """
        if not MARSILEA_AVAILABLE:
            raise ImportError("marsilea package is not available. Please install it: pip install marsilea")
        
        # Calculate communication matrix
        if signaling is not None:
            # Calculate communication matrix for specific pathway
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # Check if signaling pathways exist
            available_pathways = self.adata.var['classification'].unique()
            for pathway in signaling:
                if pathway not in available_pathways:
                    raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
            
            # Calculate communication matrix for specific pathway
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_mask = self.adata.var['classification'].isin(signaling)
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                raise ValueError(f"No interactions found for pathway(s): {signaling}")
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                    self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # Get significant interactions for this pathway
                pvals = self.adata.layers['pvalues'][i, pathway_indices]
                means = self.adata.layers['means'][i, pathway_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
            
            matrix = pathway_matrix
            heatmap_title = f"{title} - {', '.join(signaling)}"
        else:
            # Use aggregated communication matrix
            _, matrix = self.compute_aggregated_network(pvalue_threshold)
            heatmap_title = title
        
        # Create DataFrame for better labeling
        df_matrix = pd.DataFrame(matrix, 
                                index=self.cell_types, 
                                columns=self.cell_types)
        #return df_matrix
        
        # Create marsilea heatmap
        h = ma.Heatmap(df_matrix, linewidth=linewidth, 
                    cmap=color_heatmap, label="Interaction Strength")
        
        # Add row and column grouping - this is a key step!
        #h.group_rows(df_matrix.index, order=df_matrix.index.tolist())
        #h.group_cols(df_matrix.columns, order=df_matrix.columns.tolist())
        
        
        
        # Add row sums (left side)
        if add_row_sum:
            row_sums = matrix.sum(axis=1)
            h.add_left(ma.plotter.Numbers(row_sums, color="#F05454", 
                                        label="Outgoing",show_value=False))
        
        # Add column sums (top)
        if add_col_sum:
            col_sums = matrix.sum(axis=0)
            h.add_top(ma.plotter.Numbers(col_sums, color="#4A90E2",
                                    label="Incoming",show_value=False))
        
        # Add cell type color annotations
        cell_colors = self._get_cell_type_colors()
        row_colors = [cell_colors.get(ct, '#808080') for ct in self.cell_types]
        col_colors = [cell_colors.get(ct, '#808080') for ct in self.cell_types]
        
        # Add cell type color bars
        h.add_left(ma.plotter.Colors(self.cell_types,palette=row_colors),size=0.2,legend=False)
        h.add_top(ma.plotter.Colors(self.cell_types,palette=col_colors),size=0.2)
        
        # Add legends
        h.add_legends()
        
        # Add title
        h.add_title(heatmap_title)
        # Add dendrograms
        if add_dendrogram:
            h.add_dendrogram("left", colors="#2E8B57")
            h.add_dendrogram("top", colors="#2E8B57")
        
        
        return h
    
    def netVisual_heatmap_marsilea_focused(self, signaling=None, pvalue_threshold=0.05,
                                          min_interaction_threshold=0, color_heatmap="Reds", 
                                          add_dendrogram=True, add_row_sum=True, add_col_sum=True,
                                          linewidth=0.5, figsize=(8, 6), title="Communication Heatmap"):
        """
        Use marsilea package to draw focused cell-cell communication heatmap, showing only cell types with actual interactions
        
        Args:
            signaling: str, list or None
                Specific signaling pathway names
            pvalue_threshold: float
                P-value threshold for significant interactions
            min_interaction_threshold: float
                Minimum interaction strength threshold for filtering cell types
            color_heatmap: str
                Heatmap colormap
            add_dendrogram: bool
                Whether to add dendrogram
            add_row_sum: bool
                Whether to show row sums on the left
            add_col_sum: bool
                Whether to show column sums on top
            linewidth: float
                Grid line width
            figsize: tuple
                Figure size
            title: str
                Heatmap title
            
        Returns:
            h: marsilea heatmap object
        """
        if not MARSILEA_AVAILABLE:
            raise ImportError("marsilea package is not available. Please install it: pip install marsilea")
        
        # First get complete matrix
        if signaling is not None:
            # Calculate communication matrix for specific pathway 
            if isinstance(signaling, str):
                signaling = [signaling]
                
            available_pathways = self.adata.var['classification'].unique()
            for pathway in signaling:
                if pathway not in available_pathways:
                    raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
            
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_mask = self.adata.var['classification'].isin(signaling)
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                raise ValueError(f"No interactions found for pathway(s): {signaling}")
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                      self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                pvals = self.adata.layers['pvalues'][i, pathway_indices]
                means = self.adata.layers['means'][i, pathway_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
            
            matrix = pathway_matrix
            heatmap_title = f"{title} - {', '.join(signaling)} (Focused)"
        else:
            _, matrix = self.compute_aggregated_network(pvalue_threshold)
            heatmap_title = f"{title} (Focused)"
        
        # Filter cell types with actual interactions
        interaction_mask = (matrix.sum(axis=0) + matrix.sum(axis=1)) > min_interaction_threshold
        active_cell_indices = np.where(interaction_mask)[0]
        
        if len(active_cell_indices) == 0:
            raise ValueError("No cell types have interactions above the threshold")
        
        # Create filtered matrix and cell type list
        filtered_matrix = matrix[np.ix_(active_cell_indices, active_cell_indices)]
        active_cell_types = [self.cell_types[i] for i in active_cell_indices]
        
        # Create DataFrame
        df_matrix = pd.DataFrame(filtered_matrix,
                                index=active_cell_types,
                                columns=active_cell_types)
        
        # Create marsilea heatmap
        h = ma.Heatmap(df_matrix, linewidth=linewidth,
                      cmap=color_heatmap, label="Interaction Strength")
        
        # Add row and column grouping - this is a key step!
        h.group_rows(df_matrix.index, order=df_matrix.index.tolist())
        h.group_cols(df_matrix.columns, order=df_matrix.columns.tolist())
        
        
        
        # Add row sums (left side)
        if add_row_sum:
            row_sums = filtered_matrix.sum(axis=1)
            h.add_left(ma.plotter.Numbers(row_sums, color="#F05454",
                                        label="Outgoing"))
        
        # Add column sums (top)
        if add_col_sum:
            col_sums = filtered_matrix.sum(axis=0)
            h.add_top(ma.plotter.Numbers(col_sums, color="#4A90E2",
                                       label="Incoming"))
        
        # Add cell type color annotations
        cell_colors = self._get_cell_type_colors()
        row_colors = [cell_colors.get(ct, '#808080') for ct in active_cell_types]
        col_colors = [cell_colors.get(ct, '#808080') for ct in active_cell_types]
            
        # Add cell type color bars
        h.add_left(ma.plotter.Chunk(active_cell_types, fill_colors=row_colors, 
                                       rotation=90, label="Cell Types (Senders)"))
        h.add_top(ma.plotter.Chunk(active_cell_types, fill_colors=col_colors, 
                                      rotation=90, label="Cell Types (Receivers)"))
        
        # Add legends
        h.add_legends()
        # Add dendrograms
        if add_dendrogram:
            h.add_dendrogram("right", colors="#2E8B57")
            h.add_dendrogram("top", colors="#2E8B57")
        
        # Add title
        h.add_title(heatmap_title)
        
        
        return h
    
    def netVisual_chord(self, matrix, title="Chord Diagram", threshold=0, 
                       cmap='tab20', figsize=(12, 12)):
        """
        Chord diagram visualization
        
        Args:
            matrix: np.array
                Interaction matrix
            title: str
                Plot title
            threshold: float
                Minimum value to show connection
        """
        from matplotlib.path import Path
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Filter by threshold
        matrix_filtered = matrix.copy()
        matrix_filtered[matrix_filtered < threshold] = 0
        
        # Calculate positions
        n = len(self.cell_types)
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Draw cell type arcs
        # Use consistent cell type colors
        cell_colors = self._get_cell_type_colors()
        colors = [cell_colors.get(cell_type, '#1f77b4') for cell_type in self.cell_types]
        arc_height = 0.1
        
        for i, (angle, cell_type, color) in enumerate(zip(theta, self.cell_types, colors)):
            # Draw arc
            ax.barh(1, 2*np.pi/n, left=angle-np.pi/n, height=arc_height, 
                   color=color, edgecolor='white', linewidth=2)
            
            # Add labels
            label_angle = angle * 180 / np.pi
            if label_angle > 90 and label_angle < 270:
                label_angle = label_angle + 180
                ha = 'right'
            else:
                ha = 'left'
            
            ax.text(angle, 1.15, cell_type, rotation=label_angle, 
                   ha=ha, va='center', fontsize=10)
        
        # Draw connections
        for i in range(n):
            for j in range(n):
                if matrix_filtered[i, j] > 0:
                    # Source and target angles
                    theta1 = theta[i]
                    theta2 = theta[j]
                    
                    # Draw bezier curve
                    width = matrix_filtered[i, j] / matrix.max() * 0.02
                    
                    # Create path
                    verts = [
                        (theta1, 0.9),
                        (theta1, 0.5),
                        (theta2, 0.5),
                        (theta2, 0.9)
                    ]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    path = Path(verts, codes)
                    
                    patch = patches.PathPatch(path, facecolor='none', 
                                            edgecolor=colors[i], 
                                            linewidth=width*50, alpha=0.6)
                    ax.add_patch(patch)
        
        ax.set_ylim(0, 1.2)
        ax.set_title(title, fontsize=16, pad=30, y=1.08)
        ax.axis('off')
        
        plt.tight_layout()
        return fig, ax
    
    def netVisual_hierarchy(self, pathway_name=None, sources=None, targets=None,
                           pvalue_threshold=0.05, figsize=(14, 10)):
        """
        Hierarchy plot visualization
        
        Args:
            pathway_name: str
                Specific pathway to visualize (from classification)
            sources: list
                Source cell types to show
            targets: list  
                Target cell types to show
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter data
        if pathway_name:
            pathway_mask = self.adata.var['classification'] == pathway_name
            # Convert boolean mask to indices to avoid AnnData slicing issues
            pathway_indices = np.where(pathway_mask)[0]
            if len(pathway_indices) == 0:
                ax.text(0.5, 0.5, f'No interactions found for pathway: {pathway_name}', 
                       ha='center', va='center', fontsize=16)
                return fig, ax
            data_filtered = self.adata[:, pathway_indices]
        else:
            data_filtered = self.adata
        
        # Get interactions
        interactions = []
        for i, (sender, receiver) in enumerate(zip(data_filtered.obs['sender'], 
                                                  data_filtered.obs['receiver'])):
            pvals = data_filtered.layers['pvalues'][i, :]
            means = data_filtered.layers['means'][i, :]
            sig_mask = pvals < pvalue_threshold
            
            if np.any(sig_mask):
                strength = np.mean(means[sig_mask])
                interactions.append((sender, receiver, strength))
        
        if not interactions:
            ax.text(0.5, 0.5, 'No significant interactions found', 
                   ha='center', va='center', fontsize=16)
            return fig, ax
        
        # Create layout
        df_int = pd.DataFrame(interactions, columns=['source', 'target', 'weight'])
        
        # Filter by sources and targets if specified
        if sources:
            df_int = df_int[df_int['source'].isin(sources)]
        if targets:
            df_int = df_int[df_int['target'].isin(targets)]
        
        if df_int.empty:
            ax.text(0.5, 0.5, 'No interactions found for specified sources/targets', 
                   ha='center', va='center', fontsize=16)
            return fig, ax
        
        sources_unique = df_int['source'].unique()
        targets_unique = df_int['target'].unique()
        
        # Position nodes
        source_y = np.linspace(0.1, 0.9, len(sources_unique))
        target_y = np.linspace(0.1, 0.9, len(targets_unique))
        
        source_pos = {cell: (0.2, y) for cell, y in zip(sources_unique, source_y)}
        target_pos = {cell: (0.8, y) for cell, y in zip(targets_unique, target_y)}
        
        # Draw nodes
        # Get consistent cell type colors
        cell_colors = self._get_cell_type_colors()
        
        for cell, (x, y) in source_pos.items():
            cell_color = cell_colors.get(cell, '#lightblue')
            rect = FancyBboxPatch((x-0.08, y-0.03), 0.16, 0.06,
                                 boxstyle="round,pad=0.01",
                                 facecolor=cell_color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, cell, ha='center', va='center', fontsize=10)
        
        for cell, (x, y) in target_pos.items():
            cell_color = cell_colors.get(cell, '#lightcoral')
            rect = FancyBboxPatch((x-0.08, y-0.03), 0.16, 0.06,
                                 boxstyle="round,pad=0.01",
                                 facecolor=cell_color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, cell, ha='center', va='center', fontsize=10)
        
        # Draw edges
        max_weight = df_int['weight'].max()
        for _, row in df_int.iterrows():
            source = row['source']
            target = row['target']
            weight = row['weight']
            
            if source in source_pos and target in target_pos:
                arrow = ConnectionPatch(source_pos[source], target_pos[target],
                                      "data", "data",
                                      arrowstyle="->", shrinkA=10, shrinkB=10,
                                      mutation_scale=20, fc="gray",
                                      linewidth=weight/max_weight * 5,
                                      alpha=0.6)
                ax.add_artist(arrow)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Hierarchy Plot{" - " + pathway_name if pathway_name else ""}', 
                    fontsize=16)
        ax.text(0.2, -0.05, 'Sources', ha='center', fontsize=12, weight='bold')
        ax.text(0.8, -0.05, 'Targets', ha='center', fontsize=12, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig, ax
    
    def netVisual_bubble(self, sources=None, targets=None, pathways=None,
                        pvalue_threshold=0.05, figsize=(12, 10)):
        """
        Bubble plot visualization
        
        Args:
            sources: list
                Source cell types to include
            targets: list
                Target cell types to include  
            pathways: list
                Pathways to include
        """
        # Prepare data
        data_list = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # Filter by sources and targets
            if sources and sender not in sources:
                continue
            if targets and receiver not in targets:
                continue
            
            # Get pathways for this interaction
            pvals = self.adata.layers['pvalues'][i, :]
            means = self.adata.layers['means'][i, :]
            classifications = self.adata.var['classification']
            
            # Filter by pathways
            if pathways:
                pathway_mask = classifications.isin(pathways)
            else:
                pathway_mask = np.ones(len(classifications), dtype=bool)
            
            # Get significant interactions
            sig_mask = (pvals < pvalue_threshold) & pathway_mask
            
            if np.any(sig_mask):
                sig_pathways = classifications[sig_mask].unique()
                for pathway in sig_pathways:
                    pathway_specific_mask = sig_mask & (classifications == pathway)
                    mean_strength = np.mean(means[pathway_specific_mask])
                    min_pval = np.min(pvals[pathway_specific_mask])
                    
                    data_list.append({
                        'source': sender,
                        'target': receiver,
                        'pathway': pathway,
                        'mean_expression': mean_strength,
                        'pvalue': min_pval
                    })
        
        if not data_list:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No significant interactions found', 
                   ha='center', va='center', fontsize=16)
            return fig, ax
        
        df = pd.DataFrame(data_list)
        
        # Create bubble plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create interaction labels
        df['interaction'] = df['source'] + ' → ' + df['target']
        
        # Create pivot table
        pivot_mean = df.pivot_table(values='mean_expression', 
                                    index='interaction', 
                                    columns='pathway', 
                                    aggfunc='mean')
        pivot_pval = df.pivot_table(values='pvalue', 
                                   index='interaction', 
                                   columns='pathway', 
                                   aggfunc='min')
        
        # Plot bubbles
        y_labels = pivot_mean.index
        x_labels = pivot_mean.columns
        
        for i, interaction in enumerate(y_labels):
            for j, pathway in enumerate(x_labels):
                if not pd.isna(pivot_mean.loc[interaction, pathway]):
                    size = pivot_mean.loc[interaction, pathway] * 100
                    pval = pivot_pval.loc[interaction, pathway]
                    color = -np.log10(pval + 1e-10)
                    
                    ax.scatter(j, i, s=size, c=color, cmap='Reds', 
                             vmin=0, vmax=5, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        
        ax.set_xlabel('Signaling Pathways', fontsize=12)
        ax.set_ylabel('Cell-Cell Interactions', fontsize=12)
        ax.set_title('Bubble Plot of Cell-Cell Communication', fontsize=16, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)
        
        # Add size legend
        sizes = [50, 100, 200]
        labels = ['Low', 'Medium', 'High']
        legend_elements = [plt.scatter([], [], s=s, c='gray', alpha=0.6, 
                                     edgecolors='black', linewidth=0.5) 
                          for s in sizes]
        legend = ax.legend(legend_elements, labels, title="Mean Expression",
                          loc='upper left', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return fig, ax
    
    def compute_pathway_network(self, pvalue_threshold=0.05):
        """
        Compute pathway-level communication networks
        
        Returns:
            pathway_networks: dict
                Dictionary with pathway names as keys and communication matrices as values
        """
        pathways = self.adata.var['classification'].unique()
        pathway_networks = {}
        
        for pathway in pathways:
            # Get interactions for this pathway
            pathway_mask = self.adata.var['classification'] == pathway
            pathway_indices = np.where(pathway_mask)[0]
            
            # Initialize matrix for this pathway
            matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            
            # Fill matrix
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                      self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # Get significant interactions for this pathway using indices
                if len(pathway_indices) > 0:
                    pvals = self.adata.layers['pvalues'][i, pathway_indices]
                    means = self.adata.layers['means'][i, pathway_indices]
                    
                    sig_mask = pvals < pvalue_threshold
                    if np.any(sig_mask):
                        matrix[sender_idx, receiver_idx] = np.sum(means[sig_mask])
            
            pathway_networks[pathway] = matrix
        
        return pathway_networks
    
    def identify_signaling_role(self, pattern="all", pvalue_threshold=0.05):
        """
        Identify cellular signaling roles (sender, receiver, mediator, influencer)
        
        Args:
            pattern: str
                "outgoing" for sender, "incoming" for receiver, "all" for overall
        """
        count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
        
        if pattern == "outgoing":
            scores = weight_matrix.sum(axis=1)  # Sum over receivers
            role = "Sender"
        elif pattern == "incoming":
            scores = weight_matrix.sum(axis=0)  # Sum over senders
            role = "Receiver"
        else:  # all
            scores = weight_matrix.sum(axis=1) + weight_matrix.sum(axis=0)
            role = "Mediator"
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'cell_type': self.cell_types,
            'score': scores,
            'relative_score': scores / scores.max() if scores.max() > 0 else scores
        })
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use consistent cell type colors
        cell_colors = self._get_cell_type_colors()
        bar_colors = [cell_colors.get(ct, '#steelblue') for ct in df['cell_type']]
        
        bars = ax.bar(df['cell_type'], df['score'], color=bar_colors, alpha=0.8)
        
        # Highlight top contributors with darker version of their color
        top_indices = np.argsort(scores)[-3:]  # Top 3
        for idx in top_indices:
            # Make the color darker for highlighting
            original_color = bar_colors[idx]
            bars[idx].set_alpha(1.0)
            bars[idx].set_edgecolor('darkred')
            bars[idx].set_linewidth(2)
        
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel(f'{role} Score', fontsize=12)
        ax.set_title(f'Cell Signaling Role Analysis - {role}', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, df['score']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return df, fig
    
    def compute_network_similarity(self, method='functional'):
        """
        Compute pathway similarity (functional or structural similarity)
        
        Args:
            method: str
                'functional' or 'structural'
        """
        pathway_networks = self.compute_pathway_network()
        pathways = list(pathway_networks.keys())
        n_pathways = len(pathways)
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((n_pathways, n_pathways))
        
        for i, pathway1 in enumerate(pathways):
            for j, pathway2 in enumerate(pathways):
                matrix1 = pathway_networks[pathway1].flatten()
                matrix2 = pathway_networks[pathway2].flatten()
                
                if method == 'functional':
                    # Cosine similarity
                    if np.any(matrix1) and np.any(matrix2):
                        similarity = cosine_similarity([matrix1], [matrix2])[0, 0]
                    else:
                        similarity = 0
                else:  # structural
                    # Jaccard similarity based on non-zero positions
                    nonzero1 = matrix1 > 0
                    nonzero2 = matrix2 > 0
                    intersection = np.sum(nonzero1 & nonzero2)
                    union = np.sum(nonzero1 | nonzero2)
                    similarity = intersection / union if union > 0 else 0
                
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix, pathways
    
    def netEmbedding(self, method='functional', n_components=2, figsize=(10, 8)):
        """
        Pathway embedding and clustering visualization (UMAP)
        
        Args:
            method: str
                'functional' or 'structural'
            n_components: int
                Number of UMAP components
        """
        try:
            from umap import UMAP
        except ImportError:
            print("UMAP not available. Please install umap-learn: pip install umap-learn")
            print("Falling back to PCA for dimensionality reduction...")
            from sklearn.decomposition import PCA
            use_pca = True
        else:
            use_pca = False
            
        # Compute similarity
        similarity_matrix, pathways = self.compute_network_similarity(method)
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Dimensionality reduction
        if use_pca:
            # Use PCA as fallback
            reducer = PCA(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(distance_matrix)
        else:
            # UMAP embedding
            reducer = UMAP(n_components=n_components, metric='precomputed', 
                         random_state=42)
            embedding = reducer.fit_transform(distance_matrix)
        
        # K-means clustering
        n_clusters = min(4, len(pathways))  # Default to 4 clusters or less
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embedding)
        
        # Visualize
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=clusters, cmap='tab10', s=100, alpha=0.8)
        
        # Add pathway labels
        for i, pathway in enumerate(pathways):
            ax.annotate(pathway, (embedding[i, 0], embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        method_name = 'UMAP' if not use_pca else 'PCA'
        ax.set_xlabel(f'{method_name} 1', fontsize=12)
        ax.set_ylabel(f'{method_name} 2', fontsize=12)
        ax.set_title(f'Pathway {method.capitalize()} Similarity - {method_name} Embedding', 
                    fontsize=16)
        
        # Add cluster legend
        legend_elements = [plt.scatter([], [], c=plt.cm.tab10(i), s=100, 
                                     label=f'Cluster {i+1}')
                          for i in range(n_clusters)]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        return embedding, clusters, fig
    
    def netVisual_single_circle(self, cell_type, direction='outgoing', pvalue_threshold=0.05, 
                               vertex_size_max=50, edge_width_max=10, show_labels=True, 
                               cmap='Blues', figsize=(8, 8), use_sender_colors=True):
        """
        Draw circular network diagram for a single specified cell type
        Uses sender cell type colors as edge gradients
        
        Args:
            cell_type: str
                Cell type name to draw
            direction: str
                'outgoing' shows signals sent by this cell type, 'incoming' shows signals received
            pvalue_threshold: float
                P-value threshold for significant interactions
            vertex_size_max: float
                Maximum vertex size
            edge_width_max: float
                Maximum edge width
            show_labels: bool
                Whether to show cell type labels
            cmap: str
                Colormap for edges (used when use_sender_colors=False)
            figsize: tuple
                Figure size
            use_sender_colors: bool
                Whether to use sender cell type colors for edges (default: True)
        
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        if cell_type not in self.cell_types:
            raise ValueError(f"Cell type '{cell_type}' not found in data. Available cell types: {self.cell_types}")
        
        # Compute weight matrix
        _, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
        
        # Get cell type index
        cell_idx = self.cell_types.index(cell_type)
        
        # Create individual matrix based on direction
        individual_matrix = np.zeros_like(weight_matrix)
        if direction == 'outgoing':
            individual_matrix[cell_idx, :] = weight_matrix[cell_idx, :]
            title = f'{cell_type} - Outgoing Signals'
            if cmap == 'Blues' and not use_sender_colors:  # Only change default if not using sender colors
                pass
        elif direction == 'incoming':
            individual_matrix[:, cell_idx] = weight_matrix[:, cell_idx]
            title = f'{cell_type} - Incoming Signals'
            if cmap == 'Blues' and not use_sender_colors:  # Change default to red for incoming
                cmap = 'Reds'
        else:
            raise ValueError("direction must be 'outgoing' or 'incoming'")
        
        # Use the existing netVisual_circle method with sender colors
        fig, ax = self.netVisual_circle(
            matrix=individual_matrix,
            title=title,
            edge_width_max=edge_width_max,
            vertex_size_max=vertex_size_max,
            show_labels=show_labels,
            cmap=cmap,
            figsize=figsize,
            use_sender_colors=use_sender_colors
        )
        
        return fig, ax
    
    def netVisual_aggregate(self, signaling, layout='circle', vertex_receiver=None, vertex_sender=None,
                           pvalue_threshold=0.05, vertex_size_max=50, edge_width_max=10,
                           show_labels=True, cmap='Blues', figsize=(10, 8), focused_view=True,
                           use_sender_colors=True, use_curved_arrows=True, curve_strength=0.3, adjust_text=False):
        """
        Draw aggregated network diagram for specific signaling pathways (mimicking CellChat's netVisual_aggregate function)
        
        Args:
            signaling: str or list
                Signaling pathway names (from adata.var['classification'])
            layout: str
                Layout type: 'circle' or 'hierarchy'
            vertex_receiver: list or None
                Receiver cell type names list (for hierarchy layout)
            vertex_sender: list or None
                Sender cell type names list (for hierarchy layout)
            pvalue_threshold: float
                P-value threshold for significant interactions
            vertex_size_max: float
                Maximum vertex size
            edge_width_max: float
                Maximum edge width
            show_labels: bool
                Whether to show cell type labels
            cmap: str
                Colormap for edges (used when use_sender_colors=False)
            figsize: tuple
                Figure size
            focused_view: bool
                Whether to use focused view (only show cell types with interactions) for circle layout
            use_sender_colors: bool
                Whether to use different colors for different sender cell types
            use_curved_arrows: bool
                Whether to use curved arrows like CellChat (default: True)
            curve_strength: float
                Strength of the curve (0 = straight, higher = more curved)
            adjust_text: bool
                Whether to use adjust_text library to prevent label overlapping (default: False)
                If True, uses plt.text instead of nx.draw_networkx_labels
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        # Ensure signaling is in list format
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # Check if signaling pathways exist
        available_pathways = self.adata.var['classification'].unique()
        for pathway in signaling:
            if pathway not in available_pathways:
                raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
        
        # Validate cell type names
        if vertex_receiver is not None:
            invalid_receivers = [ct for ct in vertex_receiver if ct not in self.cell_types]
            if invalid_receivers:
                raise ValueError(f"Invalid receiver cell types: {invalid_receivers}. Available cell types: {self.cell_types}")
        
        if vertex_sender is not None:
            invalid_senders = [ct for ct in vertex_sender if ct not in self.cell_types]
            if invalid_senders:
                raise ValueError(f"Invalid sender cell types: {invalid_senders}. Available cell types: {self.cell_types}")
        
        # Calculate communication matrix for specific pathway
        pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
        
        # Filter specific pathway interactions
        pathway_mask = self.adata.var['classification'].isin(signaling)
        pathway_indices = np.where(pathway_mask)[0]
        
        if len(pathway_indices) == 0:
            # If no interactions found for the pathway, return empty plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No interactions found for pathway(s): {", ".join(signaling)}', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig, ax
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            sender_idx = self.cell_types.index(sender)
            receiver_idx = self.cell_types.index(receiver)
            
            # Get significant interactions for this pathway
            pvals = self.adata.layers['pvalues'][i, pathway_indices]
            means = self.adata.layers['means'][i, pathway_indices]
            
            sig_mask = pvals < pvalue_threshold
            if np.any(sig_mask):
                pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
        
        # Choose visualization method based on layout type
        if layout == 'circle':
            # Check if there are actual interactions
            if pathway_matrix.sum() == 0:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f'No significant interactions found for pathway(s): {", ".join(signaling)}', 
                       ha='center', va='center', fontsize=16)
                ax.axis('off')
                return fig, ax
            
            title = f"Signaling Pathway: {', '.join(signaling)} (Circle)"
            
            # If vertex_sender or vertex_receiver specified, need to filter matrix
            if vertex_sender is not None or vertex_receiver is not None:
                # Create filtered matrix
                filtered_matrix = np.zeros_like(pathway_matrix)
                
                for i, sender_type in enumerate(self.cell_types):
                    for j, receiver_type in enumerate(self.cell_types):
                        # Check if meets sender/receiver conditions
                        sender_ok = (vertex_sender is None) or (sender_type in vertex_sender)
                        receiver_ok = (vertex_receiver is None) or (receiver_type in vertex_receiver)
                        
                        if sender_ok and receiver_ok and pathway_matrix[i, j] > 0:
                            filtered_matrix[i, j] = pathway_matrix[i, j]
                
                pathway_matrix = filtered_matrix
                
                # Check again if there are still interactions
                if pathway_matrix.sum() == 0:
                    fig, ax = plt.subplots(figsize=figsize)
                    sender_str = f"senders: {vertex_sender}" if vertex_sender else "any senders"
                    receiver_str = f"receivers: {vertex_receiver}" if vertex_receiver else "any receivers"
                    ax.text(0.5, 0.5, f'No interactions found for pathway(s): {", ".join(signaling)}\nwith {sender_str} and {receiver_str}', 
                           ha='center', va='center', fontsize=14)
                    ax.axis('off')
                    return fig, ax
            
            # Choose appropriate circular visualization method
            if focused_view:
                fig, ax = self.netVisual_circle_focused(
                    matrix=pathway_matrix,
                    title=title,
                    edge_width_max=edge_width_max,
                    vertex_size_max=vertex_size_max,
                    show_labels=show_labels,
                    cmap=cmap,
                    figsize=figsize,
                    min_interaction_threshold=0,
                    use_sender_colors=use_sender_colors,
                    use_curved_arrows=use_curved_arrows,
                    curve_strength=curve_strength
                )
            else:
                fig, ax = self.netVisual_circle(
                    matrix=pathway_matrix,
                    title=title,
                    edge_width_max=edge_width_max,
                    vertex_size_max=vertex_size_max,
                    show_labels=show_labels,
                    cmap=cmap,
                    figsize=figsize,
                    use_sender_colors=use_sender_colors,
                    use_curved_arrows=use_curved_arrows,
                    curve_strength=curve_strength,  
                    adjust_text=adjust_text
                )
        
        elif layout == 'hierarchy':
            # Determine source and target cells
            if vertex_receiver is not None and vertex_sender is not None:
                # If both sender and receiver specified
                source_cells = vertex_sender
                target_cells = vertex_receiver
            elif vertex_receiver is not None:
                # Only receiver specified, others are senders
                target_cells = vertex_receiver
                source_cells = [ct for ct in self.cell_types if ct not in target_cells]
            elif vertex_sender is not None:
                # Only sender specified, others are receivers
                source_cells = vertex_sender
                target_cells = [ct for ct in self.cell_types if ct not in source_cells]
            else:
                # If neither specified, use all cell types with interactions
                source_cells = None
                target_cells = None
            
            title = f"Signaling Pathway: {', '.join(signaling)} (Hierarchy)"
            fig, ax = self.netVisual_hierarchy(
                pathway_name=signaling[0],  # Use first pathway name
                sources=source_cells,
                targets=target_cells,
                pvalue_threshold=pvalue_threshold,
                figsize=figsize
            )
            
            # Update title
            ax.set_title(title, fontsize=16)
        
        else:
            raise ValueError("layout must be 'circle' or 'hierarchy'")
        
        return fig, ax
    
    def get_signaling_pathways(self, min_interactions=1, pathway_pvalue_threshold=0.05, 
                              method='fisher', correction_method='fdr_bh', min_expression=0.1):
        """
        Get all significant signaling pathway lists using statistically more reliable methods to combine p-values from multiple L-R pairs
        
        Args:
            min_interactions: int
                Minimum L-R pair count threshold per pathway (default: 1)
            pathway_pvalue_threshold: float
                Pathway-level p-value threshold (default: 0.05)
            method: str
                P-value combination method: 'fisher', 'stouffer', 'min', 'mean' (default: 'fisher')
            correction_method: str
                Multiple testing correction method: 'fdr_bh', 'bonferroni', 'holm', None (default: 'fdr_bh')
            min_expression: float
                Minimum expression threshold (default: 0.1)
        
        Returns:
            pathways: list
                Significant signaling pathway list
            pathway_stats: dict
                Detailed statistics for each pathway
        """
        from scipy.stats import combine_pvalues
        from statsmodels.stats.multitest import multipletests
        import warnings
        
        pathways = [p for p in self.adata.var['classification'].unique() if pd.notna(p)]
        pathway_stats = {}
        pathway_pvalues = []
        pathway_names = []
        
        print(f"🔬 Analyzing statistical significance of {len(pathways)} signaling pathways using {method} method...")
        
        for pathway in pathways:
            pathway_mask = self.adata.var['classification'] == pathway
            pathway_lr_pairs = self.adata.var.loc[pathway_mask, 'interacting_pair'].tolist()
            
            if len(pathway_lr_pairs) < min_interactions:
                continue
                
            # Collect p-values and expression levels for this pathway across all cell pairs
            all_pathway_pvals = []
            all_pathway_means = []
            significant_cell_pairs = []
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
                pvals = self.adata.layers['pvalues'][i, pathway_mask]
                means = self.adata.layers['means'][i, pathway_mask]
                
                # Filter low expression interactions
                valid_mask = means >= min_expression
                if np.any(valid_mask):
                    valid_pvals = pvals[valid_mask]
                    valid_means = means[valid_mask]
                    
                    all_pathway_pvals.extend(valid_pvals)
                    all_pathway_means.extend(valid_means)
                    
                    # Check if there are significant interactions
                    if np.any(valid_pvals < 0.05):
                        significant_cell_pairs.append(f"{sender}|{receiver}")
            
            if len(all_pathway_pvals) == 0:
                continue
                
            all_pathway_pvals = np.array(all_pathway_pvals)
            all_pathway_means = np.array(all_pathway_means)
            
            # Combine p-values to get pathway-level significance
            try:
                if method == 'fisher':
                    # Fisher's method - suitable for independent tests
                    combined_stat, combined_pval = combine_pvalues(all_pathway_pvals, method='fisher')
                elif method == 'stouffer':
                    # Stouffer's method - can be weighted
                    weights = all_pathway_means / all_pathway_means.sum()  # Weight based on expression
                    combined_stat, combined_pval = combine_pvalues(all_pathway_pvals, method='stouffer', weights=weights)
                elif method == 'min':
                    # Minimum p-value method (needs Bonferroni correction)
                    combined_pval = np.min(all_pathway_pvals) * len(all_pathway_pvals)
                    combined_pval = min(combined_pval, 1.0)  # Cap at 1.0
                    combined_stat = -np.log10(combined_pval)
                elif method == 'mean':
                    # Average p-value (not recommended, but as reference)
                    combined_pval = np.mean(all_pathway_pvals)
                    combined_stat = -np.log10(combined_pval)
                else:
                    raise ValueError(f"Unknown method: {method}")
                    
            except Exception as e:
                warnings.warn(f"Failed to combine p-values for pathway {pathway}: {e}")
                combined_pval = 1.0
                combined_stat = 0.0
            
            # Calculate pathway statistics
            pathway_stats[pathway] = {
                'n_lr_pairs': len(pathway_lr_pairs),
                'n_tests': len(all_pathway_pvals),
                'n_significant_interactions': np.sum(all_pathway_pvals < 0.05),
                'mean_expression': np.mean(all_pathway_means),
                'max_expression': np.max(all_pathway_means),
                'combined_pvalue': combined_pval,
                'combined_statistic': combined_stat,
                'significant_cell_pairs': significant_cell_pairs,
                'lr_pairs': pathway_lr_pairs,
                'significance_rate': np.sum(all_pathway_pvals < 0.05) / len(all_pathway_pvals)
            }
            
            pathway_pvalues.append(combined_pval)
            pathway_names.append(pathway)
        
        # Multiple testing correction
        if len(pathway_pvalues) > 0 and correction_method:
            print(f"📊 Applying {correction_method} multiple testing correction...")
            try:
                corrected_results = multipletests(pathway_pvalues, alpha=pathway_pvalue_threshold, 
                                                method=correction_method)
                corrected_pvals = corrected_results[1]
                is_significant = corrected_results[0]
                
                # Update statistics
                for i, pathway in enumerate(pathway_names):
                    pathway_stats[pathway]['corrected_pvalue'] = corrected_pvals[i]
                    pathway_stats[pathway]['is_significant_corrected'] = is_significant[i]
                
                significant_pathways = [pathway_names[i] for i in range(len(pathway_names)) if is_significant[i]]
                
            except Exception as e:
                warnings.warn(f"Multiple testing correction failed: {e}")
                # Fall back to uncorrected p-values
                significant_pathways = [pathway_names[i] for i in range(len(pathway_names)) 
                                      if pathway_pvalues[i] < pathway_pvalue_threshold]
        else:
            # No multiple testing correction
            significant_pathways = [pathway_names[i] for i in range(len(pathway_names)) 
                                  if pathway_pvalues[i] < pathway_pvalue_threshold]
        
        # Sort by significance
        if len(significant_pathways) > 0:
            if correction_method:
                significant_pathways.sort(key=lambda x: pathway_stats[x]['corrected_pvalue'])
            else:
                significant_pathways.sort(key=lambda x: pathway_stats[x]['combined_pvalue'])
        
        print(f"✅ Found {len(significant_pathways)} significant pathways (out of {len(pathway_names)} pathways)")
        print(f"   - P-value combination method: {method}")
        print(f"   - Multiple testing correction: {correction_method if correction_method else 'None'}")
        print(f"   - Pathway threshold: {pathway_pvalue_threshold}")
        
        return significant_pathways, pathway_stats
    
    def plot_all_visualizations(self, pvalue_threshold=0.05, save_prefix=None):
        """
        Generate all major visualization plots
        
        Args:
            pvalue_threshold: float
                P-value threshold
            save_prefix: str
                If provided, save figures with this prefix
        """
        figures = {}
        
        # 1. Aggregated network
        count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
        
        # Circle plot
        fig1, _ = self.netVisual_circle(count_matrix, title="Number of Interactions")
        figures['circle_count'] = fig1
        
        fig2, _ = self.netVisual_circle(weight_matrix, title="Interaction Strength")
        figures['circle_weight'] = fig2
        
        # Individual cell type networks (similar to CellChat)
        fig2_1 = self.netVisual_individual_circle(pvalue_threshold=pvalue_threshold)
        figures['individual_outgoing'] = fig2_1
        
        fig2_2 = self.netVisual_individual_circle_incoming(pvalue_threshold=pvalue_threshold)
        figures['individual_incoming'] = fig2_2
        
        # 2. Heatmap
        fig3, _ = self.netVisual_heatmap(count_matrix, title="Communication Patterns")
        figures['heatmap'] = fig3
        
        # 2b. Marsilea Heatmap (if available)
        if MARSILEA_AVAILABLE:
            try:
                h_marsilea = self.netVisual_heatmap_marsilea(
                    signaling=None, 
                    pvalue_threshold=pvalue_threshold,
                    title="Communication Patterns (Marsilea)"
                )
                figures['heatmap_marsilea'] = h_marsilea
                
                h_marsilea_focused = self.netVisual_heatmap_marsilea_focused(
                    signaling=None,
                    pvalue_threshold=pvalue_threshold, 
                    title="Communication Patterns (Marsilea Focused)"
                )
                figures['heatmap_marsilea_focused'] = h_marsilea_focused
            except Exception as e:
                print(f"Warning: Failed to create Marsilea heatmaps: {e}")
        
        # 3. Chord diagram
        fig4, _ = self.netVisual_chord(weight_matrix, title="Cell-Cell Communication Network")
        figures['chord'] = fig4
        
        # 4. Hierarchy plot (example for first pathway)
        pathways = self.adata.var['classification'].unique()
        if len(pathways) > 0:
            fig5, _ = self.netVisual_hierarchy(pathway_name=pathways[0])
            figures['hierarchy'] = fig5
        
        # 5. Bubble plot
        fig6, _ = self.netVisual_bubble()
        figures['bubble'] = fig6
        
        # 6. Signaling roles
        df_sender, fig7 = self.identify_signaling_role(pattern="outgoing")
        figures['role_sender'] = fig7
        
        df_receiver, fig8 = self.identify_signaling_role(pattern="incoming")
        figures['role_receiver'] = fig8
        
        # 7. Network embedding
        embedding, clusters, fig9 = self.netEmbedding(method='functional')
        figures['embedding'] = fig9
        
        # Save figures if requested
        if save_prefix:
            for name, fig in figures.items():
                fig.savefig(f'{save_prefix}_{name}.png', dpi=300, bbox_inches='tight')
                print(f'Saved: {save_prefix}_{name}.png')
        
        return figures
    
    def netVisual_chord_cell(self, signaling=None, group_celltype=None, 
                            sources=None, targets=None,
                            pvalue_threshold=0.05, count_min=1, 
                            gap=0.03, use_gradient=True, sort="size", 
                            directed=True, cmap=None, chord_colors=None,
                            rotate_names=False, fontcolor="black", fontsize=12,
                            start_at=0, extent=360, min_chord_width=0,
                            colors=None, ax=None, figsize=(8, 8), 
                            title_name=None, save=None, normalize_to_sender=True):
        """
        Create chord diagram visualization using mpl-chord-diagram (mimicking CellChat's netVisual_chord_cell function)
        
        Args:
            signaling: str, list or None
                Specific signaling pathway names. If None, show aggregated results of all pathways
            group_celltype: dict or None
                Cell type grouping mapping, e.g., {'CellA': 'GroupX', 'CellB': 'GroupX', 'CellC': 'GroupY'}
                If None, each cell type is shown individually
            sources: list or None
                Specified sender cell type list. If None, include all cell types
            targets: list or None
                Specified receiver cell type list. If None, include all cell types
            pvalue_threshold: float
                P-value threshold for significant interactions
            count_min: int
                Minimum interaction count threshold
            gap: float
                Gap between chord diagram segments (0.03)
            use_gradient: bool
                Whether to use gradient effects (True)
            sort: str or None
                Sorting method: "size", "distance", None ("size")
            directed: bool
                Whether to show directionality (True)
            cmap: str or None
                Colormap name (None, use cell type colors)
            chord_colors: str or None
                Chord colors (None)
            rotate_names: bool
                Whether to rotate names (False)
            fontcolor: str
                Font color ("black")
            fontsize: int
                Font size (12)
            start_at: int
                Starting angle (0)
            extent: int
                Angle range covered by chord diagram (360)
            min_chord_width: int
                Minimum chord width (0)
            colors: list or None
                Custom color list (None, use cell type colors)
            ax: matplotlib.axes.Axes or None
                Matplotlib axes object (None, create new plot)
            figsize: tuple
                Figure size (8, 8)
            title_name: str or None
                Plot title (None)
            save: str or None
                Save file path (None)
            normalize_to_sender: bool
                Whether to normalize to sender for equal arc widths (True)
            
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        try:
            from ..external.mpl_chord.chord_diagram import chord_diagram
        except ImportError:
            try:
                from mpl_chord_diagram import chord_diagram
            except ImportError:
                raise ImportError("mpl-chord-diagram package is required. Please install it: pip install mpl-chord-diagram")
        
        # Calculate interaction matrix for specific pathways
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # Check if signaling pathways exist
            available_pathways = self.adata.var['classification'].unique()
            for pathway in signaling:
                if pathway not in available_pathways:
                    raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
            
            # Calculate communication matrix for specific pathways
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_mask = self.adata.var['classification'].isin(signaling)
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                raise ValueError(f"No interactions found for pathway(s): {signaling}")
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                      self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # Get significant interactions for this pathway
                pvals = self.adata.layers['pvalues'][i, pathway_indices]
                means = self.adata.layers['means'][i, pathway_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    # Use interaction count, more suitable for chord diagrams
                    pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
            
            matrix = pathway_matrix
            self.test_pathway_matrix = pathway_matrix
        else:
            # Use aggregated interaction count matrix
            count_matrix, _ = self.compute_aggregated_network(pvalue_threshold)
            matrix = count_matrix
        
        # Filter specified senders and receivers
        if sources is not None or targets is not None:
            # Validate specified cell types
            if sources is not None:
                invalid_sources = [ct for ct in sources if ct not in self.cell_types]
                if invalid_sources:
                    raise ValueError(f"Invalid source cell types: {invalid_sources}. Available: {self.cell_types}")
            
            if targets is not None:
                invalid_targets = [ct for ct in targets if ct not in self.cell_types]
                if invalid_targets:
                    raise ValueError(f"Invalid target cell types: {invalid_targets}. Available: {self.cell_types}")
            
            # Create filtered matrix
            filtered_matrix = np.zeros_like(matrix)
            for i, sender_type in enumerate(self.cell_types):
                for j, receiver_type in enumerate(self.cell_types):
                    # Check if meets sources/targets conditions
                    sender_ok = (sources is None) or (sender_type in sources)
                    receiver_ok = (targets is None) or (receiver_type in targets)
                    
                    if sender_ok and receiver_ok:
                        filtered_matrix[i, j] = matrix[i, j]
            
            matrix = filtered_matrix
        
        # Apply group_celltype grouping (if provided)
        if group_celltype is not None:
            # Validate grouping mapping
            for cell_type in self.cell_types:
                if cell_type not in group_celltype:
                    raise ValueError(f"Cell type '{cell_type}' not found in group_celltype mapping")
            
            # Get unique group names
            unique_groups = list(set(group_celltype.values()))
            group_matrix = np.zeros((len(unique_groups), len(unique_groups)))
            
            # Aggregate to group level
            for i, sender_type in enumerate(self.cell_types):
                for j, receiver_type in enumerate(self.cell_types):
                    if matrix[i, j] > 0:
                        sender_group = group_celltype[sender_type]
                        receiver_group = group_celltype[receiver_type]
                        sender_group_idx = unique_groups.index(sender_group)
                        receiver_group_idx = unique_groups.index(receiver_group)
                        group_matrix[sender_group_idx, receiver_group_idx] += matrix[i, j]
            
            # Use grouped matrix and names
            final_matrix = group_matrix
            final_names = unique_groups
        else:
            # Use original cell types
            final_matrix = matrix
            final_names = self.cell_types
        
        # Filter interactions below threshold
        final_matrix[final_matrix < count_min] = 0
        
        # Check if there are still interactions
        if final_matrix.sum() == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            
            message = f'No interactions above threshold (count_min={count_min})'
            if signaling:
                message += f' for pathway(s): {", ".join(signaling)}'
            if sources:
                message += f', sources: {sources}'
            if targets:
                message += f', targets: {targets}'
            ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
            ax.axis('off')
            if title_name:
                ax.set_title(title_name, fontsize=16, pad=20)
            return fig, ax
        
        # CellChat-style normalization: ensure equal arc width for each cell type
        if normalize_to_sender:
            # Use iterative method to ensure both row and column sums are equal
            # This is a classic method for solving bidirectional matrix normalization
            
            normalized_matrix = final_matrix.copy().astype(float)
            
            # Find rows and columns with interactions
            row_sums = normalized_matrix.sum(axis=1)
            col_sums = normalized_matrix.sum(axis=0)
            nonzero_rows = row_sums > 0
            nonzero_cols = col_sums > 0
            
            if np.any(nonzero_rows) and np.any(nonzero_cols):
                standard_sum = 100.0
                max_iterations = 15
                tolerance = 1e-4  # Relaxed tolerance, sufficient precision for practical applications
                
                for iteration in range(max_iterations):
                    # Normalize rows
                    row_sums = normalized_matrix.sum(axis=1)
                    for i in range(len(final_names)):
                        if row_sums[i] > tolerance:  # Avoid division by zero
                            scale_factor = standard_sum / row_sums[i]
                            normalized_matrix[i, :] *= scale_factor
                    
                    # Normalize columns
                    col_sums = normalized_matrix.sum(axis=0)
                    for j in range(len(final_names)):
                        if col_sums[j] > tolerance:  # Avoid division by zero
                            scale_factor = standard_sum / col_sums[j]
                            normalized_matrix[:, j] *= scale_factor
                    
                    # Check convergence
                    final_row_sums = normalized_matrix.sum(axis=1)
                    final_col_sums = normalized_matrix.sum(axis=0)
                    
                    # Calculate standard deviation of non-zero rows and columns to judge convergence
                    nonzero_final_rows = final_row_sums[final_row_sums > tolerance]
                    nonzero_final_cols = final_col_sums[final_col_sums > tolerance]
                    
                    if (len(nonzero_final_rows) > 0 and 
                        len(nonzero_final_cols) > 0):
                        row_std = np.std(nonzero_final_rows)
                        col_std = np.std(nonzero_final_cols)
                        
                        if row_std < tolerance and col_std < tolerance:
                            break
                
                final_matrix = normalized_matrix
        
        # Prepare colors
        if colors is None:
            if group_celltype is not None:
                # For grouping, assign colors to each group
                cell_colors = self._get_cell_type_colors()
                group_colors = {}
                for group in unique_groups:
                    # Use color of first cell type in this group
                    for cell_type, group_name in group_celltype.items():
                        if group_name == group:
                            group_colors[group] = cell_colors.get(cell_type, '#1f77b4')
                            break
                colors = [group_colors.get(node, '#1f77b4') for node in final_names]
            else:
                # Use cell type colors
                cell_colors = self._get_cell_type_colors()
                colors = [cell_colors.get(node, '#1f77b4') for node in final_names]
        
        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        self.test_final_matrix = final_matrix
        
        # Modify names: only hide names of cell types that neither send nor receive signals
        display_names = final_names.copy()
        if normalize_to_sender:
            # Calculate row sums (sent signals) and column sums (received signals)
            row_sums = final_matrix.sum(axis=1)
            col_sums = final_matrix.sum(axis=0)
            for i in range(len(final_names)):
                # Only hide names of cell types that neither send nor receive signals
                if row_sums[i] == 0 and col_sums[i] == 0:
                    display_names[i] = ""  # Hide name
        
        # Draw chord diagram
        chord_diagram(
            final_matrix, 
            display_names,
            ax=ax,
            gap=gap,
            use_gradient=use_gradient,
            sort=sort,
            directed=directed,
            cmap=cmap,
            chord_colors=chord_colors,
            rotate_names=rotate_names,
            fontcolor=fontcolor,
            fontsize=fontsize,
            start_at=start_at,
            extent=extent,
            min_chord_width=min_chord_width,
            colors=colors
        )
        
        # Add title
        if title_name:
            ax.set_title(title_name, fontsize=fontsize + 4, pad=20)
        
        # Save file
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"Chord diagram saved as: {save}")
        
        return fig, ax
    
    def netVisual_chord_LR(self, ligand_receptor_pairs=None, sources=None, targets=None,
                          pvalue_threshold=0.05, count_min=1, 
                          gap=0.03, use_gradient=True, sort="size", 
                          directed=True, cmap=None, chord_colors=None,
                          rotate_names=False, fontcolor="black", fontsize=12,
                          start_at=0, extent=360, min_chord_width=0,
                          colors=None, ax=None, figsize=(8, 8), 
                          title_name=None, save=None, normalize_to_sender=True):
        """
        Create chord diagram visualization for specific ligand-receptor pairs (mimicking CellChat's ligand-receptor level analysis)
        
        Args:
            ligand_receptor_pairs: str, list or None
                Specific ligand-receptor pair names. Supports following formats:
                - Single string: "LIGAND_RECEPTOR" (e.g.: "TGFB1_TGFBR1")  
                - String list: ["LIGAND1_RECEPTOR1", "LIGAND2_RECEPTOR2"]
                - If None, show aggregated results of all ligand-receptor pairs
            sources: list or None
                Specified sender cell type list. If None, include all cell types
            targets: list or None
                Specified receiver cell type list. If None, include all cell types
            pvalue_threshold: float
                P-value threshold for significant interactions
            count_min: int
                Minimum interaction count threshold
            gap: float
                Gap between chord diagram segments
            use_gradient: bool
                Whether to use gradient effects
            sort: str or None
                Sorting method: "size", "distance", None
            directed: bool
                Whether to show directionality
            cmap: str or None
                Colormap name
            chord_colors: str or None
                Chord colors
            rotate_names: bool
                Whether to rotate names
            fontcolor: str
                Font color
            fontsize: int
                Font size
            start_at: int
                Starting angle
            extent: int
                Angle range covered by chord diagram
            min_chord_width: int
                Minimum chord width
            colors: list or None
                Custom color list
            ax: matplotlib.axes.Axes or None
                Matplotlib axes object
            figsize: tuple
                Figure size
            title_name: str or None
                Plot title
            save: str or None
                Save file path
            normalize_to_sender: bool
                Whether to hide names of cell types without received signals (True)
            
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        try:
            from ..external.mpl_chord.chord_diagram import chord_diagram
        except ImportError:
            try:
                from mpl_chord_diagram import chord_diagram
            except ImportError:
                raise ImportError("mpl-chord-diagram package is required. Please install it: pip install mpl-chord-diagram")
        
        # Handle ligand-receptor pair filtering
        if ligand_receptor_pairs is not None:
            if isinstance(ligand_receptor_pairs, str):
                ligand_receptor_pairs = [ligand_receptor_pairs]
            
            # Check if ligand-receptor pairs exist
            # Assume adata.var contains ligand-receptor pair information, possibly in 'gene_name' or other columns
            if 'gene_name' in self.adata.var.columns:
                available_pairs = self.adata.var['gene_name'].unique()
            elif 'interacting_pair' in self.adata.var.columns:
                available_pairs = self.adata.var['interacting_pair'].unique()
            else:
                # If no explicit ligand-receptor pair column, use index
                available_pairs = self.adata.var.index.tolist()
            
            # Validate requested ligand-receptor pairs
            missing_pairs = []
            valid_pairs = []
            for pair in ligand_receptor_pairs:
                if pair in available_pairs:
                    valid_pairs.append(pair)
                else:
                    missing_pairs.append(pair)
            
            if missing_pairs:
                print(f"Warning: The following L-R pairs were not found: {missing_pairs}")
                print(f"Available pairs: {list(available_pairs)[:10]}...")  # Show first 10
            
            if not valid_pairs:
                raise ValueError(f"None of the specified L-R pairs were found in the data")
            
            # Calculate communication matrix for specific ligand-receptor pairs
            lr_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            
            # Filter specific ligand-receptor pair interactions
            if 'gene_name' in self.adata.var.columns:
                lr_mask = self.adata.var['gene_name'].isin(valid_pairs)
            elif 'interacting_pair' in self.adata.var.columns:
                lr_mask = self.adata.var['interacting_pair'].isin(valid_pairs)
            else:
                lr_mask = self.adata.var.index.isin(valid_pairs)
                
            lr_indices = np.where(lr_mask)[0]
            
            if len(lr_indices) == 0:
                raise ValueError(f"No interactions found for L-R pair(s): {valid_pairs}")
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                      self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # Get significant interactions for this ligand-receptor pair
                pvals = self.adata.layers['pvalues'][i, lr_indices]
                means = self.adata.layers['means'][i, lr_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    # Use average of interaction strengths as weight
                    lr_matrix[sender_idx, receiver_idx] += np.mean(means[sig_mask])
            
            matrix = lr_matrix
            title_suffix = f" - L-R: {', '.join(valid_pairs[:3])}{'...' if len(valid_pairs) > 3 else ''}"
        else:
            # Use aggregated interaction count matrix
            count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
            matrix = weight_matrix  # Use weight matrix to better reflect interaction strength
            title_suffix = " - All L-R pairs"
        
        # Filter specified senders and receivers
        if sources is not None or targets is not None:
            # Validate specified cell types
            if sources is not None:
                invalid_sources = [ct for ct in sources if ct not in self.cell_types]
                if invalid_sources:
                    raise ValueError(f"Invalid source cell types: {invalid_sources}. Available: {self.cell_types}")
            
            if targets is not None:
                invalid_targets = [ct for ct in targets if ct not in self.cell_types]
                if invalid_targets:
                    raise ValueError(f"Invalid target cell types: {invalid_targets}. Available: {self.cell_types}")
            
            # Create filtered matrix
            filtered_matrix = np.zeros_like(matrix)
            for i, sender_type in enumerate(self.cell_types):
                for j, receiver_type in enumerate(self.cell_types):
                    # Check if meets sources/targets conditions
                    sender_ok = (sources is None) or (sender_type in sources)
                    receiver_ok = (targets is None) or (receiver_type in targets)
                    
                    if sender_ok and receiver_ok:
                        filtered_matrix[i, j] = matrix[i, j]
            
            matrix = filtered_matrix
        
        # Use cell type names
        final_matrix = matrix
        final_names = self.cell_types
        
        # Filter interactions below threshold
        final_matrix[final_matrix < count_min] = 0
        
        # Check if there are still interactions
        if final_matrix.sum() == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            
            message = f'No interactions above threshold (count_min={count_min})'
            if ligand_receptor_pairs:
                message += f' for L-R pair(s): {", ".join(valid_pairs)}'
            if sources:
                message += f', sources: {sources}'
            if targets:
                message += f', targets: {targets}'
            ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
            ax.axis('off')
            if title_name:
                ax.set_title(title_name, fontsize=16, pad=20)
            return fig, ax
        
        # Prepare colors
        if colors is None:
            # Use cell type colors
            cell_colors = self._get_cell_type_colors()
            colors = [cell_colors.get(node, '#1f77b4') for node in final_names]
        
        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        self.test_final_matrix = final_matrix
        
        # Modify names: only hide names of cell types that neither send nor receive signals
        display_names = final_names.copy()
        if normalize_to_sender:
            # Calculate row sums (sent signals) and column sums (received signals)
            row_sums = final_matrix.sum(axis=1)
            col_sums = final_matrix.sum(axis=0)
            for i in range(len(final_names)):
                # Only hide names of cell types that neither send nor receive signals
                if row_sums[i] == 0 and col_sums[i] == 0:
                    display_names[i] = ""  # Hide name
        
        # Draw chord diagram
        chord_diagram(
            final_matrix, 
            display_names,
            ax=ax,
            gap=gap,
            use_gradient=use_gradient,
            sort=sort,
            directed=directed,
            cmap=cmap,
            chord_colors=chord_colors,
            rotate_names=rotate_names,
            fontcolor=fontcolor,
            fontsize=fontsize,
            start_at=start_at,
            extent=extent,
            min_chord_width=min_chord_width,
            colors=colors
        )
        
        # Add title
        if title_name is None:
            title_name = f"Ligand-Receptor Communication{title_suffix}"
        ax.set_title(title_name, fontsize=fontsize + 4, pad=20)
        
        # Save file
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"L-R Chord diagram saved as: {save}")
        
        return fig, ax
    



