import pandas as pd
import numpy as np
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


class CellChatVizPlus:

        ##########################################################

    def get_ligand_receptor_pairs(self, min_interactions=1, pvalue_threshold=0.05):
        """
        Get a list of all significant ligand-receptor pairs
        
        Args:
            min_interactions: int
                Minimum interaction count threshold
            pvalue_threshold: float
                P-value threshold for significance
        
        Returns:
            lr_pairs: list
                List of significant ligand-receptor pairs
            lr_stats: dict
                Statistics for each ligand-receptor pair
        """
        # Determine the column name for ligand-receptor pairs
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
            
            # Calculate the total number of interactions for this ligand-receptor pair
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

    def netVisual_chord_gene(self, sources_use=None, targets_use=None, 
                            signaling=None, pvalue_threshold=0.05, mean_threshold=0.1,
                            gap=0.03, use_gradient=True, sort="size", 
                            directed=True, chord_colors=None,
                            rotate_names=False, fontcolor="black", fontsize=10,
                            start_at=0, extent=360, min_chord_width=0,
                            ax=None, figsize=(12, 12), 
                            title_name=None, save=None, legend_pos_x=None,
                            show_celltype_in_name=True, show_legend=True, 
                            legend_bbox=(1.05, 1), legend_ncol=1,return_df=False):
        """
        Draw a chord diagram of all ligand-receptor pairs for specific cell types as senders (gene-level)
        Each sector represents a ligand or receptor, ligands use sender color, receptors use receiver color
        
        Args:
            sources_use: str, int, list or None
                Sender cell types. Can be:
                - String: cell type name
                - Integer: cell type index (starting from 0)
                - List: multiple cell types
                - None: all cell types as senders
            targets_use: str, int, list or None
                Receiver cell types. Can be:
                - String: cell type name
                - Integer: cell type index (starting from 0)
                - List: multiple cell types
                - None: all cell types as receivers
            signaling: str, list or None
                Specific signaling pathway name. Can be:
                - String: single pathway name
                - List: multiple pathway names
                - None: all pathways
            pvalue_threshold: float
                P-value threshold for significant interactions
            mean_threshold: float
                Mean expression intensity threshold
            gap: float
                Gap between segments in the chord diagram
            use_gradient: bool
                Whether to use gradient effect
            sort: str or None
                Sorting method: "size", "distance", None
            directed: bool
                Whether to show directionality
            chord_colors: str or None
                Chord color
            rotate_names: bool
                Whether to rotate names
            fontcolor: str
                Font color
            fontsize: int
                Font size
            start_at: int
                Starting angle
            extent: int
                Angle range covered by the chord diagram
            min_chord_width: int
                Minimum chord width
            ax: matplotlib.axes.Axes or None
                Matplotlib axis object
            figsize: tuple
                Figure size
            title_name: str or None
                Figure title
            save: str or None
                Save file path
            legend_pos_x: float or None
                Legend X position (not implemented)
            show_celltype_in_name: bool
                Whether to show cell type info in node names (default: True)
                If True, display as "Gene(CellType)"
                If False, only show gene name, but the same gene will still appear multiple times in different cell types
            show_legend: bool
                Whether to show cell type color legend (default: True)
            legend_bbox: tuple
                Legend position, format (x, y) (default: (1.05, 1))
            legend_ncol: int
                Number of legend columns (default: 1)
            
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
        
        # Check if required columns exist
        required_cols = ['gene_a', 'gene_b']
        missing_cols = [col for col in required_cols if col not in self.adata.var.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing from adata.var: {missing_cols}")
        
        # Handle signaling pathway filtering
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # Check if signaling pathways exist
            if 'classification' in self.adata.var.columns:
                available_pathways = self.adata.var['classification'].unique()
                missing_pathways = [p for p in signaling if p not in available_pathways]
                if missing_pathways:
                    print(f"Warning: The following signaling pathways were not found: {missing_pathways}")
                    print(f"Available pathways: {list(available_pathways)}")
                
                # Filter interactions containing specified pathways
                signaling_mask = self.adata.var['classification'].isin(signaling)
                signaling_indices = np.where(signaling_mask)[0]
                
                if len(signaling_indices) == 0:
                    if ax is None:
                        fig, ax = plt.subplots(figsize=figsize)
                    else:
                        fig = ax.figure
                    
                    ax.text(0.5, 0.5, f'No interactions found for signaling pathway(s): {", ".join(signaling)}', 
                           ha='center', va='center', fontsize=16)
                    ax.axis('off')
                    if title_name:
                        ax.set_title(title_name, fontsize=16, pad=20)
                    return fig, ax
            else:
                print("Warning: 'classification' column not found in adata.var. Ignoring signaling parameter.")
                signaling_indices = None
        else:
            signaling_indices = None
        
        # Handle sender cell types
        if sources_use is None:
            source_cell_types = self.cell_types
        else:
            if isinstance(sources_use, (int, str)):
                sources_use = [sources_use]
            
            source_cell_types = []
            for src in sources_use:
                if isinstance(src, int):
                    if 0 <= src < len(self.cell_types):
                        source_cell_types.append(self.cell_types[src])
                    else:
                        raise ValueError(f"Source index {src} out of range [0, {len(self.cell_types)-1}]")
                elif isinstance(src, str):
                    if src in self.cell_types:
                        source_cell_types.append(src)
                    else:
                        raise ValueError(f"Source cell type '{src}' not found. Available: {self.cell_types}")
                else:
                    raise ValueError(f"Invalid source type: {type(src)}")
        
        # Handle receiver cell types
        if targets_use is None:
            target_cell_types = self.cell_types
        else:
            if isinstance(targets_use, (int, str)):
                targets_use = [targets_use]
            
            target_cell_types = []
            for tgt in targets_use:
                if isinstance(tgt, int):
                    if 0 <= tgt < len(self.cell_types):
                        target_cell_types.append(self.cell_types[tgt])
                    else:
                        raise ValueError(f"Target index {tgt} out of range [0, {len(self.cell_types)-1}]")
                elif isinstance(tgt, str):
                    if tgt in self.cell_types:
                        target_cell_types.append(tgt)
                    else:
                        raise ValueError(f"Target cell type '{tgt}' not found. Available: {self.cell_types}")
                else:
                    raise ValueError(f"Invalid target type: {type(tgt)}")
        
        # Collect significant ligand-receptor interactions
        ligand_receptor_interactions = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # Check if sender and receiver meet the conditions
            if sender not in source_cell_types or receiver not in target_cell_types:
                continue
            
            # Get significant interactions
            pvals = self.adata.layers['pvalues'][i, :]
            means = self.adata.layers['means'][i, :]
            
            # Apply signaling pathway filtering
            if signaling_indices is not None:
                pvals = pvals[signaling_indices]
                means = means[signaling_indices]
                interaction_indices = signaling_indices
            else:
                interaction_indices = np.arange(len(pvals))
            
            sig_mask = (pvals < pvalue_threshold) & (means > mean_threshold)
            
            if np.any(sig_mask):
                # Get ligand and receptor for significant interactions
                # Use original indices to get gene info
                original_indices = interaction_indices[sig_mask]
                gene_a_values = self.adata.var['gene_a'].iloc[original_indices].values
                gene_b_values = self.adata.var['gene_b'].iloc[original_indices].values
                mean_values = means[sig_mask]
                
                for gene_a, gene_b, mean_val in zip(gene_a_values, gene_b_values, mean_values):
                    # Skip NaN values
                    if pd.isna(gene_a) or pd.isna(gene_b):
                        continue
                    
                    ligand_receptor_interactions.append({
                        'sender': sender,
                        'receiver': receiver, 
                        'ligand': gene_a,
                        'receptor': gene_b,
                        'mean_expression': mean_val
                    })
        
        if not ligand_receptor_interactions:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            
            ax.text(0.5, 0.5, 'No significant ligand-receptor interactions found\nfor the specified conditions', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            if title_name:
                ax.set_title(title_name, fontsize=16, pad=20)
            return fig, ax
        
        # Create ligand-receptor interaction DataFrame
        lr_df = pd.DataFrame(ligand_receptor_interactions)
        
        # New method: create unique node for each gene-cell type combination, allowing gene repetition
        gene_celltype_combinations = set()
        
        # Collect all ligand-cell type combinations
        for _, row in lr_df.iterrows():
            ligand = row['ligand']
            sender = row['sender']
            receptor = row['receptor']
            receiver = row['receiver']
            
            gene_celltype_combinations.add((ligand, sender, 'ligand'))
            gene_celltype_combinations.add((receptor, receiver, 'receptor'))
        
        # Group nodes by cell type, keep cell types clustered
        celltype_to_nodes = {}
        for gene, celltype, role in gene_celltype_combinations:
            if celltype not in celltype_to_nodes:
                celltype_to_nodes[celltype] = {'ligands': [], 'receptors': []}
            celltype_to_nodes[celltype][role + 's'].append(gene)
        
        # Organize node list: each node uses a unique identifier but only displays gene name
        organized_nodes = []
        organized_node_info = []  # Store node info (gene, celltype, role)
        organized_display_names = []  # Store display names
        
        # Arrange by original cell type order
        available_celltypes = [ct for ct in self.cell_types if ct in celltype_to_nodes]
        
        node_counter = 0  # For creating unique identifiers
        for celltype in available_celltypes:
            nodes = celltype_to_nodes[celltype]
            
            # Add ligands first, then receptors, ensure deduplication and sorting within the same cell type
            for ligand in sorted(set(nodes['ligands'])):
                # Use unique identifier as internal node name
                node_id = f"node_{node_counter}"
                organized_nodes.append(node_id)
                organized_node_info.append((ligand, celltype, 'ligand'))
                organized_display_names.append(ligand)  # Display name is just gene name
                node_counter += 1
            
            for receptor in sorted(set(nodes['receptors'])):
                # Use unique identifier as internal node name
                node_id = f"node_{node_counter}"
                organized_nodes.append(node_id)
                organized_node_info.append((receptor, celltype, 'receptor'))
                organized_display_names.append(receptor)  # Display name is just gene name
                node_counter += 1
        
        # Use the organized node list
        unique_genes = organized_nodes
        
        # Create mapping
        gene_to_celltype = {}
        for node_id, (gene, celltype, role) in zip(organized_nodes, organized_node_info):
            gene_to_celltype[node_id] = celltype
        
        # Create interaction matrix (ligand to receptor)
        n_genes = len(unique_genes)
        interaction_matrix = np.zeros((n_genes, n_genes))
        
        # Fill matrix - need to find corresponding node IDs
        for _, row in lr_df.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            sender = row['sender']
            receiver = row['receiver']
            
            # Find corresponding ligand node ID
            ligand_idx = None
            for i, (gene, celltype, role) in enumerate(organized_node_info):
                if gene == ligand and celltype == sender and role == 'ligand':
                    ligand_idx = i
                    break
            
            # Find corresponding receptor node ID
            receptor_idx = None
            for i, (gene, celltype, role) in enumerate(organized_node_info):
                if gene == receptor and celltype == receiver and role == 'receptor':
                    receptor_idx = i
                    break
            
            # If found, add interaction
            if ligand_idx is not None and receptor_idx is not None:
                interaction_matrix[ligand_idx, receptor_idx] += row['mean_expression']
        
        # Prepare colors: color by cell type
        cell_colors = self._get_cell_type_colors()
        gene_colors = []
        
        for node_id in unique_genes:
            associated_celltype = gene_to_celltype[node_id]
            gene_colors.append(cell_colors.get(associated_celltype, '#808080'))
        
        # Create display names
        display_names = []
        ligands = lr_df['ligand'].unique()
        receptors = lr_df['receptor'].unique()
        
        for i, node_id in enumerate(unique_genes):
            gene, celltype, role = organized_node_info[i]
            
            # Choose display format based on parameter
            if show_celltype_in_name:  # Show full name (gene+cell type)
                display_names.append(f"{gene}({celltype})")
            else:  # Only show gene name, remove parentheses
                # Use gene name directly, color explained in legend
                display_names.append(gene)
        
        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Draw chord diagram
        chord_diagram(
            interaction_matrix,
            display_names,
            ax=ax,
            gap=gap,
            use_gradient=use_gradient,
            sort=sort,
            directed=directed,
            chord_colors=chord_colors,
            rotate_names=rotate_names,
            fontcolor=fontcolor,
            fontsize=fontsize,
            start_at=start_at,
            extent=extent,
            min_chord_width=min_chord_width,
            colors=gene_colors
        )
        
        # Add title
        if title_name is None:
            source_str = ', '.join(source_cell_types) if len(source_cell_types) <= 3 else f"{len(source_cell_types)} cell types"
            target_str = ', '.join(target_cell_types) if len(target_cell_types) <= 3 else f"{len(target_cell_types)} cell types"
            title_name = f"Ligand-Receptor Interactions\nFrom: {source_str} → To: {target_str}"
            
            # Add signaling pathway info to title
            if signaling is not None:
                signaling_str = ', '.join(signaling) if len(signaling) <= 3 else f"{len(signaling)} pathways"
                title_name += f"\nSignaling: {signaling_str}"
        
        ax.set_title(title_name, fontsize=fontsize + 2, pad=20)
        
        # Add cell type color legend
        if show_legend:
            # Get involved cell types and corresponding colors
            involved_celltypes = set()
            for gene, celltype, role in organized_node_info:
                involved_celltypes.add(celltype)
            
            # Sort cell types by original order
            ordered_celltypes = [ct for ct in self.cell_types if ct in involved_celltypes]
            
            # Create legend
            legend_handles = []
            legend_labels = []
            
            for celltype in ordered_celltypes:
                color = cell_colors.get(celltype, '#808080')
                handle = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                legend_handles.append(handle)
                legend_labels.append(celltype)
            
            # Add legend
            legend = ax.legend(legend_handles, legend_labels, 
                             title='Cell Types', 
                             bbox_to_anchor=legend_bbox,
                             loc='upper left',
                             ncol=legend_ncol,
                             fontsize=fontsize-1,
                             title_fontsize=fontsize,
                             frameon=True,
                             fancybox=True,
                             shadow=True)
            
            # Adjust legend style
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.5)
        
        # Save file
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"Gene-level chord diagram saved as: {save}")
        
        if return_df:
            return fig, ax, lr_df
        else:
            return fig, ax
    
    def netVisual_bubble_marsilea(self, sources_use=None, targets_use=None, 
                                 signaling=None, pvalue_threshold=0.05, 
                                 mean_threshold=0.1, top_interactions=20,
                                 show_pvalue=True, show_mean=True, show_count=False,
                                 add_violin=False, add_dendrogram=False,
                                 group_pathways=True, figsize=(12, 8),
                                 title="Cell-Cell Communication Analysis", 
                                 remove_isolate=False, font_size=12, cmap="RdBu_r",
                                 transpose=False, scale=None,
                                 vmin=None, vmax=None, dot_size_min=None, dot_size_max=None,
                                 show_sender_colors=True, show_receiver_colors=False):
        """
        Create advanced bubble plot using Marsilea's SizedHeatmap to visualize cell-cell communication
        Similar to CellChat's netVisual_bubble function, but uses SizedHeatmap to make circle size more meaningful
        
        New features:
        - Color depth represents expression intensity (deeper red = higher expression)
        - Circle size represents statistical significance (only two sizes):
          * P < 0.01: large circle (significant)
          * P ≥ 0.01: small circle or nearly invisible (not significant)
        - Blue border marks highly significant interactions (P < 0.01)
        - Supports dual information encoding: color expression + size significance
        - Supports data scaling: 'row', 'column', or None
        - Supports sender and receiver color bars
        
        Args:
            sources_use: str, int, list or None
                Sender cell types. Can be:
                - String: cell type name
                - Integer: cell type index (starting from 0)
                - List: multiple cell types
                - None: all cell types as senders
            targets_use: str, int, list or None
                Receiver cell types. Same format as sources_use
            signaling: str, list or None
                Specific signaling pathway name. Can be:
                - String: single pathway name
                - List: multiple pathway names
                - None: all pathways
            pvalue_threshold: float
                P-value threshold for significant interactions
            mean_threshold: float
                Mean expression intensity threshold
            top_interactions: int
                Display the top N strongest interactions
            show_pvalue: bool
                Whether to show P-value information
            show_mean: bool
                Whether to show mean expression intensity
            show_count: bool
                Whether to show interaction count
            add_violin: bool
                Whether to add violin plot to show expression distribution
            add_dendrogram: bool
                Whether to add clustering tree
            group_pathways: bool
                Whether to group by signaling pathways
            figsize: tuple
                Figure size
            title: str
                Figure title
            remove_isolate: bool
                Whether to remove isolated interactions
            font_size: int
                Font size (default: 12)
            cmap: str
                Color map (default: "RdBu_r")
            Options: "Blues", "Greens", "Oranges", "Purples", "viridis", "plasma", etc.
            transpose: bool
                Whether to transpose the heatmap (default: False)
                If True, swap rows and columns: rows=L-R pairs, columns=cell type pairs
            scale: str or None
                Scaling method for the expression data (default: None)
                - 'row': Scale each row (cell type pair) to have mean=0, std=1 (Z-score)
                - 'column': Scale each column (pathway/LR pair) to have mean=0, std=1 (Z-score)
                - 'row_minmax': Min-max scaling for each row (cell type pair) to [0,1] range
                - 'column_minmax': Min-max scaling for each column (pathway/LR pair) to [0,1] range
                - None: No scaling (use raw expression values)
            vmin: float or None
                Minimum value for color scaling (default: None)
            vmax: float or None
                Maximum value for color scaling (default: None)
            show_sender_colors: bool
                Whether to show sender cell type color bar (default: True)
            show_receiver_colors: bool
                Whether to show receiver cell type color bar (default: False)
            
        Returns:
            h: marsilea plot object
        """
        try:
            import marsilea as ma
            import marsilea.plotter as mp
            from matplotlib.colors import Normalize
            from sklearn.preprocessing import normalize
        except ImportError:
            raise ImportError("marsilea and sklearn packages are required. Please install them: pip install marsilea scikit-learn")
        
        # Handle sender cell types
        if sources_use is None:
            source_cell_types = self.cell_types
        else:
            if isinstance(sources_use, (int, str)):
                sources_use = [sources_use]
            
            source_cell_types = []
            for src in sources_use:
                if isinstance(src, int):
                    if 0 <= src < len(self.cell_types):
                        source_cell_types.append(self.cell_types[src])
                    else:
                        raise ValueError(f"Source index {src} out of range [0, {len(self.cell_types)-1}]")
                elif isinstance(src, str):
                    if src in self.cell_types:
                        source_cell_types.append(src)
                    else:
                        raise ValueError(f"Source cell type '{src}' not found. Available: {self.cell_types}")
        
        # Handle receiver cell types
        if targets_use is None:
            target_cell_types = self.cell_types
        else:
            if isinstance(targets_use, (int, str)):
                targets_use = [targets_use]
            
            target_cell_types = []
            for tgt in targets_use:
                if isinstance(tgt, int):
                    if 0 <= tgt < len(self.cell_types):
                        target_cell_types.append(self.cell_types[tgt])
                    else:
                        raise ValueError(f"Target index {tgt} out of range [0, {len(self.cell_types)-1}]")
                elif isinstance(tgt, str):
                    if tgt in self.cell_types:
                        target_cell_types.append(tgt)
                    else:
                        raise ValueError(f"Target cell type '{tgt}' not found. Available: {self.cell_types}")
        
        # Handle signaling pathway filtering
        signaling_indices = None
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            if 'classification' in self.adata.var.columns:
                available_pathways = self.adata.var['classification'].unique()
                missing_pathways = [p for p in signaling if p not in available_pathways]
                if missing_pathways:
                    print(f"Warning: The following signaling pathways were not found: {missing_pathways}")
                    # Only keep existing pathways
                    signaling = [p for p in signaling if p in available_pathways]
                    if not signaling:
                        print(f"❌ Error: None of the specified signaling pathways exist in the data.")
                        print(f"Available pathways: {list(available_pathways)}")
                        return None
                
                signaling_mask = self.adata.var['classification'].isin(signaling)
                signaling_indices = np.where(signaling_mask)[0]
                
                if len(signaling_indices) == 0:
                    print(f"❌ Error: No interactions found for signaling pathway(s): {signaling}")
                    print(f"Available pathways: {list(available_pathways)}")
                    return None
            else:
                print("❌ Error: 'classification' column not found in adata.var")
                print("Cannot filter by signaling pathways")
                return None
        
        # Collect significant ligand-receptor interactions
        interactions_data = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # Check if sender and receiver meet the conditions
            if sender not in source_cell_types or receiver not in target_cell_types:
                continue
            
            # Get interaction data
            pvals = self.adata.layers['pvalues'][i, :]
            means = self.adata.layers['means'][i, :]
            
            # Apply signaling pathway filtering
            if signaling_indices is not None:
                pvals = pvals[signaling_indices]
                means = means[signaling_indices]
                var_indices = signaling_indices
            else:
                var_indices = np.arange(len(pvals))
            
            sig_mask = (pvals < pvalue_threshold) & (means > mean_threshold)
            
            if np.any(sig_mask):
                # Get significant interaction information
                original_indices = var_indices[sig_mask]
                
                for idx, (p_val, mean_val) in enumerate(zip(pvals[sig_mask], means[sig_mask])):
                    original_idx = original_indices[idx]
                    
                    # Get ligand-receptor pair information
                    if 'gene_a' in self.adata.var.columns and 'gene_b' in self.adata.var.columns:
                        ligand = self.adata.var['gene_a'].iloc[original_idx]
                        receptor = self.adata.var['gene_b'].iloc[original_idx]
                        if pd.isna(ligand) or pd.isna(receptor):
                            continue
                        lr_pair = f"{ligand}_{receptor}"
                    else:
                        lr_pair = self.adata.var.index[original_idx]
                    
                    # Get signaling pathway information
                    if 'classification' in self.adata.var.columns:
                        pathway = self.adata.var['classification'].iloc[original_idx]
                    else:
                        pathway = 'Unknown'
                    
                    interactions_data.append({
                        'source': sender,
                        'target': receiver,
                        'lr_pair': lr_pair,
                        'pathway': pathway,
                        'pvalue': p_val,
                        'mean_expression': mean_val,
                        'interaction': f"{sender} → {receiver}"
                    })
        
        if not interactions_data:
            if signaling is not None:
                print(f"❌ No significant interactions found for the specified signaling pathway(s): {signaling}")
                print(f"Try adjusting the thresholds:")
                print(f"   - pvalue_threshold (current: {pvalue_threshold})")
                print(f"   - mean_threshold (current: {mean_threshold})")
                print(f"Or check if these pathways have interactions between the specified cell types.")
            else:
                print("❌ No significant interactions found for the specified conditions")
                print(f"Try adjusting the thresholds:")
                print(f"   - pvalue_threshold (current: {pvalue_threshold})")
                print(f"   - mean_threshold (current: {mean_threshold})")
            return None
        
        # Create interaction DataFrame
        df_interactions = pd.DataFrame(interactions_data)
        
        # If specified signaling pathways, verify again if only interactions from those pathways are included
        if signaling is not None:
            pathway_in_data = df_interactions['pathway'].unique()
            unexpected_pathways = [p for p in pathway_in_data if p not in signaling]
            if unexpected_pathways:
                print(f"⚠️  Warning: Found interactions from unexpected pathways: {unexpected_pathways}")
            
            # Strict filtering: only keep interactions from specified pathways
            df_interactions = df_interactions[df_interactions['pathway'].isin(signaling)]
            
            if len(df_interactions) == 0:
                print(f"❌ After filtering, no interactions remain for signaling pathway(s): {signaling}")
                return None
        
        # Remove isolated interactions (if needed)
        if remove_isolate:
            interaction_counts = df_interactions.groupby(['source', 'target']).size()
            valid_pairs = interaction_counts[interaction_counts > 1].index
            df_interactions = df_interactions[
                df_interactions.apply(lambda x: (x['source'], x['target']) in valid_pairs, axis=1)
            ]
        
        # Select top interactions
        if top_interactions and len(df_interactions) > top_interactions:
            df_interactions = df_interactions.nlargest(top_interactions, 'mean_expression')
        
        # Create pivot table
        if group_pathways:
            # Group by signaling pathways
            pivot_mean = df_interactions.pivot_table(
                values='mean_expression', 
                index='interaction', 
                columns='pathway', 
                aggfunc='mean',
                fill_value=0
            )
            # Pathway level P-value should use a more appropriate aggregation method
            # Option 1: Median (more robust)
            # Option 2: Geometric mean 
            # Option 3: Fisher's method for combining P-values
            
            # Here, we use median as the representative P-value for pathway level (more conservative and robust)
            pivot_pval = df_interactions.pivot_table(
                values='pvalue', 
                index='interaction', 
                columns='pathway', 
                aggfunc='median',  # Use median instead of minimum
                fill_value=1
            )
            
            # If specified signaling pathways, verify if pivot table columns only contain specified pathways
            if signaling is not None:
                pivot_pathways = set(pivot_mean.columns)
                specified_pathways = set(signaling)
                if not pivot_pathways.issubset(specified_pathways):
                    unexpected_in_pivot = pivot_pathways - specified_pathways
                    print(f"⚠️  Warning: Pivot table contains unexpected pathways: {unexpected_in_pivot}")
                    # Only keep specified pathway columns
                    valid_columns = [col for col in pivot_mean.columns if col in signaling]
                    if not valid_columns:
                        print(f"❌ No valid pathway columns found for: {signaling}")
                        return None
                    pivot_mean = pivot_mean[valid_columns]
                    pivot_pval = pivot_pval[valid_columns]
        else:
            # Group by ligand-receptor pairs
            pivot_mean = df_interactions.pivot_table(
                values='mean_expression', 
                index='interaction', 
                columns='lr_pair', 
                aggfunc='mean',
                fill_value=0
            )
            pivot_pval = df_interactions.pivot_table(
                values='pvalue', 
                index='interaction', 
                columns='lr_pair', 
                aggfunc='min',
                fill_value=1
            )
        
        # Normalize expression data
        matrix_normalized = normalize(pivot_mean.to_numpy(), axis=0)
        
        # Apply scaling if specified
        if scale is not None:
            if scale not in ['row', 'column', 'row_minmax', 'column_minmax']:
                print(f"⚠️  Warning: Invalid scale parameter '{scale}'. Must be 'row', 'column', 'row_minmax', 'column_minmax', or None. Using no scaling.")
                scale = None
            
            if scale == 'row':
                # Scale each row (cell type pair) to have mean=0, std=1
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                pivot_mean_scaled = pd.DataFrame(
                    scaler.fit_transform(pivot_mean),
                    index=pivot_mean.index,
                    columns=pivot_mean.columns
                )
                print(f"📊 Applied row-wise scaling (Z-score normalization)")
                pivot_mean = pivot_mean_scaled
                
            elif scale == 'column':
                # Scale each column (pathway/LR pair) to have mean=0, std=1
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                pivot_mean_scaled = pd.DataFrame(
                    scaler.fit_transform(pivot_mean.T).T,  # Transpose, scale, then transpose back
                    index=pivot_mean.index,
                    columns=pivot_mean.columns
                )
                print(f"📊 Applied column-wise scaling (Z-score normalization)")
                pivot_mean = pivot_mean_scaled
                
            elif scale == 'row_minmax':
                # Min-max scaling for each row (cell type pair)
                means = pivot_mean.to_numpy()
                means = (means - means.min(axis=1, keepdims=True)) / (means.max(axis=1, keepdims=True) - means.min(axis=1, keepdims=True))
                pivot_mean = pd.DataFrame(means, index=pivot_mean.index, columns=pivot_mean.columns)
                print(f"📊 Applied row-wise min-max scaling")
                
            elif scale == 'column_minmax':
                # Min-max scaling for each column (pathway/LR pair)
                means = pivot_mean.to_numpy()
                means = (means - means.min(axis=0)) / (means.max(axis=0) - means.min(axis=0))
                pivot_mean = pd.DataFrame(means, index=pivot_mean.index, columns=pivot_mean.columns)
                print(f"📊 Applied column-wise min-max scaling")
        
        # Create Marsilea visualization component - using SizedHeatmap for enhanced visualization
        # Important: Calculate size and color matrices after pivot_table creation to ensure dimension matching
        
        # Prepare data: color=expression, size=significance
        expression_matrix = pivot_mean.to_numpy()
        pval_matrix = pivot_pval.to_numpy()
        
        # Color matrix: use expression, darker colors indicate higher expression
        color_matrix = expression_matrix.copy()
        # Ensure no NaN or Inf values
        color_matrix = np.nan_to_num(color_matrix, nan=0.0, posinf=color_matrix[np.isfinite(color_matrix)].max(), neginf=0.0)
        
        # Size matrix: use negative log transformation for P-value, smaller circles indicate larger P-value
        # -log10(p-value): P=0.01 → size=2, P=0.05 → size=1.3, P=0.1 → size=1
        size_matrix = -np.log10(pval_matrix + 1e-10)  # Adding small value to avoid log(0)
        
        # Normalize to a reasonable visual range (0.2 to 1.0)
        # This way, smaller P-values result in larger circles
        size_min = 0.2  # Smallest circle size (corresponds to non-significant P-value)
        size_max = 1.0  # Largest circle size (corresponds to highly significant P-value)
        
        # Normalize: map -log10(p) to [size_min, size_max] range
        if size_matrix.max() > size_matrix.min():
            size_matrix_norm = (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
            size_matrix = size_matrix_norm * (size_max - size_min) + size_min
        else:
            # When all P-values are identical, add slight random error to avoid visualization issues
            print("⚠️  Warning: All p-values are identical. Adding slight jitter for better visualization.")
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Add small random error to original P-values (does not affect statistical interpretation)
            jitter_strength = 1e-2  # Very small error, does not affect statistical interpretation
            jittered_pvals = pval_matrix + np.random.normal(0, jitter_strength, pval_matrix.shape)
            
            # Ensure P-values are still within a reasonable range [0, 1]
            jittered_pvals = np.clip(jittered_pvals, 1e-10, 1.0)
            
            # Recalculate size_matrix
            size_matrix = -np.log10(pval_matrix + 1e-10)
            
            
            # Recalculate normalization
            if size_matrix.max() > size_matrix.min():
                size_matrix_norm = (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
                size_matrix = size_matrix_norm * (size_max - size_min) + size_min
                
            else:
                # If adding error still results in the same (extreme case), use medium size
                print("⚠️  Warning: All p-values are identical after jittering. Using medium size.")
                size_matrix = np.full_like(size_matrix, (size_min + size_max) / 2)
                size_matrix=color_matrix
        

        
        # Transpose functionality - need to save original pivot for later layers
        original_pivot_mean = pivot_mean.copy()
        original_pivot_pval = pivot_pval.copy()
        
        if transpose:
            size_matrix = size_matrix.T
            color_matrix = color_matrix.T
            pivot_mean = pivot_mean.T
            pivot_pval = pivot_pval.T
            # Note: Transpose also requires recalculating matrix_normalized
            matrix_normalized = normalize(pivot_mean.to_numpy(), axis=0)
        
        # 1. Main SizedHeatmap - based on your reference code
        # Set color legend title based on scaling
        if scale == 'row':
            color_legend_title = "Expression Level (Row-scaled)"
        elif scale == 'column':
            color_legend_title = "Expression Level (Column-scaled)"
        elif scale == 'row_minmax':
            color_legend_title = "Expression Level (Row-scaled Min-max)"
        elif scale == 'column_minmax':
            color_legend_title = "Expression Level (Column-scaled Min-max)"
        else:
            color_legend_title = "Expression Level"
        
        if dot_size_min is not None and dot_size_max is not None:
            from matplotlib.colors import Normalize
            size_norm = Normalize(vmin=dot_size_min, vmax=dot_size_max)
        else:
            size_norm = None
        
        h = ma.SizedHeatmap(
            size=size_matrix,
            color=color_matrix,
            cmap=cmap,  # Using custom color map
            width=figsize[0] * 0.6, 
            height=figsize[1] * 0.7,
            legend=True,
            size_legend_kws=dict(
                colors="black",
                title="",
                labels=["p>0.05", "p<0.01"],
                show_at=[0.01, 1.0],
            ),
            vmin=vmin,
            vmax=vmax,
            size_norm=size_norm,
            color_legend_kws=dict(title=color_legend_title),
        )
        
        # 2. Optional additional significance layer
        if show_pvalue:
            try:
                # Use transposed pval_matrix to calculate significance
                current_pval_matrix = pivot_pval.to_numpy()
                highly_significant_mask = current_pval_matrix < 0.01
                if np.any(highly_significant_mask):
                    sig_layer = mp.MarkerMesh(
                        highly_significant_mask,
                        color="none",
                        edgecolor="#2E86AB",
                        linewidth=2.0,
                        label="P < 0.01"
                    )
                    h.add_layer(sig_layer)
            except Exception as e:
                print(f"Warning: Could not add significance layer: {e}")
        
        # 3. High expression marker
        if show_mean:
            try:
                high_expression_mask = matrix_normalized > 0.7
                high_mark = mp.MarkerMesh(
                    high_expression_mask, 
                    color="#DB4D6D", 
                    label="High Expression"
                )
                h.add_layer(high_mark)
            except Exception as e:
                print(f"Warning: Could not add high expression layer: {e}")

        
        
        # 4. Cell type color bar and labels - adding mp.Colors to show sender and receiver cell type colors
        # Parse cell interaction labels for cell type information (format: sender→receiver)
        cell_colors = self._get_cell_type_colors()
        
        # Create sender and receiver color mappings for each interaction pair
        sender_colors = []
        sender_names_list = []
        receiver_colors = []
        receiver_names_list = []

        if transpose:
            for interaction in pivot_mean.columns:
                if '→' in str(interaction):
                    # Parse sender and receiver
                    sender, receiver = str(interaction).split('→', 1)
                    sender = sender.strip()
                    receiver = receiver.strip()
                    
                    # Get corresponding colors
                    sender_color = cell_colors.get(sender, '#CCCCCC')
                    receiver_color = cell_colors.get(receiver, '#CCCCCC')
                    
                    sender_colors.append(sender_color)
                    sender_names_list.append(sender)
                    receiver_colors.append(receiver_color)
                    receiver_names_list.append(receiver)
                else:
                    # If not standard format, use default color
                    sender_colors.append('#CCCCCC')
                    sender_names_list.append(interaction)
                    receiver_colors.append('#CCCCCC')
                    receiver_names_list.append(interaction)

            # Add receiver color bar if requested
            if show_receiver_colors:
                receiver_palette = dict(zip(receiver_names_list, receiver_colors))
                receiver_color_bar = mp.Colors(receiver_names_list, palette=receiver_palette)
                h.add_bottom(receiver_color_bar, pad=0.05, size=0.15)
            
            # Add sender color bar if requested
            if show_sender_colors:
                sender_palette = dict(zip(sender_names_list, sender_colors))
                sender_color_bar = mp.Colors(sender_names_list, palette=sender_palette)
                h.add_bottom(sender_color_bar, pad=0.05, size=0.15)
            
            
                
        else:
            for interaction in pivot_mean.index:
                if '→' in str(interaction):
                    # Parse sender and receiver
                    sender, receiver = str(interaction).split('→', 1)
                    sender = sender.strip()
                    receiver = receiver.strip()
                    
                    # Get corresponding colors
                    sender_color = cell_colors.get(sender, '#CCCCCC')
                    receiver_color = cell_colors.get(receiver, '#CCCCCC')
                    
                    sender_colors.append(sender_color)
                    sender_names_list.append(sender)
                    receiver_colors.append(receiver_color)
                    receiver_names_list.append(receiver)
                else:
                    # If not standard format, use default color
                    sender_colors.append('#CCCCCC')
                    sender_names_list.append(interaction)
                    receiver_colors.append('#CCCCCC')
                    receiver_names_list.append(interaction)
            
            # Add sender color bar if requested
            if show_sender_colors:
                sender_palette = dict(zip(sender_names_list, sender_colors))
                sender_color_bar = mp.Colors(sender_names_list, palette=sender_palette)
                h.add_left(sender_color_bar, size=0.15, pad=0.05)
            
            # Add receiver color bar if requested
            if show_receiver_colors:
                receiver_palette = dict(zip(receiver_names_list, receiver_colors))
                receiver_color_bar = mp.Colors(receiver_names_list, palette=receiver_palette)
                h.add_left(receiver_color_bar, size=0.15, pad=0.05)
        
        # Add cell interaction labels
        cell_interaction_labels = mp.Labels(
            pivot_mean.index, 
            align="center",
            fontsize=font_size
        )
        
        h.add_left(cell_interaction_labels, pad=0.05)
        
        # 5. Ligand-receptor pair or pathway labels - based on your reference code
        lr_pathway_labels = mp.Labels(
            pivot_mean.columns,
            fontsize=font_size
        )
        h.add_bottom(lr_pathway_labels)
        
        # 6. Group by signaling pathway or function (simplified version for SizedHeatmap)
        if group_pathways and 'classification' in self.adata.var.columns:
            # Get color mapping for signaling pathways
            unique_pathways = pivot_mean.columns.tolist()
            pathway_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pathways)))
            pathway_color_map = {pathway: mcolors.to_hex(color) 
                               for pathway, color in zip(unique_pathways, pathway_colors)}
            
            # Note: Group functionality simplified for SizedHeatmap compatibility
        
        # 7. Clustering tree (with stricter safety checks)
        if add_dendrogram:
            try:
                # Check data dimensionality and quality for clustering
                can_cluster_rows = (pivot_mean.shape[0] > 2 and 
                                   not np.any(np.isnan(pivot_mean.values)) and 
                                   np.var(pivot_mean.values, axis=1).sum() > 0)
                                   
                can_cluster_cols = (pivot_mean.shape[1] > 2 and 
                                   not np.any(np.isnan(pivot_mean.values)) and 
                                   np.var(pivot_mean.values, axis=0).sum() > 0)
                
                if can_cluster_rows:
                    h.add_dendrogram("left", colors="#33A6B8")
                if group_pathways and can_cluster_cols:
                    h.add_dendrogram("bottom", colors="#B481BB")
                    
                if not can_cluster_rows and not can_cluster_cols:
                    print("Warning: Insufficient data variability for clustering. Skipping dendrograms.")
                    
            except Exception as e:
                print(f"Warning: Could not add dendrogram: {e}")
                print("Continuing without dendrograms. Consider setting add_dendrogram=False")
        
        # 8. Legend - based on your reference code
        h.add_legends()
        
        # 9. Set title
        if title:
            h.add_title(title, fontsize=font_size + 2, pad=0.02)  # Title font size is 2 larger than body text
        
        # Render figure
        h.render()
        
        print(f"📊 Visualization statistics:")
        print(f"   - Number of significant interactions: {len(df_interactions)}")
        print(f"   - Number of cell type pairs: {len(pivot_mean.index)}")
        print(f"   - {'Signaling pathways' if group_pathways else 'Ligand-receptor pairs'}: {len(pivot_mean.columns)}")
        if scale:
            if scale in ['row', 'column']:
                print(f"   - Data scaling: {scale} (Z-score normalization)")
            elif scale in ['row_minmax', 'column_minmax']:
                print(f"   - Data scaling: {scale} (Min-max normalization to [0,1])")
        else:
            print(f"   - Data scaling: None (raw expression values)")
        
        # Color bar information
        color_bar_info = []
        if show_sender_colors:
            color_bar_info.append("sender")
        if show_receiver_colors:
            color_bar_info.append("receiver")
        if color_bar_info:
            print(f"   - Color bars: {', '.join(color_bar_info)}")
        else:
            print(f"   - Color bars: None")
        
        return h

    def netVisual_bubble_lr(self, sources_use=None, targets_use=None, 
                           lr_pairs=None, pvalue_threshold=1.0, 
                           mean_threshold=0.0, show_all_pairs=True,
                           show_pvalue=True, show_mean=True, show_count=False,
                           add_violin=False, add_dendrogram=False,
                           figsize=(12, 8), title="Ligand-Receptor Communication Analysis", 
                           remove_isolate=False, font_size=12, cmap="RdBu_r",
                           transpose=False, scale=None,
                           vmin=None, vmax=None, dot_size_min=None, dot_size_max=None,
                           show_sender_colors=True, show_receiver_colors=False):
        """
        Create bubble plot to visualize specific ligand-receptor pairs in cell-cell communication
        Similar to netVisual_bubble_marsilea but focuses on specific L-R pairs instead of pathways
        
        Key differences from netVisual_bubble_marsilea:
        - Filters by specific ligand-receptor pairs instead of signaling pathways
        - Allows visualization even if the specified L-R pairs have zero expression
        - Uses more permissive default thresholds (pvalue_threshold=1.0, mean_threshold=0.0)
        - Provides show_all_pairs option to force display of all specified pairs
        
        Args:
            sources_use: str, int, list or None
                Sender cell types. Can be:
                - String: cell type name
                - Integer: cell type index (starting from 0)
                - List: multiple cell types
                - None: all cell types as senders
            targets_use: str, int, list or None
                Receiver cell types. Same format as sources_use
            lr_pairs: str, list or None
                Specific ligand-receptor pairs to visualize. Can be:
                - String: single L-R pair name (e.g., "TGFB1_TGFBR1")
                - List: multiple L-R pair names
                - None: all L-R pairs (equivalent to original function)
            pvalue_threshold: float
                P-value threshold (default: 1.0 to show all pairs)
            mean_threshold: float
                Mean expression threshold (default: 0.0 to show all pairs)
            show_all_pairs: bool
                If True, force display of all specified L-R pairs even if they have zero expression
            show_pvalue: bool
                Whether to show P-value information
            show_mean: bool
                Whether to show mean expression intensity
            show_count: bool
                Whether to show interaction count
            add_violin: bool
                Whether to add violin plot to show expression distribution
            add_dendrogram: bool
                Whether to add clustering tree
            figsize: tuple
                Figure size
            title: str
                Figure title
            remove_isolate: bool
                Whether to remove isolated interactions
            font_size: int
                Font size (default: 12)
            cmap: str
                Color map (default: "RdBu_r")
            transpose: bool
                Whether to transpose the heatmap (default: False)
            scale: str or None
                Scaling method for the expression data (default: None)
                - 'row': Scale each row (cell type pair) to have mean=0, std=1 (Z-score)
                - 'column': Scale each column (L-R pair) to have mean=0, std=1 (Z-score)
                - 'row_minmax': Min-max scaling for each row to [0,1] range
                - 'column_minmax': Min-max scaling for each column to [0,1] range
                - None: No scaling (use raw expression values)
            vmin: float or None
                Minimum value for color scaling (default: None)
            vmax: float or None
                Maximum value for color scaling (default: None)
            show_sender_colors: bool
                Whether to show sender cell type color bar (default: True)
            show_receiver_colors: bool
                Whether to show receiver cell type color bar (default: False)
            
        Returns:
            h: marsilea plot object
        """
        try:
            import marsilea as ma
            import marsilea.plotter as mp
            from matplotlib.colors import Normalize
            from sklearn.preprocessing import normalize
        except ImportError:
            raise ImportError("marsilea and sklearn packages are required. Please install them: pip install marsilea scikit-learn")
        
        # Handle sender cell types
        if sources_use is None:
            source_cell_types = self.cell_types
        else:
            if isinstance(sources_use, (int, str)):
                sources_use = [sources_use]
            
            source_cell_types = []
            for src in sources_use:
                if isinstance(src, int):
                    if 0 <= src < len(self.cell_types):
                        source_cell_types.append(self.cell_types[src])
                    else:
                        raise ValueError(f"Source index {src} out of range [0, {len(self.cell_types)-1}]")
                elif isinstance(src, str):
                    if src in self.cell_types:
                        source_cell_types.append(src)
                    else:
                        raise ValueError(f"Source cell type '{src}' not found. Available: {self.cell_types}")
        
        # Handle receiver cell types
        if targets_use is None:
            target_cell_types = self.cell_types
        else:
            if isinstance(targets_use, (int, str)):
                targets_use = [targets_use]
            
            target_cell_types = []
            for tgt in targets_use:
                if isinstance(tgt, int):
                    if 0 <= tgt < len(self.cell_types):
                        target_cell_types.append(self.cell_types[tgt])
                    else:
                        raise ValueError(f"Target index {tgt} out of range [0, {len(self.cell_types)-1}]")
                elif isinstance(tgt, str):
                    if tgt in self.cell_types:
                        target_cell_types.append(tgt)
                    else:
                        raise ValueError(f"Target cell type '{tgt}' not found. Available: {self.cell_types}")
        
        # Handle ligand-receptor pair filtering
        lr_indices = None
        if lr_pairs is not None:
            if isinstance(lr_pairs, str):
                lr_pairs = [lr_pairs]
            
            # Build L-R pair names from gene_a and gene_b columns
            if 'gene_a' in self.adata.var.columns and 'gene_b' in self.adata.var.columns:
                # Create L-R pair names: ligand_receptor format
                lr_names_in_data = []
                for i in range(len(self.adata.var)):
                    ligand = self.adata.var['gene_a'].iloc[i]
                    receptor = self.adata.var['gene_b'].iloc[i]
                    if not (pd.isna(ligand) or pd.isna(receptor)):
                        lr_names_in_data.append(f"{ligand}_{receptor}")
                    else:
                        lr_names_in_data.append(self.adata.var.index[i])
                
                # Find indices for specified L-R pairs
                lr_indices = []
                missing_pairs = []
                
                for lr_pair in lr_pairs:
                    try:
                        # Try to find exact match
                        idx = lr_names_in_data.index(lr_pair)
                        lr_indices.append(idx)
                    except ValueError:
                        # Try to find by alternative formats
                        found = False
                        # Try with different separators
                        for sep in ['_', '-', ':', ' ']:
                            alt_name = lr_pair.replace('_', sep)
                            if alt_name in lr_names_in_data:
                                idx = lr_names_in_data.index(alt_name)
                                lr_indices.append(idx)
                                found = True
                                break
                        
                        if not found:
                            # Try to find in adata.var.index directly
                            if lr_pair in self.adata.var.index:
                                idx = list(self.adata.var.index).index(lr_pair)
                                lr_indices.append(idx)
                                found = True
                        
                        if not found:
                            missing_pairs.append(lr_pair)
                
                if missing_pairs and not show_all_pairs:
                    print(f"Warning: The following L-R pairs were not found: {missing_pairs}")
                    print(f"Available L-R pairs (first 10): {lr_names_in_data[:10]}")
                    if len(lr_indices) == 0:
                        print(f"❌ Error: None of the specified L-R pairs exist in the data.")
                        return None
                
                lr_indices = np.array(lr_indices)
            else:
                print("❌ Error: 'gene_a' and 'gene_b' columns not found in adata.var")
                print("Cannot filter by L-R pairs")
                return None
        
        # Collect ligand-receptor interactions (modified to be more permissive)
        interactions_data = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # Check if sender and receiver meet the conditions
            if sender not in source_cell_types or receiver not in target_cell_types:
                continue
            
            # Get interaction data
            pvals = self.adata.layers['pvalues'][i, :]
            means = self.adata.layers['means'][i, :]
            
            # Apply L-R pair filtering
            if lr_indices is not None:
                pvals = pvals[lr_indices]
                means = means[lr_indices]
                var_indices = lr_indices
            else:
                var_indices = np.arange(len(pvals))
            
            # Modified filtering logic: more permissive when show_all_pairs=True
            if show_all_pairs and lr_pairs is not None:
                # When show_all_pairs=True, include all specified pairs regardless of significance
                sig_mask = np.ones(len(pvals), dtype=bool)
            else:
                # Standard filtering
                sig_mask = (pvals < pvalue_threshold) & (means > mean_threshold)
            
            # Always add interactions for specified L-R pairs when show_all_pairs=True
            if show_all_pairs and lr_pairs is not None:
                # Include all specified pairs
                for idx in range(len(pvals)):
                    original_idx = var_indices[idx]
                    p_val = pvals[idx]
                    mean_val = means[idx]
                    
                    # Get ligand-receptor pair information
                    if 'gene_a' in self.adata.var.columns and 'gene_b' in self.adata.var.columns:
                        ligand = self.adata.var['gene_a'].iloc[original_idx]
                        receptor = self.adata.var['gene_b'].iloc[original_idx]
                        if pd.isna(ligand) or pd.isna(receptor):
                            lr_pair = self.adata.var.index[original_idx]
                        else:
                            lr_pair = f"{ligand}_{receptor}"
                    else:
                        lr_pair = self.adata.var.index[original_idx]
                    
                    # Get signaling pathway information
                    if 'classification' in self.adata.var.columns:
                        pathway = self.adata.var['classification'].iloc[original_idx]
                    else:
                        pathway = 'Unknown'
                    
                    interactions_data.append({
                        'source': sender,
                        'target': receiver,
                        'lr_pair': lr_pair,
                        'pathway': pathway,
                        'pvalue': p_val,
                        'mean_expression': mean_val,
                        'interaction': f"{sender} → {receiver}"
                    })
            elif np.any(sig_mask):
                # Standard processing for significant interactions
                original_indices = var_indices[sig_mask]
                
                for idx, (p_val, mean_val) in enumerate(zip(pvals[sig_mask], means[sig_mask])):
                    original_idx = original_indices[idx]
                    
                    # Get ligand-receptor pair information
                    if 'gene_a' in self.adata.var.columns and 'gene_b' in self.adata.var.columns:
                        ligand = self.adata.var['gene_a'].iloc[original_idx]
                        receptor = self.adata.var['gene_b'].iloc[original_idx]
                        if pd.isna(ligand) or pd.isna(receptor):
                            lr_pair = self.adata.var.index[original_idx]
                        else:
                            lr_pair = f"{ligand}_{receptor}"
                    else:
                        lr_pair = self.adata.var.index[original_idx]
                    
                    # Get signaling pathway information
                    if 'classification' in self.adata.var.columns:
                        pathway = self.adata.var['classification'].iloc[original_idx]
                    else:
                        pathway = 'Unknown'
                    
                    interactions_data.append({
                        'source': sender,
                        'target': receiver,
                        'lr_pair': lr_pair,
                        'pathway': pathway,
                        'pvalue': p_val,
                        'mean_expression': mean_val,
                        'interaction': f"{sender} → {receiver}"
                    })
        
        if not interactions_data:
            if lr_pairs is not None:
                print(f"❌ No interactions found for the specified L-R pair(s): {lr_pairs}")
                print(f"Current thresholds:")
                print(f"   - pvalue_threshold: {pvalue_threshold}")
                print(f"   - mean_threshold: {mean_threshold}")
                print(f"   - show_all_pairs: {show_all_pairs}")
                print(f"Try setting show_all_pairs=True to force display of all specified pairs.")
            else:
                print("❌ No interactions found for the specified conditions")
                print(f"Try adjusting the thresholds:")
                print(f"   - pvalue_threshold (current: {pvalue_threshold})")
                print(f"   - mean_threshold (current: {mean_threshold})")
            return None
        
        # Create interaction DataFrame
        df_interactions = pd.DataFrame(interactions_data)
        
        # Remove isolated interactions (if needed)
        if remove_isolate:
            interaction_counts = df_interactions.groupby(['source', 'target']).size()
            valid_pairs = interaction_counts[interaction_counts > 1].index
            df_interactions = df_interactions[
                df_interactions.apply(lambda x: (x['source'], x['target']) in valid_pairs, axis=1)
            ]
        
        # Create pivot table - always group by L-R pairs for this function
        pivot_mean = df_interactions.pivot_table(
            values='mean_expression', 
            index='interaction', 
            columns='lr_pair', 
            aggfunc='mean',
            fill_value=0
        )
        pivot_pval = df_interactions.pivot_table(
            values='pvalue', 
            index='interaction', 
            columns='lr_pair', 
            aggfunc='min',
            fill_value=1
        )
        
        # Apply scaling if specified
        if scale is not None:
            if scale not in ['row', 'column', 'row_minmax', 'column_minmax']:
                print(f"⚠️  Warning: Invalid scale parameter '{scale}'. Must be 'row', 'column', 'row_minmax', 'column_minmax', or None. Using no scaling.")
                scale = None
            
            if scale == 'row':
                # Scale each row (cell type pair) to have mean=0, std=1
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                pivot_mean_scaled = pd.DataFrame(
                    scaler.fit_transform(pivot_mean),
                    index=pivot_mean.index,
                    columns=pivot_mean.columns
                )
                print(f"📊 Applied row-wise scaling (Z-score normalization)")
                pivot_mean = pivot_mean_scaled
                
            elif scale == 'column':
                # Scale each column (L-R pair) to have mean=0, std=1
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                pivot_mean_scaled = pd.DataFrame(
                    scaler.fit_transform(pivot_mean.T).T,  # Transpose, scale, then transpose back
                    index=pivot_mean.index,
                    columns=pivot_mean.columns
                )
                print(f"📊 Applied column-wise scaling (Z-score normalization)")
                pivot_mean = pivot_mean_scaled
                
            elif scale == 'row_minmax':
                # Min-max scaling for each row (cell type pair)
                means = pivot_mean.to_numpy()
                means = (means - means.min(axis=1, keepdims=True)) / (means.max(axis=1, keepdims=True) - means.min(axis=1, keepdims=True))
                pivot_mean = pd.DataFrame(means, index=pivot_mean.index, columns=pivot_mean.columns)
                print(f"📊 Applied row-wise min-max scaling")
                
            elif scale == 'column_minmax':
                # Min-max scaling for each column (L-R pair)
                means = pivot_mean.to_numpy()
                means = (means - means.min(axis=0)) / (means.max(axis=0) - means.min(axis=0))
                pivot_mean = pd.DataFrame(means, index=pivot_mean.index, columns=pivot_mean.columns)
                print(f"📊 Applied column-wise min-max scaling")
        
        # Normalize expression data for additional layers
        matrix_normalized = normalize(pivot_mean.to_numpy(), axis=0)
        
        # Prepare data for Marsilea SizedHeatmap
        expression_matrix = pivot_mean.to_numpy()
        pval_matrix = pivot_pval.to_numpy()
        
        # Color matrix: use expression
        color_matrix = expression_matrix.copy()
        color_matrix = np.nan_to_num(color_matrix, nan=0.0, posinf=color_matrix[np.isfinite(color_matrix)].max(), neginf=0.0)
        
        # Size matrix: use P-value significance
        size_matrix = -np.log10(pval_matrix + 1e-10)
        
        # Normalize size matrix
        size_min = 0.2
        size_max = 1.0
        
        if size_matrix.max() > size_matrix.min():
            size_matrix_norm = (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
            size_matrix = size_matrix_norm * (size_max - size_min) + size_min
        else:
            size_matrix = np.full_like(size_matrix, (size_min + size_max) / 2)
        
        # Transpose functionality
        if transpose:
            size_matrix = size_matrix.T
            color_matrix = color_matrix.T
            pivot_mean = pivot_mean.T
            pivot_pval = pivot_pval.T
            matrix_normalized = normalize(pivot_mean.to_numpy(), axis=0)
        
        # Set color legend title based on scaling
        if scale == 'row':
            color_legend_title = "Expression Level (Row-scaled)"
        elif scale == 'column':
            color_legend_title = "Expression Level (Column-scaled)"
        elif scale == 'row_minmax':
            color_legend_title = "Expression Level (Row-scaled Min-max)"
        elif scale == 'column_minmax':
            color_legend_title = "Expression Level (Column-scaled Min-max)"
        else:
            color_legend_title = "Expression Level"
        
        if dot_size_min is not None and dot_size_max is not None:
            from matplotlib.colors import Normalize
            size_norm = Normalize(vmin=dot_size_min, vmax=dot_size_max)
        else:
            size_norm = None
        
        # Create main SizedHeatmap
        h = ma.SizedHeatmap(
            size=size_matrix,
            color=color_matrix,
            cmap=cmap,
            width=figsize[0] * 0.6, 
            height=figsize[1] * 0.7,
            legend=True,
            size_legend_kws=dict(
                colors="black",
                title="",
                labels=["p>0.05", "p<0.01"],
                show_at=[0.01, 1.0],
            ),
            vmin=vmin,
            vmax=vmax,
            size_norm=size_norm,
            color_legend_kws=dict(title=color_legend_title),
        )
        
        # Add significance layer
        if show_pvalue:
            try:
                current_pval_matrix = pivot_pval.to_numpy()
                highly_significant_mask = current_pval_matrix < 0.01
                if np.any(highly_significant_mask):
                    sig_layer = mp.MarkerMesh(
                        highly_significant_mask,
                        color="none",
                        edgecolor="#2E86AB",
                        linewidth=2.0,
                        label="P < 0.01"
                    )
                    h.add_layer(sig_layer)
            except Exception as e:
                print(f"Warning: Could not add significance layer: {e}")
        
        # Add high expression marker
        if show_mean:
            try:
                high_expression_mask = matrix_normalized > 0.7
                high_mark = mp.MarkerMesh(
                    high_expression_mask, 
                    color="#DB4D6D", 
                    label="High Expression"
                )
                h.add_layer(high_mark)
            except Exception as e:
                print(f"Warning: Could not add high expression layer: {e}")
        
        # Add cell type color bars
        cell_colors = self._get_cell_type_colors()
        
        # Create sender and receiver color mappings
        sender_colors = []
        sender_names_list = []
        receiver_colors = []
        receiver_names_list = []

        if transpose:
            for interaction in pivot_mean.columns:
                if '→' in str(interaction):
                    sender, receiver = str(interaction).split('→', 1)
                    sender = sender.strip()
                    receiver = receiver.strip()
                    
                    sender_color = cell_colors.get(sender, '#CCCCCC')
                    receiver_color = cell_colors.get(receiver, '#CCCCCC')
                    
                    sender_colors.append(sender_color)
                    sender_names_list.append(sender)
                    receiver_colors.append(receiver_color)
                    receiver_names_list.append(receiver)
                else:
                    sender_colors.append('#CCCCCC')
                    sender_names_list.append(interaction)
                    receiver_colors.append('#CCCCCC')
                    receiver_names_list.append(interaction)

            if show_receiver_colors:
                receiver_palette = dict(zip(receiver_names_list, receiver_colors))
                receiver_color_bar = mp.Colors(receiver_names_list, palette=receiver_palette)
                h.add_bottom(receiver_color_bar, pad=0.05, size=0.15)
            
            if show_sender_colors:
                sender_palette = dict(zip(sender_names_list, sender_colors))
                sender_color_bar = mp.Colors(sender_names_list, palette=sender_palette)
                h.add_bottom(sender_color_bar, pad=0.05, size=0.15)
        else:
            for interaction in pivot_mean.index:
                if '→' in str(interaction):
                    sender, receiver = str(interaction).split('→', 1)
                    sender = sender.strip()
                    receiver = receiver.strip()
                    
                    sender_color = cell_colors.get(sender, '#CCCCCC')
                    receiver_color = cell_colors.get(receiver, '#CCCCCC')
                    
                    sender_colors.append(sender_color)
                    sender_names_list.append(sender)
                    receiver_colors.append(receiver_color)
                    receiver_names_list.append(receiver)
                else:
                    sender_colors.append('#CCCCCC')
                    sender_names_list.append(interaction)
                    receiver_colors.append('#CCCCCC')
                    receiver_names_list.append(interaction)
            
            if show_sender_colors:
                sender_palette = dict(zip(sender_names_list, sender_colors))
                sender_color_bar = mp.Colors(sender_names_list, palette=sender_palette)
                h.add_left(sender_color_bar, size=0.15, pad=0.05)
            
            if show_receiver_colors:
                receiver_palette = dict(zip(receiver_names_list, receiver_colors))
                receiver_color_bar = mp.Colors(receiver_names_list, palette=receiver_palette)
                h.add_left(receiver_color_bar, size=0.15, pad=0.05)
        
        # Add cell interaction labels
        cell_interaction_labels = mp.Labels(
            pivot_mean.index, 
            align="center",
            fontsize=font_size
        )
        h.add_left(cell_interaction_labels, pad=0.05)
        
        # Add L-R pair labels
        lr_pair_labels = mp.Labels(
            pivot_mean.columns,
            fontsize=font_size
        )
        h.add_bottom(lr_pair_labels)
        
        # Add clustering dendrograms
        if add_dendrogram:
            try:
                can_cluster_rows = (pivot_mean.shape[0] > 2 and 
                                   not np.any(np.isnan(pivot_mean.values)) and 
                                   np.var(pivot_mean.values, axis=1).sum() > 0)
                                   
                can_cluster_cols = (pivot_mean.shape[1] > 2 and 
                                   not np.any(np.isnan(pivot_mean.values)) and 
                                   np.var(pivot_mean.values, axis=0).sum() > 0)
                
                if can_cluster_rows:
                    h.add_dendrogram("left", colors="#33A6B8")
                if can_cluster_cols:
                    h.add_dendrogram("bottom", colors="#B481BB")
                    
                if not can_cluster_rows and not can_cluster_cols:
                    print("Warning: Insufficient data variability for clustering. Skipping dendrograms.")
                    
            except Exception as e:
                print(f"Warning: Could not add dendrogram: {e}")
        
        # Add legends
        h.add_legends()
        
        # Set title
        if title:
            h.add_title(title, fontsize=font_size + 2, pad=0.02)
        
        # Render figure
        h.render()
        
        print(f"📊 Ligand-Receptor Visualization Statistics:")
        print(f"   - Number of interactions displayed: {len(df_interactions)}")
        print(f"   - Number of cell type pairs: {len(pivot_mean.index)}")
        print(f"   - Number of L-R pairs: {len(pivot_mean.columns)}")
        if lr_pairs:
            print(f"   - Specified L-R pairs: {lr_pairs}")
            print(f"   - Show all pairs (even zero): {show_all_pairs}")
        if scale:
            print(f"   - Data scaling: {scale}")
        else:
            print(f"   - Data scaling: None (raw expression values)")
        
        return h
    
    def netAnalysis_computeCentrality(self, signaling=None, slot_name="netP", 
                                     pvalue_threshold=0.05, use_weight=True):
        """
        Calculate network centrality metrics (imitating CellChat's netAnalysis_computeCentrality function)
        
        Calculate the following centrality metrics and convert to CellChat-style Importance values (0-1 range):
        - out_degree: outdegree (primary sender role)
        - in_degree: indegree (primary receiver role)
        - flow_betweenness: flow betweenness (mediator role)
        - information_centrality: information centrality (influencer role)
        
        Args:
            signaling: str, list or None
                Specific signaling pathway name. If None, use aggregated network of all pathways
            slot_name: str
                Data slot name (compatible with CellChat, used here to identify calculation type)
            pvalue_threshold: float
                P-value threshold for significant interactions
            use_weight: bool
                Whether to use weights (interaction strength) for calculation
            
        Returns:
            centrality_scores: dict
                Dictionary containing various centrality metrics, all values are Importance values in 0-1 range
        """
        try:
            import networkx as nx
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path
        except ImportError:
            raise ImportError("NetworkX and SciPy are required for centrality analysis")
        
        # Calculate communication matrix
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # Check if signaling pathways exist
            if 'classification' in self.adata.var.columns:
                available_pathways = self.adata.var['classification'].unique()
                missing_pathways = [p for p in signaling if p not in available_pathways]
                if missing_pathways:
                    print(f"Warning: The following signaling pathways were not found: {missing_pathways}")
                    signaling = [p for p in signaling if p in available_pathways]
                    if not signaling:
                        raise ValueError("No valid signaling pathways provided")
                
                # Calculate communication matrix for specific pathways
                pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
                pathway_mask = self.adata.var['classification'].isin(signaling)
                pathway_indices = np.where(pathway_mask)[0]
                
                if len(pathway_indices) == 0:
                    raise ValueError(f"No interactions found for signaling pathway(s): {signaling}")
                
                for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                          self.adata.obs['receiver'])):
                    sender_idx = self.cell_types.index(sender)
                    receiver_idx = self.cell_types.index(receiver)
                    
                    pvals = self.adata.layers['pvalues'][i, pathway_indices]
                    means = self.adata.layers['means'][i, pathway_indices]
                    
                    sig_mask = pvals < pvalue_threshold
                    if np.any(sig_mask):
                        if use_weight:
                            pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
                        else:
                            pathway_matrix[sender_idx, receiver_idx] += np.sum(sig_mask)
                
                comm_matrix = pathway_matrix
            else:
                print("Warning: 'classification' column not found. Using aggregated network.")
                count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
                comm_matrix = weight_matrix if use_weight else count_matrix
        else:
            # Use aggregated network
            count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
            comm_matrix = weight_matrix if use_weight else count_matrix
        
        # Create NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_cell_types))
        
        # Add edges
        for i in range(self.n_cell_types):
            for j in range(self.n_cell_types):
                if comm_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=comm_matrix[i, j])
        
        # Calculate raw centrality metrics
        raw_centrality_scores = {}
        
        # 1. Outdegree centrality (Outdegree) - identify primary senders
        out_degree = np.array([comm_matrix[i, :].sum() for i in range(self.n_cell_types)])
        raw_centrality_scores['outdegree'] = out_degree
        
        # 2. Indegree centrality (Indegree) - identify primary receivers
        in_degree = np.array([comm_matrix[:, j].sum() for j in range(self.n_cell_types)])
        raw_centrality_scores['indegree'] = in_degree
        
        # 3. Flow betweenness (Flow Betweenness) - identify mediators
        try:
            if len(G.edges()) > 0:
                # Use NetworkX's betweenness centrality as approximation for flow betweenness
                betweenness = nx.betweenness_centrality(G, weight='weight')
                flow_betweenness = np.array([betweenness.get(i, 0) for i in range(self.n_cell_types)])
            else:
                flow_betweenness = np.zeros(self.n_cell_types)
        except:
            print("Warning: Failed to compute betweenness centrality, using zeros")
            flow_betweenness = np.zeros(self.n_cell_types)
        
        raw_centrality_scores['flow_betweenness'] = flow_betweenness
        
        # 4. Information centrality (Information Centrality) - identify influencers
        try:
            if len(G.edges()) > 0:
                # Use eigenvector centrality as approximation for information centrality
                # For directed graphs, use indegree eigenvector centrality
                eigenvector = nx.eigenvector_centrality(G.reverse(), weight='weight', max_iter=1000)
                information_centrality = np.array([eigenvector.get(i, 0) for i in range(self.n_cell_types)])
            else:
                information_centrality = np.zeros(self.n_cell_types)
        except:
            print("Warning: Failed to compute eigenvector centrality, using closeness centrality")
            try:
                closeness = nx.closeness_centrality(G.reverse(), distance='weight')
                information_centrality = np.array([closeness.get(i, 0) for i in range(self.n_cell_types)])
            except:
                print("Warning: Failed to compute closeness centrality, using zeros")
                information_centrality = np.zeros(self.n_cell_types)
        
        raw_centrality_scores['information'] = information_centrality
        
        # Convert raw centrality scores to CellChat-style Importance values (0-1 range)
        centrality_scores = {}
        for metric, scores in raw_centrality_scores.items():
            if scores.max() > scores.min() and scores.max() > 0:
                # Normalize to 0-1 range, ensure CellChat compatibility
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                # If all values are the same or all zero, set to 0
                normalized_scores = np.zeros_like(scores)
            
            centrality_scores[metric] = normalized_scores
        
        # 5. Overall centrality (Overall) - comprehensive metric (already normalized)
        overall_centrality = (centrality_scores['outdegree'] + 
                            centrality_scores['indegree'] + 
                            centrality_scores['flow_betweenness'] + 
                            centrality_scores['information']) / 4
        
        centrality_scores['overall'] = overall_centrality
        
        # Store raw scores and normalized scores
        self.raw_centrality_scores = raw_centrality_scores  # Save raw scores for debugging
        self.centrality_scores = centrality_scores  # CellChat-style Importance values
        self.centrality_matrix = comm_matrix
        
        print(f"✅ Network centrality calculation completed (CellChat-style Importance values)")
        print(f"   - Signaling pathways used: {signaling if signaling else 'All pathways'}")
        print(f"   - Weight mode: {'Weighted' if use_weight else 'Unweighted'}")
        print(f"   - Calculated metrics: outdegree, indegree, flow_betweenness, information, overall")
        print(f"   - All centrality scores normalized to 0-1 range (Importance values)")
        
        return centrality_scores
    
    def netAnalysis_signalingRole_network(self, signaling=None, measures=None,
                                        color_heatmap="RdYlBu_r", 
                                        width=12, height=8, font_size=10,
                                        title="Signaling Role Analysis",
                                        cluster_rows=True, cluster_cols=False,
                                        save=None, show_values=True):
        """
        Visualize signaling roles of cell populations (imitating CellChat's netAnalysis_signalingRole_network function)
        
        Args:
            signaling: str, list or None
                Specific signaling pathway name. If None, use stored centrality results or calculate aggregated network
            measures: list or None
                Centrality metrics to display. Default shows all metrics
            color_heatmap: str
                Heatmap color mapping
            width: float
                Figure width
            height: float
                Figure height
            font_size: int
                Font size
            title: str
                Figure title
            cluster_rows: bool
                Whether to cluster rows
            cluster_cols: bool
                Whether to cluster columns
            save: str or None
                Save path
            show_values: bool
                Whether to show values in the heatmap
            
        Returns:
            fig: matplotlib.figure.Figure
        """
        # If no pre-computed centrality scores, calculate first
        if not hasattr(self, 'centrality_scores') or signaling is not None:
            self.netAnalysis_computeCentrality(signaling=signaling)
        
        centrality_scores = self.centrality_scores
        
        # Select metrics to display
        if measures is None:
            measures = ['outdegree', 'indegree', 'flow_betweenness', 'information']
        
        # Validate metrics
        available_measures = list(centrality_scores.keys())
        invalid_measures = [m for m in measures if m not in available_measures]
        if invalid_measures:
            print(f"Warning: Invalid measures {invalid_measures}. Available: {available_measures}")
            measures = [m for m in measures if m in available_measures]
        
        if not measures:
            raise ValueError("No valid measures specified")
        
        # Create data matrix (using CellChat-style Importance values)
        data_matrix = np.array([centrality_scores[measure] for measure in measures])
        
        # Create label mapping
        measure_labels = {
            'outdegree': 'Outdegree',
            'indegree': 'Indegree', 
            'flow_betweenness': 'Flow Betweenness',
            'information': 'Information',
            'overall': 'Overall'
        }
        
        row_labels = [measure_labels.get(m, m) for m in measures]
        col_labels = self.cell_types
        
        # Create DataFrame for visualization
        df_centrality = pd.DataFrame(data_matrix, 
                                   index=row_labels, 
                                   columns=col_labels)
        
        # Use seaborn to create heatmap
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Draw heatmap using CellChat-style configuration
        sns.heatmap(df_centrality, 
                   annot=show_values, 
                   fmt='.2f' if show_values else '',  # Use 2 decimal places since it's 0-1 range
                   cmap=color_heatmap,
                   cbar_kws={'label': 'Importance'},  # CellChat-style label
                   square=False,
                   linewidths=0.5,
                   ax=ax,
                   xticklabels=True,
                   yticklabels=True,
                   vmin=0,  # Ensure color range starts from 0
                   vmax=1)  # Ensure color range ends at 1
        
        # Set labels and title
        ax.set_xlabel('Cell Groups', fontsize=font_size + 2)  # Use CellChat-style label
        ax.set_ylabel('', fontsize=font_size + 2)  # Y-axis usually doesn't show label in CellChat
        ax.set_title(title, fontsize=font_size + 4, pad=20)
        
        # Adjust font size
        ax.tick_params(axis='x', labelsize=font_size, rotation=45)
        ax.tick_params(axis='y', labelsize=font_size, rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Signaling role heatmap saved as: {save}")
        
        print(f"📊 Signaling role analysis results (Importance values 0-1):")
        for measure in measures:
            scores = centrality_scores[measure]
            if scores.max() > 0:
                top_cell_idx = np.argmax(scores)
                top_cell = self.cell_types[top_cell_idx]
                top_score = scores[top_cell_idx]
                role_description = {
                    'outdegree': 'Dominant Sender',
                    'indegree': 'Dominant Receiver',
                    'flow_betweenness': 'Mediator',
                    'information': 'Influencer',
                    'overall': 'Overall Leader'
                }
                print(f"   - {role_description.get(measure, measure)}: {top_cell} (Importance: {top_score:.3f})")
        
        return fig
    
    def netAnalysis_signalingRole_scatter(self, signaling=None, x_measure='outdegree', 
                                        y_measure='indegree', figsize=(10, 8),
                                        point_size=100, alpha=0.7, 
                                        title="Cell Signaling Roles - 2D View",
                                        save=None):
        """
        Create 2D scatter plot to visualize cell signaling roles
        
        Args:
            signaling: str, list or None
                Specific signaling pathway name
            x_measure: str
                Centrality metric used for X-axis
            y_measure: str  
                Centrality metric used for Y-axis
            figsize: tuple
                Figure size
            point_size: int
                Scatter point size
            alpha: float
                Transparency
            title: str
                Figure title
            save: str or None
                Save path
            
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        # If no pre-computed centrality scores, calculate first
        if not hasattr(self, 'centrality_scores') or signaling is not None:
            self.netAnalysis_computeCentrality(signaling=signaling)
        
        centrality_scores = self.centrality_scores
        
        # Validate metrics
        if x_measure not in centrality_scores:
            raise ValueError(f"x_measure '{x_measure}' not found in centrality scores")
        if y_measure not in centrality_scores:
            raise ValueError(f"y_measure '{y_measure}' not found in centrality scores")
        
        # Get data
        x_data = centrality_scores[x_measure]
        y_data = centrality_scores[y_measure]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get cell type colors
        cell_colors = self._get_cell_type_colors()
        colors = [cell_colors.get(ct, '#1f77b4') for ct in self.cell_types]
        
        # Draw scatter plot
        scatter = ax.scatter(x_data, y_data, 
                           c=colors, s=point_size, alpha=alpha,
                           edgecolors='black', linewidths=0.5)
        
        # Add cell type labels
        
        try:
            from adjustText import adjust_text
            
            texts = []
            for i, cell_type in enumerate(self.cell_types):
                text = ax.text(x_data[i], y_data[i], cell_type,
                             fontsize=10, alpha=0.8, ha='center', va='center',)
                texts.append(text)
            
            # Use adjust_text to prevent label overlap
            adjust_text(texts, ax=ax,
                      expand_points=(1.2, 1.2),
                      expand_text=(1.2, 1.2),
                      force_points=0.3,
                      force_text=0.3,
                      arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.8))
            
        except ImportError:
            import warnings
            warnings.warn("adjustText library not found. Using default ax.annotate instead.")
            # Fallback to original annotate method
            for i, cell_type in enumerate(self.cell_types):
                ax.annotate(cell_type, (x_data[i], y_data[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8)
        
        # Set labels and title
        measure_labels = {
            'outdegree': 'Outdegree (Sender Role)',
            'indegree': 'Indegree (Receiver Role)',
            'flow_betweenness': 'Flow Betweenness (Mediator Role)',
            'information': 'Information Centrality (Influencer Role)',
            'overall': 'Overall Centrality'
        }
        
        ax.set_xlabel(measure_labels.get(x_measure, x_measure), fontsize=12)
        ax.set_ylabel(measure_labels.get(y_measure, y_measure), fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"2D signaling role plot saved as: {save}")
        
        return fig, ax
    
    def netAnalysis_signalingRole_heatmap(self, pattern="outgoing", signaling=None, 
                                        row_scale=True, figsize=(12, 8), 
                                        cmap='RdYlBu_r', show_totals=True,
                                        title=None, save=None,min_threshold=0.1):
        """
        Create a heatmap to analyze the signaling roles of cell populations (outgoing or incoming contribution)
        Use Marsilea for modern heatmap visualization
        
        Args:
            pattern: str
                'outgoing' for outgoing signaling or 'incoming' for incoming signaling
            signaling: str, list or None
                Specific signaling pathway name. If None, analyze all pathways
            row_scale: bool
                Whether to standardize rows (show relative signaling strength)
            figsize: tuple
                Figure size
            cmap: str
                Heatmap color mapping
            show_totals: bool
                Whether to show total signaling strength bar plots
            title: str or None
                Figure title
            save: str or None
                Save path
            
        Returns:
            h: marsilea plot object
            axes: list containing marsilea object (for compatibility)
            signaling_matrix: pandas.DataFrame
                Signaling strength matrix
        """
        # Use new Marsilea implementation to replace old matplotlib implementation
        h, df = self.netVisual_signaling_heatmap(
            pattern=pattern,
            signaling=signaling,
            min_threshold=min_threshold,
            cmap=cmap,
            figsize=figsize,
            show_bars=show_totals,
            show_colors=True,
            fontsize=10,
            title=title,
            save=save
        )
        
        # If row standardization is needed, reprocess data
        if row_scale:
            from scipy.stats import zscore
            import pandas as pd
            import marsilea as ma
            import marsilea.plotter as mp
            
            # Get original signaling matrix and perform z-score standardization
            cell_matrix = self.get_signaling_matrix(
                level="cell_type", 
                pattern=pattern, 
                signaling=signaling
            )
            
            df_raw = cell_matrix.T  # Transpose: pathway x cell type
            df_scaled = df_raw.apply(zscore, axis=1).fillna(0)
            
            # Recreate standardized heatmap
            cell_colors = self._get_cell_type_colors()
            colors = [cell_colors.get(ct, '#1f77b4') for ct in df_scaled.columns]
            
            h = ma.Heatmap(df_scaled, linewidth=1, width=figsize[0], height=figsize[1], cmap=cmap)
            h.add_left(mp.Labels(df_scaled.index, fontsize=10), pad=0.1)
            h.add_bottom(mp.Colors(df_scaled.columns, palette=cell_colors), size=0.15, pad=0.02)
            h.add_bottom(mp.Labels(df_scaled.columns, fontsize=10), pad=0.1)
            
            if show_totals:
                h.add_right(mp.Bar(df_raw.mean(axis=1), color='#c2c2c2'), pad=0.1)
                h.add_top(mp.Bar(df_raw.mean(axis=0), palette=colors), pad=0.1)
            
            if title:
                h.add_title(title, fontsize=12, pad=0.02)
            elif title is None:
                direction = "Outgoing" if pattern == "outgoing" else "Incoming"
                h.add_title(f"{direction} Signaling Role Analysis", fontsize=12, pad=0.02)
            
            h.render()
            
            if save:
                h.fig.savefig(save, dpi=300, bbox_inches='tight')
                print(f"Signaling role heatmap saved as: {save}")
            
            df = df_scaled
        
        # For compatibility, return structure similar to original function
        return h, [h], df
    
    
    
    def get_signaling_matrix(self, pattern="outgoing", signaling=None, 
                           aggregation="mean", normalize=False, level="cell_type"):
        """
        Get signaling strength matrix
        
        Args:
            pattern: str
                'outgoing', 'incoming', or 'overall'
            signaling: str, list or None
                Specific signaling pathway name. If None, analyze all pathways
            aggregation: str
                Aggregation method: 'mean', 'sum', 'max'
            normalize: bool
                Whether to normalize each row
            level: str
                'cell_type' for cell type level or 'cell' for individual cell level
            
        Returns:
            matrix_df: pandas.DataFrame
                Signaling strength matrix (cell_type/cell x pathway)
        """
        import pandas as pd
        
        # Get all signaling pathways
        if 'classification' not in self.adata.var.columns:
            raise ValueError("'classification' column not found in adata.var")
        
        all_pathways = self.adata.var['classification'].unique()
        
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            pathways = [p for p in signaling if p in all_pathways]
            if not pathways:
                raise ValueError(f"No valid signaling pathways found: {signaling}")
        else:
            pathways = all_pathways
        
        if level == "cell_type":
            return self._get_celltype_signaling_matrix(pattern, pathways, aggregation, normalize)
        elif level == "cell":
            return self._get_cell_signaling_matrix(pattern, pathways, aggregation, normalize)
        else:
            raise ValueError("level must be 'cell_type' or 'cell'")
    
    def _get_celltype_signaling_matrix(self, pattern, pathways, aggregation, normalize):
        """Get cell type level signaling matrix"""
        import pandas as pd
        
        result_data = []
        
        for cell_type in self.cell_types:
            cell_data = {'cell_type': cell_type}
            
            for pathway in pathways:
                # Filter interactions for this pathway
                pathway_mask = self.adata.var['classification'] == pathway
                pathway_indices = np.where(pathway_mask)[0]
                
                if len(pathway_indices) == 0:
                    cell_data[pathway] = 0
                    continue
                
                # Get mean matrix for this pathway
                if 'means' in self.adata.layers:
                    means = self.adata.layers['means'][:, pathway_indices]
                else:
                    means = self.adata.X[:, pathway_indices]
                
                # Calculate signaling strength based on pattern
                if pattern == "outgoing":
                    mask = self.adata.obs['sender'] == cell_type
                elif pattern == "incoming":
                    mask = self.adata.obs['receiver'] == cell_type
                elif pattern == "overall":
                    mask = (self.adata.obs['sender'] == cell_type) | \
                           (self.adata.obs['receiver'] == cell_type)
                else:
                    raise ValueError("pattern must be 'outgoing', 'incoming', or 'overall'")
                
                if np.any(mask):
                    pathway_data = means[mask, :]
                    
                    if aggregation == "mean":
                        strength = np.mean(pathway_data)
                    elif aggregation == "sum":
                        strength = np.sum(pathway_data)
                    elif aggregation == "max":
                        strength = np.max(pathway_data)
                    else:
                        raise ValueError("aggregation must be 'mean', 'sum', or 'max'")
                else:
                    strength = 0
                
                cell_data[pathway] = strength
            
            result_data.append(cell_data)
        
        # Create DataFrame
        matrix_df = pd.DataFrame(result_data)
        matrix_df = matrix_df.set_index('cell_type')
        
        # Normalize
        if normalize:
            matrix_df = matrix_df.div(matrix_df.max(axis=1), axis=0).fillna(0)
        
        return matrix_df
    
    def _get_cell_signaling_matrix(self, pattern, pathways, aggregation, normalize):
        """Get single cell level signaling matrix"""
        import pandas as pd
        
        # Check if cell identifier exists
        if 'cell_id' not in self.adata.obs.columns:
            # If there is no cell_id column, use index as cell identifier
            if hasattr(self.adata.obs.index, 'name') and self.adata.obs.index.name:
                cell_ids = self.adata.obs.index.tolist()
            else:
                cell_ids = [f"Cell_{i}" for i in range(len(self.adata.obs))]
        else:
            cell_ids = self.adata.obs['cell_id'].tolist()
        
        # Get signaling strength for each cell
        result_data = []
        
        for i, cell_id in enumerate(cell_ids):
            cell_data = {'cell_id': cell_id}
            
            # Get the cell type for this cell
            if pattern == "outgoing":
                cell_type = self.adata.obs['sender'].iloc[i]
            elif pattern == "incoming":
                cell_type = self.adata.obs['receiver'].iloc[i]
            else:
                # For overall mode, consider the cell as both sender and receiver
                sender_type = self.adata.obs['sender'].iloc[i]
                receiver_type = self.adata.obs['receiver'].iloc[i]
                cell_type = f"{sender_type}-{receiver_type}"  # Combined identifier
            
            cell_data['cell_type'] = cell_type
            
            for pathway in pathways:
                # Filter interactions for this pathway
                pathway_mask = self.adata.var['classification'] == pathway
                pathway_indices = np.where(pathway_mask)[0]
                
                if len(pathway_indices) == 0:
                    cell_data[pathway] = 0
                    continue
                
                # Get signaling strength for this pathway
                if 'means' in self.adata.layers:
                    means = self.adata.layers['means'][i, pathway_indices]
                else:
                    means = self.adata.X[i, pathway_indices]
                
                # Calculate signaling strength based on aggregation method
                if aggregation == "mean":
                    strength = np.mean(means)
                elif aggregation == "sum":
                    strength = np.sum(means)
                elif aggregation == "max":
                    strength = np.max(means)
                else:
                    raise ValueError("aggregation must be 'mean', 'sum', or 'max'")
                
                cell_data[pathway] = strength
            
            result_data.append(cell_data)
        
        # Create DataFrame
        matrix_df = pd.DataFrame(result_data)
        matrix_df = matrix_df.set_index('cell_id')
        
        # Normalize
        if normalize:
            # Only normalize pathway columns
            pathway_cols = [col for col in matrix_df.columns if col != 'cell_type']
            matrix_df[pathway_cols] = matrix_df[pathway_cols].div(
                matrix_df[pathway_cols].max(axis=1), axis=0
            ).fillna(0)
        
        return matrix_df
    
    def netVisual_signaling_heatmap(self, pattern="incoming", signaling=None, 
                                  min_threshold=0.1, cmap='Greens',
                                  figsize=(4, 4), show_bars=True,
                                  show_colors=True, fontsize=10, 
                                  title=None, save=None):
        """
        Use Marsilea to create a signaling pathway heatmap, showing signaling strength of cell types
        
        Args:
            pattern: str
                'outgoing', 'incoming', or 'overall'
            signaling: str, list or None
                Specific signaling pathway name. If None, analyze all pathways
            min_threshold: float
                Minimum signaling strength threshold, pathways below this value will be filtered
            cmap: str
                Heatmap color mapping
            figsize: tuple
                Figure size (width, height)
            show_bars: bool
                Whether to show marginal bar plots
            show_colors: bool
                Whether to show cell type color bar
            fontsize: int
                Font size
            title: str or None
                Figure title
            save: str or None
                Save path
            
        Returns:
            h: marsilea plot object
            df: pandas.DataFrame
                Filtered signaling strength matrix
        """
        try:
            import marsilea as ma
            import marsilea.plotter as mp
            import scanpy as sc
        except ImportError:
            raise ImportError("marsilea and scanpy packages are required. Please install them: pip install marsilea scanpy")
        
        # Get signaling matrix (cell type x pathway)
        cell_matrix = self.get_signaling_matrix(
            level="cell_type", 
            pattern=pattern, 
            signaling=signaling
        )
        
        # Create AnnData object for filtering
        ad_signal = sc.AnnData(cell_matrix)
        ad_signal.var['mean'] = ad_signal.X.mean(axis=0)
        ad_signal.var['min'] = ad_signal.X.min(axis=0)
        
        # Filter pathways with low signaling strength
        valid_pathways = ad_signal.var['min'][ad_signal.var['min'] > min_threshold].index
        
        if len(valid_pathways) == 0:
            raise ValueError(f"No pathways found with minimum signal strength > {min_threshold}")
        
        # Get filtered data matrix (transpose: pathway x cell type)
        df = ad_signal[:, valid_pathways].to_df().T
        
        # Get cell type colors
        cell_colors = self._get_cell_type_colors()
        colors = [cell_colors.get(ct, '#1f77b4') for ct in df.columns]
        
        # Create main heatmap
        h = ma.Heatmap(
            df, 
            linewidth=1,
            width=figsize[0],
            height=figsize[1],
            cmap=cmap,
        )
        
        # Add pathway labels (left)
        h.add_left(mp.Labels(df.index, fontsize=fontsize), pad=0.1)
        
        # Add cell type color bar (bottom)
        if show_colors:
            h.add_bottom(
                mp.Colors(df.columns, palette=cell_colors), 
                size=0.15, 
                pad=0.02
            )
        
        # Add cell type labels (bottom)
        h.add_bottom(mp.Labels(df.columns, fontsize=fontsize), pad=0.1)
        
        # Add marginal bar plots
        if show_bars:
            # Right: mean signaling strength for each pathway
            h.add_right(
                mp.Bar(df.mean(axis=1), color='#c2c2c2'), 
                pad=0.1
            )
            
            # Top: mean signaling strength for each cell type (using cell type colors)
            h.add_top(
                mp.Bar(df.mean(axis=0), palette=colors), 
                pad=0.1
            )
        
        # Add title
        if title:
            h.add_title(title, fontsize=fontsize + 2, pad=0.02)
        elif title is None:
            direction = {"outgoing": "Outgoing", "incoming": "Incoming", "overall": "Overall"}
            auto_title = f"{direction.get(pattern, pattern.title())} Signaling Heatmap"
            h.add_title(auto_title, fontsize=fontsize + 2, pad=0.02)
        
        # Render figure
        h.render()
        
        # Save figure
        if save:
            h.fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Signaling heatmap saved as: {save}")
        
        # Print statistics
        print(f"📊 Heatmap statistics:")
        print(f"   - Number of pathways: {len(df.index)}")
        print(f"   - Number of cell types: {len(df.columns)}")
        print(f"   - Signal strength range: {df.values.min():.3f} - {df.values.max():.3f}")
        
        return h, df
    
    def netAnalysis_contribution(self, signaling, group_celltype=None, 
                               sources=None, targets=None,
                               pvalue_threshold=0.05, top_pairs=10,
                               figsize=(12, 8), font_size=10,
                               title=None, save=None):
        """
        分析特定信号通路中配体-受体对的贡献
        回答：哪些信号对特定细胞群的传出或传入信号贡献最大
        
        Args:
            signaling: str or list
                要分析的信号通路
            group_celltype: str or None
                要分析的特定细胞类型。如果为None，分析所有细胞类型
            sources: list or None
                关注的发送者细胞类型
            targets: list or None
                关注的接收者细胞类型
            pvalue_threshold: float
                P-value threshold
            top_pairs: int
                显示前N个贡献最大的配体-受体对
            figsize: tuple
                图形大小
            font_size: int
                字体大小
            title: str or None
                图形标题
            save: str or None
                保存路径
            
        Returns:
            fig: matplotlib.figure.Figure
            contribution_df: pandas.DataFrame
                贡献分析结果
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # 收集配体-受体贡献数据
        contributions = []
        
        # 筛选信号通路
        if 'classification' in self.adata.var.columns:
            pathway_mask = self.adata.var['classification'].isin(signaling)
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                raise ValueError(f"No interactions found for signaling pathway(s): {signaling}")
        else:
            raise ValueError("'classification' column not found in adata.var")
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # 过滤细胞类型
            if sources and sender not in sources:
                continue
            if targets and receiver not in targets:
                continue
            if group_celltype and (sender != group_celltype and receiver != group_celltype):
                continue
            
            # 获取显著交互
            pvals = self.adata.layers['pvalues'][i, pathway_indices]
            means = self.adata.layers['means'][i, pathway_indices]
            
            sig_mask = pvals < pvalue_threshold
            
            if np.any(sig_mask):
                # 获取显著交互信息
                original_indices = pathway_indices[sig_mask]
                
                for idx, (p_val, mean_val) in enumerate(zip(pvals[sig_mask], means[sig_mask])):
                    original_idx = original_indices[idx]
                    
                    # 获取配体-受体对信息
                    if 'gene_a' in self.adata.var.columns and 'gene_b' in self.adata.var.columns:
                        ligand = self.adata.var['gene_a'].iloc[original_idx]
                        receptor = self.adata.var['gene_b'].iloc[original_idx]
                        if pd.isna(ligand) or pd.isna(receptor):
                            continue
                        lr_pair = f"{ligand}_{receptor}"
                    else:
                        lr_pair = self.adata.var.index[original_idx]
                    
                    pathway = self.adata.var['classification'].iloc[original_idx]
                    
                    contributions.append({
                        'sender': sender,
                        'receiver': receiver,
                        'lr_pair': lr_pair,
                        'ligand': ligand if 'gene_a' in self.adata.var.columns else lr_pair.split('_')[0],
                        'receptor': receptor if 'gene_b' in self.adata.var.columns else lr_pair.split('_')[1],
                        'pathway': pathway,
                        'pvalue': p_val,
                        'mean_expression': mean_val,
                        'contribution': mean_val * (-np.log10(p_val + 1e-10))  # 综合表达和显著性
                    })
        
        if not contributions:
            print("No significant contributions found for the specified conditions")
            return None, None
        
        # 创建DataFrame并分析
        df_contrib = pd.DataFrame(contributions)
        
        # 按配体-受体对聚合贡献
        lr_contrib = df_contrib.groupby('lr_pair').agg({
            'contribution': 'sum',
            'mean_expression': 'mean',
            'pvalue': 'min',
            'pathway': 'first',
            'ligand': 'first',
            'receptor': 'first'
        }).reset_index()
        
        # 排序并选择前N个
        lr_contrib = lr_contrib.sort_values('contribution', ascending=False).head(top_pairs)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：配体-受体对贡献排序
        bars1 = ax1.barh(range(len(lr_contrib)), lr_contrib['contribution'], 
                        color='skyblue', alpha=0.7)
        ax1.set_yticks(range(len(lr_contrib)))
        ax1.set_yticklabels(lr_contrib['lr_pair'], fontsize=font_size)
        ax1.set_xlabel('Contribution Score', fontsize=font_size + 2)
        ax1.set_title('Top Contributing L-R Pairs', fontsize=font_size + 2)
        ax1.grid(axis='x', alpha=0.3)
        
        # 右图：按信号通路分组的贡献
        pathway_contrib = df_contrib.groupby('pathway')['contribution'].sum().sort_values(ascending=False)
        bars2 = ax2.bar(range(len(pathway_contrib)), pathway_contrib.values, 
                       color='lightcoral', alpha=0.7)
        ax2.set_xticks(range(len(pathway_contrib)))
        ax2.set_xticklabels(pathway_contrib.index, rotation=45, ha='right', fontsize=font_size)
        ax2.set_ylabel('Total Contribution', fontsize=font_size + 2)
        ax2.set_title('Contribution by Pathway', fontsize=font_size + 2)
        ax2.grid(axis='y', alpha=0.3)
        
        # 总标题
        if title is None:
            title = f"Signal Contribution Analysis"
            if group_celltype:
                title += f" - {group_celltype}"
            if sources or targets:
                title += f" ({len(signaling)} pathway(s))"
        
        fig.suptitle(title, fontsize=font_size + 4, y=0.95)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Contribution analysis saved as: {save}")
        
        print(f"📈 贡献分析结果:")
        print(f"   - 分析信号通路: {signaling}")
        print(f"   - 顶级贡献L-R对: {lr_contrib.iloc[0]['lr_pair']} (score: {lr_contrib.iloc[0]['contribution']:.3f})")
        print(f"   - 主要贡献通路: {pathway_contrib.index[0]} (total: {pathway_contrib.iloc[0]:.3f})")
        
        return fig, lr_contrib
    
    def netAnalysis_signalingRole_network_marsilea(self, signaling=None, measures=None,
                                                  color_heatmap="RdYlBu_r", 
                                                  width=12, height=6, font_size=10,
                                                  title="Signaling Role Analysis",
                                                  add_dendrogram=True, add_cell_colors=True,
                                                  add_importance_bars=True, show_values=True,
                                                  save=None,return_df=False,label_rotation=45):
        """
        使用Marsilea创建高级信号角色热图（CellChat风格的netAnalysis_signalingRole_network）
        
        Args:
            signaling: str, list or None
                特定信号通路名称。如果为None，使用存储的中心性结果或计算聚合网络
            measures: list or None
                要显示的中心性指标。默认显示所有指标
            color_heatmap: str
                热图颜色映射
            width: float
                图形宽度
            height: float
                图形高度
            font_size: int
                字体大小
            title: str
                图形标题
            add_dendrogram: bool
                是否添加聚类树
            add_cell_colors: bool
                是否添加细胞类型颜色条
            add_importance_bars: bool
                是否添加Importance值的柱状图
            show_values: bool
                是否在热图中显示数值
            save: str or None
                保存路径
            
        Returns:
            h: marsilea plot object
        """
        if not MARSILEA_AVAILABLE:
            raise ImportError("marsilea package is not available. Please install it: pip install marsilea")
        
        # 如果没有预计算的中心性分数，先计算
        if not hasattr(self, 'centrality_scores') or signaling is not None:
            self.netAnalysis_computeCentrality(signaling=signaling)
        
        centrality_scores = self.centrality_scores
        
        # 选择要显示的指标
        if measures is None:
            measures = ['outdegree', 'indegree', 'flow_betweenness', 'information']
        
        # 验证指标
        available_measures = list(centrality_scores.keys())
        invalid_measures = [m for m in measures if m not in available_measures]
        if invalid_measures:
            print(f"Warning: Invalid measures {invalid_measures}. Available: {available_measures}")
            measures = [m for m in measures if m in available_measures]
        
        if not measures:
            raise ValueError("No valid measures specified")
        
        # 创建数据矩阵（使用CellChat风格的Importance值）
        data_matrix = np.array([centrality_scores[measure] for measure in measures])
        
        # 创建标签映射
        measure_labels = {
            'outdegree': 'Outdegree',
            'indegree': 'Indegree', 
            'flow_betweenness': 'Flow Betweenness',
            'information': 'Information',
            'overall': 'Overall'
        }
        
        row_labels = [measure_labels.get(m, m) for m in measures]
        col_labels = self.cell_types
        
        # 创建DataFrame便于可视化
        df_centrality = pd.DataFrame(data_matrix, 
                                   index=row_labels, 
                                   columns=col_labels)
        
        # 创建Marsilea热图 - 修复API兼容性
        h = ma.Heatmap(
            df_centrality,
            cmap=color_heatmap,
            label="Importance",
            width=width * 0.6,
            height=height * 0.7,
            linewidth=0.5,
            vmin=0,
            vmax=1
        )
        
        # 如果支持，添加数值显示
        if show_values:
            try:
                # 尝试添加文本层显示数值
                text_matrix = df_centrality.values
                text_array = np.array([[f"{val:.2f}" for val in row] for row in text_matrix])
                h.add_layer(ma.plotter.TextMesh(text_array, fontsize=font_size-2, color="white"))
            except:
                print("Warning: Failed to add text values to heatmap")
        
        # 添加细胞类型颜色条
        if add_cell_colors:
            cell_colors = self._get_cell_type_colors()
            col_colors = [cell_colors.get(ct, '#808080') for ct in self.cell_types]
            
            try:
                h.add_top(
                    ma.plotter.Colors(
                        col_labels,
                        palette=col_colors
                    ),
                    size=0.15,
                    pad=0.02
                )
            except Exception as e:
                print(f"Warning: Failed to add cell colors: {e}")
        
        # 添加Importance值柱状图
        if add_importance_bars:
            try:
                # 右侧：每个指标的最大Importance值
                max_importance_per_measure = np.array([centrality_scores[measure].max() for measure in measures])
                h.add_right(
                    ma.plotter.Numbers(
                        max_importance_per_measure,
                        color="#E74C3C",
                        label="Max\nImportance"
                    ),
                    size=0.2,
                    pad=0.05
                )
                
                # 顶部：每个细胞类型的平均Importance值
                avg_importance_per_cell = np.array([np.mean([centrality_scores[measure][i] for measure in measures]) 
                                                  for i in range(len(self.cell_types))])
                h.add_top(
                    ma.plotter.Numbers(
                        avg_importance_per_cell,
                        color="#3498DB",
                        label="Avg Importance"
                    ),
                    size=0.2,
                    pad=0.02
                )
            except Exception as e:
                print(f"Warning: Failed to add importance bars: {e}")
        
        # 添加聚类树
        if add_dendrogram:
            try:
                # 行聚类（指标聚类）
                h.add_dendrogram("left", colors="#2ECC71")
                # 列聚类（细胞类型聚类）
                h.add_dendrogram("top", colors="#9B59B6")
            except Exception as e:
                print(f"Warning: Failed to add dendrograms: {e}")
        
        # 添加角色标签说明
        try:
            role_descriptions = {
                'Outdegree': 'Senders',
                'Indegree': 'Receivers',
                'Flow Betweenness': 'Mediators',
                'Information': 'Influencers'
            }
            
            role_labels = [role_descriptions.get(label, label) for label in row_labels]
            h.add_left(
                ma.plotter.Labels(
                    role_labels,
                    rotation=0,
                    fontsize=font_size
                ),
                size=0.3,
                pad=0.02
            )
        except Exception as e:
            print(f"Warning: Failed to add role labels: {e}")
        
        # 添加细胞类型标签
        try:
            h.add_bottom(
                ma.plotter.Labels(
                    col_labels,
                    rotation=label_rotation,
                    fontsize=font_size
                ),
                size=0.3,
                pad=0.02
            )
        except Exception as e:
            print(f"Warning: Failed to add cell type labels: {e}")
        
        # 添加图例
        try:
            h.add_legends()
        except Exception as e:
            print(f"Warning: Failed to add legends: {e}")
        
        # 添加标题
        try:
            h.add_title(title, fontsize=font_size + 4, pad=0.02)
        except Exception as e:
            print(f"Warning: Failed to add title: {e}")
        
        # 设置边距
        try:
            h.set_margin(0.1)
        except Exception as e:
            print(f"Warning: Failed to set margin: {e}")
        
        # 渲染图形
        h.render()
        
        # 保存图形
        if save:
            try:
                h.save(save, dpi=300)
                print(f"Marsilea signaling role heatmap saved as: {save}")
            except Exception as e:
                print(f"Warning: Failed to save figure: {e}")
        
        # 输出分析结果
        print(f"📊 信号角色分析结果（Marsilea可视化，Importance值 0-1）:")
        for measure in measures:
            scores = centrality_scores[measure]
            if scores.max() > 0:
                top_cell_idx = np.argmax(scores)
                top_cell = self.cell_types[top_cell_idx]
                top_score = scores[top_cell_idx]
                role_description = {
                    'outdegree': 'Dominant Sender',
                    'indegree': 'Dominant Receiver',
                    'flow_betweenness': 'Mediator',
                    'information': 'Influencer',
                    'overall': 'Overall Leader'
                }
                print(f"   - {role_description.get(measure, measure)}: {top_cell} (Importance: {top_score:.3f})")
        
        if return_df:
            return h, df_centrality
        else:
            return h
    
    def demo_curved_arrows(self, signaling_pathway=None, curve_strength=0.4, figsize=(12, 10)):
        """
        演示弯曲箭头效果的示例函数
        
        Args:
            signaling_pathway: str or None
                要可视化的信号通路，如果为None则使用聚合网络
            curve_strength: float
                箭头弯曲强度 (0-1), 0为直线，越大越弯曲
            figsize: tuple
                图片大小
        
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        print("🌸 演示CellChat风格的弯曲箭头效果...")
        print(f"📏 弯曲强度: {curve_strength} (推荐范围: 0.2-0.6)")
        
        if signaling_pathway is not None:
            # 可视化特定信号通路
            fig, ax = self.netVisual_aggregate(
                signaling=signaling_pathway,
                layout='circle',
                focused_view=True,
                use_curved_arrows=True,
                curve_strength=curve_strength,
                figsize=figsize,
                use_sender_colors=True
            )
            print(f"✨ 已生成信号通路 '{signaling_pathway}' 的弯曲箭头网络图")
        else:
            # 可视化聚合网络
            _, weight_matrix = self.compute_aggregated_network()
            fig, ax = self.netVisual_circle_focused(
                matrix=weight_matrix,
                title="Cell-Cell Communication Network (Curved Arrows)",
                use_curved_arrows=True,
                curve_strength=curve_strength,
                figsize=figsize,
                use_sender_colors=True
            )
            print("✨ 已生成聚合网络的弯曲箭头图")
        
        print("💡 提示：")
        print("  - curve_strength=0.2: 轻微弯曲")
        print("  - curve_strength=0.4: 中等弯曲（推荐）") 
        print("  - curve_strength=0.6: 强烈弯曲")
        print("  - use_curved_arrows=False: 切换回直线箭头")
        
        return fig, ax
    
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
        计算通路级别的细胞通讯强度（类似CellChat的方法）
        
        Args:
            method: str
                聚合方法: 'mean', 'sum', 'max', 'median' (default: 'mean')
            min_lr_pairs: int
                通路中最少L-R对数量 (default: 1)  
            min_expression: float
                最小表达阈值 (default: 0.1)
            
        Returns:
            pathway_communication: dict
                包含每个通路的通讯矩阵和统计信息
        """
        pathways = [p for p in self.adata.var['classification'].unique() if pd.notna(p)]
        pathway_communication = {}
        
        print(f"🔬 计算{len(pathways)}个通路的细胞通讯强度...")
        print(f"   - 聚合方法: {method}")
        print(f"   - 最小表达阈值: {min_expression}")
        
        for pathway in pathways:
            pathway_mask = self.adata.var['classification'] == pathway
            pathway_lr_pairs = self.adata.var.loc[pathway_mask, 'interacting_pair'].tolist()
            
            if len(pathway_lr_pairs) < min_lr_pairs:
                continue
                
            # 初始化通路通讯矩阵
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_pvalue_matrix = np.ones((self.n_cell_types, self.n_cell_types))
            valid_interactions_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # 获取该通路在这对细胞间的所有L-R对数据
                pathway_means = self.adata.layers['means'][i, pathway_mask]
                pathway_pvals = self.adata.layers['pvalues'][i, pathway_mask]
                
                # 过滤低表达的交互
                valid_mask = pathway_means >= min_expression
                
                if np.any(valid_mask):
                    valid_means = pathway_means[valid_mask]
                    valid_pvals = pathway_pvals[valid_mask]
                    
                    # 计算通路级别的通讯强度
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
                    
                    # 计算通路级别的p-value（使用最小p-value作为通路显著性）
                    pathway_pval = np.min(valid_pvals)
                    
                    pathway_matrix[sender_idx, receiver_idx] = pathway_strength
                    pathway_pvalue_matrix[sender_idx, receiver_idx] = pathway_pval
                    valid_interactions_matrix[sender_idx, receiver_idx] = len(valid_means)
                else:
                    # 没有有效的交互
                    pathway_matrix[sender_idx, receiver_idx] = 0
                    pathway_pvalue_matrix[sender_idx, receiver_idx] = 1.0
                    valid_interactions_matrix[sender_idx, receiver_idx] = 0
            
            # 存储通路通讯结果
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
        
        print(f"✅ 完成通路通讯强度计算，共{len(pathway_communication)}个通路")
        
        return pathway_communication
    
    def get_significant_pathways_v2(self, pathway_communication=None, 
                                   strength_threshold=0.1, pvalue_threshold=0.05, 
                                   min_significant_pairs=1):
        """
        基于通路级别通讯强度判断显著通路（更符合CellChat逻辑）
        
        Args:
            pathway_communication: dict or None
                通路通讯结果，如果为None则重新计算
            strength_threshold: float
                通路强度阈值 (default: 0.1)
            pvalue_threshold: float  
                p-value阈值 (default: 0.05)
            min_significant_pairs: int
                最少显著细胞对数量 (default: 1)
            
        Returns:
            significant_pathways: list
                显著通路列表
            pathway_summary: pd.DataFrame
                通路统计摘要
        """
        if pathway_communication is None:
            pathway_communication = self.compute_pathway_communication()
        
        pathway_summary_data = []
        significant_pathways = []
        
        for pathway, data in pathway_communication.items():
            comm_matrix = data['communication_matrix']
            pval_matrix = data['pvalue_matrix']
            
            # 通路级别统计
            total_strength = data['total_strength']
            max_strength = data['max_strength']
            mean_strength = data['mean_strength']
            
            # 使用.values确保返回numpy数组而不是pandas Series
            pval_values = pval_matrix.values
            comm_values = comm_matrix.values
            
            n_significant_pairs = np.sum((pval_values < pvalue_threshold) & (comm_values >= strength_threshold))
            n_total_pairs = np.sum(comm_values > 0)
            
            # 判断通路是否显著
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
        
        # 创建摘要DataFrame
        pathway_summary = pd.DataFrame(pathway_summary_data)
        pathway_summary = pathway_summary.sort_values('total_strength', ascending=False)
        
        print(f"📊 通路显著性分析结果:")
        print(f"   - 总通路数: {len(pathway_summary)}")
        print(f"   - 显著通路数: {len(significant_pathways)}")
        print(f"   - 强度阈值: {strength_threshold}")
        print(f"   - p-value阈值: {pvalue_threshold}")
        
        # 显示top通路
        print(f"\n🏆 Top 10通路按总强度排序:")
        print("-" * 100)
        print(f"{'Pathway':<30} {'Total':<8} {'Max':<7} {'Mean':<7} {'L-R':<4} {'Active':<6} {'Sig':<4} {'Rate':<6} {'Status'}")
        print("-" * 100)
        
        for _, row in pathway_summary.head(10).iterrows():
            status = "***" if row['is_significant'] else "   "
            print(f"{row['pathway'][:28]:<30} {row['total_strength']:<8.2f} {row['max_strength']:<7.2f} "
                  f"{row['mean_strength']:<7.2f} {row['n_lr_pairs']:<4} {row['n_active_cell_pairs']:<6} "
                  f"{row['n_significant_pairs']:<4} {row['significance_rate']:<6.2f} {status}")
        
        return significant_pathways, pathway_summary
    
    def netAnalysis_contribution(self, signaling, pvalue_threshold=0.05, 
                                mean_threshold=0.1, top_pairs=10, 
                                figsize=(10, 6), save=None):
        """
        Calculate the contribution of each ligand-receptor pair to the overall signaling pathway and visualize
        (Similar to CellChat's netAnalysis_contribution function)
        
        Args:
            signaling: str or list
                Signaling pathway name
            pvalue_threshold: float
                P-value threshold (default: 0.05)
            mean_threshold: float  
                Mean expression threshold (default: 0.1)
            top_pairs: int
                Number of top L-R pairs to display (default: 10)
            figsize: tuple
                Figure size (default: (10, 6))
            save: str or None
                Save path (default: None)
            
        Returns:
            contribution_df: pd.DataFrame
                L-R pair contribution statistics
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # Filter interactions for the specified pathway
        pathway_mask = self.adata.var['classification'].isin(signaling)
        if not pathway_mask.any():
            raise ValueError(f"No interactions found for signaling pathway(s): {signaling}")
        
        # Calculate the contribution of each L-R pair
        contributions = []
        
        for var_idx in np.where(pathway_mask)[0]:
            lr_pair = self.adata.var.iloc[var_idx]['interacting_pair']
            gene_a = self.adata.var.iloc[var_idx]['gene_a']
            gene_b = self.adata.var.iloc[var_idx]['gene_b']
            classification = self.adata.var.iloc[var_idx]['classification']
            
            # Calculate total strength and significance for this L-R pair across all cell pairs
            total_strength = 0
            significant_pairs = 0
            active_pairs = 0
            max_strength = 0
            
            for i in range(len(self.adata.obs)):
                pval = self.adata.layers['pvalues'][i, var_idx]
                mean_expr = self.adata.layers['means'][i, var_idx]
                
                if mean_expr > mean_threshold:
                    active_pairs += 1
                    total_strength += mean_expr
                    max_strength = max(max_strength, mean_expr)
                    
                    if pval < pvalue_threshold:
                        significant_pairs += 1
            
            if total_strength > 0:  # Only include active L-R pairs
                contributions.append({
                    'ligand_receptor': lr_pair,
                    'ligand': gene_a,
                    'receptor': gene_b,
                    'pathway': classification,
                    'total_strength': total_strength,
                    'max_strength': max_strength,
                    'mean_strength': total_strength / max(1, active_pairs),
                    'significant_pairs': significant_pairs,
                    'active_pairs': active_pairs,
                    'contribution_score': total_strength * significant_pairs
                })
        
        if not contributions:
            raise ValueError(f"No active L-R pairs found for pathway(s): {signaling}")
        
        # Convert to DataFrame and sort
        contribution_df = pd.DataFrame(contributions)
        contribution_df = contribution_df.sort_values('contribution_score', ascending=False)
        
        # 计算相对贡献百分比
        total_contribution = contribution_df['contribution_score'].sum()
        contribution_df['contribution_percent'] = (contribution_df['contribution_score'] / total_contribution) * 100
        
        # 选择top pairs
        top_df = contribution_df.head(top_pairs)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：贡献百分比条形图
        bars = ax1.barh(range(len(top_df)), top_df['contribution_percent'], 
                       color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_yticks(range(len(top_df)))
        ax1.set_yticklabels(top_df['ligand_receptor'], fontsize=10)
        ax1.set_xlabel('Contribution Percentage (%)')
        ax1.set_title(f'L-R Pair Contribution\n{" & ".join(signaling)}')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, percent) in enumerate(zip(bars, top_df['contribution_percent'])):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{percent:.1f}%', va='center', fontsize=9)
        
        # 右图：显著性 vs 强度散点图
        scatter = ax2.scatter(top_df['total_strength'], top_df['significant_pairs'], 
                            s=top_df['active_pairs']*20, 
                            c=top_df['contribution_percent'], 
                            cmap='viridis', alpha=0.7, edgecolors='black')
        
        # 添加L-R对标签
        for _, row in top_df.iterrows():
            ax2.annotate(row['ligand_receptor'], 
                        (row['total_strength'], row['significant_pairs']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        ax2.set_xlabel('Total Expression Strength')
        ax2.set_ylabel('Number of Significant Cell Pairs')
        ax2.set_title('L-R Pair Activity vs Significance')
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Contribution %')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Contribution analysis saved as: {save}")
        
        return contribution_df, fig, (ax1, ax2)
    
    def extractEnrichedLR(self, signaling, pvalue_threshold=0.05, 
                         mean_threshold=0.1, min_cell_pairs=1,
                         geneLR_return=False):
        """
        Extract all significant L-R pairs in the specified signaling pathway
        (Similar to CellChat's extractEnrichedLR function)
        
        Args:
            signaling: str or list
                Signaling pathway name
            pvalue_threshold: float
                P-value threshold (default: 0.05)
            mean_threshold: float
                Mean expression threshold (default: 0.1)  
            min_cell_pairs: int
                Minimum number of significant cell pairs (default: 1)
            geneLR_return: bool
                Whether to return gene-level information (default: False)
            
        Returns:
            enriched_lr: pd.DataFrame
                Significant L-R pair information
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # Filter interactions for the specified pathway
        pathway_mask = self.adata.var['classification'].isin(signaling)
        if not pathway_mask.any():
            raise ValueError(f"No interactions found for signaling pathway(s): {signaling}")
        
        enriched_pairs = []
        
        for var_idx in np.where(pathway_mask)[0]:
            lr_pair = self.adata.var.iloc[var_idx]['interacting_pair']
            gene_a = self.adata.var.iloc[var_idx]['gene_a']
            gene_b = self.adata.var.iloc[var_idx]['gene_b']
            classification = self.adata.var.iloc[var_idx]['classification']
            
            # Calculate significance statistics
            significant_cell_pairs = []
            total_strength = 0
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                     self.adata.obs['receiver'])):
                pval = self.adata.layers['pvalues'][i, var_idx]
                mean_expr = self.adata.layers['means'][i, var_idx]
                
                if pval < pvalue_threshold and mean_expr > mean_threshold:
                    significant_cell_pairs.append(f"{sender}|{receiver}")
                    total_strength += mean_expr
            
            # Only include L-R pairs that meet the criteria
            if len(significant_cell_pairs) >= min_cell_pairs:
                pair_info = {
                    'ligand_receptor': lr_pair,
                    'ligand': gene_a,
                    'receptor': gene_b, 
                    'pathway': classification,
                    'n_significant_pairs': len(significant_cell_pairs),
                    'total_strength': total_strength,
                    'mean_strength': total_strength / len(significant_cell_pairs),
                    'significant_cell_pairs': significant_cell_pairs
                }
                
                # If gene-level information is needed
                if geneLR_return:
                    # Add more detailed gene information
                    var_info = self.adata.var.iloc[var_idx]
                    for col in var_info.index:
                        if col not in pair_info:
                            pair_info[col] = var_info[col]
                
                enriched_pairs.append(pair_info)
        
        if not enriched_pairs:
            print(f"No enriched L-R pairs found for pathway(s): {signaling}")
            return pd.DataFrame()
        
        # Convert to DataFrame and sort by significance
        enriched_df = pd.DataFrame(enriched_pairs)
        enriched_df = enriched_df.sort_values(['n_significant_pairs', 'total_strength'], 
                                            ascending=[False, False])
        
        print(f"✅ Found {len(enriched_df)} enriched L-R pairs in pathway(s): {', '.join(signaling)}")
        
        return enriched_df
    
    def netVisual_individual(self, signaling, pairLR_use=None, sources_use=None, 
                           targets_use=None, layout='hierarchy', 
                           vertex_receiver=None, pvalue_threshold=0.05,
                           edge_width_max=8, vertex_size_max=50,
                           figsize=(10, 8), title=None, save=None):
        """
        Visualize cell-cell communication mediated by individual ligand-receptor pairs
        (Similar to CellChat's netVisual_individual function)
        
        Args:
            signaling: str or list
                Signaling pathway name
            pairLR_use: str, dict, or pd.Series
                L-R pair to display. Can be:
                - String: L-R pair name (e.g., "TGFB1_TGFBR1")
                - Dictionary: dictionary containing ligand and receptor
                - pandas Series: row returned by extractEnrichedLR
            sources_use: list or None
                Specified sender cell types
            targets_use: list or None  
                Specified receiver cell types
            layout: str
                Layout type: 'hierarchy', 'circle' (default: 'hierarchy')
            vertex_receiver: list or None
                Numeric vector specifying receiver positions (hierarchy layout only)
            pvalue_threshold: float
                Significance threshold (default: 0.05)
            edge_width_max: float
                Maximum edge width (default: 8)
            vertex_size_max: float
                Maximum node size (default: 50)
            figsize: tuple
                Figure size (default: (10, 8))
            title: str or None
                Figure title
            save: str or None
                Save path
            
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # Parse pairLR_use parameter
        if pairLR_use is None:
            # If not specified, select the first enriched L-R pair
            enriched_lr = self.extractEnrichedLR(signaling, pvalue_threshold)
            if enriched_lr.empty:
                raise ValueError(f"No enriched L-R pairs found for {signaling}")
            pairLR_use = enriched_lr.iloc[0]
        
        # Handle different types of pairLR_use input
        if isinstance(pairLR_use, str):
            # Assume ligand_receptor format
            ligand, receptor = pairLR_use.split('_') if '_' in pairLR_use else pairLR_use.split('-')
            lr_pair = pairLR_use
        elif isinstance(pairLR_use, dict):
            ligand = pairLR_use['ligand']
            receptor = pairLR_use['receptor'] 
            lr_pair = pairLR_use.get('ligand_receptor', f"{ligand}_{receptor}")
        elif isinstance(pairLR_use, pd.Series):
            ligand = pairLR_use['ligand']
            receptor = pairLR_use['receptor']
            lr_pair = pairLR_use['ligand_receptor']
        else:
            raise ValueError("pairLR_use must be str, dict, or pandas Series")
        
        # Find the corresponding L-R pair
        lr_mask = (self.adata.var['gene_a'] == ligand) & (self.adata.var['gene_b'] == receptor)
        if signaling:
            pathway_mask = self.adata.var['classification'].isin(signaling)
            lr_mask = lr_mask & pathway_mask
        
        if not lr_mask.any():
            raise ValueError(f"L-R pair {lr_pair} not found in pathway(s) {signaling}")
        
        var_idx = np.where(lr_mask)[0][0]
        
        # Collect significant cell-cell communications
        communications = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                 self.adata.obs['receiver'])):
            # Apply cell type filtering
            if sources_use and sender not in sources_use:
                continue
            if targets_use and receiver not in targets_use:
                continue
                
            pval = self.adata.layers['pvalues'][i, var_idx]
            mean_expr = self.adata.layers['means'][i, var_idx]
            
            if pval < pvalue_threshold and mean_expr > 0:
                communications.append({
                    'sender': sender,
                    'receiver': receiver,
                    'strength': mean_expr,
                    'pvalue': pval
                })
        
        if not communications:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No significant communication found\nfor {lr_pair}', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig, ax
        
        # Create communication DataFrame
        comm_df = pd.DataFrame(communications)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get cell type colors
        cell_colors = self._get_cell_type_colors()
        
        if layout == 'hierarchy':
            self._draw_hierarchy_plot(comm_df, ax, cell_colors, vertex_receiver, 
                                    edge_width_max, vertex_size_max)
        elif layout == 'circle':
            self._draw_circle_plot(comm_df, ax, cell_colors, edge_width_max, vertex_size_max)
        else:
            raise ValueError("layout must be 'hierarchy' or 'circle'")
        
        # Set title
        if title is None:
            title = f"{ligand} → {receptor} Communication\nPathway: {', '.join(signaling)}"
        ax.set_title(title, fontsize=14, pad=20)
        
        # Save
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Individual communication plot saved as: {save}")
        
        return fig, ax
    
    def _draw_hierarchy_plot(self, comm_df, ax, cell_colors, vertex_receiver, 
                           edge_width_max, vertex_size_max):
        """Draw hierarchy plot"""
        # Get unique senders and receivers
        senders = comm_df['sender'].unique()
        receivers = comm_df['receiver'].unique()
        
        # Set positions
        if vertex_receiver is not None:
            # User-specified receiver positions
            y_positions = {}
            for i, receiver in enumerate(receivers):
                if i < len(vertex_receiver):
                    y_positions[receiver] = vertex_receiver[i]
                else:
                    y_positions[receiver] = i + 1
        else:
            # Auto-assign positions
            y_positions = {receiver: i for i, receiver in enumerate(receivers)}
        
        # Sender positions (left side)
        sender_y = np.linspace(0, max(y_positions.values()), len(senders))
        sender_pos = {sender: (0.2, y) for sender, y in zip(senders, sender_y)}
        
        # Receiver positions (right side)
        receiver_pos = {receiver: (0.8, y_positions[receiver]) for receiver in receivers}
        
        # Draw nodes
        max_strength = comm_df['strength'].max()
        
        for sender, (x, y) in sender_pos.items():
            # Calculate node size (based on sending strength)
            sender_strength = comm_df[comm_df['sender'] == sender]['strength'].sum()
            size = (sender_strength / max_strength) * vertex_size_max + 20
            
            color = cell_colors.get(sender, '#lightblue')
            circle = plt.Circle((x, y), 0.05, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            ax.text(x-0.1, y, sender, ha='right', va='center', fontsize=10, weight='bold')
        
        for receiver, (x, y) in receiver_pos.items():
            # Calculate node size (based on receiving strength)
            receiver_strength = comm_df[comm_df['receiver'] == receiver]['strength'].sum()
            size = (receiver_strength / max_strength) * vertex_size_max + 20
            
            color = cell_colors.get(receiver, '#lightcoral')
            circle = plt.Circle((x, y), 0.05, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            ax.text(x+0.1, y, receiver, ha='left', va='center', fontsize=10, weight='bold')
        
        # Draw edges
        for _, row in comm_df.iterrows():
            sender = row['sender']
            receiver = row['receiver']
            strength = row['strength']
            
            if sender in sender_pos and receiver in receiver_pos:
                x1, y1 = sender_pos[sender]
                x2, y2 = receiver_pos[receiver]
                
                # Edge width
                width = (strength / max_strength) * edge_width_max
                
                # Draw arrow
                from matplotlib.patches import FancyArrowPatch
                arrow = FancyArrowPatch((x1+0.05, y1), (x2-0.05, y2),
                                      arrowstyle='->', mutation_scale=20,
                                      linewidth=width, color='gray', alpha=0.6)
                ax.add_patch(arrow)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, max(y_positions.values()) + 0.5)
        ax.axis('off')
    
    def _draw_circle_plot(self, comm_df, ax, cell_colors, edge_width_max, vertex_size_max):
        """Draw circular plot"""
        # Get all unique cell types
        all_cells = list(set(comm_df['sender'].tolist() + comm_df['receiver'].tolist()))
        n_cells = len(all_cells)
        
        # Create circular positions
        angles = np.linspace(0, 2*np.pi, n_cells, endpoint=False)
        positions = {cell: (np.cos(angle), np.sin(angle)) for cell, angle in zip(all_cells, angles)}
        
        # Calculate node sizes
        cell_strengths = {}
        for cell in all_cells:
            send_strength = comm_df[comm_df['sender'] == cell]['strength'].sum()
            receive_strength = comm_df[comm_df['receiver'] == cell]['strength'].sum()
            cell_strengths[cell] = send_strength + receive_strength
        
        max_strength = max(cell_strengths.values()) if cell_strengths else 1
        
        # Draw nodes
        for cell, (x, y) in positions.items():
            size = (cell_strengths[cell] / max_strength) * vertex_size_max + 100
            color = cell_colors.get(cell, '#lightgray')
            
            circle = plt.Circle((x, y), 0.1, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            ax.text(x*1.2, y*1.2, cell, ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw edges
        edge_max = comm_df['strength'].max()
        for _, row in comm_df.iterrows():
            sender = row['sender']
            receiver = row['receiver']
            strength = row['strength']
            
            if sender in positions and receiver in positions:
                x1, y1 = positions[sender]
                x2, y2 = positions[receiver]
                
                # Edge width
                width = (strength / edge_max) * edge_width_max
                
                # Draw curved arrow
                from matplotlib.patches import FancyArrowPatch, ConnectionPatch
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle='->', mutation_scale=15,
                                      linewidth=width, color='red', alpha=0.6,
                                      connectionstyle="arc3,rad=0.2")
                ax.add_patch(arrow)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def compute_communication_prob(self, pvalue_threshold=0.05, normalize=True):
        """
        Calculate cell-cell communication probability matrix (similar to CellChat's prob matrix)
        
        Args:
            pvalue_threshold: float
                P-value threshold for significant interactions
            normalize: bool
                Whether to normalize probabilities
            
        Returns:
            prob_tensor: np.ndarray
                Probability tensor, shape (n_cell_types, n_cell_types, n_pathways)
            pathway_names: list
                List of signaling pathway names
        """
        if 'classification' not in self.adata.var.columns:
            raise ValueError("'classification' column not found in adata.var")
        
        # Get all signaling pathways
        pathway_names = [p for p in self.adata.var['classification'].unique() if pd.notna(p)]
        n_pathways = len(pathway_names)
        
        # Initialize probability tensor (sender, receiver, pathway)
        prob_tensor = np.zeros((self.n_cell_types, self.n_cell_types, n_pathways))
        
        for p_idx, pathway in enumerate(pathway_names):
            # Get interactions for this pathway
            pathway_mask = self.adata.var['classification'] == pathway
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                continue
            
            # Calculate probability matrix for this pathway
            pathway_prob = np.zeros((self.n_cell_types, self.n_cell_types))
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                      self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # Get significant interactions
                pvals = self.adata.layers['pvalues'][i, pathway_indices]
                means = self.adata.layers['means'][i, pathway_indices]
                
                # Calculate probability: mean expression intensity of significant interactions
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    pathway_prob[sender_idx, receiver_idx] = np.mean(means[sig_mask])
            
            # Store in tensor
            prob_tensor[:, :, p_idx] = pathway_prob
        
        # Normalize probabilities (optional)
        if normalize:
            # Normalize each pathway so that probabilities sum to 1
            for p_idx in range(n_pathways):
                prob_sum = prob_tensor[:, :, p_idx].sum()
                if prob_sum > 0:
                    prob_tensor[:, :, p_idx] /= prob_sum
        
        # Store results
        self.prob_tensor = prob_tensor
        self.pathway_names = pathway_names
        
        return prob_tensor, pathway_names
    
    def selectK(self, pattern="outgoing", k_range=range(2, 11), nrun=5, 
               plot_results=True, figsize=(8, 6)):
        """
        选择NMF分解的最优K值（类似CellChat的selectK功能）
        
        Args:
            pattern: str
                'outgoing' or 'incoming'
            k_range: range or list
                要测试的K值范围
            nrun: int
                每个K值运行的次数
            plot_results: bool
                是否绘制评估结果
            figsize: tuple
                图形大小
            
        Returns:
            results: dict
                包含不同K值的评估指标
            optimal_k: int
                推荐的最优K值
        """
        try:
            from sklearn.decomposition import NMF
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn is required for NMF analysis. Please install: pip install scikit-learn")
        
        # 获取概率矩阵
        if not hasattr(self, 'prob_tensor'):
            self.compute_communication_prob()
        
        prob_tensor = self.prob_tensor
        
        # 准备数据矩阵
        if pattern == "outgoing":
            # 聚合为 (sender, pathway)
            data_matrix = np.sum(prob_tensor, axis=1)  # Sum over receivers
        elif pattern == "incoming":
            # 聚合为 (receiver, pathway)
            data_matrix = np.sum(prob_tensor, axis=0)  # Sum over senders
        else:
            raise ValueError("pattern must be 'outgoing' or 'incoming'")
        
        # 归一化：每列除以最大值
        data_matrix = data_matrix / (np.max(data_matrix, axis=0, keepdims=True) + 1e-10)
        
        # 过滤掉全零行
        row_sums = np.sum(data_matrix, axis=1)
        data_matrix = data_matrix[row_sums > 0, :]
        
        if data_matrix.shape[0] < 2:
            raise ValueError("Insufficient data for NMF analysis")
        
        # 评估不同K值
        results = {
            'k_values': [],
            'reconstruction_error': [],
            'silhouette_score': [],
            'explained_variance': []
        }
        
        print(f"🔍 评估K值范围: {list(k_range)}...")
        
        for k in k_range:
            if k >= min(data_matrix.shape):
                continue
                
            k_errors = []
            k_silhouettes = []
            k_variances = []
            
            for run in range(nrun):
                # 运行NMF
                nmf_model = NMF(n_components=k, init='nndsvd', random_state=42+run, 
                               max_iter=1000, alpha_W=0.1, alpha_H=0.1)
                
                try:
                    W = nmf_model.fit_transform(data_matrix)
                    H = nmf_model.components_
                    
                    # 重构误差
                    reconstruction = np.dot(W, H)
                    error = np.linalg.norm(data_matrix - reconstruction, 'fro')
                    k_errors.append(error)
                    
                    # 轮廓系数（基于W矩阵的聚类质量）
                    if k > 1:
                        labels = np.argmax(W, axis=1)
                        if len(np.unique(labels)) > 1:
                            silhouette = silhouette_score(W, labels)
                            k_silhouettes.append(silhouette)
                    
                    # 解释方差
                    total_var = np.var(data_matrix)
                    explained_var = 1 - np.var(data_matrix - reconstruction) / total_var
                    k_variances.append(explained_var)
                    
                except Exception as e:
                    print(f"Warning: NMF failed for k={k}, run={run}: {e}")
                    continue
            
            if k_errors:
                results['k_values'].append(k)
                results['reconstruction_error'].append(np.mean(k_errors))
                results['silhouette_score'].append(np.mean(k_silhouettes) if k_silhouettes else 0)
                results['explained_variance'].append(np.mean(k_variances))
        
        if not results['k_values']:
            raise ValueError("No valid K values found")
        
        # 选择最优K：综合考虑重构误差和轮廓系数
        scores = []
        for i, k in enumerate(results['k_values']):
            # 标准化指标
            error_norm = 1 - (results['reconstruction_error'][i] / max(results['reconstruction_error']))
            silhouette_norm = results['silhouette_score'][i] if results['silhouette_score'][i] > 0 else 0
            variance_norm = results['explained_variance'][i]
            
            # 综合评分
            combined_score = 0.4 * error_norm + 0.3 * silhouette_norm + 0.3 * variance_norm
            scores.append(combined_score)
        
        optimal_idx = np.argmax(scores)
        optimal_k = results['k_values'][optimal_idx]
        
        # 可视化结果
        if plot_results:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # 重构误差
            axes[0, 0].plot(results['k_values'], results['reconstruction_error'], 'bo-')
            axes[0, 0].set_xlabel('Number of patterns (K)')
            axes[0, 0].set_ylabel('Reconstruction Error')
            axes[0, 0].set_title('NMF Reconstruction Error')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 轮廓系数
            axes[0, 1].plot(results['k_values'], results['silhouette_score'], 'ro-')
            axes[0, 1].set_xlabel('Number of patterns (K)')
            axes[0, 1].set_ylabel('Silhouette Score')
            axes[0, 1].set_title('Clustering Quality')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 解释方差
            axes[1, 0].plot(results['k_values'], results['explained_variance'], 'go-')
            axes[1, 0].set_xlabel('Number of patterns (K)')
            axes[1, 0].set_ylabel('Explained Variance')
            axes[1, 0].set_title('Variance Explained')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 综合评分
            axes[1, 1].plot(results['k_values'], scores, 'mo-')
            axes[1, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_xlabel('Number of patterns (K)')
            axes[1, 1].set_ylabel('Combined Score')
            axes[1, 1].set_title(f'Overall Score (Optimal K={optimal_k})')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        print(f"📊 K值选择结果:")
        print(f"   - 推荐最优K值: {optimal_k}")
        print(f"   - 重构误差: {results['reconstruction_error'][optimal_idx]:.4f}")
        print(f"   - 轮廓系数: {results['silhouette_score'][optimal_idx]:.4f}")
        print(f"   - 解释方差: {results['explained_variance'][optimal_idx]:.4f}")
        
        return results, optimal_k
    
    def identifyCommunicationPatterns(self, pattern="outgoing", k=None, 
                                    heatmap_show=True, figsize=(15, 6), 
                                    font_size=10, save=None, 
                                    color_heatmap="RdYlBu_r", title=None):
        """
        识别细胞通信模式使用NMF分解（类似CellChat的identifyCommunicationPatterns功能）
        
        Args:
            pattern: str
                'outgoing' or 'incoming'
            k: int or None
                NMF分解的模式数量，如果为None则需要先运行selectK
            heatmap_show: bool
                是否显示热图
            figsize: tuple
                图形大小
            font_size: int
                字体大小
            save: str or None
                保存路径
            color_heatmap: str
                热图颜色方案
            title: str or None
                图形标题
            
        Returns:
            patterns: dict
                包含细胞模式和信号模式的结果
            fig: matplotlib.figure.Figure or None
                可视化图形
        """
        try:
            from sklearn.decomposition import NMF
            from sklearn.preprocessing import normalize
        except ImportError:
            raise ImportError("scikit-learn is required for NMF analysis. Please install: pip install scikit-learn")
        
        if k is None:
            raise ValueError("Please provide k value or run selectK() first to determine optimal k")
        
        # 获取概率矩阵
        if not hasattr(self, 'prob_tensor'):
            self.compute_communication_prob()
        
        prob_tensor = self.prob_tensor
        
        # 准备数据矩阵
        if pattern == "outgoing":
            # 聚合为 (sender, pathway)
            data_matrix = np.sum(prob_tensor, axis=1)  # Sum over receivers
            cell_labels = self.cell_types
        elif pattern == "incoming":
            # 聚合为 (receiver, pathway)  
            data_matrix = np.sum(prob_tensor, axis=0)  # Sum over senders
            cell_labels = self.cell_types
        else:
            raise ValueError("pattern must be 'outgoing' or 'incoming'")
        
        print(f"🔍 原始数据矩阵形状: {data_matrix.shape}")
        print(f"   - 数据范围: {data_matrix.min():.4f} - {data_matrix.max():.4f}")
        print(f"   - 非零元素比例: {(data_matrix > 0).sum() / data_matrix.size:.2%}")
        
        # 改进的数据预处理
        # 1. 过滤掉全零行和列
        row_sums = np.sum(data_matrix, axis=1)
        col_sums = np.sum(data_matrix, axis=0)
        
        valid_rows = row_sums > 0
        valid_cols = col_sums > 0
        
        data_filtered = data_matrix[valid_rows, :][:, valid_cols]
        valid_cell_labels = [cell_labels[i] for i in range(len(cell_labels)) if valid_rows[i]]
        valid_pathway_names = [self.pathway_names[i] for i in range(len(self.pathway_names)) if valid_cols[i]]
        
        print(f"🔧 过滤后数据形状: {data_filtered.shape}")
        
        if data_filtered.shape[0] < k:
            raise ValueError(f"Not enough valid cell types ({data_filtered.shape[0]}) for k={k} patterns")
        
        if data_filtered.shape[1] < k:
            raise ValueError(f"Not enough valid pathways ({data_filtered.shape[1]}) for k={k} patterns")
        
        # 2. 改进的归一化策略：使用CellChat风格的按行归一化
        # 每行除以该行的最大值，类似CellChat的sweep操作
        row_max = np.max(data_filtered, axis=1, keepdims=True)
        row_max[row_max == 0] = 1  # 避免除零
        data_normalized = data_filtered / row_max
        
        print(f"📊 归一化后数据范围: {data_normalized.min():.4f} - {data_normalized.max():.4f}")
        
        # 3. 添加小量随机噪声避免完全相同的行（这在真实数据中很少见）
        np.random.seed(42)
        noise_level = data_normalized.std() * 0.01  # 1%的噪声
        data_normalized += np.random.normal(0, noise_level, data_normalized.shape)
        data_normalized = np.clip(data_normalized, 0, None)  # 确保非负
        
        # 执行NMF分解
        print(f"🔬 执行NMF分解 (k={k}, pattern={pattern})...")
        
        # 改进的NMF参数：降低正则化，增加迭代次数
        nmf_model = NMF(
            n_components=k, 
            init='nndsvd',  # 更好的初始化
            random_state=42, 
            max_iter=2000,  # 增加迭代次数
            alpha_W=0.01,   # 降低W的正则化
            alpha_H=0.01,   # 降低H的正则化
            beta_loss='frobenius',
            tol=1e-6
        )
        
        W = nmf_model.fit_transform(data_normalized)  # (cells, patterns)
        H = nmf_model.components_  # (patterns, pathways)
        
        print(f"   - 收敛状态: {'已收敛' if nmf_model.n_iter_ < nmf_model.max_iter else '未完全收敛'}")
        print(f"   - 迭代次数: {nmf_model.n_iter_}")
        
        # 检查分解质量
        reconstruction = np.dot(W, H)
        reconstruction_error = np.linalg.norm(data_normalized - reconstruction, 'fro')
        relative_error = reconstruction_error / np.linalg.norm(data_normalized, 'fro')
        
        print(f"   - 重构误差: {reconstruction_error:.4f}")
        print(f"   - 相对误差: {relative_error:.4f}")
        
        # 4. 改进的归一化：采用CellChat的标准化方法
        # W矩阵：每行归一化（每个细胞在所有模式中的贡献和为1）
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        
        # H矩阵：每列归一化（每个通路在所有模式中的贡献和为1）
        H_norm = H / (H.sum(axis=0, keepdims=True) + 1e-10)
        
        # 检查结果的多样性
        pattern_diversity = []
        for i in range(k):
            # 计算每个模式的熵（多样性指标）
            w_entropy = -np.sum(W_norm[:, i] * np.log(W_norm[:, i] + 1e-10))
            h_entropy = -np.sum(H_norm[i, :] * np.log(H_norm[i, :] + 1e-10))
            pattern_diversity.append((w_entropy, h_entropy))
        
        print(f"📈 模式多样性分析:")
        for i, (w_ent, h_ent) in enumerate(pattern_diversity):
            print(f"   - Pattern {i+1}: 细胞多样性={w_ent:.2f}, 通路多样性={h_ent:.2f}")
        
        # 创建模式标签
        pattern_labels = [f"Pattern {i+1}" for i in range(k)]
        
        # 创建结果DataFrame
        cell_patterns_df = pd.DataFrame(
            W_norm, 
            index=valid_cell_labels, 
            columns=pattern_labels
        )
        
        signaling_patterns_df = pd.DataFrame(
            H_norm.T,  # 转置: (pathways, patterns)
            index=valid_pathway_names,
            columns=pattern_labels
        )
        
        # 存储结果
        patterns = {
            'cell': cell_patterns_df,
            'signaling': signaling_patterns_df,
            'W_matrix': W_norm,
            'H_matrix': H_norm,
            'pattern': pattern,
            'k': k,
            'reconstruction_error': reconstruction_error,
            'relative_error': relative_error,
            'pattern_diversity': pattern_diversity,
            'valid_cells': valid_cell_labels,
            'valid_pathways': valid_pathway_names
        }
        
        self.communication_patterns = patterns
        
        # 可视化
        fig = None
        if heatmap_show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 细胞模式热图
            im1 = ax1.imshow(W_norm, cmap=color_heatmap, aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(k))
            ax1.set_xticklabels(pattern_labels, fontsize=font_size)
            ax1.set_yticks(range(len(valid_cell_labels)))
            ax1.set_yticklabels(valid_cell_labels, fontsize=font_size-1)
            ax1.set_title('Cell Patterns', fontsize=font_size + 2)
            ax1.set_xlabel('Communication Patterns', fontsize=font_size)
            
            # 添加细胞类型颜色条
            cell_colors = self._get_cell_type_colors()
            colors = [cell_colors.get(ct, '#808080') for ct in valid_cell_labels]
            
            # 在左侧添加颜色条
            for i, color in enumerate(colors):
                rect = plt.Rectangle((-0.8, i-0.4), 0.6, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=0.5)
                ax1.add_patch(rect)
            
            # 信号模式热图
            im2 = ax2.imshow(H_norm.T, cmap=color_heatmap, aspect='auto', vmin=0, vmax=1)
            ax2.set_xticks(range(k))
            ax2.set_xticklabels(pattern_labels, fontsize=font_size)
            ax2.set_yticks(range(len(valid_pathway_names)))
            ax2.set_yticklabels(valid_pathway_names, fontsize=font_size-1, rotation=0)
            ax2.set_title('Signaling Patterns', fontsize=font_size + 2)
            ax2.set_xlabel('Communication Patterns', fontsize=font_size)
            
            # 添加colorbar
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Contribution')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Contribution')
            
            # 总标题
            if title is None:
                title = f"Communication Pattern Analysis ({pattern.title()}) - RE={relative_error:.3f}"
            fig.suptitle(title, fontsize=font_size + 4, y=0.95)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(save, dpi=300, bbox_inches='tight')
                print(f"Communication patterns saved as: {save}")
        
        # 输出分析结果
        print(f"✅ 通信模式识别完成:")
        print(f"   - 模式数量: {k}")
        print(f"   - 分析方向: {pattern}")
        print(f"   - 重构误差: {reconstruction_error:.4f}")
        print(f"   - 相对误差: {relative_error:.4f}")
        print(f"   - 有效细胞类型: {len(valid_cell_labels)}")
        print(f"   - 有效信号通路: {len(valid_pathway_names)}")
        
        # 显示主要模式特征 - 改进版本
        print(f"\n🔍 模式特征分析:")
        for i in range(k):
            pattern_name = pattern_labels[i]
            
            # 主要细胞类型 - 显示更有意义的差异
            cell_scores = cell_patterns_df[pattern_name]
            if cell_scores.max() > 0.1:  # 只显示有意义的贡献
                top_cells = cell_scores.nlargest(3)
                cell_str = ", ".join([f"{ct}({score:.3f})" for ct, score in top_cells.items() if score > 0.05])
            else:
                cell_str = "低贡献模式"
            
            # 主要信号通路
            pathway_scores = signaling_patterns_df[pattern_name]
            if pathway_scores.max() > 0.1:
                top_pathways = pathway_scores.nlargest(3)
                pathway_str = ", ".join([f"{pw}({score:.3f})" for pw, score in top_pathways.items() if score > 0.05])
            else:
                pathway_str = "低贡献模式"
            
            w_ent, h_ent = pattern_diversity[i]
            print(f"   - {pattern_name} (多样性: 细胞={w_ent:.2f}, 通路={h_ent:.2f}):")
            print(f"     * 主要细胞: {cell_str}")
            print(f"     * 主要通路: {pathway_str}")
        
        return patterns, fig
    
    def computeNetSimilarity(self, similarity_type="functional", k=None, thresh=None):
        """
        计算信号网络之间的相似性（类似CellChat的computeNetSimilarity功能）
        
        Args:
            similarity_type: str
                相似性类型: "functional" or "structural"
            k: int or None
                SNN平滑的邻居数量，如果为None则自动计算
            thresh: float or None
                过滤阈值，去除低于该分位数的交互
            
        Returns:
            similarity_matrix: pd.DataFrame
                信号网络相似性矩阵
        """
        # 获取概率矩阵
        if not hasattr(self, 'prob_tensor'):
            self.compute_communication_prob()
        
        prob_tensor = self.prob_tensor
        n_pathways = prob_tensor.shape[2]
        
        # 自动设置k值
        if k is None:
            if n_pathways <= 25:
                k = int(np.ceil(np.sqrt(n_pathways)))
            else:
                k = int(np.ceil(np.sqrt(n_pathways))) + 1
        
        # 应用阈值过滤
        if thresh is not None:
            non_zero_values = prob_tensor[prob_tensor != 0]
            if len(non_zero_values) > 0:
                threshold_value = np.quantile(non_zero_values, thresh)
                prob_tensor = prob_tensor.copy()
                prob_tensor[prob_tensor < threshold_value] = 0
        
        print(f"🔍 计算{similarity_type}相似性 (k={k}, n_pathways={n_pathways})...")
        
        # 初始化相似性矩阵
        similarity_matrix = np.zeros((n_pathways, n_pathways))
        
        if similarity_type == "functional":
            # 计算功能相似性（基于Jaccard指数）
            for i in range(n_pathways - 1):
                for j in range(i + 1, n_pathways):
                    # 获取二进制矩阵（是否有交互）
                    Gi = (prob_tensor[:, :, i] > 0).astype(int)
                    Gj = (prob_tensor[:, :, j] > 0).astype(int)
                    
                    # 计算Jaccard相似性
                    intersection = np.sum(Gi * Gj)
                    union = np.sum(Gi + Gj - Gi * Gj)
                    
                    if union > 0:
                        jaccard_sim = intersection / union
                    else:
                        jaccard_sim = 0
                    
                    similarity_matrix[i, j] = jaccard_sim
            
            # 对称化矩阵
            similarity_matrix = similarity_matrix + similarity_matrix.T
            np.fill_diagonal(similarity_matrix, 1.0)
            
        elif similarity_type == "structural":
            # 计算结构相似性
            for i in range(n_pathways - 1):
                for j in range(i + 1, n_pathways):
                    Gi = (prob_tensor[:, :, i] > 0).astype(int)
                    Gj = (prob_tensor[:, :, j] > 0).astype(int)
                    
                    # 计算结构距离（简化版本）
                    # 使用Hamming距离的归一化版本
                    diff_matrix = np.abs(Gi - Gj)
                    total_positions = Gi.size
                    hamming_distance = np.sum(diff_matrix) / total_positions
                    
                    # 转换为相似性（距离越小，相似性越高）
                    structural_sim = 1 - hamming_distance
                    similarity_matrix[i, j] = structural_sim
            
            # 对称化矩阵
            similarity_matrix = similarity_matrix + similarity_matrix.T
            np.fill_diagonal(similarity_matrix, 1.0)
            
        else:
            raise ValueError("similarity_type must be 'functional' or 'structural'")
        
        # SNN平滑（简化版本）
        similarity_smoothed = self._apply_snn_smoothing(similarity_matrix, k)
        
        # 创建DataFrame
        similarity_df = pd.DataFrame(
            similarity_smoothed,
            index=self.pathway_names,
            columns=self.pathway_names
        )
        
        # 存储结果
        if not hasattr(self, 'net_similarity'):
            self.net_similarity = {}
        self.net_similarity[similarity_type] = similarity_df
        
        print(f"✅ 网络相似性计算完成:")
        print(f"   - 相似性类型: {similarity_type}")
        print(f"   - 相似性范围: {similarity_df.values.min():.3f} - {similarity_df.values.max():.3f}")
        print(f"   - 平均相似性: {similarity_df.values.mean():.3f}")
        
        return similarity_df
    
    def _apply_snn_smoothing(self, similarity_matrix, k):
        """
        应用共享最近邻（SNN）平滑
        
        Args:
            similarity_matrix: np.ndarray
                原始相似性矩阵
            k: int
                邻居数量
            
        Returns:
            smoothed_matrix: np.ndarray
                平滑后的相似性矩阵
        """
        n = similarity_matrix.shape[0]
        snn_matrix = np.zeros_like(similarity_matrix)
        
        # 对每个节点找k个最近邻
        for i in range(n):
            # 获取相似性分数并排序
            similarities = similarity_matrix[i, :]
            # 不包括自己，找到k个最近邻
            neighbor_indices = np.argsort(similarities)[::-1][1:k+1]
            
            for j in range(n):
                if i != j:
                    # 计算共享邻居数量
                    j_neighbors = np.argsort(similarity_matrix[j, :])[::-1][1:k+1]
                    shared_neighbors = len(set(neighbor_indices) & set(j_neighbors))
                    
                    # SNN相似性
                    snn_matrix[i, j] = shared_neighbors / k
        
        # 应用SNN权重
        prune_threshold = 1/15  # 类似CellChat的prune.SNN参数
        snn_matrix[snn_matrix < prune_threshold] = 0
        
        # 与原始相似性矩阵相乘
        smoothed_matrix = similarity_matrix * snn_matrix
        
        return smoothed_matrix
    
    def netVisual_diffusion(self, similarity_type="functional", layout='spring',
                           node_size_factor=500, edge_width_factor=5,
                           figsize=(12, 10), title=None, save=None,
                           show_labels=True, font_size=12):
        """
        可视化信号网络相似性和扩散模式
        
        Args:
            similarity_type: str
                使用的相似性类型
            layout: str
                网络布局: 'spring', 'circular', 'kamada_kawai'
            node_size_factor: float
                节点大小因子
            edge_width_factor: float
                边宽度因子
            figsize: tuple
                图形大小
            title: str or None
                图标题
            save: str or None
                保存路径
            show_labels: bool
                是否显示标签
            font_size: int
                字体大小
            
        Returns:
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for network visualization. Please install: pip install networkx")
        
        if not hasattr(self, 'net_similarity') or similarity_type not in self.net_similarity:
            print(f"Computing {similarity_type} similarity first...")
            self.computeNetSimilarity(similarity_type=similarity_type)
        
        similarity_df = self.net_similarity[similarity_type]
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        pathways = similarity_df.index.tolist()
        G.add_nodes_from(pathways)
        
        # 添加边（只保留高相似性的边）
        threshold = similarity_df.values.mean() + similarity_df.values.std()
        
        for i, pathway1 in enumerate(pathways):
            for j, pathway2 in enumerate(pathways):
                if i < j:  # 避免重复边
                    weight = similarity_df.loc[pathway1, pathway2]
                    if weight > threshold:
                        G.add_edge(pathway1, pathway2, weight=weight)
        
        # 计算布局
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 可视化
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算节点大小（基于平均相似性）
        node_similarities = similarity_df.mean(axis=1)
        node_sizes = [node_similarities[pathway] * node_size_factor for pathway in pathways]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, 
                              edgecolors='black', linewidths=1, ax=ax)
        
        # 绘制边
        edges = G.edges()
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            edge_widths = [w * edge_width_factor for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, 
                                  alpha=0.6, edge_color='gray', ax=ax)
        
        # 添加标签
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=font_size, 
                                   font_weight='bold', ax=ax)
        
        # 设置标题
        if title is None:
            title = f"Signaling Network Similarity ({similarity_type.title()})"
        ax.set_title(title, fontsize=font_size + 4, pad=20)
        ax.axis('off')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=10, label='Signaling Pathway'),
            plt.Line2D([0], [0], color='gray', linewidth=3, label='High Similarity')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Network similarity plot saved as: {save}")
        
        return fig, ax
    
    def identifyOverExpressedGenes(self, signaling, patterns=None,
                                  min_expression=0.1, pvalue_threshold=0.05):
        """
        识别在特定模式中过表达的基因
        
        Args:
            signaling: str or list
                信号通路名称
            patterns: list or None
                要分析的模式编号，如果为None则分析所有模式
            min_expression: float
                最小表达阈值
            pvalue_threshold: float
                显著性阈值
            
        Returns:
            overexpressed_genes: dict
                每个模式中过表达的基因
        """
        if not hasattr(self, 'communication_patterns'):
            raise ValueError("Please run identifyCommunicationPatterns() first")
        
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # 获取模式信息
        cell_patterns = self.communication_patterns['cell']
        signaling_patterns = self.communication_patterns['signaling']
        
        if patterns is None:
            patterns = list(range(len(cell_patterns.columns)))
        
        overexpressed_genes = {}
        
        # 分析每个模式
        for pattern_idx in patterns:
            pattern_name = f"Pattern {pattern_idx + 1}"
            
            if pattern_name not in cell_patterns.columns:
                continue
            
            # 获取该模式中贡献最大的细胞类型
            top_cells = cell_patterns[pattern_name].nlargest(3).index.tolist()
            
            # 获取该模式中贡献最大的信号通路
            pattern_pathways = signaling_patterns[pattern_name].nlargest(5).index.tolist()
            
            # 找到交集通路
            relevant_pathways = list(set(pattern_pathways) & set(signaling))
            
            if not relevant_pathways:
                continue
            
            # 收集过表达基因
            pattern_genes = {'ligands': set(), 'receptors': set()}
            
            # 筛选相关通路的基因
            pathway_mask = self.adata.var['classification'].isin(relevant_pathways)
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                     self.adata.obs['receiver'])):
                # 只考虑top细胞类型
                if sender not in top_cells and receiver not in top_cells:
                    continue
                
                # 获取显著交互
                pvals = self.adata.layers['pvalues'][i, pathway_mask]
                means = self.adata.layers['means'][i, pathway_mask]
                
                sig_mask = (pvals < pvalue_threshold) & (means > min_expression)
                
                if np.any(sig_mask):
                    # 获取基因信息
                    pathway_indices = np.where(pathway_mask)[0]
                    sig_indices = pathway_indices[sig_mask]
                    
                    for idx in sig_indices:
                        gene_a = self.adata.var['gene_a'].iloc[idx]
                        gene_b = self.adata.var['gene_b'].iloc[idx]
                        
                        if pd.notna(gene_a):
                            pattern_genes['ligands'].add(gene_a)
                        if pd.notna(gene_b):
                            pattern_genes['receptors'].add(gene_b)
            
            overexpressed_genes[pattern_name] = {
                'ligands': list(pattern_genes['ligands']),
                'receptors': list(pattern_genes['receptors']),
                'top_cells': top_cells,
                'relevant_pathways': relevant_pathways
            }
        
        return overexpressed_genes