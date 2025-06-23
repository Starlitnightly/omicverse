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

class CellChatViz:
    """
    CellChat-like visualization for CellPhoneDB AnnData
    """
    
    def __init__(self, adata, palette=None):
        """
        Initialize with CellPhoneDB AnnData object
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object with CellPhoneDB results
            - obs: 'sender', 'receiver'
            - var: interaction information including 'classification'
            - layers: 'pvalues', 'means'
        palette : dict or list, optional
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
        
        Parameters:
        -----------
        palette : dict, list, or None
            Color palette specification
            
        Returns:
        --------
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
        """获取细胞类型颜色映射，确保所有方法使用相同的颜色"""
        # 如果已经有缓存的颜色，直接返回
        if self._color_cache is not None:
            return self._color_cache
        
        cell_type_colors = {}
        
        # 优先使用用户提供的palette
        if self.palette is not None:
            cell_type_colors.update(self.palette)
        
        # 如果palette没有覆盖所有细胞类型，尝试从adata.uns中获取颜色信息
        missing_types = set(self.cell_types) - set(cell_type_colors.keys())
        if missing_types:
            color_keys = [key for key in self.adata.uns.keys() if key.endswith('_colors')]
            
            # 寻找可能的细胞类型颜色
            for key in color_keys:
                # 提取细胞类型名称 (去掉'_colors'后缀)
                celltype_key = key.replace('_colors', '')
                if celltype_key in self.adata.obs.columns:
                    categories = self.adata.obs[celltype_key].cat.categories
                    colors = self.adata.uns[key]
                    # 只保留在我们细胞类型列表中的颜色
                    for i, cat in enumerate(categories):
                        if cat in missing_types and i < len(colors):
                            cell_type_colors[cat] = colors[i]
                    break
        
        # 如果仍有缺失的颜色，使用默认颜色映射
        missing_types = set(self.cell_types) - set(cell_type_colors.keys())
        if missing_types:
            # 使用固定的颜色映射确保一致性
            import matplotlib.cm as cm
            # 使用tab20颜色映射，但确保稳定的颜色分配
            tab20_colors = cm.tab20(np.linspace(0, 1, 20))
            from ..pl._palette import palette_56
            
            # 为缺失的细胞类型分配颜色
            for i, ct in enumerate(sorted(missing_types)):
                cell_type_colors[ct] = palette_56[i]
        
        # 确保颜色映射的稳定性：按细胞类型名称排序
        sorted_colors = {}
        for ct in sorted(self.cell_types):
            sorted_colors[ct] = cell_type_colors[ct]
        
        # 缓存颜色映射
        self._color_cache = sorted_colors
        
        return sorted_colors
    
    def _create_custom_colormap(self, cell_color):
        """
        Create a custom colormap based on cell type color
        
        Parameters:
        -----------
        cell_color : str
            Base color for the cell type
        
        Returns:
        --------
        cmap : matplotlib.colors.LinearSegmentedColormap
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
        绘制弯曲的箭头，模仿CellChat的旋转开花效果
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Matplotlib axes object
        start_pos : tuple
            Starting position (x, y)
        end_pos : tuple
            Ending position (x, y)
        weight : float
            Edge weight
        max_weight : float
            Maximum weight for normalization
        color : str or tuple
            Edge color
        edge_width_max : float
            Maximum edge width
        curve_strength : float
            Strength of the curve (0 = straight, higher = more curved)
        arrowsize : float
            Size of the arrow head
        """
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.patches import ConnectionPatch
        import matplotlib.patches as patches
        
        # 计算箭头宽度
        width = (weight / max_weight) * edge_width_max
        
        # 计算中点和法向量来创建弯曲效果
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # 计算向量
        dx = end_x - start_x
        dy = end_y - start_y
        
        # 计算中点
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # 计算垂直向量（用于弯曲）
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # 标准化垂直向量
            perp_x = -dy / length
            perp_y = dx / length
            
            # 添加弯曲偏移
            curve_offset = curve_strength * length
            control_x = mid_x + perp_x * curve_offset
            control_y = mid_y + perp_y * curve_offset
            
            # 创建弯曲路径
            from matplotlib.path import Path
            import matplotlib.patches as patches
            
            # 定义贝塞尔曲线路径
            verts = [
                (start_x, start_y),  # 起点
                (control_x, control_y),  # 控制点
                (end_x, end_y),  # 终点
            ]
            
            codes = [
                Path.MOVETO,  # 移动到起点
                Path.CURVE3,  # 二次贝塞尔曲线
                Path.CURVE3,  # 二次贝塞尔曲线
            ]
            
            path = Path(verts, codes)
            
            # 绘制弯曲的线条
            patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                                    linewidth=width, alpha=0.7)
            ax.add_patch(patch)
            
            # 添加箭头头部
            # 计算箭头方向
            arrow_dx = end_x - control_x
            arrow_dy = end_y - control_y
            arrow_length = np.sqrt(arrow_dx**2 + arrow_dy**2)
            
            if arrow_length > 0:
                # 标准化方向向量
                arrow_dx /= arrow_length
                arrow_dy /= arrow_length
                
                # 箭头头部大小
                head_length = arrowsize * 0.01
                head_width = arrowsize * 0.008
                
                # 计算箭头头部的三个点
                # 箭头尖端
                tip_x = end_x
                tip_y = end_y
                
                # 箭头底部的两个点
                base_x = tip_x - arrow_dx * head_length
                base_y = tip_y - arrow_dy * head_length
                
                left_x = base_x - arrow_dy * head_width
                left_y = base_y + arrow_dx * head_width
                right_x = base_x + arrow_dy * head_width
                right_y = base_y - arrow_dx * head_width
                
                # 绘制箭头头部
                triangle = plt.Polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], 
                                     color=color, alpha=0.8)
                ax.add_patch(triangle)
    
    def _draw_self_loop(self, ax, pos, weight, max_weight, color, edge_width_max):
        """
        绘制自环（自己到自己的连接）
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Matplotlib axes object
        pos : tuple
            Position (x, y)
        weight : float
            Edge weight
        max_weight : float
            Maximum weight for normalization
        color : str or tuple
            Edge color
        edge_width_max : float
            Maximum edge width
        """
        import matplotlib.patches as patches
        
        x, y = pos
        width = (weight / max_weight) * edge_width_max
        
        # 创建一个小圆圈作为自环
        radius = 0.15
        circle = patches.Circle((x + radius, y), radius, fill=False, 
                              edgecolor=color, linewidth=width, alpha=0.7)
        ax.add_patch(circle)
        
        # 添加小箭头
        arrow_x = x + radius + radius * 0.7
        arrow_y = y
        arrow = patches.FancyArrowPatch((arrow_x - 0.05, arrow_y), (arrow_x, arrow_y),
                                      arrowstyle='->', mutation_scale=10, 
                                      color=color, alpha=0.8)
        ax.add_patch(arrow)
    
    def compute_aggregated_network(self, pvalue_threshold=0.05, use_means=True):
        """
        计算聚合的细胞通讯网络
        
        Parameters:
        -----------
        pvalue_threshold : float
            P-value threshold for significant interactions
        use_means : bool
            Whether to use mean expression values as weights
        
        Returns:
        --------
        count_matrix : np.array
            Number of interactions between cell types
        weight_matrix : np.array
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
        圆形网络图可视化（类似CellChat的circle plot）
        使用发送者细胞类型颜色作为边的渐变颜色
        
        Parameters:
        -----------
        matrix : np.array
            Interaction matrix (count or weight)
        title : str
            Plot title
        edge_width_max : float
            Maximum edge width
        vertex_size_max : float
            Maximum vertex size
        show_labels : bool
            Whether to show cell type labels
        cmap : str
            Colormap for edges (used when use_sender_colors=False)
        figsize : tuple
            Figure size
        use_sender_colors : bool
            Whether to use different colors for different sender cell types (default: True)
        use_curved_arrows : bool
            Whether to use curved arrows like CellChat (default: True)
        curve_strength : float
            Strength of the curve (0 = straight, higher = more curved)
        adjust_text : bool
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
                        
                        # 自环处理
                        if u == v:
                            # 绘制自环
                            self._draw_self_loop(ax, start_pos, weight, max_weight, 
                                               sender_color, edge_width_max)
                        else:
                            # 绘制弯曲箭头
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
    
    def netVisual_circle_focused(self, matrix, title="Cell-Cell Communication Network", 
                                edge_width_max=10, vertex_size_max=50, show_labels=True,
                                cmap='Blues', figsize=(10, 10), min_interaction_threshold=0,
                                use_sender_colors=True, use_curved_arrows=True, curve_strength=0.3):
        """
        绘制聚焦的圆形网络图，只显示有实际交互的细胞类型
        
        Parameters:
        -----------
        matrix : np.array
            Interaction matrix (count or weight)
        title : str
            Plot title
        edge_width_max : float
            Maximum edge width
        vertex_size_max : float
            Maximum vertex size
        show_labels : bool
            Whether to show cell type labels
        cmap : str
            Colormap for edges (used when use_sender_colors=False)
        figsize : tuple
            Figure size
        min_interaction_threshold : float
            Minimum interaction strength to include cell type
        use_sender_colors : bool
            Whether to use different colors for different sender cell types
        use_curved_arrows : bool
            Whether to use curved arrows like CellChat (default: True)
        curve_strength : float
            Strength of the curve (0 = straight, higher = more curved)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        # 找到有实际交互的细胞类型
        interaction_mask = (matrix.sum(axis=0) + matrix.sum(axis=1)) > min_interaction_threshold
        active_cell_indices = np.where(interaction_mask)[0]
        
        if len(active_cell_indices) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No interactions above threshold', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig, ax
        
        # 创建筛选后的矩阵和细胞类型列表
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
                        
                        # 自环处理
                        if u == v:
                            # 绘制自环
                            self._draw_self_loop(ax, start_pos, weight, max_weight, 
                                               sender_color, edge_width_max)
                        else:
                            # 绘制弯曲箭头
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
        为每个细胞类型绘制单独的圆形网络图，显示其向外发送的信号
        模仿CellChat的功能，使用发送者细胞类型颜色作为边的渐变
        
        Parameters:
        -----------
        pvalue_threshold : float
            P-value threshold for significant interactions
        vertex_size_max : float
            Maximum vertex size
        edge_width_max : float
            Maximum edge width (consistent across all plots for comparison)
        show_labels : bool
            Whether to show cell type labels
        cmap : str
            Colormap for edges (used when use_sender_colors=False)
        figsize : tuple
            Figure size
        ncols : int
            Number of columns in subplot layout
        use_sender_colors : bool
            Whether to use sender cell type colors for edges (default: True)
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
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
        为每个细胞类型绘制单独的圆形网络图，显示其接收的信号
        使用发送者细胞类型颜色作为边的渐变
        
        Parameters:
        -----------
        pvalue_threshold : float
            P-value threshold for significant interactions
        vertex_size_max : float
            Maximum vertex size
        edge_width_max : float
            Maximum edge width
        show_labels : bool
            Whether to show cell type labels
        cmap : str
            Colormap for edges (used when use_sender_colors=False)
        figsize : tuple
            Figure size
        ncols : int
            Number of columns in subplot layout
        use_sender_colors : bool
            Whether to use sender cell type colors for edges (default: True)
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
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
        热图可视化细胞间通讯
        
        Parameters:
        -----------
        matrix : np.array
            Interaction matrix
        title : str
            Plot title
        cmap : str
            Colormap
        show_values : bool
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
        使用marsilea包绘制细胞间通讯热图（模仿CellChat的netVisual_heatmap功能）
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称。如果为None，显示所有通路的聚合结果
        pvalue_threshold : float
            P-value threshold for significant interactions
        color_heatmap : str
            热图颜色映射
        add_dendrogram : bool
            是否添加树状图
        add_row_sum : bool
            是否在左侧显示行总和
        add_col_sum : bool
            是否在顶部显示列总和  
        linewidth : float
            网格线宽度
        figsize : tuple
            图形大小
        title : str
            热图标题
            
        Returns:
        --------
        h : marsilea heatmap object
        """
        if not MARSILEA_AVAILABLE:
            raise ImportError("marsilea package is not available. Please install it: pip install marsilea")
        
        # 计算通讯矩阵
        if signaling is not None:
            # 计算特定通路的通讯矩阵
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # 检查信号通路是否存在
            available_pathways = self.adata.var['classification'].unique()
            for pathway in signaling:
                if pathway not in available_pathways:
                    raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
            
            # 计算特定通路的通讯矩阵
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_mask = self.adata.var['classification'].isin(signaling)
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                raise ValueError(f"No interactions found for pathway(s): {signaling}")
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                    self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # 获取该通路的显著交互
                pvals = self.adata.layers['pvalues'][i, pathway_indices]
                means = self.adata.layers['means'][i, pathway_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
            
            matrix = pathway_matrix
            heatmap_title = f"{title} - {', '.join(signaling)}"
        else:
            # 使用聚合的通讯矩阵
            _, matrix = self.compute_aggregated_network(pvalue_threshold)
            heatmap_title = title
        
        # 创建DataFrame以便更好地标记
        df_matrix = pd.DataFrame(matrix, 
                                index=self.cell_types, 
                                columns=self.cell_types)

        #return df_matrix
        
        # 创建marsilea热图
        h = ma.Heatmap(df_matrix, linewidth=linewidth, 
                    cmap=color_heatmap, label="Interaction Strength")
        
        # 添加行列分组 - 这是关键步骤！
        #h.group_rows(df_matrix.index, order=df_matrix.index.tolist())
        #h.group_cols(df_matrix.columns, order=df_matrix.columns.tolist())
        
        
        
        # 添加行总和（左侧）
        # 添加行总和（左侧）
        if add_row_sum:
            row_sums = matrix.sum(axis=1)
            h.add_left(ma.plotter.Numbers(row_sums, color="#F05454", 
                                        label="Outgoing",show_value=False))
        
        # 添加列总和（顶部）
        if add_col_sum:
            col_sums = matrix.sum(axis=0)
            h.add_top(ma.plotter.Numbers(col_sums, color="#4A90E2",
                                    label="Incoming",show_value=False))
        
        # 添加细胞类型颜色注释
        cell_colors = self._get_cell_type_colors()
        row_colors = [cell_colors.get(ct, '#808080') for ct in self.cell_types]
        col_colors = [cell_colors.get(ct, '#808080') for ct in self.cell_types]
        
        # 添加细胞类型颜色条
        h.add_left(ma.plotter.Colors(self.cell_types,palette=row_colors),size=0.2,legend=False)
        h.add_top(ma.plotter.Colors(self.cell_types,palette=col_colors),size=0.2)
        
        # 添加图例
        h.add_legends()
        
        # 添加标题
        h.add_title(heatmap_title)

        # 添加树状图
        if add_dendrogram:
            h.add_dendrogram("left", colors="#2E8B57")
            h.add_dendrogram("top", colors="#2E8B57")
        
        
        return h
    
    def netVisual_heatmap_marsilea_focused(self, signaling=None, pvalue_threshold=0.05,
                                          min_interaction_threshold=0, color_heatmap="Reds", 
                                          add_dendrogram=True, add_row_sum=True, add_col_sum=True,
                                          linewidth=0.5, figsize=(8, 6), title="Communication Heatmap"):
        """
        使用marsilea包绘制聚焦的细胞间通讯热图，只显示有实际交互的细胞类型
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称
        pvalue_threshold : float
            P-value threshold for significant interactions
        min_interaction_threshold : float
            最小交互强度阈值，用于筛选细胞类型
        color_heatmap : str
            热图颜色映射
        add_dendrogram : bool
            是否添加树状图
        add_row_sum : bool
            是否在左侧显示行总和
        add_col_sum : bool
            是否在顶部显示列总和
        linewidth : float
            网格线宽度
        figsize : tuple
            图形大小
        title : str
            热图标题
            
        Returns:
        --------
        h : marsilea heatmap object
        """
        if not MARSILEA_AVAILABLE:
            raise ImportError("marsilea package is not available. Please install it: pip install marsilea")
        
        # 首先获取完整矩阵
        if signaling is not None:
            # 计算特定通路的通讯矩阵 
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
        
        # 筛选有实际交互的细胞类型
        interaction_mask = (matrix.sum(axis=0) + matrix.sum(axis=1)) > min_interaction_threshold
        active_cell_indices = np.where(interaction_mask)[0]
        
        if len(active_cell_indices) == 0:
            raise ValueError("No cell types have interactions above the threshold")
        
        # 创建筛选后的矩阵和细胞类型列表
        filtered_matrix = matrix[np.ix_(active_cell_indices, active_cell_indices)]
        active_cell_types = [self.cell_types[i] for i in active_cell_indices]
        
        # 创建DataFrame
        df_matrix = pd.DataFrame(filtered_matrix,
                                index=active_cell_types,
                                columns=active_cell_types)
        
        # 创建marsilea热图
        h = ma.Heatmap(df_matrix, linewidth=linewidth,
                      cmap=color_heatmap, label="Interaction Strength")
        
        # 添加行列分组 - 这是关键步骤！
        h.group_rows(df_matrix.index, order=df_matrix.index.tolist())
        h.group_cols(df_matrix.columns, order=df_matrix.columns.tolist())
        
        
        
        # 添加行总和（左侧）
        if add_row_sum:
            row_sums = filtered_matrix.sum(axis=1)
            h.add_left(ma.plotter.Numbers(row_sums, color="#F05454",
                                        label="Outgoing"))
        
        # 添加列总和（顶部）
        if add_col_sum:
            col_sums = filtered_matrix.sum(axis=0)
            h.add_top(ma.plotter.Numbers(col_sums, color="#4A90E2",
                                       label="Incoming"))
        
        # 添加细胞类型颜色注释
            cell_colors = self._get_cell_type_colors()
        row_colors = [cell_colors.get(ct, '#808080') for ct in active_cell_types]
        col_colors = [cell_colors.get(ct, '#808080') for ct in active_cell_types]
            
        # 添加细胞类型颜色条
        h.add_left(ma.plotter.Chunk(active_cell_types, fill_colors=row_colors, 
                                       rotation=90, label="Cell Types (Senders)"))
        h.add_top(ma.plotter.Chunk(active_cell_types, fill_colors=col_colors, 
                                      rotation=90, label="Cell Types (Receivers)"))
        
        # 添加图例
        h.add_legends()

        # 添加树状图
        if add_dendrogram:
            h.add_dendrogram("right", colors="#2E8B57")
            h.add_dendrogram("top", colors="#2E8B57")
        
        # 添加标题
        h.add_title(heatmap_title)
        
        
        return h
    
    def netVisual_chord(self, matrix, title="Chord Diagram", threshold=0, 
                       cmap='tab20', figsize=(12, 12)):
        """
        弦图可视化（Chord diagram）
        
        Parameters:
        -----------
        matrix : np.array
            Interaction matrix
        title : str
            Plot title
        threshold : float
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
        层次图可视化（Hierarchy plot）
        
        Parameters:
        -----------
        pathway_name : str
            Specific pathway to visualize (from classification)
        sources : list
            Source cell types to show
        targets : list  
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
        气泡图可视化
        
        Parameters:
        -----------
        sources : list
            Source cell types to include
        targets : list
            Target cell types to include  
        pathways : list
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
        计算通路级别的通讯网络
        
        Returns:
        --------
        pathway_networks : dict
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
        识别细胞的信号角色（sender, receiver, mediator, influencer）
        
        Parameters:
        -----------
        pattern : str
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
        计算通路间的相似性（功能或结构相似性）
        
        Parameters:
        -----------
        method : str
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
        通路嵌入和聚类可视化（UMAP）
        
        Parameters:
        -----------
        method : str
            'functional' or 'structural'
        n_components : int
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
        为指定的单个细胞类型绘制圆形网络图
        使用发送者细胞类型颜色作为边的渐变
        
        Parameters:
        -----------
        cell_type : str
            要绘制的细胞类型名称
        direction : str
            'outgoing' 显示该细胞类型发送的信号, 'incoming' 显示接收的信号
        pvalue_threshold : float
            P-value threshold for significant interactions
        vertex_size_max : float
            Maximum vertex size
        edge_width_max : float
            Maximum edge width
        show_labels : bool
            Whether to show cell type labels
        cmap : str
            Colormap for edges (used when use_sender_colors=False)
        figsize : tuple
            Figure size
        use_sender_colors : bool
            Whether to use sender cell type colors for edges (default: True)
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
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
        绘制特定信号通路的聚合网络图（模仿CellChat的netVisual_aggregate功能）
        
        Parameters:
        -----------
        signaling : str or list
            信号通路名称（来自adata.var['classification']）
        layout : str
            布局类型：'circle' 或 'hierarchy'
        vertex_receiver : list or None
            接收者细胞类型名称列表（用于hierarchy布局）
        vertex_sender : list or None
            发送者细胞类型名称列表（用于hierarchy布局）
        pvalue_threshold : float
            P-value threshold for significant interactions
        vertex_size_max : float
            Maximum vertex size
        edge_width_max : float
            Maximum edge width
        show_labels : bool
            Whether to show cell type labels
        cmap : str
            Colormap for edges (used when use_sender_colors=False)
        figsize : tuple
            Figure size
        focused_view : bool
            Whether to use focused view (only show cell types with interactions) for circle layout
        use_sender_colors : bool
            Whether to use different colors for different sender cell types
        use_curved_arrows : bool
            Whether to use curved arrows like CellChat (default: True)
        curve_strength : float
            Strength of the curve (0 = straight, higher = more curved)
        adjust_text : bool
            Whether to use adjust_text library to prevent label overlapping (default: False)
            If True, uses plt.text instead of nx.draw_networkx_labels
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        # 确保signaling是列表格式
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # 检查信号通路是否存在
        available_pathways = self.adata.var['classification'].unique()
        for pathway in signaling:
            if pathway not in available_pathways:
                raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
        
        # 验证细胞类型名称
        if vertex_receiver is not None:
            invalid_receivers = [ct for ct in vertex_receiver if ct not in self.cell_types]
            if invalid_receivers:
                raise ValueError(f"Invalid receiver cell types: {invalid_receivers}. Available cell types: {self.cell_types}")
        
        if vertex_sender is not None:
            invalid_senders = [ct for ct in vertex_sender if ct not in self.cell_types]
            if invalid_senders:
                raise ValueError(f"Invalid sender cell types: {invalid_senders}. Available cell types: {self.cell_types}")
        
        # 计算特定通路的通讯矩阵
        pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
        
        # 筛选特定通路的交互
        pathway_mask = self.adata.var['classification'].isin(signaling)
        pathway_indices = np.where(pathway_mask)[0]
        
        if len(pathway_indices) == 0:
            # 如果没有找到该通路的交互，返回空图
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No interactions found for pathway(s): {", ".join(signaling)}', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig, ax
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            sender_idx = self.cell_types.index(sender)
            receiver_idx = self.cell_types.index(receiver)
            
            # 获取该通路的显著交互
            pvals = self.adata.layers['pvalues'][i, pathway_indices]
            means = self.adata.layers['means'][i, pathway_indices]
            
            sig_mask = pvals < pvalue_threshold
            if np.any(sig_mask):
                pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
        
        # 根据布局类型选择可视化方法
        if layout == 'circle':
            # 检查是否有实际的交互
            if pathway_matrix.sum() == 0:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f'No significant interactions found for pathway(s): {", ".join(signaling)}', 
                       ha='center', va='center', fontsize=16)
                ax.axis('off')
                return fig, ax
            
            title = f"Signaling Pathway: {', '.join(signaling)} (Circle)"
            
            # 如果指定了vertex_sender或vertex_receiver，需要筛选矩阵
            if vertex_sender is not None or vertex_receiver is not None:
                # 创建筛选后的矩阵
                filtered_matrix = np.zeros_like(pathway_matrix)
                
                for i, sender_type in enumerate(self.cell_types):
                    for j, receiver_type in enumerate(self.cell_types):
                        # 检查是否符合sender/receiver条件
                        sender_ok = (vertex_sender is None) or (sender_type in vertex_sender)
                        receiver_ok = (vertex_receiver is None) or (receiver_type in vertex_receiver)
                        
                        if sender_ok and receiver_ok and pathway_matrix[i, j] > 0:
                            filtered_matrix[i, j] = pathway_matrix[i, j]
                
                pathway_matrix = filtered_matrix
                
                # 再次检查是否还有交互
                if pathway_matrix.sum() == 0:
                    fig, ax = plt.subplots(figsize=figsize)
                    sender_str = f"senders: {vertex_sender}" if vertex_sender else "any senders"
                    receiver_str = f"receivers: {vertex_receiver}" if vertex_receiver else "any receivers"
                    ax.text(0.5, 0.5, f'No interactions found for pathway(s): {", ".join(signaling)}\nwith {sender_str} and {receiver_str}', 
                           ha='center', va='center', fontsize=14)
                    ax.axis('off')
                    return fig, ax
            
            # 选择合适的圆形可视化方法
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
                    curve_strength=curve_strength,
                    adjust_text=adjust_text 
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
            # 确定源细胞和目标细胞
            if vertex_receiver is not None and vertex_sender is not None:
                # 如果同时指定了发送者和接收者
                source_cells = vertex_sender
                target_cells = vertex_receiver
            elif vertex_receiver is not None:
                # 只指定了接收者，其余为发送者
                target_cells = vertex_receiver
                source_cells = [ct for ct in self.cell_types if ct not in target_cells]
            elif vertex_sender is not None:
                # 只指定了发送者，其余为接收者
                source_cells = vertex_sender
                target_cells = [ct for ct in self.cell_types if ct not in source_cells]
            else:
                # 如果都没有指定，使用所有有交互的细胞类型
                source_cells = None
                target_cells = None
            
            title = f"Signaling Pathway: {', '.join(signaling)} (Hierarchy)"
            fig, ax = self.netVisual_hierarchy(
                pathway_name=signaling[0],  # 使用第一个通路名称
                sources=source_cells,
                targets=target_cells,
                pvalue_threshold=pvalue_threshold,
                figsize=figsize
            )
            
            # 更新标题
            ax.set_title(title, fontsize=16)
        
        else:
            raise ValueError("layout must be 'circle' or 'hierarchy'")
        
        return fig, ax
    
    def get_signaling_pathways(self, min_interactions=1, pathway_pvalue_threshold=0.05, 
                              method='fisher', correction_method='fdr_bh', min_expression=0.1):
        """
        获取所有显著的信号通路列表，使用统计学上更可靠的方法组合多个L-R对的p-values
        
        Parameters:
        -----------
        min_interactions : int
            每个通路最小L-R对数量阈值 (default: 1)
        pathway_pvalue_threshold : float
            通路级别的p-value阈值 (default: 0.05)
        method : str
            P-value组合方法: 'fisher', 'stouffer', 'min', 'mean' (default: 'fisher')
        correction_method : str
            多重检验校正方法: 'fdr_bh', 'bonferroni', 'holm', None (default: 'fdr_bh')
        min_expression : float
            最小表达量阈值 (default: 0.1)
        
        Returns:
        --------
        pathways : list
            显著信号通路列表
        pathway_stats : dict
            每个通路的详细统计信息
        """
        from scipy.stats import combine_pvalues
        from statsmodels.stats.multitest import multipletests
        import warnings
        
        pathways = [p for p in self.adata.var['classification'].unique() if pd.notna(p)]
        pathway_stats = {}
        pathway_pvalues = []
        pathway_names = []
        
        print(f"🔬 使用{method}方法分析{len(pathways)}个信号通路的统计显著性...")
        
        for pathway in pathways:
            pathway_mask = self.adata.var['classification'] == pathway
            pathway_lr_pairs = self.adata.var.loc[pathway_mask, 'interacting_pair'].tolist()
            
            if len(pathway_lr_pairs) < min_interactions:
                continue
                
            # 收集该通路在所有细胞对中的p-values和表达量
            all_pathway_pvals = []
            all_pathway_means = []
            significant_cell_pairs = []
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], self.adata.obs['receiver'])):
                pvals = self.adata.layers['pvalues'][i, pathway_mask]
                means = self.adata.layers['means'][i, pathway_mask]
                
                # 过滤低表达的相互作用
                valid_mask = means >= min_expression
                if np.any(valid_mask):
                    valid_pvals = pvals[valid_mask]
                    valid_means = means[valid_mask]
                    
                    all_pathway_pvals.extend(valid_pvals)
                    all_pathway_means.extend(valid_means)
                    
                    # 检查是否有显著的相互作用
                    if np.any(valid_pvals < 0.05):
                        significant_cell_pairs.append(f"{sender}|{receiver}")
            
            if len(all_pathway_pvals) == 0:
                continue
                
            all_pathway_pvals = np.array(all_pathway_pvals)
            all_pathway_means = np.array(all_pathway_means)
            
            # 组合p-values以获得通路级别的显著性
            try:
                if method == 'fisher':
                    # Fisher's method - 适用于独立检验
                    combined_stat, combined_pval = combine_pvalues(all_pathway_pvals, method='fisher')
                elif method == 'stouffer':
                    # Stouffer's method - 可以加权
                    weights = all_pathway_means / all_pathway_means.sum()  # 基于表达量加权
                    combined_stat, combined_pval = combine_pvalues(all_pathway_pvals, method='stouffer', weights=weights)
                elif method == 'min':
                    # 最小p-value方法 (需要Bonferroni校正)
                    combined_pval = np.min(all_pathway_pvals) * len(all_pathway_pvals)
                    combined_pval = min(combined_pval, 1.0)  # Cap at 1.0
                    combined_stat = -np.log10(combined_pval)
                elif method == 'mean':
                    # 平均p-value (不推荐，但作为参考)
                    combined_pval = np.mean(all_pathway_pvals)
                    combined_stat = -np.log10(combined_pval)
                else:
                    raise ValueError(f"Unknown method: {method}")
                    
            except Exception as e:
                warnings.warn(f"Failed to combine p-values for pathway {pathway}: {e}")
                combined_pval = 1.0
                combined_stat = 0.0
            
            # 计算通路统计信息
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
        
        # 多重检验校正
        if len(pathway_pvalues) > 0 and correction_method:
            print(f"📊 应用{correction_method}多重检验校正...")
            try:
                corrected_results = multipletests(pathway_pvalues, alpha=pathway_pvalue_threshold, 
                                                method=correction_method)
                corrected_pvals = corrected_results[1]
                is_significant = corrected_results[0]
                
                # 更新统计信息
                for i, pathway in enumerate(pathway_names):
                    pathway_stats[pathway]['corrected_pvalue'] = corrected_pvals[i]
                    pathway_stats[pathway]['is_significant_corrected'] = is_significant[i]
                
                significant_pathways = [pathway_names[i] for i in range(len(pathway_names)) if is_significant[i]]
                
            except Exception as e:
                warnings.warn(f"Multiple testing correction failed: {e}")
                # 回退到未校正的p-values
                significant_pathways = [pathway_names[i] for i in range(len(pathway_names)) 
                                      if pathway_pvalues[i] < pathway_pvalue_threshold]
        else:
            # 不进行多重检验校正
            significant_pathways = [pathway_names[i] for i in range(len(pathway_names)) 
                                  if pathway_pvalues[i] < pathway_pvalue_threshold]
        
        # 按显著性排序
        if len(significant_pathways) > 0:
            if correction_method:
                significant_pathways.sort(key=lambda x: pathway_stats[x]['corrected_pvalue'])
            else:
                significant_pathways.sort(key=lambda x: pathway_stats[x]['combined_pvalue'])
        
        print(f"✅ 发现{len(significant_pathways)}个显著通路 (总共{len(pathway_names)}个通路)")
        print(f"   - P-value组合方法: {method}")
        print(f"   - 多重检验校正: {correction_method if correction_method else 'None'}")
        print(f"   - 通路阈值: {pathway_pvalue_threshold}")
        
        return significant_pathways, pathway_stats
    
    def plot_all_visualizations(self, pvalue_threshold=0.05, save_prefix=None):
        """
        生成所有主要的可视化图
        
        Parameters:
        -----------
        pvalue_threshold : float
            P-value threshold
        save_prefix : str
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
        
        # Individual cell type networks (类似CellChat)
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
        使用mpl-chord-diagram创建弦图可视化（模仿CellChat的netVisual_chord_cell功能）
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称。如果为None，显示所有通路的聚合结果
        group_celltype : dict or None
            细胞类型分组映射，例如 {'CellA': 'GroupX', 'CellB': 'GroupX', 'CellC': 'GroupY'}
            如果为None，每个细胞类型单独显示
        sources : list or None
            指定的发送者细胞类型列表。如果为None，包含所有细胞类型
        targets : list or None
            指定的接收者细胞类型列表。如果为None，包含所有细胞类型
        pvalue_threshold : float
            P-value threshold for significant interactions
        count_min : int
            最小交互计数阈值
        gap : float
            弦图各段之间的间隙 (0.03)
        use_gradient : bool
            是否使用渐变效果 (True)
        sort : str or None
            排序方式: "size", "distance", None ("size")
        directed : bool
            是否显示方向性 (True)
        cmap : str or None
            颜色映射名称 (None，使用细胞类型颜色)
        chord_colors : str or None
            弦的颜色 (None)
        rotate_names : bool
            是否旋转名称 (False)
        fontcolor : str
            字体颜色 ("black")
        fontsize : int
            字体大小 (12)
        start_at : int
            起始角度 (0)
        extent : int
            弦图覆盖的角度范围 (360)
        min_chord_width : int
            最小弦宽度 (0)
        colors : list or None
            自定义颜色列表 (None，使用细胞类型颜色)
        ax : matplotlib.axes.Axes or None
            matplotlib轴对象 (None，创建新图)
        figsize : tuple
            图形大小 (8, 8)
        title_name : str or None
            图标题 (None)
        save : str or None
            保存文件路径 (None)
        normalize_to_sender : bool
            是否对发送者进行归一化以获得相等的弧宽度 (True)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        try:
            from ..externel.mpl_chord.chord_diagram import chord_diagram
        except ImportError:
            try:
                from mpl_chord_diagram import chord_diagram
            except ImportError:
                raise ImportError("mpl-chord-diagram package is required. Please install it: pip install mpl-chord-diagram")
        
        # 计算特定通路的交互矩阵
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # 检查信号通路是否存在
            available_pathways = self.adata.var['classification'].unique()
            for pathway in signaling:
                if pathway not in available_pathways:
                    raise ValueError(f"Pathway '{pathway}' not found. Available pathways: {list(available_pathways)}")
            
            # 计算特定通路的通讯矩阵
            pathway_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            pathway_mask = self.adata.var['classification'].isin(signaling)
            pathway_indices = np.where(pathway_mask)[0]
            
            if len(pathway_indices) == 0:
                raise ValueError(f"No interactions found for pathway(s): {signaling}")
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                      self.adata.obs['receiver'])):
                sender_idx = self.cell_types.index(sender)
                receiver_idx = self.cell_types.index(receiver)
                
                # 获取该通路的显著交互
                pvals = self.adata.layers['pvalues'][i, pathway_indices]
                means = self.adata.layers['means'][i, pathway_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    # 使用交互数量，更适合弦图
                    pathway_matrix[sender_idx, receiver_idx] += np.sum(means[sig_mask])
            
            matrix = pathway_matrix
            self.test_pathway_matrix = pathway_matrix
        else:
            # 使用聚合的交互计数矩阵
            count_matrix, _ = self.compute_aggregated_network(pvalue_threshold)
            matrix = count_matrix
        
        # 过滤指定的发送者和接收者
        if sources is not None or targets is not None:
            # 验证指定的细胞类型
            if sources is not None:
                invalid_sources = [ct for ct in sources if ct not in self.cell_types]
                if invalid_sources:
                    raise ValueError(f"Invalid source cell types: {invalid_sources}. Available: {self.cell_types}")
            
            if targets is not None:
                invalid_targets = [ct for ct in targets if ct not in self.cell_types]
                if invalid_targets:
                    raise ValueError(f"Invalid target cell types: {invalid_targets}. Available: {self.cell_types}")
            
            # 创建过滤后的矩阵
            filtered_matrix = np.zeros_like(matrix)
            for i, sender_type in enumerate(self.cell_types):
                for j, receiver_type in enumerate(self.cell_types):
                    # 检查是否符合sources/targets条件
                    sender_ok = (sources is None) or (sender_type in sources)
                    receiver_ok = (targets is None) or (receiver_type in targets)
                    
                    if sender_ok and receiver_ok:
                        filtered_matrix[i, j] = matrix[i, j]
            
            matrix = filtered_matrix
        
        # 应用group_celltype分组（如果提供）
        if group_celltype is not None:
            # 验证分组映射
            for cell_type in self.cell_types:
                if cell_type not in group_celltype:
                    raise ValueError(f"Cell type '{cell_type}' not found in group_celltype mapping")
            
            # 获取唯一的组名
            unique_groups = list(set(group_celltype.values()))
            group_matrix = np.zeros((len(unique_groups), len(unique_groups)))
            
            # 聚合到组级别
            for i, sender_type in enumerate(self.cell_types):
                for j, receiver_type in enumerate(self.cell_types):
                    if matrix[i, j] > 0:
                        sender_group = group_celltype[sender_type]
                        receiver_group = group_celltype[receiver_type]
                        sender_group_idx = unique_groups.index(sender_group)
                        receiver_group_idx = unique_groups.index(receiver_group)
                        group_matrix[sender_group_idx, receiver_group_idx] += matrix[i, j]
            
            # 使用分组后的矩阵和名称
            final_matrix = group_matrix
            final_names = unique_groups
        else:
            # 使用原始细胞类型
            final_matrix = matrix
            final_names = self.cell_types
        
        # 过滤低于阈值的交互
        final_matrix[final_matrix < count_min] = 0
        
        # 检查是否还有交互
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
        
        # CellChat风格的归一化：确保每个细胞类型的弧宽度相等
        if normalize_to_sender:
            # 使用迭代方法确保行和列总和都相等
            # 这是解决矩阵双向归一化的经典方法
            
            normalized_matrix = final_matrix.copy().astype(float)
            
            # 找到有交互的行和列
            row_sums = normalized_matrix.sum(axis=1)
            col_sums = normalized_matrix.sum(axis=0)
            nonzero_rows = row_sums > 0
            nonzero_cols = col_sums > 0
            
            if np.any(nonzero_rows) and np.any(nonzero_cols):
                standard_sum = 100.0
                max_iterations = 15
                tolerance = 1e-4  # 放宽容差，实际应用中这个精度已足够
                
                for iteration in range(max_iterations):
                    # 归一化行
                    row_sums = normalized_matrix.sum(axis=1)
                    for i in range(len(final_names)):
                        if row_sums[i] > tolerance:  # 避免除零
                            scale_factor = standard_sum / row_sums[i]
                            normalized_matrix[i, :] *= scale_factor
                    
                    # 归一化列
                    col_sums = normalized_matrix.sum(axis=0)
                    for j in range(len(final_names)):
                        if col_sums[j] > tolerance:  # 避免除零
                            scale_factor = standard_sum / col_sums[j]
                            normalized_matrix[:, j] *= scale_factor
                    
                    # 检查收敛性
                    final_row_sums = normalized_matrix.sum(axis=1)
                    final_col_sums = normalized_matrix.sum(axis=0)
                    
                    # 计算非零行列的标准差，判断是否收敛
                    nonzero_final_rows = final_row_sums[final_row_sums > tolerance]
                    nonzero_final_cols = final_col_sums[final_col_sums > tolerance]
                    
                    if (len(nonzero_final_rows) > 0 and 
                        len(nonzero_final_cols) > 0):
                        row_std = np.std(nonzero_final_rows)
                        col_std = np.std(nonzero_final_cols)
                        
                        if row_std < tolerance and col_std < tolerance:
                            break
                
                final_matrix = normalized_matrix
        
        # 准备颜色
        if colors is None:
            if group_celltype is not None:
                # 对于分组，为每个组分配颜色
                cell_colors = self._get_cell_type_colors()
                group_colors = {}
                for group in unique_groups:
                    # 使用该组中第一个细胞类型的颜色
                    for cell_type, group_name in group_celltype.items():
                        if group_name == group:
                            group_colors[group] = cell_colors.get(cell_type, '#1f77b4')
                            break
                colors = [group_colors.get(node, '#1f77b4') for node in final_names]
            else:
                # 使用细胞类型颜色
                cell_colors = self._get_cell_type_colors()
                colors = [cell_colors.get(node, '#1f77b4') for node in final_names]
        
        # 创建图形
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        self.test_final_matrix = final_matrix
        
        # 修改名称：只隐藏既不发送也不接收信号的细胞类型名字
        display_names = final_names.copy()
        if normalize_to_sender:
            # 计算行总和（发送的信号）和列总和（接收的信号）
            row_sums = final_matrix.sum(axis=1)
            col_sums = final_matrix.sum(axis=0)
            for i in range(len(final_names)):
                # 只有既不发送也不接收信号的细胞类型才隐藏名字
                if row_sums[i] == 0 and col_sums[i] == 0:
                    display_names[i] = ""  # 隐藏名字
        
        # 绘制弦图
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
        
        # 添加标题
        if title_name:
            ax.set_title(title_name, fontsize=fontsize + 4, pad=20)
        
        # 保存文件
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
        为特定配体-受体对创建弦图可视化（模仿CellChat的配体-受体级别分析）
        
        Parameters:
        -----------
        ligand_receptor_pairs : str, list or None
            特定配体-受体对名称。支持以下格式：
            - 单个字符串: "LIGAND_RECEPTOR" (例如: "TGFB1_TGFBR1")  
            - 字符串列表: ["LIGAND1_RECEPTOR1", "LIGAND2_RECEPTOR2"]
            - 如果为None，显示所有配体-受体对的聚合结果
        sources : list or None
            指定的发送者细胞类型列表。如果为None，包含所有细胞类型
        targets : list or None
            指定的接收者细胞类型列表。如果为None，包含所有细胞类型
        pvalue_threshold : float
            P-value threshold for significant interactions
        count_min : int
            最小交互计数阈值
        gap : float
            弦图各段之间的间隙
        use_gradient : bool
            是否使用渐变效果
        sort : str or None
            排序方式: "size", "distance", None
        directed : bool
            是否显示方向性
        cmap : str or None
            颜色映射名称
        chord_colors : str or None
            弦的颜色
        rotate_names : bool
            是否旋转名称
        fontcolor : str
            字体颜色
        fontsize : int
            字体大小
        start_at : int
            起始角度
        extent : int
            弦图覆盖的角度范围
        min_chord_width : int
            最小弦宽度
        colors : list or None
            自定义颜色列表
        ax : matplotlib.axes.Axes or None
            matplotlib轴对象
        figsize : tuple
            图形大小
        title_name : str or None
            图标题
        save : str or None
            保存文件路径
        normalize_to_sender : bool
            是否隐藏没有接收信号的细胞类型名字 (True)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        try:
            from ..externel.mpl_chord.chord_diagram import chord_diagram
        except ImportError:
            try:
                from mpl_chord_diagram import chord_diagram
            except ImportError:
                raise ImportError("mpl-chord-diagram package is required. Please install it: pip install mpl-chord-diagram")
        
        # 处理配体-受体对筛选
        if ligand_receptor_pairs is not None:
            if isinstance(ligand_receptor_pairs, str):
                ligand_receptor_pairs = [ligand_receptor_pairs]
            
            # 检查配体-受体对是否存在
            # 假设adata.var中包含配体-受体对信息，可能在'gene_name'或其他列中
            if 'gene_name' in self.adata.var.columns:
                available_pairs = self.adata.var['gene_name'].unique()
            elif 'interacting_pair' in self.adata.var.columns:
                available_pairs = self.adata.var['interacting_pair'].unique()
            else:
                # 如果没有明确的配体-受体对列，使用索引
                available_pairs = self.adata.var.index.tolist()
            
            # 验证请求的配体-受体对
            missing_pairs = []
            valid_pairs = []
            for pair in ligand_receptor_pairs:
                if pair in available_pairs:
                    valid_pairs.append(pair)
                else:
                    missing_pairs.append(pair)
            
            if missing_pairs:
                print(f"Warning: The following L-R pairs were not found: {missing_pairs}")
                print(f"Available pairs: {list(available_pairs)[:10]}...")  # 显示前10个
            
            if not valid_pairs:
                raise ValueError(f"None of the specified L-R pairs were found in the data")
            
            # 计算特定配体-受体对的通讯矩阵
            lr_matrix = np.zeros((self.n_cell_types, self.n_cell_types))
            
            # 筛选特定配体-受体对的交互
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
                
                # 获取该配体-受体对的显著交互
                pvals = self.adata.layers['pvalues'][i, lr_indices]
                means = self.adata.layers['means'][i, lr_indices]
                
                sig_mask = pvals < pvalue_threshold
                if np.any(sig_mask):
                    # 使用交互强度的平均值作为权重
                    lr_matrix[sender_idx, receiver_idx] += np.mean(means[sig_mask])
            
            matrix = lr_matrix
            title_suffix = f" - L-R: {', '.join(valid_pairs[:3])}{'...' if len(valid_pairs) > 3 else ''}"
        else:
            # 使用聚合的交互计数矩阵
            count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
            matrix = weight_matrix  # 使用权重矩阵以更好地反映交互强度
            title_suffix = " - All L-R pairs"
        
        # 过滤指定的发送者和接收者
        if sources is not None or targets is not None:
            # 验证指定的细胞类型
            if sources is not None:
                invalid_sources = [ct for ct in sources if ct not in self.cell_types]
                if invalid_sources:
                    raise ValueError(f"Invalid source cell types: {invalid_sources}. Available: {self.cell_types}")
            
            if targets is not None:
                invalid_targets = [ct for ct in targets if ct not in self.cell_types]
                if invalid_targets:
                    raise ValueError(f"Invalid target cell types: {invalid_targets}. Available: {self.cell_types}")
            
            # 创建过滤后的矩阵
            filtered_matrix = np.zeros_like(matrix)
            for i, sender_type in enumerate(self.cell_types):
                for j, receiver_type in enumerate(self.cell_types):
                    # 检查是否符合sources/targets条件
                    sender_ok = (sources is None) or (sender_type in sources)
                    receiver_ok = (targets is None) or (receiver_type in targets)
                    
                    if sender_ok and receiver_ok:
                        filtered_matrix[i, j] = matrix[i, j]
            
            matrix = filtered_matrix
        
        # 使用细胞类型名称
        final_matrix = matrix
        final_names = self.cell_types
        
        # 过滤低于阈值的交互
        final_matrix[final_matrix < count_min] = 0
        
        # 检查是否还有交互
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
        
        # 准备颜色
        if colors is None:
            # 使用细胞类型颜色
            cell_colors = self._get_cell_type_colors()
            colors = [cell_colors.get(node, '#1f77b4') for node in final_names]
        
        # 创建图形
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        self.test_final_matrix = final_matrix
        
        # 修改名称：只隐藏既不发送也不接收信号的细胞类型名字
        display_names = final_names.copy()
        if normalize_to_sender:
            # 计算行总和（发送的信号）和列总和（接收的信号）
            row_sums = final_matrix.sum(axis=1)
            col_sums = final_matrix.sum(axis=0)
            for i in range(len(final_names)):
                # 只有既不发送也不接收信号的细胞类型才隐藏名字
                if row_sums[i] == 0 and col_sums[i] == 0:
                    display_names[i] = ""  # 隐藏名字
        
        # 绘制弦图
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
        
        # 添加标题
        if title_name is None:
            title_name = f"Ligand-Receptor Communication{title_suffix}"
        ax.set_title(title_name, fontsize=fontsize + 4, pad=20)
        
        # 保存文件
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"L-R Chord diagram saved as: {save}")
        
        return fig, ax

    def get_ligand_receptor_pairs(self, min_interactions=1, pvalue_threshold=0.05):
        """
        获取所有显著的配体-受体对列表
        
        Parameters:
        -----------
        min_interactions : int
            最小交互数量阈值
        pvalue_threshold : float
            P-value threshold for significance
        
        Returns:
        --------
        lr_pairs : list
            显著配体-受体对列表
        lr_stats : dict
            每个配体-受体对的统计信息
        """
        # 确定配体-受体对的列名
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
            
            # 计算该配体-受体对的总交互数
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
                            legend_bbox=(1.05, 1), legend_ncol=1):
        """
        绘制特定细胞类型作为发送者的所有配体-受体对弦图（基于基因级别）
        每个区域代表一个配体或受体，配体使用发送者颜色，受体使用接收者颜色
        
        Parameters:
        -----------
        sources_use : str, int, list or None
            发送者细胞类型。可以是：
            - 字符串：细胞类型名称
            - 整数：细胞类型索引（从0开始）
            - 列表：多个细胞类型
            - None：所有细胞类型作为发送者
        targets_use : str, int, list or None
            接收者细胞类型。可以是：
            - 字符串：细胞类型名称
            - 整数：细胞类型索引（从0开始）
            - 列表：多个细胞类型
            - None：所有细胞类型作为接收者
        signaling : str, list or None
            特定信号通路名称。可以是：
            - 字符串：单个信号通路名称
            - 列表：多个信号通路名称
            - None：所有信号通路
        pvalue_threshold : float
            P-value threshold for significant interactions
        mean_threshold : float
            平均表达强度阈值
        gap : float
            弦图各段之间的间隙
        use_gradient : bool
            是否使用渐变效果
        sort : str or None
            排序方式: "size", "distance", None
        directed : bool
            是否显示方向性
        chord_colors : str or None
            弦的颜色
        rotate_names : bool
            是否旋转名称
        fontcolor : str
            字体颜色
        fontsize : int
            字体大小
        start_at : int
            起始角度
        extent : int
            弦图覆盖的角度范围
        min_chord_width : int
            最小弦宽度
        ax : matplotlib.axes.Axes or None
            matplotlib轴对象
        figsize : tuple
            图形大小
        title_name : str or None
            图标题
        save : str or None
            保存文件路径
        legend_pos_x : float or None
            图例X位置（暂未实现）
        show_celltype_in_name : bool
            是否在节点名称中显示细胞类型信息 (default: True)
            如果True，显示为 "基因名(细胞类型)"
            如果False，只显示基因名，但同一基因在不同细胞类型中仍会重复出现
        show_legend : bool
            是否显示细胞类型颜色图例 (default: True)
        legend_bbox : tuple
            图例位置，格式为 (x, y) (default: (1.05, 1))
        legend_ncol : int
            图例列数 (default: 1)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        try:
            from ..externel.mpl_chord.chord_diagram import chord_diagram
        except ImportError:
            try:
                from mpl_chord_diagram import chord_diagram
            except ImportError:
                raise ImportError("mpl-chord-diagram package is required. Please install it: pip install mpl-chord-diagram")
        
        # 验证必需的列是否存在
        required_cols = ['gene_a', 'gene_b']
        missing_cols = [col for col in required_cols if col not in self.adata.var.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing from adata.var: {missing_cols}")
        
        # 处理信号通路过滤
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # 检查信号通路是否存在
            if 'classification' in self.adata.var.columns:
                available_pathways = self.adata.var['classification'].unique()
                missing_pathways = [p for p in signaling if p not in available_pathways]
                if missing_pathways:
                    print(f"Warning: The following signaling pathways were not found: {missing_pathways}")
                    print(f"Available pathways: {list(available_pathways)}")
                
                # 过滤出包含指定信号通路的交互
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
        
        # 处理发送者细胞类型
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
        
        # 处理接收者细胞类型
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
        
        # 收集显著的配体-受体交互
        ligand_receptor_interactions = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # 检查是否符合发送者和接收者条件
            if sender not in source_cell_types or receiver not in target_cell_types:
                continue
            
            # 获取显著交互
            pvals = self.adata.layers['pvalues'][i, :]
            means = self.adata.layers['means'][i, :]
            
            # 应用信号通路过滤
            if signaling_indices is not None:
                pvals = pvals[signaling_indices]
                means = means[signaling_indices]
                interaction_indices = signaling_indices
            else:
                interaction_indices = np.arange(len(pvals))
            
            sig_mask = (pvals < pvalue_threshold) & (means > mean_threshold)
            
            if np.any(sig_mask):
                # 获取显著交互的配体和受体
                # 使用原始索引来获取基因信息
                original_indices = interaction_indices[sig_mask]
                gene_a_values = self.adata.var['gene_a'].iloc[original_indices].values
                gene_b_values = self.adata.var['gene_b'].iloc[original_indices].values
                mean_values = means[sig_mask]
                
                for gene_a, gene_b, mean_val in zip(gene_a_values, gene_b_values, mean_values):
                    # 跳过NaN值
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
        
        # 创建配体-受体交互DataFrame
        lr_df = pd.DataFrame(ligand_receptor_interactions)
        
        # 新方法：为每个基因-细胞类型组合创建唯一节点，允许基因重复出现
        gene_celltype_combinations = set()
        
        # 收集所有配体-细胞类型组合
        for _, row in lr_df.iterrows():
            ligand = row['ligand']
            sender = row['sender']
            receptor = row['receptor']
            receiver = row['receiver']
            
            gene_celltype_combinations.add((ligand, sender, 'ligand'))
            gene_celltype_combinations.add((receptor, receiver, 'receptor'))
        
        # 按细胞类型分组节点，保持细胞类型聚集
        celltype_to_nodes = {}
        for gene, celltype, role in gene_celltype_combinations:
            if celltype not in celltype_to_nodes:
                celltype_to_nodes[celltype] = {'ligands': [], 'receptors': []}
            celltype_to_nodes[celltype][role + 's'].append(gene)
        
        # 组织节点列表：每个节点使用唯一标识符但显示时只显示基因名
        organized_nodes = []
        organized_node_info = []  # 存储节点信息 (gene, celltype, role)
        organized_display_names = []  # 存储显示名称
        
        # 按照原始细胞类型顺序排列
        available_celltypes = [ct for ct in self.cell_types if ct in celltype_to_nodes]
        
        node_counter = 0  # 用于创建唯一标识符
        for celltype in available_celltypes:
            nodes = celltype_to_nodes[celltype]
            
            # 先添加配体，再添加受体，并确保在同一细胞类型内去重和排序
            for ligand in sorted(set(nodes['ligands'])):
                # 使用唯一标识符作为内部节点名
                node_id = f"node_{node_counter}"
                organized_nodes.append(node_id)
                organized_node_info.append((ligand, celltype, 'ligand'))
                organized_display_names.append(ligand)  # 显示名称只是基因名
                node_counter += 1
            
            for receptor in sorted(set(nodes['receptors'])):
                # 使用唯一标识符作为内部节点名
                node_id = f"node_{node_counter}"
                organized_nodes.append(node_id)
                organized_node_info.append((receptor, celltype, 'receptor'))
                organized_display_names.append(receptor)  # 显示名称只是基因名
                node_counter += 1
        
        # 使用组织后的节点列表
        unique_genes = organized_nodes
        
        # 创建映射
        gene_to_celltype = {}
        for node_id, (gene, celltype, role) in zip(organized_nodes, organized_node_info):
            gene_to_celltype[node_id] = celltype
        
        # 创建交互矩阵（配体到受体）
        n_genes = len(unique_genes)
        interaction_matrix = np.zeros((n_genes, n_genes))
        
        # 填充矩阵 - 需要找到对应的节点ID
        for _, row in lr_df.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            sender = row['sender']
            receiver = row['receiver']
            
            # 找到对应的配体节点ID
            ligand_idx = None
            for i, (gene, celltype, role) in enumerate(organized_node_info):
                if gene == ligand and celltype == sender and role == 'ligand':
                    ligand_idx = i
                    break
            
            # 找到对应的受体节点ID
            receptor_idx = None
            for i, (gene, celltype, role) in enumerate(organized_node_info):
                if gene == receptor and celltype == receiver and role == 'receptor':
                    receptor_idx = i
                    break
            
            # 如果找到了对应的节点，就添加交互
            if ligand_idx is not None and receptor_idx is not None:
                interaction_matrix[ligand_idx, receptor_idx] += row['mean_expression']
        
        # 准备颜色：根据细胞类型着色
        cell_colors = self._get_cell_type_colors()
        gene_colors = []
        
        for node_id in unique_genes:
            associated_celltype = gene_to_celltype[node_id]
            gene_colors.append(cell_colors.get(associated_celltype, '#808080'))
        
        # 创建显示名称
        display_names = []
        ligands = lr_df['ligand'].unique()
        receptors = lr_df['receptor'].unique()
        
        for i, node_id in enumerate(unique_genes):
            gene, celltype, role = organized_node_info[i]
            
            # 根据参数选择显示格式
            if show_celltype_in_name:  # 显示完整名称（基因名+细胞类型）
                display_names.append(f"{gene}({celltype})")
            else:  # 只显示基因名，完全去掉括号
                # 直接使用基因名，颜色由图例说明
                display_names.append(gene)
        
        # 创建图形
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # 绘制弦图
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
        
        # 添加标题
        if title_name is None:
            source_str = ', '.join(source_cell_types) if len(source_cell_types) <= 3 else f"{len(source_cell_types)} cell types"
            target_str = ', '.join(target_cell_types) if len(target_cell_types) <= 3 else f"{len(target_cell_types)} cell types"
            title_name = f"Ligand-Receptor Interactions\nFrom: {source_str} → To: {target_str}"
            
            # 添加信号通路信息到标题
            if signaling is not None:
                signaling_str = ', '.join(signaling) if len(signaling) <= 3 else f"{len(signaling)} pathways"
                title_name += f"\nSignaling: {signaling_str}"
        
        ax.set_title(title_name, fontsize=fontsize + 2, pad=20)
        
        # 添加细胞类型颜色图例
        if show_legend:
            # 获取涉及的细胞类型和对应颜色
            involved_celltypes = set()
            for gene, celltype, role in organized_node_info:
                involved_celltypes.add(celltype)
            
            # 按原始顺序排序细胞类型
            ordered_celltypes = [ct for ct in self.cell_types if ct in involved_celltypes]
            
            # 创建图例
            legend_handles = []
            legend_labels = []
            
            for celltype in ordered_celltypes:
                color = cell_colors.get(celltype, '#808080')
                handle = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                legend_handles.append(handle)
                legend_labels.append(celltype)
            
            # 添加图例
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
            
            # 调整图例样式
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.5)
        
        # 保存文件
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"Gene-level chord diagram saved as: {save}")
        
        return fig, ax
    
    def netVisual_bubble_marsilea(self, sources_use=None, targets_use=None, 
                                 signaling=None, pvalue_threshold=0.05, 
                                 mean_threshold=0.1, top_interactions=20,
                                 show_pvalue=True, show_mean=True, show_count=False,
                                 add_violin=False, add_dendrogram=False,
                                 group_pathways=True, figsize=(12, 8),
                                 title="Cell-Cell Communication Analysis", 
                                 remove_isolate=False, font_size=12, cmap="RdBu_r",
                                 transpose=False):
        """
        使用Marsilea的SizedHeatmap创建高级气泡图来可视化细胞间通讯
        类似CellChat的netVisual_bubble功能，但使用SizedHeatmap使圆圈大小更有意义
        
        新功能特性:
        - 颜色深度代表表达强度 (红色越深表达越高)
        - 圆圈大小代表统计显著性 (只有两种大小)：
          * P < 0.01: 大圆圈 (显著)
          * P ≥ 0.01: 小圆圈或几乎看不见 (不显著)
        - 蓝色边框标记高度显著的交互 (P < 0.01)
        - 支持双重信息编码：颜色表达量+大小显著性
        
        Parameters:
        -----------
        sources_use : str, int, list or None
            发送者细胞类型。可以是：
            - 字符串：细胞类型名称
            - 整数：细胞类型索引（从0开始）
            - 列表：多个细胞类型
            - None：所有细胞类型作为发送者
        targets_use : str, int, list or None
            接收者细胞类型。同sources_use格式
        signaling : str, list or None
            特定信号通路名称。可以是：
            - 字符串：单个信号通路名称
            - 列表：多个信号通路名称
            - None：所有信号通路
        pvalue_threshold : float
            P-value threshold for significant interactions
        mean_threshold : float
            平均表达强度阈值
        top_interactions : int
            显示最强的前N个交互
        show_pvalue : bool
            是否显示P值信息
        show_mean : bool
            是否显示平均表达强度
        show_count : bool
            是否显示交互计数
        add_violin : bool
            是否添加小提琴图显示表达分布
        add_dendrogram : bool
            是否添加聚类树
        group_pathways : bool
            是否按信号通路分组
        figsize : tuple
            图形大小
        title : str
            图标题
        remove_isolate : bool
            是否移除孤立的交互
        font_size : int
            字体大小 (default: 12)
        cmap : str
            颜色映射 (default: "RdBu_r")
            可选: "Blues", "Greens", "Oranges", "Purples", "viridis", "plasma"等
        transpose : bool
            是否转置热图 (default: False)
            如果True，行列互换：行=L-R对，列=细胞类型对
            
        Returns:
        --------
        h : marsilea plot object
        """
        try:
            import marsilea as ma
            import marsilea.plotter as mp
            from matplotlib.colors import Normalize
            from sklearn.preprocessing import normalize
        except ImportError:
            raise ImportError("marsilea and sklearn packages are required. Please install them: pip install marsilea scikit-learn")
        
        # 处理发送者细胞类型
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
        
        # 处理接收者细胞类型
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
        
        # 处理信号通路过滤
        signaling_indices = None
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            if 'classification' in self.adata.var.columns:
                available_pathways = self.adata.var['classification'].unique()
                missing_pathways = [p for p in signaling if p not in available_pathways]
                if missing_pathways:
                    print(f"Warning: The following signaling pathways were not found: {missing_pathways}")
                    # 只保留存在的通路
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
        
        # 收集显著的配体-受体交互
        interactions_data = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                  self.adata.obs['receiver'])):
            # 检查是否符合发送者和接收者条件
            if sender not in source_cell_types or receiver not in target_cell_types:
                continue
            
            # 获取交互数据
            pvals = self.adata.layers['pvalues'][i, :]
            means = self.adata.layers['means'][i, :]
            
            # 应用信号通路过滤
            if signaling_indices is not None:
                pvals = pvals[signaling_indices]
                means = means[signaling_indices]
                var_indices = signaling_indices
            else:
                var_indices = np.arange(len(pvals))
            
            sig_mask = (pvals < pvalue_threshold) & (means > mean_threshold)
            
            if np.any(sig_mask):
                # 获取显著交互信息
                original_indices = var_indices[sig_mask]
                
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
                    
                    # 获取信号通路信息
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
        
        # 创建交互DataFrame
        df_interactions = pd.DataFrame(interactions_data)
        
        # 如果指定了信号通路，再次验证是否只包含指定通路的交互
        if signaling is not None:
            pathway_in_data = df_interactions['pathway'].unique()
            unexpected_pathways = [p for p in pathway_in_data if p not in signaling]
            if unexpected_pathways:
                print(f"⚠️  Warning: Found interactions from unexpected pathways: {unexpected_pathways}")
            
            # 严格过滤：只保留指定通路的交互
            df_interactions = df_interactions[df_interactions['pathway'].isin(signaling)]
            
            if len(df_interactions) == 0:
                print(f"❌ After filtering, no interactions remain for signaling pathway(s): {signaling}")
                return None
        
        # 移除孤立交互（如果需要）
        if remove_isolate:
            interaction_counts = df_interactions.groupby(['source', 'target']).size()
            valid_pairs = interaction_counts[interaction_counts > 1].index
            df_interactions = df_interactions[
                df_interactions.apply(lambda x: (x['source'], x['target']) in valid_pairs, axis=1)
            ]
        
        # 选择最强的交互
        if top_interactions and len(df_interactions) > top_interactions:
            df_interactions = df_interactions.nlargest(top_interactions, 'mean_expression')
        
        # 创建透视表
        if group_pathways:
            # 按信号通路分组
            pivot_mean = df_interactions.pivot_table(
                values='mean_expression', 
                index='interaction', 
                columns='pathway', 
                aggfunc='mean',
                fill_value=0
            )
            # 通路级别P值应该使用更合适的聚合方法
            # 选项1: 使用中位数 (更稳健)
            # 选项2: 使用几何平均数 
            # 选项3: 使用费舍尔合并P值方法
            
            # 这里使用中位数作为通路级别的代表P值 (更保守和稳健)
            pivot_pval = df_interactions.pivot_table(
                values='pvalue', 
                index='interaction', 
                columns='pathway', 
                aggfunc='median',  # 使用中位数而不是最小值
                fill_value=1
            )
            
            # 如果指定了信号通路，验证透视表的列只包含指定的通路
            if signaling is not None:
                pivot_pathways = set(pivot_mean.columns)
                specified_pathways = set(signaling)
                if not pivot_pathways.issubset(specified_pathways):
                    unexpected_in_pivot = pivot_pathways - specified_pathways
                    print(f"⚠️  Warning: Pivot table contains unexpected pathways: {unexpected_in_pivot}")
                    # 只保留指定的通路列
                    valid_columns = [col for col in pivot_mean.columns if col in signaling]
                    if not valid_columns:
                        print(f"❌ No valid pathway columns found for: {signaling}")
                        return None
                    pivot_mean = pivot_mean[valid_columns]
                    pivot_pval = pivot_pval[valid_columns]
        else:
            # 按配体-受体对分组
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
        
        # 标准化表达数据
        matrix_normalized = normalize(pivot_mean.to_numpy(), axis=0)
        
        # 创建Marsilea可视化组件 - 使用SizedHeatmap增强可视化
        # 重要：在pivot_table创建之后计算size和color矩阵，确保维度匹配
        
        # 准备数据：颜色=表达量，大小=P值显著性
        expression_matrix = pivot_mean.to_numpy()
        pval_matrix = pivot_pval.to_numpy()
        
        # 颜色矩阵：使用表达量，颜色越深表示表达越高
        color_matrix = expression_matrix.copy()
        # 确保没有NaN或Inf值
        color_matrix = np.nan_to_num(color_matrix, nan=0.0, posinf=color_matrix[np.isfinite(color_matrix)].max(), neginf=0.0)
        
        # 大小矩阵：使用负对数转换P值，P值越小圆圈越大
        # -log10(p-value): P=0.01 → size=2, P=0.05 → size=1.3, P=0.1 → size=1
        size_matrix = -np.log10(pval_matrix + 1e-10)  # 添加小值避免log(0)
        
        # 归一化到合理的视觉范围 (0.2 到 1.0)
        # 这样P值越小，圆圈越大
        size_min = 0.2  # 最小圆圈大小 (对应不显著的P值)
        size_max = 1.0  # 最大圆圈大小 (对应高度显著的P值)
        
        # 归一化: 将-log10(p)映射到[size_min, size_max]范围
        if size_matrix.max() > size_matrix.min():
            size_matrix_norm = (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
            size_matrix = size_matrix_norm * (size_max - size_min) + size_min
        else:
            # 当所有P值相同时，添加轻微的随机误差避免可视化问题
            print("⚠️  Warning: All p-values are identical. Adding slight jitter for better visualization.")
            
            # 设置随机种子以保证结果可重现
            np.random.seed(42)
            
            # 在原始P值基础上添加很小的随机误差（不影响统计意义）
            jitter_strength = 1e-2  # 非常小的误差，不会影响统计解释
            jittered_pvals = pval_matrix + np.random.normal(0, jitter_strength, pval_matrix.shape)
            
            # 确保P值仍在合理范围内 [0, 1]
            jittered_pvals = np.clip(jittered_pvals, 1e-10, 1.0)
            
            # 重新计算size_matrix
            size_matrix = -np.log10(pval_matrix + 1e-10)
            
            
            # 重新归一化
            if size_matrix.max() > size_matrix.min():
                size_matrix_norm = (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
                size_matrix = size_matrix_norm * (size_max - size_min) + size_min
                
            else:
                # 如果添加误差后仍然相同（极端情况），使用中等大小
                print("⚠️  Warning: All p-values are identical after jittering. Using medium size.")
                size_matrix = np.full_like(size_matrix, (size_min + size_max) / 2)
                size_matrix=color_matrix
        

        
        # 转置功能 - 需要保存原始pivot用于后续层
        original_pivot_mean = pivot_mean.copy()
        original_pivot_pval = pivot_pval.copy()
        
        if transpose:
            size_matrix = size_matrix.T
            color_matrix = color_matrix.T
            pivot_mean = pivot_mean.T
            pivot_pval = pivot_pval.T
            # 注意：转置后 matrix_normalized 也需要重新计算
            matrix_normalized = normalize(pivot_mean.to_numpy(), axis=0)
        
        # 1. 主要的SizedHeatmap - 基于您的参考代码改进
        h = ma.SizedHeatmap(
            size=size_matrix,
            color=color_matrix,
            cmap=cmap,  # 使用自定义颜色映射
            width=figsize[0] * 0.6, 
            height=figsize[1] * 0.7,
            legend=True,
            size_legend_kws=dict(
                colors="black",
                title="",
                labels=["p>0.05", "p<0.01"],
                show_at=[0.01, 1.0],
            ),
            color_legend_kws=dict(title="Expression Level"),
        )
        
        # 2. 可选的额外显著性标记层
        if show_pvalue:
            try:
                # 使用转置后的pval_matrix计算显著性
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
        
        # 3. 高表达标记
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
        
        # 4. 细胞类型标签 - 基于您的参考代码
        cell_interaction_labels = mp.Labels(
            pivot_mean.index, 
            align="center",
            fontsize=font_size
        )
        h.add_left(cell_interaction_labels)
        
        # 5. 配体-受体对或通路标签 - 基于您的参考代码
        lr_pathway_labels = mp.Labels(
            pivot_mean.columns,
            fontsize=font_size
        )
        h.add_bottom(lr_pathway_labels)
        
        # 6. 按信号通路或功能分组 (simplified version for SizedHeatmap)
        if group_pathways and 'classification' in self.adata.var.columns:
            # 获取信号通路的颜色映射
            unique_pathways = pivot_mean.columns.tolist()
            pathway_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pathways)))
            pathway_color_map = {pathway: mcolors.to_hex(color) 
                               for pathway, color in zip(unique_pathways, pathway_colors)}
            
            # Note: Group functionality simplified for SizedHeatmap compatibility
        
        # 7. 聚类树 (带更严格的安全检查)
        if add_dendrogram:
            try:
                # 检查数据维度和质量是否足够进行聚类
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
        
        # 8. 图例 - 基于您的参考代码
        h.add_legends()
        
        # 9. 设置标题
        if title:
            h.add_title(title, fontsize=font_size + 2, pad=0.02)  # 标题字体比正文大2
        
        # 渲染图形
        h.render()
        
        print(f"📊 可视化统计:")
        print(f"   - 显著交互数量: {len(df_interactions)}")
        print(f"   - 细胞类型对: {len(pivot_mean.index)}")
        print(f"   - {'信号通路' if group_pathways else '配体-受体对'}: {len(pivot_mean.columns)}")
        
        return h
    
    def netAnalysis_computeCentrality(self, signaling=None, slot_name="netP", 
                                     pvalue_threshold=0.05, use_weight=True):
        """
        计算网络中心性指标（模仿CellChat的netAnalysis_computeCentrality功能）
        
        计算以下中心性指标并转换为CellChat风格的Importance值（0-1范围）：
        - out_degree: 出度（主要发送者角色）
        - in_degree: 入度（主要接收者角色）
        - flow_betweenness: 流中介性（中介者角色）
        - information_centrality: 信息中心性（影响者角色）
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称。如果为None，使用所有通路的聚合网络
        slot_name : str
            数据插槽名称（兼容CellChat，这里用于标识计算类型）
        pvalue_threshold : float
            P-value threshold for significant interactions
        use_weight : bool
            是否使用权重（交互强度）进行计算
            
        Returns:
        --------
        centrality_scores : dict
            包含各种中心性指标的字典，所有值均为0-1范围的Importance值
        """
        try:
            import networkx as nx
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path
        except ImportError:
            raise ImportError("NetworkX and SciPy are required for centrality analysis")
        
        # 计算通讯矩阵
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            
            # 检查信号通路是否存在
            if 'classification' in self.adata.var.columns:
                available_pathways = self.adata.var['classification'].unique()
                missing_pathways = [p for p in signaling if p not in available_pathways]
                if missing_pathways:
                    print(f"Warning: The following signaling pathways were not found: {missing_pathways}")
                    signaling = [p for p in signaling if p in available_pathways]
                    if not signaling:
                        raise ValueError("No valid signaling pathways provided")
                
                # 计算特定通路的通讯矩阵
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
            # 使用聚合网络
            count_matrix, weight_matrix = self.compute_aggregated_network(pvalue_threshold)
            comm_matrix = weight_matrix if use_weight else count_matrix
        
        # 创建NetworkX图
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_cell_types))
        
        # 添加边
        for i in range(self.n_cell_types):
            for j in range(self.n_cell_types):
                if comm_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=comm_matrix[i, j])
        
        # 计算原始中心性指标
        raw_centrality_scores = {}
        
        # 1. 出度中心性 (Outdegree) - 识别主要发送者
        out_degree = np.array([comm_matrix[i, :].sum() for i in range(self.n_cell_types)])
        raw_centrality_scores['outdegree'] = out_degree
        
        # 2. 入度中心性 (Indegree) - 识别主要接收者
        in_degree = np.array([comm_matrix[:, j].sum() for j in range(self.n_cell_types)])
        raw_centrality_scores['indegree'] = in_degree
        
        # 3. 流中介性 (Flow Betweenness) - 识别中介者
        try:
            if len(G.edges()) > 0:
                # 使用NetworkX的中介中心性作为流中介性的近似
                betweenness = nx.betweenness_centrality(G, weight='weight')
                flow_betweenness = np.array([betweenness.get(i, 0) for i in range(self.n_cell_types)])
            else:
                flow_betweenness = np.zeros(self.n_cell_types)
        except:
            print("Warning: Failed to compute betweenness centrality, using zeros")
            flow_betweenness = np.zeros(self.n_cell_types)
        
        raw_centrality_scores['flow_betweenness'] = flow_betweenness
        
        # 4. 信息中心性 (Information Centrality) - 识别影响者
        try:
            if len(G.edges()) > 0:
                # 使用eigenvector centrality作为信息中心性的近似
                # 对于有向图，使用入度的eigenvector centrality
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
        
        # 将原始中心性分数转换为CellChat风格的Importance值（0-1范围）
        centrality_scores = {}
        for metric, scores in raw_centrality_scores.items():
            if scores.max() > scores.min() and scores.max() > 0:
                # 标准化到0-1范围，确保CellChat兼容性
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                # 如果所有值相同或都为0，则设为0
                normalized_scores = np.zeros_like(scores)
            
            centrality_scores[metric] = normalized_scores
        
        # 5. 总中心性 (Overall) - 综合指标（已标准化）
        overall_centrality = (centrality_scores['outdegree'] + 
                            centrality_scores['indegree'] + 
                            centrality_scores['flow_betweenness'] + 
                            centrality_scores['information']) / 4
        
        centrality_scores['overall'] = overall_centrality
        
        # 存储原始分数和标准化分数
        self.raw_centrality_scores = raw_centrality_scores  # 保存原始分数用于调试
        self.centrality_scores = centrality_scores  # CellChat风格的Importance值
        self.centrality_matrix = comm_matrix
        
        print(f"✅ 网络中心性计算完成（CellChat风格Importance值）")
        print(f"   - 使用信号通路: {signaling if signaling else 'All pathways'}")
        print(f"   - 权重模式: {'Weighted' if use_weight else 'Unweighted'}")
        print(f"   - 计算指标: outdegree, indegree, flow_betweenness, information, overall")
        print(f"   - 所有中心性分数已标准化到0-1范围（Importance值）")
        
        return centrality_scores
    
    def netAnalysis_signalingRole_network(self, signaling=None, measures=None,
                                        color_heatmap="RdYlBu_r", 
                                        width=12, height=8, font_size=10,
                                        title="Signaling Role Analysis",
                                        cluster_rows=True, cluster_cols=False,
                                        save=None, show_values=True):
        """
        可视化细胞群的信号传导角色（模仿CellChat的netAnalysis_signalingRole_network功能）
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称。如果为None，使用存储的中心性结果或计算聚合网络
        measures : list or None
            要显示的中心性指标。默认显示所有指标
        color_heatmap : str
            热图颜色映射
        width : float
            图形宽度
        height : float
            图形高度
        font_size : int
            字体大小
        title : str
            图形标题
        cluster_rows : bool
            是否对行进行聚类
        cluster_cols : bool
            是否对列进行聚类
        save : str or None
            保存路径
        show_values : bool
            是否在热图中显示数值
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        """
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
        
        # 使用seaborn创建热图
        fig, ax = plt.subplots(figsize=(width, height))
        
        # 绘制热图，使用CellChat风格的配置
        sns.heatmap(df_centrality, 
                   annot=show_values, 
                   fmt='.2f' if show_values else '',  # 使用2位小数，因为是0-1范围
                   cmap=color_heatmap,
                   cbar_kws={'label': 'Importance'},  # CellChat风格的标签
                   square=False,
                   linewidths=0.5,
                   ax=ax,
                   xticklabels=True,
                   yticklabels=True,
                   vmin=0,  # 确保颜色范围从0开始
                   vmax=1)  # 确保颜色范围到1结束
        
        # 设置标签和标题
        ax.set_xlabel('Cell Groups', fontsize=font_size + 2)  # 使用CellChat风格的标签
        ax.set_ylabel('', fontsize=font_size + 2)  # CellChat中Y轴通常不显示标签
        ax.set_title(title, fontsize=font_size + 4, pad=20)
        
        # 调整字体大小
        ax.tick_params(axis='x', labelsize=font_size, rotation=45)
        ax.tick_params(axis='y', labelsize=font_size, rotation=0)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Signaling role heatmap saved as: {save}")
        
        print(f"📊 信号角色分析结果（Importance值 0-1）:")
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
        创建2D散点图来可视化细胞的信号传导角色
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称
        x_measure : str
            X轴使用的中心性指标
        y_measure : str  
            Y轴使用的中心性指标
        figsize : tuple
            图形大小
        point_size : int
            散点大小
        alpha : float
            透明度
        title : str
            图形标题
        save : str or None
            保存路径
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        # 如果没有预计算的中心性分数，先计算
        if not hasattr(self, 'centrality_scores') or signaling is not None:
            self.netAnalysis_computeCentrality(signaling=signaling)
        
        centrality_scores = self.centrality_scores
        
        # 验证指标
        if x_measure not in centrality_scores:
            raise ValueError(f"x_measure '{x_measure}' not found in centrality scores")
        if y_measure not in centrality_scores:
            raise ValueError(f"y_measure '{y_measure}' not found in centrality scores")
        
        # 获取数据
        x_data = centrality_scores[x_measure]
        y_data = centrality_scores[y_measure]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取细胞类型颜色
        cell_colors = self._get_cell_type_colors()
        colors = [cell_colors.get(ct, '#1f77b4') for ct in self.cell_types]
        
        # 绘制散点图
        scatter = ax.scatter(x_data, y_data, 
                           c=colors, s=point_size, alpha=alpha,
                           edgecolors='black', linewidths=0.5)
        
        # 添加细胞类型标签
        try:
            from adjustText import adjust_text
            
            texts = []
            for i, cell_type in enumerate(self.cell_types):
                text = ax.text(x_data[i], y_data[i], cell_type,
                             fontsize=10, alpha=0.8, ha='center', va='center',)
                texts.append(text)
            
            # 使用adjust_text防止标签重叠
            adjust_text(texts, ax=ax,
                      expand_points=(1.2, 1.2),
                      expand_text=(1.2, 1.2),
                      force_points=0.3,
                      force_text=0.3,
                      arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.8))
            
        except ImportError:
            import warnings
            warnings.warn("adjustText library not found. Using default ax.annotate instead.")
            # 回退到原始的annotate方法
            for i, cell_type in enumerate(self.cell_types):
                ax.annotate(cell_type, (x_data[i], y_data[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8)
        
        # 设置标签和标题
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
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save:
            fig.savefig(save, dpi=300, bbox_inches='tight')
            print(f"2D signaling role plot saved as: {save}")
        
        return fig, ax
    
    def netAnalysis_contribution(self, signaling, group_celltype=None, 
                               sources=None, targets=None,
                               pvalue_threshold=0.05, top_pairs=10,
                               figsize=(12, 8), font_size=10,
                               title=None, save=None):
        """
        分析特定信号通路中配体-受体对的贡献
        回答：哪些信号对特定细胞群的传出或传入信号贡献最大
        
        Parameters:
        -----------
        signaling : str or list
            要分析的信号通路
        group_celltype : str or None
            要分析的特定细胞类型。如果为None，分析所有细胞类型
        sources : list or None
            关注的发送者细胞类型
        targets : list or None
            关注的接收者细胞类型
        pvalue_threshold : float
            P-value threshold
        top_pairs : int
            显示前N个贡献最大的配体-受体对
        figsize : tuple
            图形大小
        font_size : int
            字体大小
        title : str or None
            图形标题
        save : str or None
            保存路径
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        contribution_df : pandas.DataFrame
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
                                                  save=None):
        """
        使用Marsilea创建高级信号角色热图（CellChat风格的netAnalysis_signalingRole_network）
        
        Parameters:
        -----------
        signaling : str, list or None
            特定信号通路名称。如果为None，使用存储的中心性结果或计算聚合网络
        measures : list or None
            要显示的中心性指标。默认显示所有指标
        color_heatmap : str
            热图颜色映射
        width : float
            图形宽度
        height : float
            图形高度
        font_size : int
            字体大小
        title : str
            图形标题
        add_dendrogram : bool
            是否添加聚类树
        add_cell_colors : bool
            是否添加细胞类型颜色条
        add_importance_bars : bool
            是否添加Importance值的柱状图
        show_values : bool
            是否在热图中显示数值
        save : str or None
            保存路径
            
        Returns:
        --------
        h : marsilea plot object
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
                    rotation=45,
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
        
        return h
    
    def demo_curved_arrows(self, signaling_pathway=None, curve_strength=0.4, figsize=(12, 10)):
        """
        演示弯曲箭头效果的示例函数
        
        Parameters:
        -----------
        signaling_pathway : str or None
            要可视化的信号通路，如果为None则使用聚合网络
        curve_strength : float
            箭头弯曲强度 (0-1), 0为直线，越大越弯曲
        figsize : tuple
            图片大小
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
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
        
        Parameters:
        -----------
        count_min : int
            Minimum count threshold to filter interactions (default: 1)
            
        Returns:
        --------
        mean_matrix : pd.DataFrame
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
        
        Parameters:
        -----------
        count_min : int
            Minimum count threshold to filter interactions (default: 1)
            
        Returns:
        --------
        pvalue_matrix : pd.DataFrame
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
        
        Parameters:
        -----------
        pathway_stats : dict
            Dictionary returned from get_signaling_pathways
        show_details : bool
            Whether to show detailed statistics for each pathway
        
        Returns:
        --------
        summary_df : pd.DataFrame
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
        
        Parameters:
        -----------
        method : str
            聚合方法: 'mean', 'sum', 'max', 'median' (default: 'mean')
        min_lr_pairs : int
            通路中最少L-R对数量 (default: 1)  
        min_expression : float
            最小表达阈值 (default: 0.1)
            
        Returns:
        --------
        pathway_communication : dict
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
        
        Parameters:
        -----------
        pathway_communication : dict or None
            通路通讯结果，如果为None则重新计算
        strength_threshold : float
            通路强度阈值 (default: 0.1)
        pvalue_threshold : float  
            p-value阈值 (default: 0.05)
        min_significant_pairs : int
            最少显著细胞对数量 (default: 1)
            
        Returns:
        --------
        significant_pathways : list
            显著通路列表
        pathway_summary : pd.DataFrame
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
        计算每个配体-受体对对整体信号通路的贡献并可视化
        (类似CellChat的netAnalysis_contribution功能)
        
        Parameters:
        -----------
        signaling : str or list
            信号通路名称
        pvalue_threshold : float
            P-value阈值 (default: 0.05)
        mean_threshold : float  
            平均表达阈值 (default: 0.1)
        top_pairs : int
            显示的top L-R对数量 (default: 10)
        figsize : tuple
            图形大小 (default: (10, 6))
        save : str or None
            保存路径 (default: None)
            
        Returns:
        --------
        contribution_df : pd.DataFrame
            L-R对贡献统计
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # 过滤指定通路的交互
        pathway_mask = self.adata.var['classification'].isin(signaling)
        if not pathway_mask.any():
            raise ValueError(f"No interactions found for signaling pathway(s): {signaling}")
        
        # 计算每个L-R对的贡献
        contributions = []
        
        for var_idx in np.where(pathway_mask)[0]:
            lr_pair = self.adata.var.iloc[var_idx]['interacting_pair']
            gene_a = self.adata.var.iloc[var_idx]['gene_a']
            gene_b = self.adata.var.iloc[var_idx]['gene_b']
            classification = self.adata.var.iloc[var_idx]['classification']
            
            # 计算这个L-R对在所有细胞对中的总强度和显著性
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
            
            if total_strength > 0:  # 只包含有活性的L-R对
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
        
        # 转换为DataFrame并排序
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
        提取指定信号通路中的所有显著L-R对
        (类似CellChat的extractEnrichedLR功能)
        
        Parameters:
        -----------
        signaling : str or list
            信号通路名称
        pvalue_threshold : float
            P-value阈值 (default: 0.05)
        mean_threshold : float
            平均表达阈值 (default: 0.1)  
        min_cell_pairs : int
            最少显著细胞对数量 (default: 1)
        geneLR_return : bool
            是否返回基因级别信息 (default: False)
            
        Returns:
        --------
        enriched_lr : pd.DataFrame
            显著的L-R对信息
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # 过滤指定通路的交互
        pathway_mask = self.adata.var['classification'].isin(signaling)
        if not pathway_mask.any():
            raise ValueError(f"No interactions found for signaling pathway(s): {signaling}")
        
        enriched_pairs = []
        
        for var_idx in np.where(pathway_mask)[0]:
            lr_pair = self.adata.var.iloc[var_idx]['interacting_pair']
            gene_a = self.adata.var.iloc[var_idx]['gene_a']
            gene_b = self.adata.var.iloc[var_idx]['gene_b']
            classification = self.adata.var.iloc[var_idx]['classification']
            
            # 计算显著性统计
            significant_cell_pairs = []
            total_strength = 0
            
            for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                     self.adata.obs['receiver'])):
                pval = self.adata.layers['pvalues'][i, var_idx]
                mean_expr = self.adata.layers['means'][i, var_idx]
                
                if pval < pvalue_threshold and mean_expr > mean_threshold:
                    significant_cell_pairs.append(f"{sender}|{receiver}")
                    total_strength += mean_expr
            
            # 只包含满足条件的L-R对
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
                
                # 如果需要基因级别信息
                if geneLR_return:
                    # 添加更详细的基因信息
                    var_info = self.adata.var.iloc[var_idx]
                    for col in var_info.index:
                        if col not in pair_info:
                            pair_info[col] = var_info[col]
                
                enriched_pairs.append(pair_info)
        
        if not enriched_pairs:
            print(f"No enriched L-R pairs found for pathway(s): {signaling}")
            return pd.DataFrame()
        
        # 转换为DataFrame并按显著性排序
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
        可视化单个配体-受体对介导的细胞间通讯
        (类似CellChat的netVisual_individual功能)
        
        Parameters:
        -----------
        signaling : str or list
            信号通路名称
        pairLR_use : str, dict, or pd.Series
            要显示的L-R对。可以是：
            - 字符串：L-R对名称 (如 "TGFB1_TGFBR1")
            - 字典：包含ligand和receptor的字典
            - pandas Series：extractEnrichedLR返回的行
        sources_use : list or None
            指定的发送者细胞类型
        targets_use : list or None  
            指定的接收者细胞类型
        layout : str
            布局类型：'hierarchy', 'circle' (default: 'hierarchy')
        vertex_receiver : list or None
            指定接收者位置的数值向量(仅hierarchy布局)
        pvalue_threshold : float
            显著性阈值 (default: 0.05)
        edge_width_max : float
            最大边宽度 (default: 8)
        vertex_size_max : float
            最大节点大小 (default: 50)
        figsize : tuple
            图形大小 (default: (10, 8))
        title : str or None
            图标题
        save : str or None
            保存路径
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        if isinstance(signaling, str):
            signaling = [signaling]
        
        # 解析pairLR_use参数
        if pairLR_use is None:
            # 如果未指定，选择第一个enriched L-R对
            enriched_lr = self.extractEnrichedLR(signaling, pvalue_threshold)
            if enriched_lr.empty:
                raise ValueError(f"No enriched L-R pairs found for {signaling}")
            pairLR_use = enriched_lr.iloc[0]
        
        # 处理不同类型的pairLR_use输入
        if isinstance(pairLR_use, str):
            # 假设是ligand_receptor格式
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
        
        # 找到对应的L-R对
        lr_mask = (self.adata.var['gene_a'] == ligand) & (self.adata.var['gene_b'] == receptor)
        if signaling:
            pathway_mask = self.adata.var['classification'].isin(signaling)
            lr_mask = lr_mask & pathway_mask
        
        if not lr_mask.any():
            raise ValueError(f"L-R pair {lr_pair} not found in pathway(s) {signaling}")
        
        var_idx = np.where(lr_mask)[0][0]
        
        # 收集显著的细胞间通讯
        communications = []
        
        for i, (sender, receiver) in enumerate(zip(self.adata.obs['sender'], 
                                                 self.adata.obs['receiver'])):
            # 应用细胞类型过滤
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
        
        # 创建通讯DataFrame
        comm_df = pd.DataFrame(communications)
        
        # 创建图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取细胞类型颜色
        cell_colors = self._get_cell_type_colors()
        
        if layout == 'hierarchy':
            self._draw_hierarchy_plot(comm_df, ax, cell_colors, vertex_receiver, 
                                    edge_width_max, vertex_size_max)
        elif layout == 'circle':
            self._draw_circle_plot(comm_df, ax, cell_colors, edge_width_max, vertex_size_max)
        else:
            raise ValueError("layout must be 'hierarchy' or 'circle'")
        
        # 设置标题
        if title is None:
            title = f"{ligand} → {receptor} Communication\nPathway: {', '.join(signaling)}"
        ax.set_title(title, fontsize=14, pad=20)
        
        # 保存
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Individual communication plot saved as: {save}")
        
        return fig, ax
    
    def _draw_hierarchy_plot(self, comm_df, ax, cell_colors, vertex_receiver, 
                           edge_width_max, vertex_size_max):
        """绘制层次图"""
        # 获取唯一的发送者和接收者
        senders = comm_df['sender'].unique()
        receivers = comm_df['receiver'].unique()
        
        # 设置位置
        if vertex_receiver is not None:
            # 用户指定接收者位置
            y_positions = {}
            for i, receiver in enumerate(receivers):
                if i < len(vertex_receiver):
                    y_positions[receiver] = vertex_receiver[i]
                else:
                    y_positions[receiver] = i + 1
        else:
            # 自动分配位置
            y_positions = {receiver: i for i, receiver in enumerate(receivers)}
        
        # 发送者位置（左侧）
        sender_y = np.linspace(0, max(y_positions.values()), len(senders))
        sender_pos = {sender: (0.2, y) for sender, y in zip(senders, sender_y)}
        
        # 接收者位置（右侧）
        receiver_pos = {receiver: (0.8, y_positions[receiver]) for receiver in receivers}
        
        # 绘制节点
        max_strength = comm_df['strength'].max()
        
        for sender, (x, y) in sender_pos.items():
            # 计算节点大小（基于发送强度）
            sender_strength = comm_df[comm_df['sender'] == sender]['strength'].sum()
            size = (sender_strength / max_strength) * vertex_size_max + 20
            
            color = cell_colors.get(sender, '#lightblue')
            circle = plt.Circle((x, y), 0.05, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            ax.text(x-0.1, y, sender, ha='right', va='center', fontsize=10, weight='bold')
        
        for receiver, (x, y) in receiver_pos.items():
            # 计算节点大小（基于接收强度）
            receiver_strength = comm_df[comm_df['receiver'] == receiver]['strength'].sum()
            size = (receiver_strength / max_strength) * vertex_size_max + 20
            
            color = cell_colors.get(receiver, '#lightcoral')
            circle = plt.Circle((x, y), 0.05, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            ax.text(x+0.1, y, receiver, ha='left', va='center', fontsize=10, weight='bold')
        
        # 绘制边
        for _, row in comm_df.iterrows():
            sender = row['sender']
            receiver = row['receiver']
            strength = row['strength']
            
            if sender in sender_pos and receiver in receiver_pos:
                x1, y1 = sender_pos[sender]
                x2, y2 = receiver_pos[receiver]
                
                # 边宽度
                width = (strength / max_strength) * edge_width_max
                
                # 绘制箭头
                from matplotlib.patches import FancyArrowPatch
                arrow = FancyArrowPatch((x1+0.05, y1), (x2-0.05, y2),
                                      arrowstyle='->', mutation_scale=20,
                                      linewidth=width, color='gray', alpha=0.6)
                ax.add_patch(arrow)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, max(y_positions.values()) + 0.5)
        ax.axis('off')
    
    def _draw_circle_plot(self, comm_df, ax, cell_colors, edge_width_max, vertex_size_max):
        """绘制圆形图"""
        # 获取所有唯一的细胞类型
        all_cells = list(set(comm_df['sender'].tolist() + comm_df['receiver'].tolist()))
        n_cells = len(all_cells)
        
        # 创建圆形位置
        angles = np.linspace(0, 2*np.pi, n_cells, endpoint=False)
        positions = {cell: (np.cos(angle), np.sin(angle)) for cell, angle in zip(all_cells, angles)}
        
        # 计算节点大小
        cell_strengths = {}
        for cell in all_cells:
            send_strength = comm_df[comm_df['sender'] == cell]['strength'].sum()
            receive_strength = comm_df[comm_df['receiver'] == cell]['strength'].sum()
            cell_strengths[cell] = send_strength + receive_strength
        
        max_strength = max(cell_strengths.values()) if cell_strengths else 1
        
        # 绘制节点
        for cell, (x, y) in positions.items():
            size = (cell_strengths[cell] / max_strength) * vertex_size_max + 100
            color = cell_colors.get(cell, '#lightgray')
            
            circle = plt.Circle((x, y), 0.1, color=color, alpha=0.7, zorder=3)
            ax.add_patch(circle)
            ax.text(x*1.2, y*1.2, cell, ha='center', va='center', fontsize=10, weight='bold')
        
        # 绘制边
        edge_max = comm_df['strength'].max()
        for _, row in comm_df.iterrows():
            sender = row['sender']
            receiver = row['receiver']
            strength = row['strength']
            
            if sender in positions and receiver in positions:
                x1, y1 = positions[sender]
                x2, y2 = positions[receiver]
                
                # 边宽度
                width = (strength / edge_max) * edge_width_max
                
                # 绘制弯曲箭头
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
    