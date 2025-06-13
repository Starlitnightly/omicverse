# 扩展版本：支持所有模式的3D可视化
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from scipy.integrate import odeint

# 兼容性导入
try:
    import plotly.colors.qualitative as pq
except ImportError:
    try:
        import plotly.colors as pc
        pq = pc.qualitative
    except ImportError:
        # 如果plotly不可用，提供备用颜色
        class MockColors:
            Plotly = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
            Light24 = ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB']
            Dark24 = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100', '#750D86', '#EB663B', '#511CFB']
            Alphabet = ['#AA0DFE', '#3283FE', '#85660D', '#782AB6', '#565656', '#1C8356', '#16FF32', '#F7E1A0', '#E2E2E2', '#1CBE4F']
        pq = MockColors()

def plot_cell_communication_3D(
    adata,
    database_name: str = None,
    pathway_name: str = None,
    lr_pair = None,
    keys = None,
    plot_method: str = "cell",
    background: str = "summary",
    background_legend: bool = False,
    clustering: str = None,
    summary: str = "sender",
    cmap: str = "coolwarm",
    cluster_cmap: dict = None,
    pos_idx: np.ndarray = np.array([0, 1], int),  # 现在支持[0,1,2]
    ndsize: float = 1,
    scale: float = 1.0,
    normalize_v: bool = False,
    normalize_v_quantile: float = 0.95,
    arrow_color: str = "#333333",
    grid_density: float = 1.0,
    grid_knn: int = None,
    grid_scale: float = 1.0,
    grid_thresh: float = 1.0,
    grid_width: float = 0.005,
    stream_density: float = 1.0,
    stream_linewidth: float = 1,
    stream_cutoff_perc: float = 5,
    filename: str = None,
    ax = None,
    # 新增3D参数
    grid_cutoff_perc: float = 90,
    html_3d: bool = False,  # 强制生成HTML 3D输出
    camera_eye: dict = None  # 3D相机位置
):
    """
    扩展版本：支持所有模式的3D可视化
    
    新参数:
    html_3d : bool, default False
        如果为True，生成3D HTML而不是matplotlib图
    camera_eye : dict, optional
        3D相机位置，例如 {'x': 1.5, 'y': 1.5, 'z': 1.5}
    """
    
    # 检测是否为3D模式
    is_3d = len(pos_idx) == 3 or html_3d
    
    # 原有数据提取逻辑保持不变
    if not keys is None:
        ncell = adata.shape[0]
        V = np.zeros([ncell, len(pos_idx)], float)
        signal_sum = np.zeros([ncell], float)
        for key in keys:
            if summary == 'sender':
                V_key = adata.obsm['commot_sender_vf-'+database_name+'-'+key]
                # 处理2D->3D扩展
                if len(pos_idx) == 3 and V_key.shape[1] == 2:
                    V_key_extended = np.zeros([V_key.shape[0], 3])
                    V_key_extended[:, :2] = V_key
                    V += V_key_extended[:, pos_idx]
                else:
                    V += V_key[:, pos_idx]
                signal_sum += adata.obsm['commot-'+database_name+"-sum-sender"]['s-'+key]
            elif summary == 'receiver':
                V_key = adata.obsm['commot_receiver_vf-'+database_name+'-'+key]
                if len(pos_idx) == 3 and V_key.shape[1] == 2:
                    V_key_extended = np.zeros([V_key.shape[0], 3])
                    V_key_extended[:, :2] = V_key
                    V += V_key_extended[:, pos_idx]
                else:
                    V += V_key[:, pos_idx]
                signal_sum += adata.obsm['commot-'+database_name+"-sum-receiver"]['r-'+key]
        V = V / float(len(keys))
        signal_sum = signal_sum / float(len(keys))
    else:
        # 单个通路处理
        if not lr_pair is None:
            vf_name = database_name+'-'+lr_pair[0]+'-'+lr_pair[1]
            sum_name = lr_pair[0]+'-'+lr_pair[1]
        elif not pathway_name is None:
            vf_name = database_name+'-'+pathway_name
            sum_name = pathway_name
        else:
            vf_name = database_name+'-total-total'
            sum_name = 'total-total'
        
        if summary == 'sender':
            V_raw = adata.obsm['commot_sender_vf-'+vf_name]
            signal_sum = adata.obsm['commot-'+database_name+"-sum-sender"]['s-'+sum_name]
        elif summary == 'receiver':
            V_raw = adata.obsm['commot_receiver_vf-'+vf_name]
            signal_sum = adata.obsm['commot-'+database_name+"-sum-receiver"]['r-'+sum_name]
        
        # 处理维度扩展
        if len(pos_idx) == 3 and V_raw.shape[1] == 2:
            V = np.zeros([V_raw.shape[0], 3])
            V[:, :2] = V_raw
            V = V[:, pos_idx]
        else:
            V = V_raw[:, pos_idx]
    
    # 获取空间坐标
    X = adata.obsm["spatial"][:, pos_idx]
    
    # 向量预处理
    if normalize_v:
        V = V / np.quantile(np.linalg.norm(V, axis=1), normalize_v_quantile)
    
    # 3D HTML模式 (支持所有方法)
    if is_3d:
        return create_3d_html_plot(
            X, V, signal_sum, adata, plot_method, background, clustering, 
            cmap, cluster_cmap, summary, ndsize, scale,
            grid_density, grid_knn, grid_scale, grid_thresh,
            stream_density, stream_linewidth, stream_cutoff_perc,
            arrow_color, filename, camera_eye, background_legend,
            grid_cutoff_perc
        )
    
    # 原有2D模式 - 调用原来的函数逻辑
    else:
        return plot_original_2d(
            X, V, signal_sum, adata, plot_method, background, clustering,
            background_legend, cmap, cluster_cmap, summary, ndsize, scale,
            grid_density, grid_knn, grid_scale, grid_thresh, grid_width,
            stream_density, stream_linewidth, stream_cutoff_perc,
            arrow_color, filename, ax
        )

def create_3d_html_plot(X, V, signal_sum, adata, plot_method, background, clustering, 
                        cmap, cluster_cmap, summary, ndsize, scale,
                        grid_density, grid_knn, grid_scale, grid_thresh,
                        stream_density, stream_linewidth, stream_cutoff_perc,
                        arrow_color, filename, camera_eye, background_legend,
                        cutoff_perc=90):
    """创建3D HTML可视化 - 支持所有绘图模式"""
    
    fig = go.Figure()
    
    # 添加背景细胞点
    add_3d_background(fig, X, signal_sum, adata, background, clustering, 
                      cmap, cluster_cmap, ndsize, background_legend)
    
    # 根据绘图方法添加向量可视化
    if plot_method == "cell":
        add_3d_cell_vectors(fig, X, V, scale, arrow_color)
    elif plot_method == "grid":
        add_3d_grid_vectors(fig, X, V, grid_density, grid_knn, grid_scale, 
                           grid_thresh, scale, arrow_color, cutoff_perc)
    elif plot_method == "stream":
        add_3d_streamlines(fig, X, V, stream_density, stream_linewidth, 
                          stream_cutoff_perc, arrow_color)
    
    # 配置3D场景
    configure_3d_scene(fig, camera_eye, filename)
    
    return fig

def add_3d_background(fig, X, signal_sum, adata, background, clustering, 
                      cmap, cluster_cmap, ndsize, background_legend):
    """添加3D背景细胞点"""
    
    if background == 'summary':
        # 按信号强度着色
        print(f"Signal sum range: {signal_sum.min():.3f} - {signal_sum.max():.3f}")
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode='markers',
            marker=dict(
                size=ndsize * 5,
                color=signal_sum,
                colorscale=cmap,
                opacity=0.8,
                line=dict(width=0),
                colorbar=dict(
                    title="Signal Strength",
                    titleside="right",
                    thickness=15,
                    len=0.7
                ),
                cmin=signal_sum.min(),
                cmax=signal_sum.max()
            ),
            name='Cells (Signal Strength)',
            showlegend=background_legend,
            text=[f'Cell {i}<br>Signal: {signal_sum[i]:.3f}' for i in range(len(signal_sum))],
            hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
    
    elif background == 'cluster' and clustering:
        # 按细胞群着色
        labels = np.array(adata.obs[clustering], str)
        unique_labels = np.unique(labels)
        
        # 获取颜色映射
        if cluster_cmap is None:
            colors = get_plotly_colors(cmap, len(unique_labels))
            cluster_cmap = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for label in unique_labels:
            idx = labels == label
            fig.add_trace(go.Scatter3d(
                x=X[idx, 0], y=X[idx, 1], z=X[idx, 2],
                mode='markers',
                marker=dict(
                    size=ndsize * 3,
                    color=cluster_cmap[label],
                    opacity=0.6,
                    line=dict(width=0)
                ),
                name=f'Cluster {label}',
                showlegend=background_legend,
                text=[f'Cell {i}<br>Cluster: {label}' for i in np.where(idx)[0]],
                hovertemplate='<b>%{text}</b><br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))

def add_3d_cell_vectors(fig, X, V, scale, arrow_color):
    """添加3D细胞级向量（cell模式）"""
    
    # 过滤有效向量
    vector_magnitude = np.linalg.norm(V, axis=1)
    valid_mask = vector_magnitude > np.percentile(vector_magnitude[vector_magnitude > 0], 10)
    
    if np.any(valid_mask):
        X_valid = X[valid_mask]
        V_valid = V[valid_mask]
        
        fig.add_trace(go.Cone(
            x=X_valid[:, 0], y=X_valid[:, 1], z=X_valid[:, 2],
            u=V_valid[:, 0] * scale, v=V_valid[:, 1] * scale, w=V_valid[:, 2] * scale,
            sizemode="absolute",
            sizeref=0.5,
            colorscale=[[0, arrow_color], [1, arrow_color]],
            showscale=False,
            opacity=0.8,
            name='Cell Communication Vectors',
            hovertemplate='Cell Vector<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>Direction: (%{u:.2f}, %{v:.2f}, %{w:.2f})<extra></extra>'
        ))
        print(f"Added {len(X_valid)} cell vectors")

def add_3d_grid_vectors(fig, X, V, grid_density, grid_knn, grid_scale, 
                       grid_thresh, scale, arrow_color, cutoff_perc=90):
    """添加3D网格向量（grid模式）"""
    
    # 创建3D网格并插值
    X_grid, V_grid = create_3d_grid_interpolation(
        X, V, grid_density, grid_knn, grid_scale, grid_thresh
    )
    
    original_vector_count = len(X_grid)

    # 添加通信向量
    if original_vector_count > 0:
        # 新增：过滤掉小的向量来隐藏小箭头
        vector_magnitudes = np.linalg.norm(V_grid, axis=1)
        
        # 仅在有正值向量时进行过滤
        positive_magnitudes = vector_magnitudes[vector_magnitudes > 0]
        if len(positive_magnitudes) > 0:
            # 将阈值设置为20%，意味着最小的20%的箭头将被隐藏
            threshold = np.percentile(positive_magnitudes, cutoff_perc)
            mask = vector_magnitudes > threshold
            X_grid = X_grid[mask]
            V_grid = V_grid[mask]

        if len(X_grid) == 0:
            print("All grid vectors were filtered out due to small magnitude.")
            return
            
        fig.add_trace(go.Cone(
            x=X_grid[:, 0], y=X_grid[:, 1], z=X_grid[:, 2],
            u=V_grid[:, 0] * scale, v=V_grid[:, 1] * scale, w=V_grid[:, 2] * scale,
            sizemode="absolute",
            sizeref=1,
            colorscale=[[0, arrow_color], [1, arrow_color]],
            showscale=False,
            opacity=0.8,
            name='Grid Communication Vectors',
            hovertemplate='Grid Vector<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>Direction: (%{u:.2f}, %{v:.2f}, %{w:.2f})<extra></extra>'
        ))
        print(f"Added {len(X_grid)} grid vectors (filtered from {original_vector_count})")

def add_3d_streamlines(fig, X, V, stream_density, stream_linewidth, 
                      stream_cutoff_perc, arrow_color):
    """添加3D流线（stream模式）- 使用Plotly Streamtube"""
    
    print("Generating 3D streamtubes...")
    
    # 方法1: 使用Plotly原生Streamtube (推荐)
    try:
        add_plotly_streamtubes(fig, X, V, stream_density, stream_linewidth, arrow_color)
        print("✅ Using Plotly Streamtubes")
        return
    except Exception as e:
        print(f"⚠️ Streamtube failed ({e}), falling back to manual streamlines")
    
    # 方法2: 备用 - 手动绘制流线
    add_manual_streamlines(fig, X, V, stream_density, stream_linewidth, 
                          stream_cutoff_perc, arrow_color)

def add_plotly_streamtubes(fig, X, V, stream_density, stream_linewidth, arrow_color):
    """使用Plotly原生Streamtube功能"""
    
    # 创建规则网格用于Streamtube
    X_grid, V_grid = create_regular_3d_grid(X, V, stream_density)
    
    if X_grid is None:
        raise ValueError("Cannot create regular grid for streamtubes")
    
    # 获取网格尺寸
    x_unique = np.unique(X_grid[:, 0])
    y_unique = np.unique(X_grid[:, 1])  
    z_unique = np.unique(X_grid[:, 2])
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    
    # 重塑向量场为网格形状
    U = V_grid[:, 0].reshape((nx, ny, nz))
    V_comp = V_grid[:, 1].reshape((nx, ny, nz))
    W = V_grid[:, 2].reshape((nx, ny, nz))
    
    # 创建起始点
    starts_x, starts_y, starts_z = create_streamtube_starts(x_unique, y_unique, z_unique, stream_density)
    
    # 添加Streamtube
    fig.add_trace(go.Streamtube(
        x=x_unique,
        y=y_unique, 
        z=z_unique,
        u=U,
        v=V_comp,
        w=W,
        starts=dict(
            x=starts_x,
            y=starts_y,
            z=starts_z
        ),
        sizeref=stream_linewidth * 0.1,
        colorscale=[[0, arrow_color], [1, arrow_color]],
        showscale=False,
        maxdisplayed=int(100 * stream_density),  # 控制显示的流管数量
        name='Communication Streamtubes',
        hovertemplate='Streamtube<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

def create_regular_3d_grid(X, V, density=1.0):
    """为Streamtube创建规则网格"""
    
    # 计算空间边界
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = X[:, 2].min(), X[:, 2].max()
    
    # 扩展边界
    padding = 0.1
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range  
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range
    
    # 创建规则网格
    n_grid = max(8, int(12 * density))
    
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    z_grid = np.linspace(z_min, z_max, max(6, n_grid//2))
    
    mesh_x, mesh_y, mesh_z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = np.column_stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()])
    
    # 使用K近邻插值向量场到规则网格
    nbrs = NearestNeighbors(n_neighbors=min(8, len(X)), algorithm='ball_tree')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(grid_points)
    
    # 高斯权重插值
    sigma = min(x_range, y_range, z_range) / n_grid
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weight_sums = weights.sum(axis=1)
    
    V_grid = np.zeros([len(grid_points), 3])
    for dim in range(3):
        if dim < V.shape[1]:
            V_values = V[indices, dim]
            V_grid[:, dim] = (V_values * weights).sum(axis=1) / np.maximum(weight_sums, 1e-10)
    
    return grid_points, V_grid

def create_streamtube_starts(x_unique, y_unique, z_unique, density):
    """创建Streamtube起始点"""
    
    # 在边界面创建起始点
    n_starts_per_face = max(3, int(8 * density))
    
    starts_x, starts_y, starts_z = [], [], []
    
    # X边界面
    for x_val in [x_unique[0], x_unique[-1]]:
        y_starts = np.linspace(y_unique[0], y_unique[-1], n_starts_per_face)
        z_starts = np.linspace(z_unique[0], z_unique[-1], n_starts_per_face)
        yy, zz = np.meshgrid(y_starts, z_starts)
        starts_x.extend([x_val] * len(yy.flatten()))
        starts_y.extend(yy.flatten())
        starts_z.extend(zz.flatten())
    
    # Y边界面
    for y_val in [y_unique[0], y_unique[-1]]:
        x_starts = np.linspace(x_unique[0], x_unique[-1], n_starts_per_face)
        z_starts = np.linspace(z_unique[0], z_unique[-1], n_starts_per_face)
        xx, zz = np.meshgrid(x_starts, z_starts)
        starts_x.extend(xx.flatten())
        starts_y.extend([y_val] * len(xx.flatten()))
        starts_z.extend(zz.flatten())
    
    return starts_x, starts_y, starts_z

def add_manual_streamlines(fig, X, V, stream_density, stream_linewidth, 
                          stream_cutoff_perc, arrow_color):
    """备用方案：手动绘制流线"""
    
    # 创建流线起始点
    streamline_starts = create_3d_streamline_starts(X, V, stream_density, stream_cutoff_perc)
    
    if len(streamline_starts) == 0:
        print("No valid streamline starting points found")
        return
    
    # 为每个起始点生成流线
    all_streamlines = []
    
    for start_point in streamline_starts:
        streamline = integrate_3d_streamline(start_point, X, V)
        if len(streamline) > 2:  # 至少需要3个点
            all_streamlines.append(streamline)
    
    print(f"Generated {len(all_streamlines)} manual streamlines")
    
    # 绘制流线
    for i, streamline in enumerate(all_streamlines):
        fig.add_trace(go.Scatter3d(
            x=streamline[:, 0], 
            y=streamline[:, 1], 
            z=streamline[:, 2],
            mode='lines',
            line=dict(
                color=arrow_color,
                width=stream_linewidth * 2,
            ),
            opacity=0.7,
            name=f'Streamline {i+1}' if i < 5 else '',
            showlegend=i < 5,
            hovertemplate='Streamline<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
    
    # 添加流线起始点
    fig.add_trace(go.Scatter3d(
        x=streamline_starts[:, 0], 
        y=streamline_starts[:, 1], 
        z=streamline_starts[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.8
        ),
        name='Streamline Origins',
        showlegend=True,
        hovertemplate='Streamline Start<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))

def create_3d_streamline_starts(X, V, stream_density, stream_cutoff_perc):
    """创建3D流线起始点"""
    
    # 计算向量强度
    vector_magnitude = np.linalg.norm(V, axis=1)
    
    # 过滤掉弱向量
    magnitude_threshold = np.percentile(vector_magnitude, stream_cutoff_perc)
    strong_vector_mask = vector_magnitude > magnitude_threshold
    
    if not np.any(strong_vector_mask):
        return np.array([]).reshape(0, 3)
    
    # 在强向量区域创建起始点
    X_strong = X[strong_vector_mask]
    
    # 根据密度参数采样起始点
    n_starts = max(5, int(len(X_strong) * stream_density * 0.1))
    n_starts = min(n_starts, len(X_strong))
    
    # 使用k-means样采样或均匀采样
    if n_starts < len(X_strong):
        indices = np.random.choice(len(X_strong), n_starts, replace=False)
        streamline_starts = X_strong[indices]
    else:
        streamline_starts = X_strong
    
    print(f"Created {len(streamline_starts)} streamline starting points")
    return streamline_starts

def integrate_3d_streamline(start_point, X, V, max_length=50, step_size=0.1):
    """积分生成3D流线"""
    
    # 使用K近邻插值向量场
    def vector_field_interpolator(pos):
        pos = pos.reshape(1, -1)
        
        # 找到最近的邻居
        nbrs = NearestNeighbors(n_neighbors=min(5, len(X)), algorithm='ball_tree')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(pos)
        
        # 使用距离权重插值
        weights = 1.0 / (distances[0] + 1e-10)
        weights = weights / weights.sum()
        
        # 插值向量
        interpolated_v = np.sum(V[indices[0]] * weights[:, None], axis=0)
        return interpolated_v
    
    # 积分流线
    streamline = [start_point.copy()]
    current_pos = start_point.copy()
    
    for step in range(max_length):
        # 获取当前位置的向量
        try:
            current_v = vector_field_interpolator(current_pos)
            
            # 检查向量是否有效
            if np.linalg.norm(current_v) < 1e-6:
                break
            
            # 归一化并应用步长
            current_v = current_v / np.linalg.norm(current_v) * step_size
            
            # 更新位置
            next_pos = current_pos + current_v
            streamline.append(next_pos.copy())
            current_pos = next_pos
            
            # 检查是否超出边界
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            if np.any(current_pos < X_min - 0.5*(X_max-X_min)) or np.any(current_pos > X_max + 0.5*(X_max-X_min)):
                break
                
        except Exception as e:
            print(f"Streamline integration error: {e}")
            break
    
    return np.array(streamline)

def create_3d_grid_interpolation(X, V, grid_density, grid_knn, grid_scale, grid_thresh):
    """创建3D网格插值"""
    
    # 计算空间范围
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    ranges = maxs - mins
    
    # 扩展边界
    padding = 0.1
    mins -= padding * ranges
    maxs += padding * ranges
    
    # 创建3D网格
    n_grid = max(8, int(15 * grid_density))  # 控制网格密度
    
    if X.shape[1] == 3:
        # 真3D网格
        x_grid = np.linspace(mins[0], maxs[0], n_grid)
        y_grid = np.linspace(mins[1], maxs[1], n_grid)
        z_grid = np.linspace(mins[2], maxs[2], n_grid)
        mesh_x, mesh_y, mesh_z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        grid_points = np.column_stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()])
    else:
        # 2D数据扩展到3D
        x_grid = np.linspace(mins[0], maxs[0], n_grid)
        y_grid = np.linspace(mins[1], maxs[1], n_grid)
        z_grid = np.linspace(0, ranges.mean() * 0.5, max(4, n_grid//3))
        mesh_x, mesh_y, mesh_z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        grid_points = np.column_stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()])
    
    # K近邻插值
    if grid_knn is None:
        grid_knn = max(8, min(50, X.shape[0] // 10))
    
    # 确保有足够的邻居
    grid_knn = min(grid_knn, X.shape[0])
    
    nbrs = NearestNeighbors(n_neighbors=grid_knn, algorithm='ball_tree')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(grid_points)
    
    # 高斯权重
    grid_size = np.mean(ranges) / n_grid
    sigma = grid_size * grid_scale
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weight_sums = weights.sum(axis=1)
    
    # 插值向量
    V_grid = np.zeros([len(grid_points), V.shape[1]])
    for dim in range(V.shape[1]):
        V_values = V[indices, dim]  # shape: (n_grid_points, grid_knn)
        V_grid[:, dim] = (V_values * weights).sum(axis=1) / np.maximum(weight_sums, 1e-10)
    
    # 应用阈值
    threshold = np.percentile(weight_sums, 50) * grid_thresh  # 降低阈值
    valid_mask = weight_sums > threshold
    
    # 过滤掉零向量
    vector_magnitude = np.linalg.norm(V_grid, axis=1)
    magnitude_threshold = np.percentile(vector_magnitude[vector_magnitude > 0], 10) if np.any(vector_magnitude > 0) else 0
    valid_mask &= vector_magnitude > magnitude_threshold
    
    return grid_points[valid_mask], V_grid[valid_mask]

def configure_3d_scene(fig, camera_eye, filename):
    """配置3D场景"""
    
    scene_config = dict(
        xaxis=dict(title='X', showgrid=True),
        yaxis=dict(title='Y', showgrid=True),
        zaxis=dict(title='Z', showgrid=True),
        aspectmode='cube'
    )
    
    if camera_eye:
        scene_config['camera'] = dict(eye=camera_eye)
    else:
        scene_config['camera'] = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    
    fig.update_layout(
        title='3D Cell Communication',
        scene=scene_config,
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # 保存HTML文件
    if filename:
        html_filename = filename if filename.endswith('.html') else filename + '.html'
        fig.write_html(html_filename, include_plotlyjs=True)
        print(f"3D visualization saved as '{html_filename}'")

def get_plotly_colors(colormap_name, n_colors):
    """获取Plotly颜色映射 - 兼容版本"""
    try:
        color_maps = {
            'Plotly': pq.Plotly,
            'Light24': pq.Light24, 
            'Dark24': pq.Dark24,
            'Alphabet': pq.Alphabet
        }
        colors = color_maps.get(colormap_name, pq.Plotly)
    except AttributeError:
        # 备用颜色方案
        backup_colors = {
            'Plotly': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'],
            'Light24': ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB'],
            'Dark24': ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100', '#750D86', '#EB663B', '#511CFB'],
            'Alphabet': ['#AA0DFE', '#3283FE', '#85660D', '#782AB6', '#565656', '#1C8356', '#16FF32', '#F7E1A0', '#E2E2E2', '#1CBE4F']
        }
        colors = backup_colors.get(colormap_name, backup_colors['Plotly'])
    
    return [colors[i % len(colors)] for i in range(n_colors)]

def plot_original_2d(X, V, signal_sum, adata, plot_method, background, clustering,
                     background_legend, cmap, cluster_cmap, summary, ndsize, scale,
                     grid_density, grid_knn, grid_scale, grid_thresh, grid_width,
                     stream_density, stream_linewidth, stream_cutoff_perc,
                     arrow_color, filename, ax):
    """原始2D绘图逻辑的简化版本"""
    
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 背景点
    if background == 'summary':
        scatter = ax.scatter(X[:, 0], X[:, 1], s=ndsize*20, c=signal_sum, 
                           cmap=cmap, linewidth=0, alpha=0.7)
        if background_legend:
            plt.colorbar(scatter, ax=ax, label='Signal Strength')
    
    elif background == 'cluster' and clustering:
        labels = adata.obs[clustering].astype('category')
        unique_labels = labels.cat.categories
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = cluster_cmap[label] if cluster_cmap and label in cluster_cmap else f'C{i}'
            ax.scatter(X[mask, 0], X[mask, 1], s=ndsize*20, c=color, 
                      label=label, linewidth=0, alpha=0.7)
        
        if background_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加向量
    if plot_method == "cell":
        # 过滤零向量
        V_plot = V.copy()
        valid_mask = np.linalg.norm(V_plot, axis=1) > 1e-6
        if np.any(valid_mask):
            ax.quiver(X[valid_mask, 0], X[valid_mask, 1], 
                     V_plot[valid_mask, 0], V_plot[valid_mask, 1],
                     scale=scale, scale_units='xy', angles='xy',
                     color=arrow_color, alpha=0.8, width=0.003)
    
    elif plot_method == "grid":
        # 2D网格插值 (简化版)
        X_grid, V_grid = create_2d_grid_interpolation(X[:, :2], V[:, :2], grid_density, grid_knn, grid_scale, grid_thresh)
        if len(X_grid) > 0:
            ax.quiver(X_grid[:, 0], X_grid[:, 1], V_grid[:, 0], V_grid[:, 1],
                     scale=scale, scale_units='xy', angles='xy',
                     color=arrow_color, alpha=0.8, width=grid_width)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    
    return ax

def create_2d_grid_interpolation(X, V, grid_density, grid_knn, grid_scale, grid_thresh):
    """2D网格插值"""
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    n_grid = int(30 * grid_density)
    x_grid = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, n_grid)
    y_grid = np.linspace(y_min - 0.1*y_range, y_max + 0.1*y_range, n_grid)
    
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([mesh_x.flatten(), mesh_y.flatten()])
    
    if grid_knn is None:
        grid_knn = max(5, X.shape[0] // 20)
    grid_knn = min(grid_knn, X.shape[0])
    
    nbrs = NearestNeighbors(n_neighbors=grid_knn)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(grid_points)
    
    grid_size = min(x_range, y_range) / n_grid
    weights = np.exp(-0.5 * (distances / (grid_size * grid_scale)) ** 2)
    weight_sums = weights.sum(axis=1)
    
    V_grid = np.zeros([len(grid_points), V.shape[1]])
    for dim in range(V.shape[1]):
        V_grid[:, dim] = (V[indices, dim] * weights).sum(axis=1) / np.maximum(weight_sums, 1e-10)
    
    threshold = np.percentile(weight_sums, 70) * grid_thresh
    valid_mask = weight_sums > threshold
    valid_mask &= np.linalg.norm(V_grid, axis=1) > np.percentile(np.linalg.norm(V_grid, axis=1), 20)
    
    return grid_points[valid_mask], V_grid[valid_mask]

# 使用示例 - 现在支持所有3D模式
def example_usage(adata):
    """使用示例 - 演示所有3D模式"""
    
    print("=== 生成所有3D模式的可视化 ===")
    
    # 1. Cell模式 - 3D
    print("\n1. 生成3D Cell模式...")
    fig_cell = plot_cell_communication(
        adata,
        database_name='cellchat',
        plot_method='cell',  # cell模式
        background='summary',
        pos_idx=np.array([0, 1, 2]),
        html_3d=True,
        background_legend=True,
        filename='cell_comm_3d_cell.html',
        camera_eye={'x': 2, 'y': 2, 'z': 1.5},
        scale=2.0  # 放大箭头便于观察
    )
    
    # 2. Grid模式 - 3D  
    print("\n2. 生成3D Grid模式...")
    fig_grid = plot_cell_communication(
        adata,
        database_name='cellchat',
        plot_method='grid',  # grid模式
        background='summary',
        pos_idx=np.array([0, 1, 2]),
        html_3d=True,
        background_legend=True,
        filename='cell_comm_3d_grid.html',
        camera_eye={'x': 2, 'y': 2, 'z': 1.5},
        grid_density=1.2
    )
    
    # 3. Stream模式 - 3D (使用Plotly Streamtubes!)
    print("\n3. 生成3D Stream模式 (Streamtubes)...")
    fig_stream = plot_cell_communication(
        adata,
        database_name='cellchat',
        plot_method='stream',  # stream模式 - 现在使用Streamtubes!
        background='cluster',
        clustering='leiden',
        pos_idx=np.array([0, 1, 2]),
        html_3d=True,
        background_legend=True,
        filename='cell_comm_3d_streamtubes.html',
        camera_eye={'x': 2, 'y': 2, 'z': 1.5},
        stream_density=1.0,    # 更好的密度控制
        stream_linewidth=3,    # Streamtube粗细
        stream_cutoff_perc=20
    )
    
    print("\n=== 完成！生成的文件：===")
    print("- cell_comm_3d_cell.html        (3D Cell模式 - 单细胞向量)")
    print("- cell_comm_3d_grid.html        (3D Grid模式 - 网格插值)")  
    print("- cell_comm_3d_streamtubes.html (3D Stream模式 - Plotly Streamtubes!)")
    print("\n🎯 Stream模式特别说明：")
    print("- 现在使用Plotly原生Streamtube功能")
    print("- 具有体积感的3D流管可视化")
    print("- 自动流线积分和渲染")
    print("- 如果Streamtube失败，会自动降级到手动流线")
    print("\n在浏览器中打开HTML文件，可以:")
    print("- 拖拽旋转3D场景")
    print("- 鼠标悬停查看详细信息")
    print("- 滚轮缩放")
    print("- 使用图例控制显示/隐藏元素")
    
    return fig_cell, fig_grid, fig_stream

# 快速测试函数
def test_all_3d_modes(adata, database_name='cellchat'):
    """快速测试所有3D模式"""
    
    print("Testing all 3D visualization modes...")
    
    modes = ['cell', 'grid', 'stream']
    for mode in modes:
        print(f"\nTesting {mode} mode...")
        try:
            fig = plot_cell_communication(
                adata,
                database_name=database_name,
                plot_method=mode,
                background='summary',
                pos_idx=np.array([0, 1, 2]),
                html_3d=True,
                filename=f'test_3d_{mode}_streamtubes.html' if mode == 'stream' else f'test_3d_{mode}.html',
                scale=1.5 if mode == 'cell' else 1.0,
                stream_density=0.5 if mode == 'stream' else 1.0
            )
            print(f"✅ {mode} mode successful")
        except Exception as e:
            print(f"❌ {mode} mode failed: {e}")
    
    print("\nTest completed!")

def test_signal_visualization(adata, database_name='cellchat', summary='sender'):
    """快速测试signal_sum可视化是否正常"""
    
    # 检查数据
    if summary == 'sender':
        signal_key = 's-total-total'
        signal_data = adata.obsm[f'commot-{database_name}-sum-sender']
    else:
        signal_key = 'r-total-total' 
        signal_data = adata.obsm[f'commot-{database_name}-sum-receiver']
    
    if signal_key in signal_data.columns:
        signal_values = signal_data[signal_key].values
        print(f"✅ Signal data found: {signal_key}")
        print(f"   Range: {signal_values.min():.3f} - {signal_values.max():.3f}")
        print(f"   Mean: {signal_values.mean():.3f}")
        print(f"   Non-zero values: {np.sum(signal_values > 0)}/{len(signal_values)}")
        return signal_values
    else:
        print(f"❌ Signal data NOT found: {signal_key}")
        print(f"   Available keys: {list(signal_data.columns)}")
        return None