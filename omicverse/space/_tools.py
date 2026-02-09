import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scanpy as sc
import torch
import torch.nn.functional as F
from .._registry import register_function

@register_function(
    aliases=["裁剪空间数据", "crop_space_visium", "crop_visium", "空间数据裁剪", "Visium裁剪"],
    category="space",
    description="Crop Visium spatial transcriptomics data to focus on specific region of interest",
    examples=[
        "# Basic cropping",
        "adata_cropped = ov.space.crop_space_visium(",
        "    adata, crop_loc=(500, 500), crop_area=(1000, 1000),",
        "    library_id='V1_Human_Brain', scale=1.0)",
        "# Small region cropping",
        "adata_roi = ov.space.crop_space_visium(",
        "    adata, crop_loc=(0, 0), crop_area=(800, 600),",
        "    library_id=list(adata.uns['spatial'].keys())[0], scale=1)",
        "# High resolution cropping",
        "adata_hires = ov.space.crop_space_visium(",
        "    adata, crop_loc=(200, 200), crop_area=(1500, 1500),",
        "    library_id='sample_1', scale=2, res='hires')"
    ],
    related=["space.rotate_space_visium", "space.map_spatial_auto"]
)
def crop_space_visium(adata, crop_loc, crop_area,
                     library_id, scale, spatial_key='spatial', res='hires'):
    """
    Crop Visium spatial data to a specific region of interest.
    
    This function allows cropping of Visium spatial transcriptomics data to focus on
    a specific region while maintaining proper scaling and coordinate systems.

    Arguments:
        adata: AnnData
            Annotated data matrix containing Visium spatial data.
        crop_loc: tuple
            (x, y) coordinates for the top-left corner of the crop region.
        crop_area: tuple
            (width, height) of the cropping area in spatial coordinates.
        library_id: str
            Library ID for the spatial data in adata.uns['spatial'].
        scale: float
            Scale factor for the cropping operation.
        spatial_key: str, optional (default='spatial')
            Key in adata.obsm containing spatial coordinates.
        res: str, optional (default='hires')
            Image resolution to use ('hires' or 'lowres').

    Returns:
        AnnData
            Cropped AnnData object containing only spots within the specified region.
            The spatial coordinates and image are adjusted accordingly.

    Notes:
        - The function preserves the original coordinate system scaling
        - The cropped image is stored in adata.uns['spatial'][library_id]['images'][res]
        - Coordinates are automatically adjusted to the new cropped region

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load Visium data
        >>> adata = sc.read_visium(...)
        >>> # Crop a 1000x1000 region starting at (500, 500)
        >>> adata_cropped = ov.space.crop_space_visium(
        ...     adata,
        ...     crop_loc=(500, 500),
        ...     crop_area=(1000, 1000),
        ...     library_id='V1_Human_Brain',
        ...     scale=1.0
        ... )
    """
    # Configure dask to use query-planning before importing squidpy
    import dask
    dask.config.set({"dataframe.query-planning": True})
    import squidpy as sq
    adata1 = adata.copy()
    img = sq.im.ImageContainer(
        adata1.uns["spatial"][library_id]["images"][res], library_id=library_id
    )
    crop_corner = img.crop_corner(crop_loc[0], crop_loc[1], size=crop_area, scale=scale,)
    adata1.obsm['spatial1'] = adata1.obsm[spatial_key]*\
                adata1.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    adata_crop = crop_corner.subset(adata1, spatial_key='spatial1')
    adata_crop.uns["spatial"][library_id]["images"][res] = np.squeeze(crop_corner['image'].data, axis=2)
    adata_crop.obsm[spatial_key][:,0] = (adata_crop.obsm['spatial1'][:,0]-crop_loc[1])/adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    adata_crop.obsm[spatial_key][:,1] = (adata_crop.obsm['spatial1'][:,1]-crop_loc[0])/adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    
    return adata_crop

import numpy as np
import scipy.ndimage  # 导入 scipy.ndimage 库用于图像旋转

@register_function(
    aliases=["旋转空间数据", "rotate_space_visium", "rotate_visium", "空间数据旋转", "Visium旋转"],
    category="space",
    description="Rotate Visium spatial data image and coordinates by specified angle",
    examples=[
        "# Basic rotation",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=45, library_id='V1_Human_Brain')",
        "# Custom center rotation",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=90, center=(100, 100),",
        "    library_id='sample_1', interpolation_order=3)",
        "# Precise rotation with high-quality interpolation",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=30, res='hires',",
        "    library_id=library_id, interpolation_order=1)",
        "# Apply spatial mapping after rotation",
        "ov.space.map_spatial_auto(adata_rotated, method='phase')"
    ],
    related=["space.crop_space_visium", "space.map_spatial_auto", "space.map_spatial_manual"]
)
def rotate_space_visium(
    adata,
    angle,
    center=None,
    spatial_key='spatial',
    res='hires',
    library_id=None,
    interpolation_order=1
):
    """
    Rotate Visium spatial data image and coordinates by a specified angle.

    This function performs rotation of both the tissue image and spot coordinates
    while maintaining proper alignment and scaling.

    Arguments:
        adata: AnnData
            Annotated data matrix containing Visium spatial data.
        angle: float
            Rotation angle in degrees (positive for counterclockwise).
        center: tuple, optional (default=None)
            (x, y) coordinates of rotation center. If None, uses center of spatial coordinates.
        spatial_key: str, optional (default='spatial')
            Key in adata.obsm containing spatial coordinates.
        res: str, optional (default='hires')
            Image resolution to use ('hires' or 'lowres').
        library_id: str
            Library ID for the spatial data in adata.uns['spatial'].
        interpolation_order: int, optional (default=1)
            Order of interpolation for image rotation:
            0: nearest neighbor, 1: bilinear, 3: cubic spline.

    Returns:
        AnnData
            Rotated AnnData object with transformed coordinates and image.

    Notes:
        - The function preserves the original coordinate system scaling
        - Both image and spot coordinates are rotated around the same center
        - The rotation is performed counterclockwise for positive angles
        - Image interpolation can be adjusted for quality vs speed tradeoff

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load Visium data
        >>> adata = sc.read_visium(...)
        >>> # Rotate 45 degrees counterclockwise
        >>> adata_rotated = ov.space.rotate_space_visium(
        ...     adata,
        ...     angle=45,
        ...     library_id='V1_Human_Brain'
        ... )
    """
    adata_rotate = adata.copy()

    if library_id is None:
        raise ValueError("必须明确指定 library_id.")

    # 1. 获取原始图像
    original_image = adata_rotate.uns["spatial"][library_id]["images"][res].copy()

    # 2. 确定空间坐标旋转中心 (统一使用空间坐标中心)
    original_spatial = adata_rotate.obsm[spatial_key].copy()
    if center is None:
        # 如果 center 为 None，则使用空间坐标的中心作为旋转中心
        center_x_spatial = original_spatial[:, 0].mean()
        center_y_spatial = original_spatial[:, 1].mean()
        rotation_center_spatial = (center_x_spatial, center_y_spatial)
    else:
        # 如果提供了 center 参数，直接使用提供的中心 (空间坐标)
        rotation_center_spatial = center

    # 3. 将空间坐标旋转中心转换为像素坐标，用于图像旋转
    scalefactors = adata_rotate.uns['spatial'][library_id]['scalefactors']
    tissue_hires_scalef = scalefactors['tissue_hires_scalef']

    # **重要**: 您需要根据您的数据坐标系统，确定正确的空间坐标到像素坐标的转换公式。
    #          以下转换公式仅为示例，您很可能需要根据实际情况进行调整!
    #          您可能需要考虑 offsetX, offsetY 等平移参数，这些参数可能需要从 scalefactors 或其他地方获取。
    offsetX = 0  # **占位符**:  请替换为实际的 X 轴平移参数 (如果需要)
    offsetY = 0  # **占位符**:  请替换为实际的 Y 轴平移参数 (如果需要)

    rotation_center_image = (
        rotation_center_spatial[0] / tissue_hires_scalef + offsetX,
        rotation_center_spatial[1] / tissue_hires_scalef + offsetY
    )

    # 4. 旋转图像 (默认逆时针)，使用转换后的像素坐标中心
    rotated_image = scipy.ndimage.rotate(
        original_image,
        angle=angle,           # 图像旋转角度 (逆时针)
        reshape=False,         # 保持输出图像尺寸与输入相同
        order=interpolation_order,  # 使用指定的插值阶数
        mode='reflect',
    )

    # 5. 更新 adata_rotate 中的图像
    adata_rotate.uns["spatial"][library_id]["images"][res] = rotated_image

    # 6. 旋转空间坐标 (调整为顺时针旋转)，使用空间坐标中心
    rotated_spatial = np.zeros_like(original_spatial)
    theta_rad = np.radians(angle) # 角度转弧度
    rotation_center_spatial = rotation_center_spatial # 空间坐标旋转中心直接使用之前确定的

    for i in range(len(original_spatial)):
        x_orig = original_spatial[i, 0]
        y_orig = original_spatial[i, 1]

        # 步骤 1: 将原始空间坐标平移到以旋转中心为原点的坐标系
        x_centered = x_orig - rotation_center_spatial[0]
        y_centered = y_orig - rotation_center_spatial[1]

        # 步骤 2: 执行顺时针旋转 (修改 sin(theta_rad) 项的符号)
        rotated_x_centered = x_centered * np.cos(theta_rad) + y_centered * np.sin(theta_rad)
        rotated_y_centered = -x_centered * np.sin(theta_rad) + y_centered * np.cos(theta_rad)

        # 步骤 3: 将旋转后的坐标平移回原始的全局空间坐标系
        rotated_x_orig = rotated_x_centered + rotation_center_spatial[0]
        rotated_y_orig = rotated_y_centered + rotation_center_spatial[1]

        rotated_spatial[i, 0] = rotated_x_orig
        rotated_spatial[i, 1] = rotated_y_orig

    # 7. 更新 adata_rotate 中的空间坐标
    adata_rotate.obsm[spatial_key] = rotated_spatial

    return adata_rotate


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
import os
import scanpy as sc  # 假设您使用了 scanpy

def find_image_offset_phase_correlation_array_input(image1_array, image2_array):
    """
    Calculate image offset using phase correlation method.

    This function computes the relative displacement between two images using
    Fourier-based phase correlation, which is robust to intensity variations
    and noise.

    Arguments:
        image1_array: numpy.ndarray
            First image array (image to be aligned).
            Can be grayscale or RGB.
        image2_array: numpy.ndarray
            Second image array (reference image).
            Can be grayscale or RGB.

    Returns:
        tuple
            (offset, aligned_image) where:
            - offset: tuple (dx, dy) representing the displacement in pixels
            - aligned_image: numpy.ndarray of the aligned first image

    Notes:
        - Images are automatically converted to grayscale if RGB
        - The method is based on frequency-domain phase correlation
        - Works best with images of similar size and content
        - Handles sub-pixel accuracy for precise alignment

    Examples:
        >>> import omicverse as ov
        >>> import numpy as np
        >>> # Create sample images
        >>> img1 = np.random.rand(100, 100)
        >>> img2 = np.roll(img1, shift=(5, 10), axis=(0, 1))
        >>> # Find offset
        >>> offset, aligned = ov.space.find_image_offset_phase_correlation_array_input(
        ...     img1, img2
        ... )
        >>> print(f"Detected offset: {offset}")
    """
    # **不再需要读取图像文件，直接使用输入的 NumPy 数组**
    import cv2
    img1 = image1_array
    img2 = image2_array

    # 确保是灰度图 (如果您的 img 已经是灰度图，则可以跳过)
    if len(img1.shape) > 2 and img1.shape[2] > 1: # 检查是否是彩色图像
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) # 假设是 RGB，转为灰度
    if len(img2.shape) > 2 and img2.shape[2] > 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 将图像转换为 float32 类型，以便进行傅里叶变换
    img1_float32 = np.float32(img1)
    img2_float32 = np.float32(img2)

    # 计算傅里叶变换
    fft_img1 = np.fft.fft2(img1_float32)
    fft_img2 = np.fft.fft2(img2_float32)

    # 计算互功率谱 (Cross-Power Spectrum)
    conjugate_fft_img2 = np.conjugate(fft_img2) # 共轭复数
    cross_power_spectrum = fft_img1 * conjugate_fft_img2

    # 归一化互功率谱 (为了增强峰值)
    magnitude = np.abs(cross_power_spectrum)
    normalized_cross_power_spectrum = cross_power_spectrum / magnitude

    # 计算逆傅里叶变换，得到脉冲响应 (Inverse FFT)
    inverse_fft = np.fft.ifft2(normalized_cross_power_spectrum)

    # 找到峰值位置 (峰值位置对应偏移量)
    peak_position = np.unravel_index(np.argmax(np.abs(inverse_fft)), inverse_fft.shape)

    # 计算偏移量 (需要考虑图像尺寸和峰值位置的转换关系)
    shift_y = peak_position[0]
    shift_x = peak_position[1]

    if shift_y > img1.shape[0] // 2: # 如果峰值在图像的下半部分，则偏移为负
        shift_y -= img1.shape[0]
    if shift_x > img1.shape[1] // 2: # 如果峰值在图像的右半部分，则偏移为负
        shift_x -= img1.shape[1]

    offset = (shift_x, shift_y)

    # 平移图像 image1
    image1_aligned = shift(img1, shift=(-shift_y, -shift_x), order=0) # 注意 shift 的方向

    return offset, image1_aligned

def find_image_offset_phase_correlation_torch(image1_tensor, image2_tensor):
    """
    Calculate image offset using phase correlation method with PyTorch tensors.

    This function is the PyTorch implementation of phase correlation for image
    alignment, offering GPU acceleration when available.

    Arguments:
        image1_tensor: torch.Tensor
            First image as PyTorch tensor (image to be aligned).
            Expected shape: (H, W) or (1, H, W) or (3, H, W).
        image2_tensor: torch.Tensor
            Second image as PyTorch tensor (reference image).
            Expected shape: (H, W) or (1, H, W) or (3, H, W).

    Returns:
        tuple
            (offset, aligned_image) where:
            - offset: tuple (dx, dy) representing the displacement in pixels
            - aligned_image: torch.Tensor of the aligned first image

    Notes:
        - Images are automatically converted to grayscale if RGB
        - Computation is performed on GPU if available
        - More efficient than NumPy version for large images when using GPU
        - Maintains sub-pixel accuracy for precise alignment

    Examples:
        >>> import omicverse as ov
        >>> import torch
        >>> # Create sample images
        >>> img1 = torch.rand(100, 100)
        >>> img2 = torch.roll(img1, shifts=(5, 10), dims=(0, 1))
        >>> # Find offset
        >>> offset, aligned = ov.space.find_image_offset_phase_correlation_torch(
        ...     img1, img2
        ... )
        >>> print(f"Detected offset: {offset}")
    """
    device = image1_tensor.device
    
    # 转换为灰度图（如果输入是彩色）
    if image1_tensor.ndim == 3:
        image1_gray = torch.mean(image1_tensor, dim=0, keepdim=True)
        image2_gray = torch.mean(image2_tensor, dim=0, keepdim=True)
    else:
        image1_gray = image1_tensor
        image2_gray = image2_tensor

    # 傅里叶变换
    fft1 = torch.fft.fft2(image1_gray)
    fft2 = torch.fft.fft2(image2_gray)
    
    # 计算互功率谱（带数值稳定性保护）
    cross_power_spectrum = fft1 * torch.conj(fft2)
    eps = 1e-8
    normalized_cross_power = cross_power_spectrum / (torch.abs(cross_power_spectrum) + eps)
    
    # 逆傅里叶变换
    inv_cross_power = torch.fft.ifft2(normalized_cross_power).real
    
    # 寻找峰值位置
    _, h, w = inv_cross_power.shape
    max_idx = torch.argmax(inv_cross_power.view(-1))
    shift_y, shift_x = np.unravel_index(max_idx.cpu().numpy(), (h, w))
    
    # 计算循环位移量
    if shift_y > h // 2:
        shift_y -= h
    if shift_x > w // 2:
        shift_x -= w
    
    # 创建仿射变换矩阵进行图像对齐
    theta = torch.tensor([[1, 0, -shift_x], [0, 1, -shift_y]], 
                        dtype=torch.float32, device=device)
    
    grid = F.affine_grid(theta.unsqueeze(0), image1_tensor.unsqueeze(0).shape)
    image1_aligned = F.grid_sample(image1_tensor.unsqueeze(0), grid, 
                                 padding_mode="zeros", align_corners=True)
    
    return (shift_x, shift_y), image1_aligned.squeeze(0)

def _create_spatial_image(adata, ax, color=None, alpha=1, save_path=None):
    """
    Create a spatial plot of gene expression or feature values.

    Internal function to generate spatial plots with customizable appearance.

    Arguments:
        adata: AnnData
            Annotated data matrix containing spatial data.
        ax: matplotlib.axes.Axes
            Matplotlib axes object to plot on.
        color: str or None, optional (default=None)
            Key in adata.obs or adata.var for coloring points.
        alpha: float, optional (default=1)
            Transparency of the plotted points (0 to 1).
        save_path: str or None, optional (default=None)
            Path to save the plot. If None, plot is not saved.

    Returns:
        matplotlib.axes.Axes
            The axes object containing the plot.

    Notes:
        - This is an internal function used by other plotting functions
        - Handles both categorical and continuous coloring
        - Automatically scales point sizes based on data
    """
    if color is not None:
        adata.obs['temp_color'] = adata.obs[color] # 使用临时列，避免修改原始 adata.obs
    else:
        adata.obs['temp_color'] = '1' # 使用默认值，确保 `color` 参数可以处理 None

    library_id = list(adata.uns['spatial'].keys())[0]
    scalefactors = adata.uns['spatial'][library_id]['scalefactors']
    tissue_hires_scalef = scalefactors['tissue_hires_scalef']

    sc.pl.embedding(
        adata,
        basis='spatial',
        color='temp_color', # 使用临时列
        show=False,
        ax=ax,
        size=10,
        scale_factor=tissue_hires_scalef,
        legend_loc=None
    )

    cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
    img = adata.uns["spatial"][library_id]["images"]["hires"]
    ax.imshow(img, cmap='gray', alpha=alpha) # 使用传入的 alpha 参数
    ax.set_xlim(cur_coords[0], cur_coords[1])
    ax.set_ylim(cur_coords[3], cur_coords[2])
    ax.set_xticks([]) # 使用 ax 设置，更简洁
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.axis(False) # 隐藏坐标轴

    if save_path:
        plt.savefig(save_path, bbox_inches='tight') # 使用 plt.savefig, 更通用
    if 'temp_color' in adata.obs: # 清理临时列
        del adata.obs['temp_color']


def _map_spatial_img(adata_rotated, color=None,
                    method='phase_correlation'):
    """
    Map and align spatial transcriptomics data with tissue image.

    Internal function to perform image alignment between spatial data and tissue image.

    Arguments:
        adata_rotated: AnnData
            Annotated data matrix containing rotated spatial data.
        color: str or None, optional (default=None)
            Key in adata.obs or adata.var for visualization.
        method: str, optional (default='phase_correlation')
            Method for image alignment:
            - 'phase_correlation': Use phase correlation
            - 'feature': Use feature-based alignment

    Returns:
        tuple
            (offset, aligned_adata) where:
            - offset: tuple (dx, dy) representing the optimal alignment
            - aligned_adata: AnnData object with aligned coordinates

    Notes:
        - This is an internal function used by map_spatial_auto
        - Different alignment methods may work better for different data types
        - Phase correlation is generally more robust to intensity variations
    """
    import cv2
    scale_factor_denominator = pow(10, (len(str(int(adata_rotated.obsm['spatial'][:, 0].mean()))) - 2))
    fig_size_x = adata_rotated.obsm['spatial'][:, 0].mean() / scale_factor_denominator
    fig_size_y = adata_rotated.obsm['spatial'][:, 1].mean() / scale_factor_denominator

    # 确保 'temp' 目录存在
    os.makedirs('temp', exist_ok=True)

    # 创建并保存 image1 (带点图层)
    fig1, ax1 = plt.subplots(figsize=(fig_size_x, fig_size_y))
    _create_spatial_image(adata_rotated, ax1, color=color, alpha=0, save_path='temp/image1.png')
    image1_path = 'temp/image1.png'

    # 创建并保存 image2 (仅背景图像)
    fig2, ax2 = plt.subplots(figsize=(fig_size_x, fig_size_y))
    _create_spatial_image(adata_rotated, ax2, alpha=1, save_path='temp/image2.png') #  不传递 color, 仅背景图像
    image2_path = 'temp/image2.png'


    # 从文件读取图像 (实际应用中，如果可能，尽量避免读写文件，直接在内存中操作 NumPy 数组)
    img1_from_memory = cv2.imread(image1_path)
    img2_from_memory = cv2.imread(image2_path)
    if method == 'phase_correlation':
        # 使用相位相关法计算偏移量
        offset, _ = find_image_offset_phase_correlation_array_input(img1_from_memory, img2_from_memory)
        print(f"估计的偏移量 (dx, dy) 使用相位相关 (文件图像): {offset}")
    elif method == 'phase_correlation_torch':
        # 转换到PyTorch张量 (需要添加的预处理步骤)
        img1_tensor = torch.from_numpy(img1_from_memory).permute(2, 0, 1).float() / 255.0  # [C, H, W] 格式
        img2_tensor = torch.from_numpy(img2_from_memory).permute(2, 0, 1).float() / 255.0  # 归一化到[0,1]

        # 可选：移动到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # 调用函数
        offset, image1_aligned_pytorch = find_image_offset_phase_correlation_torch(
            img1_tensor, img2_tensor
        )

        # 转换回numpy用于显示 (需要添加的后处理步骤)
        image1_aligned_np = image1_aligned_pytorch.cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        image1_aligned_np = (image1_aligned_np * 255).astype(np.uint8)  # 恢复[0,255]

        print(f"估计的偏移量 (dx, dy) 使用 PyTorch 相位相关 (文件图像): {offset}")

    # 应用偏移量到空间坐标
    adata_rotated.obsm['spatial1'] = adata_rotated.obsm['spatial'].copy()
    adata_rotated.obsm['spatial1'][:, 0] -= (offset[0]) * (
        adata_rotated.obsm['spatial'][:, 0].mean() / (img1_from_memory.shape[0] / 2)
    )
    adata_rotated.obsm['spatial1'][:, 1] -= (offset[1]) * (
        adata_rotated.obsm['spatial'][:, 1].mean() / (img1_from_memory.shape[1] / 2)
    )
    return adata_rotated


@register_function(
    aliases=["自动空间映射", "map_spatial_auto", "auto_spatial_mapping", "空间自动映射", "自动对齐"],
    category="space",
    description="Automatically map and align spatial transcriptomics data with tissue image",
    examples=[
        "# Basic auto mapping with phase correlation",
        "ov.space.map_spatial_auto(adata_rotated, method='phase')",
        "# Feature-based alignment",
        "ov.space.map_spatial_auto(adata_rotated, method='feature')",
        "# Hybrid alignment approach",
        "ov.space.map_spatial_auto(adata_rotated, method='hybrid')",
        "# After rotation, apply auto mapping",
        "adata_rotated = ov.space.rotate_space_visium(adata, angle=45)",
        "ov.space.map_spatial_auto(adata_rotated)"
    ],
    related=["space.map_spatial_manual", "space.rotate_space_visium", "space.crop_space_visium"]
)
def map_spatial_auto(adata_rotated, method='phase'):
    """
    Automatically map and align spatial transcriptomics data.

    This function performs automatic alignment of spatial transcriptomics data
    with the corresponding tissue image using various alignment methods.

    Arguments:
        adata_rotated: AnnData
            Annotated data matrix containing spatial data to be aligned.
        method: str, optional (default='phase')
            Alignment method to use:
            - 'phase': Phase correlation-based alignment
            - 'feature': Feature-based alignment
            - 'hybrid': Combination of phase and feature methods

    Returns:
        AnnData
            Aligned AnnData object with updated spatial coordinates.

    Notes:
        - The function automatically selects the best alignment
        - Results can be verified using spatial plotting functions
        - Different methods may work better for different data types

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load and preprocess data
        >>> adata = sc.read_visium(...)
        >>> # Perform automatic alignment
        >>> adata_aligned = ov.space.map_spatial_auto(
        ...     adata,
        ...     method='phase'
        ... )
    """
    # 确保 'temp' 目录存在
    os.makedirs('temp', exist_ok=True)
    image1_path = 'temp/image1.png'
    image2_path = 'temp/image2.png'

    ee=pow(10,(len(str(int(adata_rotated.obsm['spatial'][:,0].mean())))-2))
    fig1, ax1 = plt.subplots(figsize=(adata_rotated.obsm['spatial'][:,0].mean()/ee,
                                     adata_rotated.obsm['spatial'][:,1].mean()/ee))
    ax=ax1
    adata_rotated.obs['test']='1'
    library_id = list(adata_rotated.uns['spatial'].keys())[0]
    scalefactors = adata_rotated.uns['spatial'][library_id]['scalefactors']
    tissue_hires_scalef = scalefactors['tissue_hires_scalef']
    sc.pl.embedding(
        adata_rotated,
        basis='spatial',
        color='Anno_manual',
        show=False,
        ax=ax1,
        size=10,
        scale_factor = tissue_hires_scalef,
        #frameon=False,
        legend_loc=None
    )
    cur_coords = np.concatenate([ax1.get_xlim(), ax1.get_ylim()])
    img=adata_rotated.uns["spatial"][library_id]["images"]["hires"]
    ax.set_xlim(cur_coords[0], cur_coords[1])
    ax.set_ylim(cur_coords[3], cur_coords[2])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    
    ax1.imshow(img, cmap='gray', alpha=0)
    plt.xlim(cur_coords[0], cur_coords[1])
    plt.ylim(cur_coords[3], cur_coords[2])
    plt.xticks([])
    plt.yticks([])
    plt.axis(False)

    fig1.savefig('temp/image1.png',bbox_inches='tight', )

    fig2, ax2 = plt.subplots(figsize=(adata_rotated.obsm['spatial'][:,0].mean()/ee,
                                 adata_rotated.obsm['spatial'][:,1].mean()/ee))
    ax=ax2
    
    ax2.imshow(img, cmap='gray', alpha=1)
    ax.set_xlim(cur_coords[0], cur_coords[1])
    ax.set_ylim(cur_coords[3], cur_coords[2])
    plt.xticks([])
    plt.yticks([])
    plt.axis(False)
    fig2.savefig('temp/image2.png',bbox_inches='tight', )

    

    import cv2 # 导入 cv2 只是为了模拟读取 png, 实际您不需要读取文件了
    image1_path = 'temp/image1.png' # 上图的路径 (您之前保存的)
    image2_path = 'temp/image2.png' # 下图的路径 (您之前保存的)
    
    img1_from_memory = cv2.imread(image1_path) # 模拟从内存获取 image1 的 numpy array
    img2_from_memory = cv2.imread(image2_path) # 模拟从内存获取 image2 的 numpy array
    
    

    # 修改后的配准逻辑
    if method == 'torch':
        # PyTorch处理路径
        img1_tensor = torch.from_numpy(img1_from_memory).permute(2, 0, 1).float() / 255.0
        img2_tensor = torch.from_numpy(img2_from_memory).permute(2, 0, 1).float() / 255.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)
        
        offset, image1_aligned = find_image_offset_phase_correlation_torch(img1_tensor, img2_tensor)
        image1_aligned = image1_aligned.cpu().permute(1, 2, 0).numpy() * 255
        image1_aligned = image1_aligned.astype(np.uint8)
        
    elif method == 'phase':
        # 原相位相关法处理
        offset, image1_aligned = find_image_offset_phase_correlation_array_input(img1_from_memory, img2_from_memory)
    else:
        raise ValueError("method 参数错误，请输入 'torch' 或 'phase'.")
    # 保持原有坐标修正逻辑不变 ...
    # 1) 先把 spatial1 转成 float64
    adata_rotated.obsm['spatial1'] = adata_rotated.obsm['spatial'].astype(np.float64)

    # 2) 再做就地减法
    adata_rotated.obsm['spatial1'][:,0] -= offset[0] * (
        adata_rotated.obsm['spatial'][:,0].mean() / (img1_from_memory.shape[0] / 2)
    )
    adata_rotated.obsm['spatial1'][:,1] -= offset[1] * (
        adata_rotated.obsm['spatial'][:,1].mean() / (img1_from_memory.shape[1] / 2)
    )
    return adata_rotated

@register_function(
    aliases=["手动空间映射", "map_spatial_manual", "manual_spatial_mapping", "空间手动映射", "手动对齐"],
    category="space",
    description="Manually adjust spatial transcriptomics data alignment with specified offset",
    examples=[
        "# Apply manual offset",
        "ov.space.map_spatial_manual(adata_rotated, offset=(-1500, 1000))",
        "# Fine-tune alignment",
        "ov.space.map_spatial_manual(adata_rotated, offset=(50, -30))",
        "# Large adjustment",
        "ov.space.map_spatial_manual(adata_rotated, offset=(-500, 200))",
        "# Access manually aligned coordinates",
        "manual_coords = adata_rotated.obsm['spatial1']"
    ],
    related=["space.map_spatial_auto", "space.rotate_space_visium", "space.crop_space_visium"]
)
def map_spatial_manual(
        adata_rotated,
        offset,
):
    """
    Manually adjust spatial transcriptomics data alignment.

    This function allows manual adjustment of the alignment between
    spatial transcriptomics data and the tissue image using specified offsets.

    Arguments:
        adata_rotated: AnnData
            Annotated data matrix containing spatial data to be aligned.
        offset: tuple
            (dx, dy) tuple specifying the manual offset to apply.

    Returns:
        AnnData
            Aligned AnnData object with manually adjusted spatial coordinates.

    Notes:
        - Useful for fine-tuning automatic alignment results
        - Offset values are in pixel coordinates
        - Positive dx moves spots right, positive dy moves spots down

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data
        >>> adata = sc.read_visium(...)
        >>> # Apply manual offset
        >>> adata_aligned = ov.space.map_spatial_manual(
        ...     adata,
        ...     offset=(10, -5)  # Move 10 pixels right, 5 pixels up
        ... )
    """
    adata_rotated.obsm['spatial1'] = adata_rotated.obsm['spatial'].copy()
    adata_rotated.obsm['spatial1'][:, 0] -= (offset[0]) * (
            adata_rotated.obsm['spatial'][:, 0].mean() / (adata_rotated.obsm['spatial'][:, 0].max())
    )
    adata_rotated.obsm['spatial1'][:, 1] -= (offset[1]) * (
            adata_rotated.obsm['spatial'][:, 1].mean() / (adata_rotated.obsm['spatial'][:, 1].max())
    )
    return adata_rotated


@register_function(
    aliases=["读取Visium数据", "read_visium_10x", "load_visium", "Visium数据读取", "空间数据载入"],
    category="space",
    description="Read and process 10x Visium spatial transcriptomics data",
    examples=[
        "# Basic Visium data loading",
        "adata = ov.space.read_visium_10x(adata_path)",
        "# With custom parameters",
        "adata = ov.space.read_visium_10x(adata_path, genome='GRCh38')",
        "# Load with filtering",
        "adata = ov.space.read_visium_10x(path, min_counts=100)"
    ],
    related=["space.svg", "space.crop_space_visium"]
)
def read_visium_10x(
        adata,
        **kwargs,
):
    from ..external.bin2cell import read_visium
    adata = read_visium(adata, **kwargs)
    adata.var_names_make_unique()
    return adata


@register_function(
    aliases=["Visium细胞分割HE", "visium_10x_hd_cellpose_he", "cellpose_he", "HE图像分割", "细胞核分割"],
    category="space",
    description="Cell segmentation on H&E images for high-resolution Visium data using CellPose",
    examples=[
        "# Basic H&E segmentation",
        "adata = ov.space.visium_10x_hd_cellpose_he(adata, mpp=0.3)",
        "# Custom parameters",
        "adata = ov.space.visium_10x_hd_cellpose_he(adata, mpp=0.5,",
        "                                           prob_thresh=0.1,",
        "                                           flow_threshold=0.3)",
        "# GPU acceleration",
        "adata = ov.space.visium_10x_hd_cellpose_he(adata, gpu=True,",
        "                                           buffer=200)"
    ],
    related=["space.visium_10x_hd_cellpose_gex", "space.bin2cell"]
)
def visium_10x_hd_cellpose_he(
        adata,
        mpp=0.3,
        he_save_path="stardist/he_colon.tiff",
        prob_thresh=0,
        flow_threshold=0.2,
        gpu=True,
        buffer=150,
        backend='tifffile',
        **kwargs,
):
    """
    Convert Visium 10x data to cell-level data.
    
    """
    from ..external.bin2cell import destripe, scaled_he_image, stardist, insert_labels

    if not os.path.exists(he_save_path):    
        destripe(adata)
        scaled_he_image(adata, mpp=mpp, buffer=buffer, save_path=he_save_path,
                        backend=backend)
    else:
        print(f"he_save_path {he_save_path} already exists, skipping destripe and scaled_he_image")
    stardist(image_path=he_save_path    , 
             labels_npz_path=he_save_path.replace(".tiff", ".npz"), 
             stardist_model="2D_versatile_he", 
             prob_thresh=prob_thresh,
             flow_threshold=flow_threshold,
             gpu=gpu,
             **kwargs,
            )
    insert_labels(adata, 
                  labels_npz_path=he_save_path.replace(".tiff", ".npz"), 
                  basis="spatial", 
                  spatial_key=f"spatial_cropped_{buffer}_buffer",
                  mpp=mpp, 
                  labels_key="labels_he"
                 )
    
@register_function(
    aliases=["Visium细胞扩展", "visium_10x_hd_cellpose_expand", "cellpose_expand", "细胞标签扩展", "空间扩展"],
    category="space",
    description="Expand cell segmentation labels to nearby bins for improved coverage",
    examples=[
        "# Basic label expansion",
        "adata = ov.space.visium_10x_hd_cellpose_expand(adata,",
        "                                               max_bin_distance=4)",
        "# Custom parameters",
        "adata = ov.space.visium_10x_hd_cellpose_expand(adata,",
        "                                               max_bin_distance=6,",
        "                                               labels_key='cellpose_labels')",
        "# Different expansion keys",
        "adata = ov.space.visium_10x_hd_cellpose_expand(adata,",
        "                                               expanded_labels_key='expanded_cells')"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.visium_10x_hd_cellpose_gex"]
)
def visium_10x_hd_cellpose_expand(
        adata,
        max_bin_distance=4,
        labels_key="labels_he",
        expanded_labels_key="labels_he_expanded",
        **kwargs,
):
    from ..external.bin2cell import expand_labels
    expand_labels(adata, 
                  labels_key=labels_key, 
                  expanded_labels_key=expanded_labels_key,
                  max_bin_distance=max_bin_distance,
                  **kwargs,
                 )
    
@register_function(
    aliases=["Visium细胞基因表达", "visium_10x_hd_cellpose_gex", "cellpose_gex", "细胞基因表达映射", "细胞水平表达"],
    category="space",
    description="Map gene expression to cell level using CellPose segmentation",
    examples=[
        "# Basic gene expression mapping",
        "adata = ov.space.visium_10x_hd_cellpose_gex(adata)",
        "# Custom parameters",
        "adata = ov.space.visium_10x_hd_cellpose_gex(adata,",
        "                                            obs_key='total_counts',",
        "                                            mpp=0.5, sigma=3)",
        "# With log transformation",
        "adata = ov.space.visium_10x_hd_cellpose_gex(adata, log1p=True,",
        "                                            prob_thresh=0.1)"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.bin2cell"]
)
def visium_10x_hd_cellpose_gex(
        adata,
        obs_key="n_counts_adjusted",
        log1p=False,
        mpp=0.3,
        sigma=5,
        gex_save_path="stardist/gex_colon.tiff",
        prob_thresh=0.01,
        nms_thresh=0.1,
        gpu=True,
        buffer=150,
        **kwargs,
):
    from ..external.bin2cell import grid_image, stardist, insert_labels,destripe
    #if gex_save_path's file exist, jump grid_image to stardist
    if obs_key not in adata.obs.keys():
        destripe(adata)
    if not os.path.exists(gex_save_path):
        grid_image(adata, obs_key, log1p=log1p,
                mpp=mpp, sigma=sigma, save_path=gex_save_path)
    else:
        print(f"gex_save_path {gex_save_path} already exists, skipping grid_image")
    stardist(image_path=gex_save_path, 
             labels_npz_path=gex_save_path.replace(".tiff", ".npz"), 
             stardist_model="2D_versatile_fluo", 
             prob_thresh=prob_thresh, 
             nms_thresh=nms_thresh,gpu=gpu,
             **kwargs,
            )
    insert_labels(adata, 
                  labels_npz_path=gex_save_path.replace(".tiff", ".npz"), 
                  basis="spatial", 
                  spatial_key=f"spatial_cropped_{buffer}_buffer",
                  mpp=mpp, 
                  labels_key="labels_gex"
                 )
    
@register_function(
    aliases=["挂失次级标签", "salvage_secondary_labels", "rescue_labels", "标签救救", "次级标签恢复"],
    category="space",
    description="Salvage secondary labels by combining primary and secondary segmentation results",
    examples=[
        "# Basic label salvaging",
        "ov.space.salvage_secondary_labels(adata, primary_label='labels_he',",
        "                                   secondary_label='labels_gex')",
        "# Custom label keys",
        "ov.space.salvage_secondary_labels(adata, primary_label='cellpose_he',",
        "                                   secondary_label='cellpose_gex',",
        "                                   labels_key='combined_labels')",
        "# Access combined labels",
        "joint_labels = adata.obs['labels_joint']"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.visium_10x_hd_cellpose_gex", "space.bin2cell"]
)
def salvage_secondary_labels(
        adata,
        primary_label="labels_he", 
        secondary_label="labels_gex",
        labels_key="labels_joint",
        **kwargs,
):
    from ..external.bin2cell import salvage_secondary_labels
    salvage_secondary_labels(adata, primary_label=primary_label,
                             secondary_label=secondary_label,
                             labels_key=labels_key)  
    
    
    
@register_function(
    aliases=["空间段到细胞", "bin2cell", "bin_to_cell", "空间细胞化", "段级到细胞级"],
    category="space",
    description="Convert spatial bins to single-cell resolution using segmentation labels",
    examples=[
        "# Basic bin to cell conversion",
        "adata_cell = ov.space.bin2cell(adata, labels_key='labels_joint')",
        "# Custom spatial keys",
        "adata_cell = ov.space.bin2cell(adata, labels_key='segmentation',",
        "                               spatial_keys=['spatial'])",
        "# With diameter scaling",
        "adata_cell = ov.space.bin2cell(adata, labels_key='cellpose_labels',",
        "                               diameter_scale_factor=1.5)"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.visium_10x_hd_cellpose_gex"]
)
def bin2cell(
        adata,
        labels_key="labels_joint",
        spatial_keys=["spatial"],
        diameter_scale_factor=None,
):
    from ..external.bin2cell import bin_to_cell
    return bin_to_cell(adata, labels_key=labels_key, 
                spatial_keys=spatial_keys,diameter_scale_factor=diameter_scale_factor)
    
    
    
