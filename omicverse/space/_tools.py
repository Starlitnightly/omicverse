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

def crop_space_visium(adata,crop_loc,crop_area,
               library_id,scale,spatial_key='spatial',res='hires'):
    import squidpy as sq
    adata1=adata.copy()
    img = sq.im.ImageContainer(
        adata1.uns["spatial"][library_id]["images"][res], library_id=library_id
    )
    crop_corner = img.crop_corner(crop_loc[0], crop_loc[1], size=crop_area,scale=scale,)
    adata1.obsm['spatial1']=adata1.obsm[spatial_key]*\
                adata1.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    adata_crop = crop_corner.subset(adata1,spatial_key='spatial1')
    adata_crop.uns["spatial"][library_id]["images"][res]=np.squeeze(crop_corner['image'].data,axis=2)
    adata_crop.obsm[spatial_key][:,0]=(adata_crop.obsm['spatial1'][:,0]-crop_loc[1])/adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    adata_crop.obsm[spatial_key][:,1]=(adata_crop.obsm['spatial1'][:,1]-crop_loc[0])/adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    
    return adata_crop

import numpy as np
import scipy.ndimage  # 导入 scipy.ndimage 库用于图像旋转

def rotate_space_visium(
    adata,
    angle,
    center=None, # 新增 center 参数，用于指定旋转中心，默认为 None (空间坐标中心)
    spatial_key='spatial',
    res='hires',
    library_id=None,
    interpolation_order=1 # 新增 interpolation_order 参数，控制插值阶数
):
    """
    旋转 Visium 空间数据的整个图像和空间坐标 (坐标旋转方向调整为顺时针)。

    Args:
        adata: AnnData 对象，包含空间数据。
        angle: 旋转角度，单位为度。正值为角度值 (例如 45, 90)。
               代码内部会将坐标旋转方向调整为顺时针，以匹配图像旋转方向 (假设图像旋转为逆时针)。
        center: 旋转中心坐标 (x, y)，**空间坐标**。如果为 None，则使用空间坐标的中心作为旋转中心。
                坐标单位应与 `adata.obsm[spatial_key]` 的空间坐标单位一致。
        spatial_key: 空间坐标存储的键值，默认为 'spatial'。
        res: 图像分辨率，默认为 'hires'。
        library_id: library_id of interest. 必须明确指定。
        interpolation_order: 图像旋转插值的阶数，默认为 1 (双线性插值)。
                                常用的取值范围为 0-5，数值越大，插值效果越平滑，但计算量也越大。
                                0: 最近邻插值，1: 双线性插值，3: 三次样条插值等。

    Returns:
        adata_rotate: 旋转后的 AnnData 对象。
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
    使用相位相关法计算两张图像之间的偏移量 (仅平移)。
    **直接接受 NumPy 数组作为输入。**

    参数:
        image1_array:  第一张图像的 NumPy 数组 (需要移动的图像)
        image2_array:  第二张图像的 NumPy 数组 (作为参考的图像)

    返回值:
        offset:  偏移量 (元组 (dx, dy))，表示 image1 相对于 image2 的偏移
        image1_aligned:  移动后的 image1，使其与 image2 对齐
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
    使用相位相关法计算两张图像之间的偏移量 (仅平移)。PyTorch版本。
    
    参数:
        image1_tensor:  Tensor [C, H, W] 或 [H, W] (需要移动的图像)
        image2_tensor:  Tensor [C, H, W] 或 [H, W] (参考图像)
    
    返回值:
        offset:  偏移量元组 (dx, dy)
        image1_aligned:  对齐后的图像 Tensor
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
    创建并保存空间位置图像的辅助函数。

    参数:
        adata: AnnData 对象
        ax:  Matplotlib 轴对象
        color: (可选) 用于着色的列名
        alpha: (可选) imshow 的透明度
        save_path: (可选) 保存图像的路径，如果为 None 则不保存
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
    绘制空间图像，并使用相位相关法校正两幅图像的偏移。

    参数:
        adata_rotated:  AnnData 对象，包含空间数据
        color: (可选) 用于空间着色的列名

    返回值:
        adata_rotated:  校正偏移后的 AnnData 对象
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


def map_spatial_auto(adata_rotated, method='phase'):
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

def map_spatial_manual(
        adata_rotated,
        offset,
):
    """
    Manually map spatial coordinates based on the given offset.
    Arguments:
        adata_rotated: AnnData object with rotated spatial coordinates.
        offset: Tuple (dx, dy) representing the offset.
    Returns:
        adata_rotated: AnnData object with updated spatial coordinates.
    """
    adata_rotated.obsm['spatial1'] = adata_rotated.obsm['spatial'].copy()
    adata_rotated.obsm['spatial1'][:, 0] -= (offset[0]) * (
            adata_rotated.obsm['spatial'][:, 0].mean() / (adata_rotated.obsm['spatial'][:, 0].max())
    )
    adata_rotated.obsm['spatial1'][:, 1] -= (offset[1]) * (
            adata_rotated.obsm['spatial'][:, 1].mean() / (adata_rotated.obsm['spatial'][:, 1].max())
    )
    return adata_rotated
    

