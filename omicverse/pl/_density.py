import numpy as np
import scanpy as sc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def calculate_gene_density(adata,
    features,
    basis='X_umap',
    dims=(0, 1),
    method='scipy',
    adjust=1,):
    """
    根据基因表达作为权重，在指定的二维嵌入上计算加权核密度估计（gene density），
    并使用 scanpy 进行可视化。支持的 feature 可在 adata.obs（如元数据）中存在，
    或者在基因表达矩阵（adata.var_names）中查找。

    参数：
    adata: AnnData 对象
    features: list，待计算密度的特征（例如基因）名称列表
    basis: string，嵌入名称，默认 'X_umap'
    dims: tuple，二维嵌入中使用的列索引，默认 (0, 1)
    method: 选用的核密度估计方法，目前仅支持“scipy”（通过 gaussian_kde）
    adjust: 数值，用于调整带宽，默认为 1
    cmap: 颜色映射，默认 'viridis'
    point_size: 点的大小，默认 20
    grid_steps: 计算网格的步数（目前未使用，但可扩展用于绘制密度轮廓）
    show: 是否立即展示图形，默认 True

    返回：
    将每个 feature 的密度信息添加为 adata.obs 中的新列，并分别展示图形。
    """

    # 检查二维嵌入维度
    if len(dims) != 2:
        raise ValueError("只能绘制二维嵌入，请确保 dims 长度为 2")

    # 获取细胞嵌入（二维），例如 UMAP 坐标
    if basis not in adata.obsm.keys():
        raise ValueError(f"在 adata.obsm 中未找到 basis: {basis}")
    embeddings = adata.obsm[basis][:, dims]  # shape: (n_cells, 2)

    # 对每个 feature 分别计算密度
    for feature in features:
        # 判断 feature 是在 adata.obs 中还是在基因表达矩阵中
        if feature in adata.obs.columns:
            # 假设已经预先计算好、存放在 obs 中的数值（例如某种评分）
            weights = adata.obs[feature].to_numpy()
        elif feature in adata.var_names:
            # 从表达矩阵中提取，注意：如果数据为稀疏格式需转换为 array
            weights = adata[:, feature].X.toarray().reshape(-1)
        else:
            raise ValueError(f"未在 adata.obs 或 adata.var_names 中找到 feature: {feature}")

        # 检查嵌入及权重是否存在 NaN 或 inf
        valid = np.isfinite(weights) & np.all(np.isfinite(embeddings), axis=1)
        if not np.all(valid):
            emb_valid = embeddings[valid, :]
            weights_valid = weights[valid]
        else:
            emb_valid = embeddings
            weights_valid = weights

        # 使用 gaussian_kde 计算加权核密度估计
        try:
            # 注意：gaussian_kde 接受的数据要求 shape=(n_dim, n_samples)
            kde = gaussian_kde(emb_valid.T, weights=weights_valid, bw_method=adjust)
        except Exception as e:
            print(f"在特征 {feature} 的密度估计过程中出错: {e}")
            continue

        # 计算每个细胞所在位置的密度值
        density = kde(embeddings.T)
        # 将密度值保存到 adata.obs 中，新列名称为 "density_<feature>"
        density_col = f"density_{feature}"
        adata.obs[density_col] = density
        print(f"The density have been stored in adata.obs['{density_col}']")


