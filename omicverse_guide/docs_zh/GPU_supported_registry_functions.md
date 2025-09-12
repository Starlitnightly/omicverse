# 已注册函数 GPU 支持一览

## 说明
- [x]: 支持 GPU 加速（原生/参数可启用/按环境自动启用）
- [ ]: 暂不支持 GPU 加速或仅 CPU 实现

提示：建议先根据需要初始化 `ov.settings.gpu_init()`（RAPIDS）或 `ov.settings.cpu_gpu_mixed_init()`（混合）。实际是否用到 GPU 亦取决于依赖是否安装 GPU 版本。

## Settings
- [x] `ov.settings.gpu_init`：初始化 RAPIDS/GPU 环境
- [x] `ov.settings.cpu_gpu_mixed_init`：初始化 CPU-GPU 混合模式

## Preprocessing（ov.pp）
- [x] `ov.pp.anndata_to_GPU`：将 AnnData 数据迁移到 GPU（RAPIDS）
- [ ] `ov.pp.anndata_to_CPU`：将数据迁回 CPU
- [x] `ov.pp.preprocess`：预处理（gpu <span class="tag tag-rapids">rapids</span>）
  - 子流程：`mode='shiftlog|pearson'`
    - [x] normalize_total/log1p（gpu <span class="tag tag-rapids">rapids</span>）
    - [x] HVGs=pearson_residuals（gpu <span class="tag tag-rapids">rapids</span>）
  - 子流程：`mode='pearson|pearson'`
    - [x] normalize_pearson_residuals（gpu <span class="tag tag-rapids">rapids</span>）
    - [x] HVGs=pearson_residuals（gpu <span class="tag tag-rapids">rapids</span>）
- [x] `ov.pp.scale`：标准化（gpu <span class="tag tag-rapids">rapids</span>）
- [x] `ov.pp.pca`：PCA（gpu <span class="tag tag-rapids">rapids</span> | <span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>|<span class="tag tag-mlx">mlx</span>]）
- `ov.pp.neighbors`：邻居图（按 method 标注）
  - [ ] method='umap'（UMAP 邻接估计，CPU）
  - [ ] method='gauss'（高斯核，CPU）
  - [x] method='rapids'（gpu <span class="tag tag-rapids">rapids</span>）
- `ov.pp.umap`：UMAP（按实现标注）
  - [ ] Scanpy UMAP（settings.mode='cpu'）
  - [x] RAPIDS UMAP（settings.mode='gpu'，gpu <span class="tag tag-rapids">rapids</span>）
  - [x] PyMDE/torch 路径（settings.mode='cpu-gpu-mixed'，<span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>]）
- [x] `ov.pp.qc`：质控（gpu <span class="tag tag-rapids">rapids</span> | <span class="tag tag-mixed">cpu-gpu-mixed</span>[<span class="tag tag-torch">torch</span>]）
- [ ] `ov.pp.score_genes_cell_cycle`：细胞周期评分
- [ ] `ov.pp.sude`：SUDE 降维（CPU 实现）

## Utils（ov.utils）
- [x] `ov.utils.mde`：MDE 最小失真嵌入（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- `ov.utils.cluster`：多算法聚类（逐项标注如下）
  - [ ] Leiden（Scanpy 实现，CPU）
  - [ ] Louvain（Scanpy 实现，CPU）
  - [ ] KMeans（scikit-learn，CPU）
  - [ ] GMM/mclust（scikit-learn，CPU）
  - [ ] mclust_R（R 包 mclust，CPU）
  - [ ] schist（schist 库，CPU）
  - [ ] scICE（当前调用中 `use_gpu=False`）
- [ ] `ov.utils.refine_label`：邻域投票精化标签
- [ ] `ov.utils.weighted_knn_trainer`：KNN 训练
- [ ] `ov.utils.weighted_knn_transfer`：KNN 标签迁移

## Single-cell（ov.single）
- `ov.single.batch_correction`：批次校正（按方法标注）
  - [ ] harmony（Harmony，CPU）
  - [ ] combat（Scanpy Combat，CPU）
  - [ ] scanorama（Scanorama，CPU）
  - [x] scVI（scvi-tools，GPU 可用）
  - [ ] CellANOVA（CPU）
- [x] `ov.single.MetaCell`：SEACells（`use_gpu=True` 可用）
- `ov.single.TrajInfer`：轨迹推断（按方法标注）
  - [ ] palantir（CPU）
  - [ ] diffusion_map（CPU）
  - [ ] slingshot（CPU）
- [x] `ov.single.Fate`：TimeFateKernel（内部 torch，自动 CUDA）
- [x] `ov.single.pyCEFCON`：CEFCON 驱动因子（`cuda>=0` 指定 GPU）
- [x] `ov.single.gptcelltype_local`：本地 LLM 注释（`device_map='cuda'`）
- [ ] `ov.single.cNMF`：cNMF（CPU 实现）
- [ ] `ov.single.CellVote`：多方法投票
  - [ ] scsa_anno（SCSA 注释，CPU）
  - [ ] gpt_anno（在线 GPT 注释，CPU/网络）
  - [ ] gbi_anno（GPTBioInsightor，CPU/网络）
  - [ ] popv_anno（PopV 注释，CPU）
- [ ] `ov.single.gptcelltype`：在线 GPT 注释
- [ ] `ov.single.mouse_hsc_nestorowa16`：加载数据
- [ ] `ov.single.load_human_prior_interaction_network`：加载先验网络
- [ ] `ov.single.convert_human_to_mouse_network`：物种基因符号转换

## Spatial（ov.space）
- [x] `ov.space.pySTAGATE`：STAGATE 空间聚类（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- `ov.space.clusters`：多方法空间聚类（按方法标注）
  - [x] STAGATE（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
  - [x] GraphST（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
  - [x] CAST（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
  - [x] BINARY（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- [ ] `ov.space.merge_cluster`：类群合并
- [ ] `ov.space.Cal_Spatial_Net`：空间邻域网络构建
- [x] `ov.space.pySTAligner`：STAligner 空间整合（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- [x] `ov.space.pySpaceFlow`：SpaceFlow 空间表征（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- `ov.space.Tangram`：Tangram 空间解卷积（按模式标注）
  - [x] mode='clusters'（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
  - [x] mode='cells'（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- [ ] `ov.space.svg`：空间变异基因（依赖预处理/统计，非显式 GPU）
- [x] `ov.space.CAST`：CAST 融合（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
- [ ] `ov.space.crop_space_visium`：裁剪图像/坐标
- [ ] `ov.space.rotate_space_visium`：旋转图像/坐标
- `ov.space.map_spatial_auto`：自动空间映射（按方法标注）
  - [x] method='torch'（<span class="tag tag-all">all</span>[<span class="tag tag-torch">torch</span>]）
  - [ ] method='phase'（NumPy/CPU）
  - [ ] method='feature'（特征点匹配，CPU）
  - [ ] method='hybrid'（混合管线，CPU）
- [ ] `ov.space.map_spatial_manual`：手动偏移
- [ ] `ov.space.read_visium_10x`：读取数据
- [x] `ov.space.visium_10x_hd_cellpose_he`：HE 图像细胞分割（`gpu=True`）
- [ ] `ov.space.visium_10x_hd_cellpose_expand`：标签扩展
- [x] `ov.space.visium_10x_hd_cellpose_gex`：GEX 图像分割/映射（`gpu=True`）
- [ ] `ov.space.salvage_secondary_labels`：合并标签
- [ ] `ov.space.bin2cell`：bin 到细胞级

## External（ov.external）
- [x] `ov.external.GraphST.GraphST`：GraphST（`device` 可用 GPU）
- [ ] `ov.bulk.pyWGCNA`：WGCNA（CPU 实现）

## Plotting（ov.pl）
- [ ] `ov.pl.*`（`_single/_bulk/_density/_dotplot/_violin/_general/_palette` 等）：绘图接口

## Bulk（ov.bulk）
- [ ] `ov.bulk.*`（`_Deseq2/_Enrichment/_combat/_network/_tcga` 等）：统计/富集/网络分析
