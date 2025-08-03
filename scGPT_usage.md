# scGPT 完整使用指南

基于 omicverse/external/scllm 模块的统一 scGPT 接口使用指南。

## 🎯 概述

本指南涵盖了使用 omicverse.external.scllm 模块进行 scGPT 相关操作的完整流程，包括：

- **基本使用**: 细胞嵌入、细胞类型注释
- **模型微调**: 在参考数据上训练分类器
- **批次积分**: 去除批次效应的完整工作流
- **端到端工作流**: 自动化的完整流程

## 🏗️ 架构简介

### 模块结构
```
omicverse/external/scllm/
├── __init__.py         # 统一导入接口
├── base.py            # 基础抽象类和配置
├── scgpt_model.py     # scGPT 模型实现
├── model_factory.py   # 模型工厂和管理器
└── scgpt/            # scGPT 核心组件
    ├── model/        # Transformer 模型
    ├── tokenizer/    # 基因词汇表和分词器
    ├── preprocess/   # 数据预处理
    └── utils/        # 工具函数
```

### 核心类
- **SCLLMManager**: 高级管理接口，提供最简单的使用方式
- **ScGPTModel**: scGPT 模型的具体实现
- **ModelFactory**: 模型工厂，支持创建不同类型的模型
- **ModelConfig**: 模型配置管理
- **TaskConfig**: 任务特定的配置（annotation, integration, generation）

## 🚀 快速开始

### 基本安装要求

```python
# 必需的基础依赖
import omicverse as ov
import scanpy as sc
import pandas as pd
import numpy as np
```

### 最简单的使用方法

```python
# 1. 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 2. 创建 scGPT 管理器
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/scgpt/model"  # 包含 vocab.json, best_model.pt, args.json
)

# 3. 获取细胞嵌入
embeddings = manager.get_embeddings(adata)
print(f"细胞嵌入维度: {embeddings.shape}")

# 4. 将嵌入添加到 adata 用于下游分析
adata.obsm['X_scgpt'] = embeddings

# 5. 使用嵌入进行聚类和可视化
sc.pp.neighbors(adata, use_rep='X_scgpt')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts'])
```

## 📚 详细使用教程

### 1. 基本使用 - 细胞嵌入

#### 方法 1: 使用 SCLLMManager (推荐)

```python
import omicverse as ov
import scanpy as sc

# 加载数据
adata = sc.read_h5ad("single_cell_data.h5ad")

# 创建管理器
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/scgpt/model"
)

# 获取细胞嵌入
embeddings = manager.get_embeddings(adata)

# 添加到 adata 中
adata.obsm['X_scgpt'] = embeddings

# 下游分析
sc.pp.neighbors(adata, use_rep='X_scgpt')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts'])
```

#### 方法 2: 便捷函数

```python
# 直接加载模型
model = ov.external.scllm.load_scgpt("/path/to/model")
embeddings = model.get_embeddings(adata)

# 一行代码获取嵌入
embeddings = ov.external.scllm.load_scgpt("/path/to/model").get_embeddings(adata)
```

### 2. 细胞类型注释 - 完整工作流

#### 数据准备

```python
# 加载参考数据 (带细胞类型标注)
reference_adata = sc.read_h5ad("reference_with_celltypes.h5ad")
print(f"细胞类型: {reference_adata.obs['celltype'].unique()}")

# 加载查询数据 (待预测)
query_adata = sc.read_h5ad("query_data.h5ad")
print(f"查询数据: {query_adata.n_obs} 细胞")
```

#### 模型微调

```python
from sklearn.model_selection import train_test_split

# 方法 1: 使用便捷函数
result = ov.external.scllm.fine_tune_scgpt(
    train_adata=reference_adata,
    model_path="/path/to/pretrained/scgpt",
    save_path="/path/to/finetuned_model",
    epochs=15,
    batch_size=32,
    lr=1e-4,
    validation_split=0.2
)
print(f"微调完成! 最佳准确率: {result['results']['best_accuracy']:.4f}")

# 方法 2: 手动控制
# 分割数据
train_idx, val_idx = train_test_split(
    range(reference_adata.n_obs),
    test_size=0.2,
    stratify=reference_adata.obs['celltype'],
    random_state=42
)

train_adata = reference_adata[train_idx].copy()
val_adata = reference_adata[val_idx].copy()

# 创建管理器并微调
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/pretrained/scgpt"
)

fine_tune_results = manager.fine_tune(
    train_adata=train_adata,
    valid_adata=val_adata,
    epochs=20,
    batch_size=64,
    lr=5e-5
)

# 保存微调后的模型
manager.save_model("/path/to/finetuned_model")
```

#### 细胞类型预测

```python
# 方法 1: 使用便捷函数
results = ov.external.scllm.predict_celltypes_workflow(
    query_adata=query_adata,
    finetuned_model_path="/path/to/finetuned_model",
    save_predictions=True
)

print("预测完成!")
print(f"预测的细胞类型: {np.unique(results['predicted_celltypes'])}")
print("细胞类型分布:")
print(query_adata.obs['predicted_celltype'].value_counts())

# 方法 2: 手动控制
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/finetuned_model"
)

# 加载细胞类型映射
manager.model.load_celltype_mapping("/path/to/finetuned_model")

# 预测
prediction_results = manager.model.predict_celltypes(query_adata)

# 添加结果
query_adata.obs['predicted_celltype'] = prediction_results['predicted_celltypes']
query_adata.obs['predicted_celltype_id'] = prediction_results['predictions']
```

#### 端到端注释工作流

```python
# 一次性完成微调和预测
results = ov.external.scllm.end_to_end_scgpt_annotation(
    reference_adata=reference_adata,
    query_adata=query_adata,
    pretrained_model_path="/path/to/pretrained/scgpt",
    save_finetuned_path="/path/to/finetuned_model",
    epochs=15,
    batch_size=32,
    lr=1e-4,
    validation_split=0.2
)

print("端到端流程完成!")
print(f"微调最佳准确率: {results['fine_tune_results']['best_accuracy']:.4f}")
print(f"预测了 {len(results['prediction_results']['predicted_celltypes'])} 个细胞")
```

### 3. 批次积分 (Integration) - 去除批次效应

#### 核心技术

scGPT Integration 使用多种先进技术：

1. **DAB (Domain Adversarial Batch)** - 域对抗批次校正
2. **DSBN (Domain-Specific Batch Normalization)** - 域特异性批次归一化  
3. **ECS (Elastic Cell Similarity)** - 弹性细胞相似性
4. **GEPC (Gene Expression Prediction for Cells)** - 细胞基因表达预测
5. **更高的掩码比例** (0.4 vs 0.0)

#### 数据准备

```python
# 加载训练数据 (包含批次信息)
train_adata = sc.read_h5ad("train_with_batches.h5ad")
query_adata = sc.read_h5ad("query_with_batches.h5ad")

# 检查批次信息
print("训练数据批次分布:")
print(train_adata.obs['batch'].value_counts())
print("查询数据批次分布:")
print(query_adata.obs['batch'].value_counts())
```

#### 训练积分模型

```python
# 方法 1: 使用便捷函数
result = ov.external.scllm.train_integration_scgpt(
    train_adata=train_adata,
    model_path="/path/to/pretrained/scgpt",
    batch_key="batch",
    save_path="integration_model",
    epochs=20,
    mask_ratio=0.4,  # Integration 使用更高的掩码比例
    do_dab=True,     # 启用域对抗批次校正
    do_mvc=True,     # 启用 GEPC
    do_ecs=True,     # 启用弹性细胞相似性
    domain_spec_batchnorm=True,  # 启用 DSBN
    dab_weight=1.0,
    ecs_weight=10.0,
    gepc_weight=1.0
)

# 方法 2: 详细控制
from sklearn.model_selection import train_test_split

# 分割训练和验证集
train_idx, val_idx = train_test_split(
    range(train_adata.n_obs),
    test_size=0.2,
    stratify=train_adata.obs['batch'],
    random_state=42
)

train_split = train_adata[train_idx].copy()
val_split = train_adata[val_idx].copy()

# 创建管理器并训练
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/pretrained/scgpt"
)

integration_results = manager.model.train_integration(
    train_adata=train_split,
    valid_adata=val_split,
    batch_key="batch",
    epochs=25,
    batch_size=32,
    lr=1e-4,
    mask_ratio=0.4,
    # Integration 特定参数
    do_dab=True,
    do_mvc=True,
    do_ecs=True,
    domain_spec_batchnorm=True,
    dab_weight=1.0,
    ecs_weight=10.0,
    gepc_weight=1.0
)

print(f"Integration 训练完成! 最佳损失: {integration_results['best_loss']:.4f}")

# 保存模型
manager.save_model("my_integration_model")
```

#### 执行批次积分

```python
# 方法 1: 使用便捷函数
results = ov.external.scllm.integrate_batches_workflow(
    query_adata=query_adata,
    integration_model_path="integration_model",
    batch_key="batch"
)

# 积分后的嵌入已自动添加到 query_adata.obsm['X_scgpt_integrated']

# 方法 2: 手动控制
integration_manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="my_integration_model"
)

# 执行积分
integration_results = integration_manager.model.predict(
    query_adata, 
    task="integration",
    batch_key="batch",
    mask_ratio=0.4
)

# 获取积分后的嵌入
integrated_embeddings = integration_results['embeddings']
query_adata.obsm['X_scgpt_integrated'] = integrated_embeddings
```

#### 端到端积分工作流

```python
# 一键完成训练和积分
results = ov.external.scllm.end_to_end_scgpt_integration(
    train_adata=train_adata,
    query_adata=query_adata,
    pretrained_model_path="/path/to/pretrained/scgpt",
    batch_key="batch",
    save_integration_path="scgpt_integration_model",
    epochs=20,
    validation_split=0.2,
    # Integration 特定参数
    mask_ratio=0.4,
    do_dab=True,
    do_mvc=True, 
    do_ecs=True,
    domain_spec_batchnorm=True
)

print(f"✅ Integration 完成!")
print(f"训练损失: {results['train_results']['best_loss']:.4f}")
print(f"积分细胞数: {results['integration_results']['integration_stats']['total_cells']}")
```

#### 使用预训练模型的批次积分

```python
# 如果只有预训练模型，可以使用后处理方法
results = ov.external.scllm.integrate_with_scgpt(
    query_adata=query_adata,
    model_path="/path/to/pretrained/scgpt",
    batch_key="batch",
    correction_method="combat",  # 可选: 'combat', 'mnn', 'center_scale', 'none'
    save_embeddings=True
)

# 积分结果保存在 query_adata.obsm['X_scgpt_integrated']
```

### 4. 结果分析和可视化

#### 训练过程可视化

```python
import matplotlib.pyplot as plt

# 可视化训练历史
history = fine_tune_results['training_history']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(history['train_loss'], label='Training Loss', color='blue')
ax1.plot(history['val_loss'], label='Validation Loss', color='red')
ax1.set_title('Loss Curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 准确率曲线
ax2.plot(history['train_acc'], label='Training Accuracy', color='blue')
ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
ax2.set_title('Accuracy Curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

#### 细胞类型预测结果可视化

```python
import seaborn as sns

# 细胞类型分布
plt.figure(figsize=(10, 6))
celltype_counts = query_adata.obs['predicted_celltype'].value_counts()
sns.barplot(y=celltype_counts.index, x=celltype_counts.values, palette='viridis')
plt.title('Predicted Cell Type Distribution')
plt.xlabel('Number of Cells')
plt.ylabel('Cell Type')
plt.tight_layout()
plt.show()

# UMAP 可视化
if 'X_scgpt' in query_adata.obsm:
    sc.pp.neighbors(query_adata, use_rep='X_scgpt')
    sc.tl.umap(query_adata)
    
    sc.pl.umap(query_adata, color='predicted_celltype', 
               title='scGPT Predicted Cell Types',
               palette='tab20')
```

#### 批次积分效果评估

```python
# 可视化批次混合度
sc.pl.umap(query_adata, color='batch', title='After Integration')

# 计算批次混合指标 (需要安装 scib 包)
try:
    import scib
    
    # 批次校正指标
    silhouette_batch = scib.me.silhouette_batch(
        query_adata.obsm['X_scgpt_integrated'], 
        query_adata.obs['batch']
    )
    
    # 生物学保存指标 (如果有细胞类型信息)
    if 'celltype' in query_adata.obs:
        ari_celltype = scib.me.ari(
            query_adata.obs['celltype'], 
            query_adata.obs['celltype']  # 或聚类结果
        )
        print(f"ARI (Cell Type): {ari_celltype:.3f}")
    
    print(f"Silhouette Batch: {silhouette_batch:.3f}")
    
except ImportError:
    print("安装 scib 包以获得更多评估指标: pip install scib")

# 简单的批次混合度评估
from sklearn.neighbors import NearestNeighbors

embeddings = query_adata.obsm['X_scgpt_integrated']
batches = query_adata.obs['batch'].values

# 计算每个细胞最近邻中不同批次的比例
nn = NearestNeighbors(n_neighbors=50)
nn.fit(embeddings)
distances, indices = nn.kneighbors(embeddings)

batch_mixing_scores = []
for i, cell_batch in enumerate(batches):
    neighbor_batches = batches[indices[i]]
    different_batch_ratio = (neighbor_batches != cell_batch).mean()
    batch_mixing_scores.append(different_batch_ratio)

print(f"平均批次混合度: {np.mean(batch_mixing_scores):.3f}")
```

## ⚙️ 高级配置

### 自定义模型配置

```python
from omicverse.external.scllm import ModelConfig

# 自定义模型配置
custom_config = ModelConfig(
    embsize=256,      # 嵌入维度
    nhead=4,          # 注意力头数
    nlayers=6,        # Transformer 层数
    dropout=0.1,      # Dropout 率
    n_bins=51,        # 表达值分箱数
    max_seq_len=3001  # 最大序列长度
)

# 使用自定义配置
manager = ov.external.scllm.SCLLMManager(
    model_type="scgpt",
    model_path="/path/to/model",
    **custom_config.to_dict()
)
```

### 任务特定配置

```python
from omicverse.external.scllm import TaskConfig

# 获取预定义的任务配置
annotation_config = TaskConfig.get_task_config("annotation")
integration_config = TaskConfig.get_task_config("integration")
generation_config = TaskConfig.get_task_config("generation")

# 微调模型使用任务配置
results = manager.fine_tune(
    train_adata=train_data,
    valid_adata=valid_data,
    task="annotation",
    **annotation_config
)
```

### 集成参数详解

```python
# Integration 特定参数
integration_results = manager.model.train_integration(
    train_adata=train_adata,
    batch_key="batch",
    
    # 基本训练参数
    epochs=25,              # Integration 通常需要更多轮次
    batch_size=32,          # 批量大小
    lr=1e-4,               # 学习率
    
    # Integration 核心参数
    mask_ratio=0.4,         # 掩码比例 (Integration 用 0.4, Annotation 用 0.0)
    
    # 技术开关
    do_dab=True,            # 域对抗批次校正
    do_mvc=True,            # Gene Expression Prediction for Cells  
    do_ecs=True,            # 弹性细胞相似性
    domain_spec_batchnorm=True,  # 域特异性批次归一化
    
    # 损失权重
    dab_weight=1.0,         # DAB 损失权重
    ecs_weight=10.0,        # ECS 损失权重 (通常较高)
    gepc_weight=1.0,        # GEPC 损失权重
)
```

## 📝 使用注意事项

### 模型文件要求

模型目录应包含以下文件：

```
your_model_directory/
├── vocab.json          # 词汇表文件 (必需)
├── best_model.pt       # 模型权重文件 (必需)
└── args.json           # 模型配置文件 (可选，但推荐)
```

### 数据要求

- **输入**: AnnData 对象，包含基因表达数据
- **基因命名**: 确保基因名称与模型词汇表匹配
- **预处理**: 系统会自动进行必要的预处理 (归一化、分箱等)
- **批次信息**: Integration 任务需要在 `adata.obs` 中包含批次标签

### 数据预处理智能检测

系统会自动检测数据的归一化状态：

```python
# 系统会自动检测以下情况：
# - 数据是否已归一化到 10k 或 1M
# - 数据是否已经 log 变换
# - 数据是否已经过预处理

# 用户可以手动控制：
manager.get_embeddings(
    adata, 
    skip_normalization=True,    # 跳过归一化
    force_normalization=True,   # 强制归一化
    data_is_raw=False          # 指定数据不是原始计数
)
```

## 🔧 故障排除

### 常见问题和解决方案

#### 1. 模型加载问题

```python
# 检查模型文件
import os
model_path = "/path/to/your/model"

required_files = ["vocab.json", "model.pt", "args.json"]
for file in required_files:
    file_path = os.path.join(model_path, file)
    if os.path.exists(file_path):
        print(f"✓ {file} exists")
    else:
        print(f"❌ {file} missing")
```

#### 2. 基因匹配检查

```python
# 检查数据与词汇表的匹配度
manager = ov.external.scllm.SCLLMManager("scgpt", model_path)

vocab_genes = set(manager.model.vocab.get_itos())
data_genes = set(adata.var_names)
overlap = vocab_genes.intersection(data_genes)

print(f"词汇表基因数: {len(vocab_genes)}")
print(f"数据基因数: {len(data_genes)}")
print(f"重叠基因数: {len(overlap)}")
print(f"匹配率: {len(overlap)/len(data_genes)*100:.1f}%")

if len(overlap) / len(data_genes) < 0.5:
    print("⚠️  基因匹配率较低，可能影响模型性能")
```

#### 3. 内存优化

```python
import torch

# 检查 GPU 状态
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 对于内存较小的 GPU，使用更小的批量大小
    batch_size = 16 if torch.cuda.get_device_properties(0).total_memory < 8e9 else 32
else:
    print("使用 CPU，建议 batch_size=8")
    batch_size = 8

# 在微调时使用调整后的批量大小
fine_tune_results = manager.fine_tune(
    train_adata=train_adata,
    batch_size=batch_size,
    # 其他参数...
)
```

## 📈 性能优化建议

### 1. 数据预处理优化

```python
# 预先过滤低质量细胞和基因
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 选择高变基因以减少计算量
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata_hvg = adata[:, adata.var.highly_variable].copy()
```

### 2. 批量处理

```python
# 对于多个数据集的批量处理
datasets = ["dataset1.h5ad", "dataset2.h5ad", "dataset3.h5ad"]
results = []

for dataset_path in datasets:
    adata = sc.read_h5ad(dataset_path)
    result = manager.model.predict_celltypes(adata)
    results.append(result)
    print(f"完成 {dataset_path}")
```

## 🎯 最佳实践

### 1. 数据准备
- 确保所有数据都有一致的批次标注 (对于 Integration)
- 批次间应该有足够的细胞数量
- 基因名称在所有批次间保持一致

### 2. 参数调优
- **mask_ratio**: Integration 通常使用 0.4，比 annotation 的 0.0 更高
- **epochs**: Integration 需要更多训练轮次 (20-30)  
- **loss weights**: ECS 权重通常设置较高 (10.0)

### 3. 验证和评估
- 使用验证集监控训练过程
- 检查批次混合度和生物学信息保存
- 使用 UMAP 可视化验证效果

## 🚀 与其他方法的对比

| 方法 | 优势 | 适用场景 |
|------|------|----------|
| **scGPT Integration** | 保留生物学信息，多种技术结合 | 复杂批次效应，大规模数据 |
| Harmony | 快速，简单 | 简单批次效应 |
| Scanorama | 处理不同技术平台 | 跨平台整合 |
| ComBat | 传统方法，稳定 | 简单线性批次效应 |

## 💡 优势总结

### 与原始 scGPT 教程的对比

| 项目 | 原始方法 | 新接口 |
|------|---------|--------|
| 代码长度 | 200+ 行 | 3-10 行 |
| 参数配置 | 手动设置 30+ 参数 | 自动配置 + 可选自定义 |
| 错误处理 | 用户自行处理 | 内置错误处理 |
| 模型管理 | 手动管理组件 | 统一管理接口 |
| 扩展性 | 单一模型 | 多模型支持架构 |
| 任务支持 | 手动配置 | 预配置的任务特定参数 |

### 主要优势

1. **大幅简化使用**: 从 200+ 行代码减少到几行
2. **统一接口**: 不同模型使用相同的 API
3. **任务导向**: 预配置的最佳实践参数
4. **易于扩展**: 模块化设计支持新模型
5. **错误处理**: 优雅处理依赖项问题
6. **向后兼容**: 不影响现有的 scGPT 使用方式
7. **智能预处理**: 自动检测数据状态并适配

这个统一的接口让研究人员可以专注于科学问题，而不是技术细节！

## 📚 完整示例脚本

### 细胞类型注释完整示例

```python
#!/usr/bin/env python3
"""
scGPT 细胞类型注释完整示例
"""

import omicverse as ov
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    # 1. 加载数据
    print("📚 加载数据...")
    reference_adata = sc.read_h5ad("reference.h5ad")
    query_adata = sc.read_h5ad("query.h5ad")
    
    print(f"参考数据: {reference_adata.n_obs} cells × {reference_adata.n_vars} genes")
    print(f"查询数据: {query_adata.n_obs} cells × {query_adata.n_vars} genes")
    print(f"细胞类型: {reference_adata.obs['celltype'].nunique()} 种")
    
    # 2. 端到端注释工作流
    print("\n🎯 执行端到端注释...")
    results = ov.external.scllm.end_to_end_scgpt_annotation(
        reference_adata=reference_adata,
        query_adata=query_adata,
        pretrained_model_path="path/to/scgpt/model",
        save_finetuned_path="finetuned_scgpt_model",
        epochs=15,
        batch_size=32,
        lr=1e-4,
        validation_split=0.2
    )
    
    print(f"✅ 注释完成!")
    print(f"微调最佳准确率: {results['fine_tune_results']['best_accuracy']:.4f}")
    print(f"预测了 {len(results['prediction_results']['predicted_celltypes'])} 个细胞")
    
    # 3. 可视化结果
    print("\n📊 可视化结果...")
    
    # 细胞类型分布
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    celltype_counts = query_adata.obs['predicted_celltype'].value_counts()
    sns.barplot(y=celltype_counts.index, x=celltype_counts.values, palette='viridis')
    plt.title('Predicted Cell Type Distribution')
    plt.xlabel('Number of Cells')
    plt.ylabel('Cell Type')
    plt.tight_layout()
    plt.savefig('predicted_celltype_distribution.pdf')
    plt.show()
    
    # UMAP 可视化
    if 'embeddings' in results['prediction_results']:
        query_adata.obsm['X_scgpt'] = results['prediction_results']['embeddings']
        sc.pp.neighbors(query_adata, use_rep='X_scgpt')
        sc.tl.umap(query_adata)
        
        sc.pl.umap(query_adata, color='predicted_celltype', 
                   title='scGPT Predicted Cell Types',
                   save='_scgpt_predictions.pdf')
    
    # 4. 保存结果
    print("\n💾 保存结果...")
    query_adata.write("query_with_predictions.h5ad")
    
    print("\n🎉 注释示例完成!")
    return results

if __name__ == "__main__":
    main()
```

### 批次积分完整示例

```python
#!/usr/bin/env python3
"""
scGPT Integration 完整示例
"""

import omicverse as ov
import scanpy as sc
import pandas as pd
import numpy as np

def integration_example():
    print("🚀 scGPT Integration 示例")
    
    # 1. 加载数据
    print("\n📚 加载数据...")
    train_adata = sc.read_h5ad("train_with_batches.h5ad")
    query_adata = sc.read_h5ad("query_with_batches.h5ad")
    
    print(f"训练数据: {train_adata.n_obs} 细胞")
    print(f"查询数据: {query_adata.n_obs} 细胞")
    print(f"训练数据批次: {train_adata.obs['batch'].nunique()}")
    print(f"查询数据批次: {query_adata.obs['batch'].nunique()}")
    
    # 2. 端到端 Integration 工作流
    print("\n🎯 执行端到端 Integration...")
    results = ov.external.scllm.end_to_end_scgpt_integration(
        train_adata=train_adata,
        query_adata=query_adata,
        pretrained_model_path="path/to/pretrained/scgpt",
        batch_key="batch",
        save_integration_path="scgpt_integration_model",
        epochs=20,
        validation_split=0.2,
        # Integration 特定参数
        mask_ratio=0.4,
        do_dab=True,
        do_mvc=True, 
        do_ecs=True,
        domain_spec_batchnorm=True
    )
    
    print(f"✅ Integration 完成!")
    print(f"训练损失: {results['train_results']['best_loss']:.4f}")
    print(f"积分细胞数: {results['integration_results']['integration_stats']['total_cells']}")
    
    # 3. 可视化 Integration 结果
    print("\n📊 可视化结果...")
    
    # 使用积分后的嵌入进行 UMAP
    sc.pp.neighbors(query_adata, use_rep='X_scgpt_integrated')
    sc.tl.umap(query_adata)
    
    # 可视化批次效应去除效果
    sc.pl.umap(query_adata, color='batch', 
               title='scGPT Integration - Batches',
               save='_scgpt_integration_batches.pdf')
    
    # 如果有细胞类型信息，也可以可视化
    if 'celltype' in query_adata.obs:
        sc.pl.umap(query_adata, color='celltype',
                   title='scGPT Integration - Cell Types', 
                   save='_scgpt_integration_celltypes.pdf')
    
    # 4. 保存结果
    print("\n💾 保存结果...")
    query_adata.write("query_integrated.h5ad")
    
    print("\n🎉 Integration 示例完成!")
    return results

if __name__ == "__main__":
    integration_example()
```

这个统一的文档涵盖了 scGPT 的所有主要功能，用户可以根据自己的需求选择合适的方法和参数配置！