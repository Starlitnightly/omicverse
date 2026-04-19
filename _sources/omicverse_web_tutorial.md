# OmicVerse Web 使用教程

OmicVerse Web 是一个基于浏览器的单细胞 / 多组学分析平台，将 OmicVerse 的全部分析能力封装成可视化界面，无需编写代码即可完成从数据导入到结果输出的完整流程。

> 📚 For English version, please check [Web Tutorial (English)](omicverse_web_tutorial_en.md)

---

## 目录

1. [安装](#1-安装)
2. [启动服务](#2-启动服务)
3. [界面概览](#3-界面概览)
4. [上传数据](#4-上传数据)
5. [预处理](#5-预处理)
6. [降维与可视化](#6-降维与可视化)
7. [聚类](#7-聚类)
8. [细胞注释](#8-细胞注释)
9. [差异基因分析](#9-差异基因分析)
10. [差异细胞组成分析](#10-差异细胞组成分析)
11. [轨迹分析](#11-轨迹分析)
12. [代码执行器](#12-代码执行器)
13. [AI Agent](#13-ai-agent)
14. [文件管理与终端](#14-文件管理与终端)
15. [远程服务器部署](#15-远程服务器部署)

---

## 1. 安装

OmicVerse Web 提供两种安装方式。

### 方式一：PyPI 安装（推荐）

```bash
pip install omicverseweb
```

### 方式二：从源码仓库安装

适合需要最新开发版本或希望参与贡献的用户：

```bash
git clone https://github.com/Starlitnightly/omicverse-web.git
cd omicverse-web
pip install -e .
```

> **推荐**：在独立的 conda 环境中安装，避免依赖冲突。
>
> ```bash
> conda create -n omicverse python=3.10
> conda activate omicverse
> pip install omicverseweb
> ```

---

## 2. 启动服务

安装完成后，在终端执行：

```bash
omicverse-web
```

服务默认在 `http://localhost:5050` 启动（端口被占用时自动顺延至 5051、5052 …）。看到以下输出即表示启动成功：

```
* OmicVerse Web running on http://localhost:5050
```

在浏览器中打开该地址即可进入平台。

### 可选参数

```bash
omicverse-web --port 8080          # 指定端口
omicverse-web --no-debug           # 关闭调试模式（生产环境推荐）
omicverse-web --remote             # 远程模式（配合 SSH 隧道）
```

---

## 3. 界面概览

打开 `http://localhost:5050` 后，点击首页的 **Launch Analysis** 按钮进入主分析界面。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318031649002.png#gh-light-mode-only)
![黑夜模式](#gh-dark-mode-only)

主分析界面分为三个区域：

**左侧边栏**
- 📁 文件浏览器：管理本地文件，支持右键菜单操作
- 🧬 变量查看器：实时查看内核中的变量
- 💻 终端面板：浏览器内置 shell
- 📊 内存监控：实时查看内存占用

**顶部选项卡**
按分析流程排列：预处理 → 可视化 → 聚类 → 注释 → DEG → DCT → 轨迹

**右侧主区域**
当前选项卡的操作面板和结果展示区

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318031836591.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318031908737.png#gh-dark-mode-only)

---

## 4. 上传数据

点击顶部工具栏的 **Upload** 按钮（或将文件拖入文件浏览器），选择本地的 `.h5ad` 文件。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032115377.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032028522.png#gh-dark-mode-only)

上传成功后，状态栏会显示数据基本信息：

```
✓ 数据已加载
  细胞数：8,542 | 基因数：33,538
```

> **支持格式**：标准 AnnData `.h5ad` 文件。
> 如需从其他格式转换，可先在代码执行器中使用 `scanpy` 读取后保存为 `.h5ad`。

---

## 5. 预处理

切换到 **Preprocessing** 选项卡，按以下顺序依次执行各步骤。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032219322.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032232087.png#gh-dark-mode-only)

### 5.1 过滤细胞和基因

点击 **Filter Cells / Filter Genes** 工具卡，设置过滤阈值：

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| Min genes per cell | 每个细胞最少检测到的基因数 | 200 |
| Max genes per cell | 上限（去除潜在双细胞） | 5000 |
| Min cells per gene | 每个基因至少在几个细胞中表达 | 3 |
| Max mito % | 线粒体基因比例上限 | 0.2 |

点击 **Run** 执行，右侧会展示过滤前后的细胞/基因数量对比。

### 5.2 标准化

点击 **Normalize**，将每个细胞的总 UMI 计数归一化到统一基数（默认 10,000）。

### 5.3 对数变换

点击 **Log1p**，执行 log(1+x) 变换，压缩数据分布。

### 5.4 高变基因选择

点击 **HVG**，选取信息量最丰富的基因用于后续分析：

| 参数 | 推荐值 |
|------|--------|
| Top genes | 2000 |
| Flavor | seurat_v3 |

### 5.5 数据缩放

点击 **Scale**，对每个基因做 z-score 标准化，`max_value` 默认为 10。

---

## 6. 降维与可视化

切换到 **Visualization** 选项卡。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032408823.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032415885.png#gh-dark-mode-only)

### 6.1 PCA

点击 **PCA**，设置主成分数（默认 50），点击 Run。

### 6.2 构建邻居图

点击 **Neighbors**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| n_neighbors | 15 | 近邻数，影响 UMAP 的局部/全局结构权衡 |
| n_pcs | 40 | 使用前 N 个 PC |

### 6.3 UMAP

点击 **UMAP**，执行后在右侧画布中即可看到嵌入图。

### 6.4 调整可视化参数

在图表上方的控制栏可以：
- 切换 **Color by**：按基因表达量、细胞元数据或聚类着色
- 调整 **点大小** 和 **透明度**
- 切换渲染模式（标准 / 栅格化 / **GPU**）

> **GPU 渲染模式**：细胞数超过 10 万时推荐开启，基于 WebGL 实现实时流畅交互。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032522189.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032529435.png#gh-dark-mode-only)

---

## 7. 聚类

切换到 **Clustering** 选项卡。

### Leiden 聚类（推荐）

| 参数 | 说明 |
|------|------|
| Resolution | 分辨率，值越大聚类越细（推荐 0.3–1.0） |

点击 **Run** 后，聚类结果自动写入 `adata.obs['leiden']`，UMAP 图同步更新着色。

---

## 8. 细胞注释

切换到 **Annotation** 选项卡，提供三种注释方式。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032617456.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032626889.png#gh-dark-mode-only)

### 8.1 CellTypist（推荐，基于预训练模型）

选择与样本类型匹配的模型：

| 模型 | 适用场景 |
|------|----------|
| Immune_All_Low.pkl | 免疫细胞细粒度分类 |
| Immune_All_High.pkl | 免疫细胞粗粒度分类 |
| Human_Lung_Atlas.pkl | 人肺细胞 |

勾选 **Majority Voting**（邻域投票，提升一致性），点击 **Run**。

### 8.2 SCSA（数据库匹配）

选择物种（Human / Mouse），指定聚类列（`leiden`），点击 **Run**。

### 8.3 AI 辅助注释（GPT-4）

填写 OpenAI API Key，设置每个 cluster 展示的 marker 基因数（默认 10），点击 **Run**。AI 会根据 marker 基因推断细胞类型并生成文字解释。

---

## 9. 差异基因分析

切换到 **DEG** 选项卡。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032751136.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032757474.png#gh-dark-mode-only)

### 9.1 设置对比

1. **Group by**：选择分组依据列（如 `cell_type` 或 `leiden`）
2. **Group 1**：选择实验组
3. **Group 2**：选择对照组（或选 `rest` 代表其余所有细胞）
4. **Method**：选择统计方法（wilcoxon / t-test / mannwhitney）

点击 **Analyze**。

### 9.2 查看结果

分析完成后自动展示：

- **火山图**：横轴 Log2FC，纵轴 -log10(p-value)，点击数据点高亮该基因

- **结果表格**：列出所有差异基因及统计量，可按 FDR 或 Log2FC 排序

- **Violin 图**：在表格中点击任意基因，右侧自动展示该基因在两组中的表达分布

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032909171.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318032915420.png#gh-dark-mode-only)

---

## 10. 差异细胞组成分析

切换到 **DCT** 选项卡，分析不同样本条件下各细胞类型比例的变化。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033033250.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033039148.png#gh-dark-mode-only)

### 配置参数

| 参数 | 说明 |
|------|------|
| Cell type column | 细胞类型列名（如 `cell_type`） |
| Sample column | 样本标识列名 |
| Condition column | 条件列名（如 `disease` vs `control`） |
| Reference cell type | 参照细胞类型（sccoda 必填） |
| Method | sccoda 或 Milo |

点击 **Run** 后展示：
- **组成条形图**：各样本的细胞类型比例堆叠图

- **效应量图**：显示各细胞类型在条件间的显著变化

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033117650.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033123985.png#gh-dark-mode-only)

---

## 12. 代码执行器

如需自定义分析，点击右上角 **Code** 按钮打开内置代码编辑器。

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033338748.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033346015.png#gh-dark-mode-only)

内核预注入以下变量，可直接使用：

```python
sc      # scanpy
pd      # pandas
np      # numpy
plt     # matplotlib.pyplot
odata   # 当前 AnnData 对象
```

示例：

```python
# 查看数据结构
print(odata)

# 查看某个基因的表达分布
import matplotlib.pyplot as plt
sc.pl.violin(odata, keys='CD3D', groupby='leiden')
plt.show()

# 保存当前分析结果
odata.write_h5ad('result.h5ad')
```

按 **Shift+Enter** 执行代码，输出实时显示在编辑器下方。

---

## 13. AI Agent

AI Agent 可以理解自然语言，自动生成并执行分析代码。

### 13.1 配置

点击侧边栏 **Agent** 图标，展开配置面板：

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033453297.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033500763.png#gh-dark-mode-only)

| 字段 | 说明 |
|------|------|
| API Key | Claude / OpenAI API Key |
| Model | 如 `claude-opus-4-6`、`gpt-4o` |
| Endpoint | 自定义 API 端点（可选，兼容 OpenAI 格式） |

### 13.2 使用方式

在对话框中输入任务描述，按 **Send**：

**示例提示词：**

```
对当前数据做 leiden 聚类（分辨率 0.5），然后用 UMAP 展示结果，按聚类着色
```

```
分析 CD4 T cells 和 CD8 T cells 之间的差异基因，找出 top 20，画火山图
```

```
我的数据有多个批次（batch 列），请用 Harmony 做批次校正并重新计算 UMAP
```

Agent 会逐步展示：思考过程 → 生成代码 → 执行 → 返回结果图表。

---

## 14. 文件管理与终端

### 14.1 文件浏览器

点击左侧边栏 📁 图标展开文件浏览器。

支持操作（右键菜单）：

- 新建文件夹 / 文件
- 重命名 / 删除 / 复制 / 粘贴
- 双击打开 `.h5ad`、`.ipynb`、文本文件和图片

<!-- 截图：文件浏览器右键菜单 -->
> 📷 *[截图：文件浏览器和右键菜单]*

### 14.2 内置终端

点击左侧边栏 💻 图标，创建 shell 会话（bash / zsh），可执行任意命令：

```bash
# 安装额外包
pip install harmonypy

# 查看 GPU 状态
nvidia-smi

# 运行自定义脚本
python my_analysis.py
```

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033635132.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033640745.png#gh-dark-mode-only)

### 14.3 包管理

在 **Environment** 面板中搜索并安装 Python 包，无需切换到终端：

![日间模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033741251.png#gh-light-mode-only)
![黑夜模式](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260318033752197.png#gh-dark-mode-only)

---

## 15. 远程服务器部署

如需在远程服务器上运行，通过 SSH 隧道在本地浏览器访问。

### 服务器端

```bash
# 安装
pip install omicverseweb

# 启动（远程模式，仅监听本地回环）
omicverse-web --remote --no-debug
```

### 本地建立 SSH 隧道

```bash
ssh -L 5050:127.0.0.1:5050 username@your-server.com -N
```

然后在本地浏览器打开 `http://localhost:5050` 即可。

### 后台持久运行

```bash
nohup omicverse-web --remote --no-debug > omicverse_web.log 2>&1 &
```

---

## 参考资料

- [OmicVerse GitHub](https://github.com/Starlitnightly/omicverse)
- [OmicVerse Web GitHub](https://github.com/Starlitnightly/omicverse-web)
- [PyPI: omicverseweb](https://pypi.org/project/omicverseweb/)
- 论文：*OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing*, Nature Communications (2024), 15:5983
