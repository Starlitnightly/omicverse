# OmicVerse Skills 阶段性成果报告

## 项目概述

为 OmicVerse 的 Claude Code skill 系统构建完整的领域知识技能库，使 AI agent 在执行生物信息学分析任务时能够正确调用 OmicVerse API，避免退化到 scanpy/scvelo 等默认库。

## 完成进度

### 第一轮：全量改进（25 个已有 skills）— 已完成

对项目初始的 25 个 skills 进行系统性重写：
- 重写 description（80 字符内前置关键词，优化 bag-of-words 匹配）
- 添加 Defensive Validation（前置断言检查）
- 添加 Troubleshooting 条目（带具体错误信息）
- 为 24 个 skill 创建 reference.md（快速拷贝代码块）
- 统一 CORRECT/WRONG 模式的 Critical API Reference

### 第二轮：新增 skills（7 个）— 已完成

| Skill | 行数 | 覆盖模块 | 填补的空白 |
|-------|------|----------|-----------|
| `fm-foundation-models` | 192+174 | `ov.fm` (22 个模型) | 零覆盖 → 完整 6 步工作流 |
| `single-scenic-grn` | 185+109 | `ov.single.SCENIC` (49KB) | 仅 18 行提及 → 独立 3 阶段流水线 |
| `single-cellfate-analysis` | 228+120 | `ov.single.Fate` (50KB) | 零覆盖 → 7 步 ATR 流水线 |
| `single-popv-annotation` | 237+199 | `ov.single` POPV 注释 | 零覆盖 → 群体级细胞类型注释工作流 |
| `biocontext-knowledge` | 204+163 | `ov.biocontext` (23 函数) | 零覆盖 → 49 个数据库查询 |
| `data-io-loading` | 217+197 | `ov.io` (重构后) | 零覆盖 → 替代 scanpy IO |
| `datasets-loading` | 187+231 | `ov.datasets` | 零覆盖 → 内置数据集加载与示例入口 |

### 第二轮：已有 skill 扩展 — 已完成

| Skill | 变更 | 内容 |
|-------|------|------|
| `single-trajectory` | +76 行 SKILL.md, +53 行 ref | 4 个 velocity 后端（dynamo/latentvelo/graphvelo） |
| `single-downstream-analysis` | +1 行交叉引用 | 指向独立 SCENIC skill |
| **10 个 skills 批量更新** | 20 处替换 | `sc.read_*` → `ov.io.*`/`ov.read()` |

### 第三轮：IO 迁移（`sc.read_*` 清理）— 已完成

跨 10 个 skill 文件完成 20 处 scanpy IO 引用替换：

| 替换类型 | 数量 | 说明 |
|----------|------|------|
| `sc.read_h5ad()` → `ov.read()` | 11 | 包括代码块和注释 |
| `sc.read_10x_mtx()` → `ov.io.read_10x_mtx()` | 3 | 同时移除 `cache=True`（ov 不支持） |
| `sc.read_visium()` → `ov.io.spatial.read_visium()` | 2 | 注意 spatial 子模块路径 |
| `sc.read()` → `ov.read()` | 4 | 通用读取 |

验证结果：`grep -rn 'sc\.read' skills/ | grep -v data-io-loading` 返回 0 条。仅 `data-io-loading/SKILL.md` 的迁移表保留 scanpy 作为反面示例。

## 当前 Skill 库全貌

**总计 32 个 skills**，5,892 行 SKILL.md + 3,247 行 reference.md = **9,139 行领域知识**

### 按领域分类

| 领域 | Skills | 数量 |
|------|--------|------|
| **Single-cell** | preprocessing, clustering, annotation, trajectory, cellfate, scenic-grn, cellphone-db, multiomics, popv | 9 |
| **Spatial** | spatial-tutorials, single-to-spatial-mapping | 2 |
| **Bulk RNA-seq** | deg, deseq2, combat, wgcna, stringdb-ppi, trajblend, tcga | 7 |
| **Foundation Models** | fm-foundation-models | 1 |
| **Knowledge Queries** | biocontext-knowledge | 1 |
| **Data I/O** | data-io-loading, datasets-loading, fastq-analysis | 3 |
| **Data Engineering** | data-transform, data-viz-plots, data-stats-analysis, data-export-excel, data-export-pdf | 5 |
| **Enrichment** | gsea-enrichment | 1 |
| **Visualization** | plotting-visualization | 1 |
| **Cross-cutting** | single-downstream-analysis (聚合型) | 1 |

### Skill 匹配机制

```
prompt_builder.py:200 — descriptions 截断到 80 字符进入系统 prompt
tool_runtime.py:494-500 — bag-of-words 匹配：
  searchable = f"{meta.name} {meta.description} {meta.slug}".lower()
  每个查询词在 searchable 中出现则 +1 分
  取 Top 2 加载完整内容（最多 4000 字符）
```

所有 skill 的 description 已针对此机制优化：
- 前 80 字符包含最具区分度的关键词
- 名称/slug 与 description 形成互补（避免重复浪费匹配位）
- 技术术语（scGPT, SCENIC, CellFateGenie, BioContext）放在前列

## 关键设计决策记录

1. **FM 不拆分** — 22 个模型用一个 skill，因为 `ov.fm` 有统一 6 步 API，拆分反而降低匹配率
2. **SCENIC 独立** — 虽然 downstream-analysis 提到了 SCENIC，但 49KB 模块 + 外部数据库下载需要独立引导
3. **Velocity 并入 trajectory** — 避免 "velocity" 查询同时匹配两个竞争 skill
4. **IO skill 核心定位** — 不只是 API 参考，更是"行为纠正"——用迁移表强化 agent 不用 scanpy
5. **`cache=True` 移除** — OmicVerse 的 `read_10x_mtx` 不支持此参数，替换时必须删除

## 未来可选方向

- **Description 自动化优化**：用 `scripts.run_loop` 对高优先级 skills 跑 trigger eval
- **Benchmark agent 侧同步**：`search_skills`/`search_functions` 注册表可能需要同步更新
- **新模块覆盖**：如果 OmicVerse 新增模块（如 `ov.multimodal`、`ov.perturbation`），可按此模式扩展
- **其他 skill 内的 scanpy 用法审计**：目前只清理了 `sc.read_*`，`sc.pp.*`/`sc.tl.*`/`sc.pl.*` 是否也有对应的 OmicVerse 替代值得排查
