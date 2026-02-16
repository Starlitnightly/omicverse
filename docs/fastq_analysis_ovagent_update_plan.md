# OmicVerse FASTQ分析 — ovagent更新计划

## 优先级排序原则

按照 **影响面 × 实施复杂度 × 依赖关系** 排序。先修底层数据，再修上层逻辑，最后补高级抽象。

---

## P1 (最高优先级): 修复 `parallel_fastq_dump` 分类错误

**影响**: Agent在按category查找alignment函数时会漏掉这个函数
**文件**: `omicverse/alignment/kb_api.py:685`
**改动量**: 1行

```python
# 当前 (第688行)
category="utils",

# 修改为
category="alignment",
```

**依赖**: 无。独立修改，0风险。

---

## P2: 为alignment函数添加prerequisite元数据

**影响**: Agent无法做前置条件验证，无法自动推荐workflow链
**文件**: 5个alignment模块文件的 `@register_function` 装饰器
**改动量**: ~60行新增

### 具体修改

**`prefetch.py`** — 工作流起点，无requires:
```python
@register_function(
    aliases=["prefetch", "sra_prefetch", "sra-download"],
    category="alignment",
    description="Download SRA accessions with prefetch and verify integrity via vdb-validate.",
    examples=[...],
    related=["alignment.fqdump", "alignment.fastp", "alignment.STAR", "alignment.featureCount"],
    produces={"uns": ["sra_files"]},
)
```

**`fq_dump.py`** — 依赖prefetch（可选）:
```python
@register_function(
    ...,
    prerequisites={"optional_functions": ["prefetch"]},
    produces={"uns": ["fastq_files"]},
)
```

**`fastp.py`** — 依赖fastq输入:
```python
@register_function(
    ...,
    prerequisites={"functions": ["fqdump"], "optional_functions": ["prefetch"]},
    produces={"uns": ["cleaned_fastq"]},
)
```

**`STAR.py`** — 依赖质控后的fastq:
```python
@register_function(
    ...,
    prerequisites={"functions": ["fastp"], "optional_functions": ["fqdump"]},
    produces={"uns": ["bam_files"]},
)
```

**`featureCount.py`** — 依赖BAM文件:
```python
@register_function(
    ...,
    prerequisites={"functions": ["STAR"]},
    produces={"uns": ["count_matrix"]},
)
```

**注意**: alignment函数操作的是文件系统而非AnnData，所以 `requires`/`produces` 使用 `uns` 键作为语义标记。实际检查依赖registry的 `get_prerequisite_chain()`，不依赖 `check_prerequisites(adata)`。

---

## P3: 扩展WorkflowEscalator支持alignment工作流

**影响**: Agent在遇到复杂FASTQ分析请求时能正确识别并升级为高级工作流
**文件**: `omicverse/utils/inspector/workflow_escalator.py`
**改动量**: ~40行

### 3.1 在 `HIGH_LEVEL_FUNCTIONS` 字典中新增 (第89行之后)
```python
'bulk_rnaseq_pipeline': {
    'replaces': ['prefetch', 'fqdump', 'fastp', 'STAR', 'featureCount'],
    'code_template': (
        "# Bulk RNA-seq pipeline\n"
        "fq = ov.alignment.fqdump(sra_ids, output_dir='fastq')\n"
        "clean = ov.alignment.fastp(samples, output_dir='fastp')\n"
        "bams = ov.alignment.STAR(samples, genome_dir=genome_dir, output_dir='star')\n"
        "counts = ov.alignment.featureCount(bam_items, gtf=gtf, output_dir='counts')"
    ),
    'description': 'Complete bulk RNA-seq alignment pipeline',
    'estimated_time': '10-30 minutes',
    'estimated_time_seconds': 1200,
},
```

### 3.2 在 `COMPLEX_TRIGGERS` 集合中新增
```python
COMPLEX_TRIGGERS = {
    'qc', 'preprocess', 'batch_correct', 'highly_variable_genes',
    'normalize', 'combat', 'harmony', 'scanorama',
    # alignment triggers
    'prefetch', 'fqdump', 'fastp', 'STAR', 'featureCount',
    'align', 'alignment', 'fastq', 'mapping',
}
```

### 3.3 在 `_get_default_code()` 方法中新增模板
```python
'prefetch': "ov.alignment.prefetch(sra_ids, output_dir='prefetch')",
'fqdump': "ov.alignment.fqdump(sra_ids, output_dir='fastq')",
'fastp': "ov.alignment.fastp(samples, output_dir='fastp')",
'STAR': "ov.alignment.STAR(samples, genome_dir='index', output_dir='star')",
'featureCount': "ov.alignment.featureCount(bam_items, gtf='genes.gtf', output_dir='counts')",
```

### 3.4 新增 `_can_use_bulk_rnaseq_pipeline()` 方法
```python
def _can_use_bulk_rnaseq_pipeline(self, missing_prerequisites):
    pipeline_replaces = set(self.HIGH_LEVEL_FUNCTIONS['bulk_rnaseq_pipeline']['replaces'])
    missing_set = set(missing_prerequisites)
    return len(missing_set & pipeline_replaces) >= 2
```

### 3.5 在 `_escalate_to_high_level()` 方法中新增判断分支
在batch_correct检查之后新增alignment pipeline判断。

---

## P4: 创建FASTQ分析skill文件

**影响**: Agent的Priority 2 (Skills Workflow) 路径无法处理FASTQ分析请求
**文件**: 新建 `.claude/skills/fastq-analysis/SKILL.md` 和 `reference.md`
**改动量**: 新建2个文件，约300行

### SKILL.md 结构
```yaml
---
name: fastq-analysis-pipeline
title: FASTQ analysis and RNA-seq alignment with omicverse
description: Guide through omicverse's alignment module for SRA downloading,
  FASTQ quality control, STAR alignment, and gene quantification pipelines.
---
```

Body包含:
1. **Overview** — 介绍 `ov.alignment` 模块的7个核心函数
2. **Instructions** — 分步指导 (SRA下载 → QC → 比对 → 定量 → 单细胞路径)
3. **Critical API Reference** — 关键参数和陷阱
4. **Examples** — bulk和单细胞两条完整pipeline
5. **Troubleshooting** — 常见错误和auto_install机制

### reference.md 结构
- 环境安装 (conda命令)
- 每个函数的copy-paste-ready代码模板
- 完整bulk RNA-seq pipeline代码
- 完整10x单细胞pipeline代码

---

## P5: 统一 `kb_api.py` 代码风格

**影响**: 代码一致性和可维护性；减少重复逻辑
**文件**: `omicverse/alignment/kb_api.py`
**改动量**: ~50行重构

### 5.1 移除独立的 `Colors` 类 (第23-33行)
删除 `Colors` 类，print语句改为普通输出（与其他alignment模块一致）。

### 5.2 将 `_which_kb()` 迁移到使用 `resolve_executable`
### 5.3 将 `_run_kb()` 迁移到使用 `run_cmd`
### 5.4 同理处理 `_which_parallel_fastq_dump()` 和 `_run_parallel_fastq_dump()`

**风险控制**: 这是重构，需要确保功能不变。

---

## P6 (最低优先级): 添加端到端pipeline高级函数

**影响**: 用户体验提升，但当前各函数已可独立使用
**文件**: 新建 `omicverse/alignment/pipeline.py`
**改动量**: ~100行新代码

新增 `bulk_rnaseq_pipeline()` 函数，串联 `prefetch → fqdump → fastp → STAR → featureCount`。

**依赖**: 需要P1-P4先完成。

---

## 总体执行顺序图

```
P1 (分类修复)     ──── 5min  ──── 独立，无依赖
      ↓
P2 (prerequisite) ── 30min ──── 依赖P1
      ↓
P3 (Escalator)   ── 45min ──── 依赖P2
  ↓ (可并行)
P4 (Skill文件)   ── 60min ──── 与P3并行
  ↓
P5 (代码统一)    ── 45min ──── 独立，可与P3/P4并行
      ↓
P6 (Pipeline函数) ── 60min ──── 依赖P1-P4全部完成
```
