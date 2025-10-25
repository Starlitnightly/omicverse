# OmicVerse Enhanced Alignment Pipeline

## 🚀 新功能概述

增强的OmicVerse比对管道现在支持多种输入类型，包括：

1. **SRA数据** - 原有的公共数据库数据
2. **直接FASTQ文件** - 用户自己的FASTQ文件

## ✨ 主要特性

### 🔧 统一输入接口
- 自动检测输入类型
- 支持多种数据源的统一处理
- 灵活的样本ID分配机制

### 🏢 fastq数据输入支持
- 自动发现FASTQ文件
- 智能样本ID提取
- 双端测序自动配对
- 文件完整性验证

### 🔍 增强的工具检查
- 自动检测必需软件
- 提供安装指引
- 支持自动安装

### ⚙️ 灵活的配置系统
- YAML/JSON配置文件
- 多种预设配置模板
- 运行时参数调整

### 🚀 多种下载模式
- **prefetch模式**: 使用NCBI SRA Toolkit (默认)
- **iseq模式**: 使用iseq工具，支持多数据库、Aspera加速、直接下载gzip等高级功能
- **iseq功能优化**: 多数据下载支持以列表形式输入

## 📋 快速开始

### 1. 基础使用

```python

from omicverse.bulk import geo_data_preprocess, fq_data_preprocess


# 支持以SRA Run List txt文本路径输入
result = geo_data_preprocess(input_data ="./srr_list.txt")
                            
# 支持以单个/多个 SRA ID 输入
data_list = ["SRR123456","SRR123457"]
result = geo_data_preprocess( input_data = data_list)
                            
# 支持以fastq数据输入

fastq_files=[
        "./work/fasterq/SRR12544421/SRR12544419_1.fastq.gz",
        "./work/fasterq/SRR12544421/SRR12544419_2.fastq.gz",
        "./work/fasterq/SRR12544421/SRR12544421_1.fastq.gz",
        "./work/fasterq/SRR12544421/SRR12544421_2.fastq.gz",
]
result = fq_data_preprocess( input_data =fastq_files )
```

### 2. 进阶使用

```python
from omicverse.bulk import geo_data_preprocess, fq_data_preprocess,AlignmentConfig

cfg = AlignmentConfig(
    work_root="work",  # 数据分析保存根目录
    threads=64,        # CPU资源
    genome="human",    #数据来源organism
    download_method="prefetch", #下载方式，可选 "iseq"
    memory = "128G",    #内存资源
    fastp_enabled=True, #QC选项
    gzip_fastq=True,    # fastq数据压缩选项
)
result = geo_data_preprocess( input_data = data_list, config =cfg)
result = fq_data_preprocess( input_data = data_list, config =cfg)

```

### 2. 下载模式选择

#### prefetch模式 (默认)
使用NCBI SRA Toolkit进行数据下载，适合大多数情况：

#### iseq模式
使用iseq工具进行数据下载，支持更多高级功能：

```python
# 自定义iseq配置
config = {
    "work_root": "work",
    "download_method": "iseq",
    "iseq_gzip": True,           # 下载gzip格式的FASTQ文件
    "iseq_aspera": True,         # 使用Aspera加速
    "iseq_database": "ena",      # 选择数据库: ena, sra
    "iseq_protocol": "ftp",      # 选择协议: ftp, https
    "iseq_parallel": 8,          # 并行下载数
    "iseq_threads": 16           # 处理线程数
}

result = run_analysis("SRR123456", config=config)
```

#### iseq命令行示例
以下是iseq支持的命令行选项，可以在配置中使用：

```bash
# 基本下载
iseq -i SRR123456

# 批量下载并gzip压缩
iseq -i SRR_Acc_List.txt -g

# 使用Aspera加速下载gzip文件
iseq -i PRJNA211801 -a -g

# 指定数据库和协议
iseq -i SRR123456 -d ena -r ftp -g

# 并行下载
iseq -i accession_list.txt -p 10 -g
```



### 工具参数

```yaml
star_params:
  gencode_release: "v44"
  sjdb_overhang: 149

fastp_params:
  qualified_quality_phred: 20
  length_required: 50

featurecounts_params:
  simple: true
  by: "gene_id"
```

## 📁 输入格式

### SRA数据
- 单个SRR编号：`"SRR123456"`
- 多个SRR编号：`["SRR123456", "SRR789012"]`
- GEO accession：`"GSE123456"`

### FASTQ文件
- 配对文件：`["sample_R1.fastq.gz", "sample_R2.fastq.gz"]`
- 多个配对：`[sample1_R1.fastq.gz", "sample1_R2.fastq.gz", "sample2_R1.fastq.gz", "sample2_R2.fastq.gz"]`

## 🔍 样本ID处理

### 自动提取
系统会自动从文件名提取样本ID：
- `sample1_R1.fastq.gz` → `sample1`
- `Tumor_001_L001_R1_001.fastq.gz` → `Tumor_001`
- `Sample_A_R1.fastq.gz` → `Sample_A`


## 🔧 工具要求

### 必需软件
- **sra-tools**: SRA数据下载
- **STAR**: RNA-seq比对
- **fastp**: 质量控制
- **featureCounts**: 基因定量
- **samtools**: BAM文件处理
- **entrez-direct**: 元数据获取

### 安装检查
```python
from omicverse.bulk._alignment import check_all_tools

# 检查工具
results = check_all_tools()
for tool, (available, path) in results.items():
    print(f"{tool}: {'✅' if available else '❌'}")
```

### 自动安装
```python
# 自动安装缺失工具（需要conda环境）
results = check_all_tools(auto_install=True)
```

## 📊 输出结果

### 结果结构
```python
result = {
    "type": "company",  # 输入类型
    "fastq_input": [(sample_id, fq1_path, fq2_path), ...],
    "fastq_qc": [(sample_id, clean_fq1, clean_fq2), ...],
    "bam": [(sample_id, bam_path, index_dir), ...],
    "counts": {
        "tables": [(sample_id, count_file), ...],
        "matrix": matrix_file_path
    }
}
```

### 文件组织
```
work/
├── meta/                    
│   ├── sample_metadata.csv  # 样本元数据
│   └── sample_id_mapping.json # ID映射
├── prefetch/                    
│   ├── SRRID  # 样本元数据
│       └── SRRID.sra # 原始文件
├── fasterq/                    
│   ├── SRRID  
│       └── SRRID_R1.fastq.gz # 原始文件
│       └── SRRID_R2.fastq.gz # 原始文件
├── index/                    
│   ├── _cache  # 自动检测数据下载对应index
├── fastp/                   # 质控结果
│   ├── Sample_001/
│   │   ├── Sample_001_clean_R1.fastq.gz
│   │   └── Sample_001_clean_R2.fastq.gz
│   └── fastp_reports/
├── star/                    # 比对结果
│   ├── Sample_001/
│   │   ├── Aligned.sortedByCoord.out.bam
│   │   └── Sample_001.sorted.bam
│   └── logs/
└── counts/                  # 定量结果
    ├── Sample_001/
    │   └── Sample_001.counts.txt
    └── matrix.auto.csv      # 合并矩阵
```

## 🚨 错误处理

### 常见错误

1. **文件不存在**
   ```
   Error: File not found: /path/to/sample.fastq.gz
   Solution: 检查文件路径是否正确
   ```

2. **样本ID冲突**
   ```
   Error: Duplicate sample IDs detected
   Solution: 使用不同的sample_prefix或手动指定样本ID
   ```

### 容错处理
- 部分样本失败时继续处理其他样本
- 自动重试机制（可配置）
- 详细的错误日志

## 🔬 高级功能

### 自定义样本ID
```python
# 手动指定样本ID映射
fastq_pairs = [
    ("Patient_001_Tumor", "/path/to/tumor_R1.fastq.gz", "/path/to/tumor_R2.fastq.gz"),
    ("Patient_001_Normal", "/path/to/normal_R1.fastq.gz", "/path/to/normal_R2.fastq.gz")
]

result = pipeline.run_from_fastq(fastq_pairs)
```

### 批量处理
```python
# 多个目录批量处理
data_dirs = ["/path/to/batch1", "/path/to/batch2", "/path/to/batch3"]

for i, data_dir in enumerate(data_dirs):
    result = pipeline.run_pipeline(
        data_dir,
        input_type="company",
        sample_prefix=f"Batch{i+1}"
    )
```

### 质量控制参数
```python
config = AlignmentConfig(
    fastp_params={
        "qualified_quality_phred": 30,  # 提高质量阈值
        "length_required": 75,          # 增加最小长度
        "detect_adapter_for_pe": True   # 自动检测接头
    }
)
```

## 📈 性能优化

### 并行处理
```python
config = AlignmentConfig(
    threads=16,           # 增加线程数
    max_workers=4,        # 并行样本数
    continue_on_error=True # 错误继续
)
```

### 内存管理
```python
config = AlignmentConfig(
    memory="32G",         # 增加内存
    retry_attempts=3,     # 重试次数
    retry_delay=10        # 重试延迟
)
```

## 🔗 相关文件

- `alignment.py` - 核心管道类
- `pipeline_config.py` - 配置管理
- `tools_check.py` - 工具检查

## 📞 支持

如有问题，请：
1. 检查工具是否正确安装
2. 查看日志文件获取详细错误信息
3. 确保输入数据格式正确
4. 参考示例代码

## 📄 更新日志

### v2.0.0
- ✅ 新增直接FASTQ文件支持
- ✅ 新增自动输入类型检测
- ✅ 增强工具检查和安装指引
- ✅ 新增统一配置系统
- ✅ 改进错误处理和日志系统
- ✅ 新增样本ID标准化机制