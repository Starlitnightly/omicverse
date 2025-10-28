# OmicVerse Enhanced Alignment Pipeline

## 🚀 New Feature Overview

The enhanced OmicVerse alignment pipeline now supports multiple input types, including:

1. **SRA data** - Original public repository data
2. **Direct FASTQ files** - User-provided FASTQ files

## ✨ Key Features

### 🔧 Unified Input Interface
- Automatically detect input type
- Unified processing across data sources
- Flexible sample ID assignment mechanism

### 🏢 FASTQ Input Support
- Automatically discover FASTQ files
- Intelligent sample ID extraction
- Automatic pairing for paired-end sequencing
- File integrity validation

### 🔍 Enhanced Tool Checks
- Automatically detect required software
- Provide installation guidance
- Support automatic installation

### ⚙️ Flexible Configuration System
- YAML/JSON configuration files
- Multiple preset configuration templates
- Runtime parameter adjustments

### 🚀 Multiple Download Modes
- **prefetch mode**: Use the NCBI SRA Toolkit (default)
- **iseq mode**: Use the iseq tool with multi-database support, Aspera acceleration, direct gzip downloads, and more
- **iseq enhancements**: Batch downloads accept list-style inputs

## 📋 Quick Start

### 1. Basic Usage

```python

from omicverse.bulk import geo_data_preprocess, fq_data_preprocess


# Accept an SRA Run List text file path
result = geo_data_preprocess(input_data ="./srr_list.txt")

# Accept single or multiple SRA IDs
data_list = ["SRR123456","SRR123457"]
result = geo_data_preprocess( input_data = data_list)

# Accept FASTQ data input

fastq_files=[
        "./work/fasterq/SRR12544421/SRR12544419_1.fastq.gz",
        "./work/fasterq/SRR12544421/SRR12544419_2.fastq.gz",
        "./work/fasterq/SRR12544421/SRR12544421_1.fastq.gz",
        "./work/fasterq/SRR12544421/SRR12544421_2.fastq.gz",
]
result = fq_data_preprocess( input_data =fastq_files )
```

### 2. Advanced Usage

```python
from omicverse.bulk import geo_data_preprocess, fq_data_preprocess,AlignmentConfig

cfg = AlignmentConfig(
    work_root="work",  # Root directory for analysis outputs
    threads=64,        # CPU resources
    genome="human",    # Source organism
    download_method="prefetch", # Download method; set to "iseq" to use iseq
    memory = "128G",    # Memory allocation
    fastp_enabled=True, # QC option
    gzip_fastq=True,    # FASTQ compression option
)
result = geo_data_preprocess( input_data = data_list, config =cfg)
result = fq_data_preprocess( input_data = data_list, config =cfg)

```

### 2. Download Mode Selection

#### prefetch mode (default)
Use the NCBI SRA Toolkit for downloads; suitable for most scenarios.

#### iseq mode
Use the iseq tool for downloads with additional advanced capabilities:

```python
# Custom iseq configuration
config = {
    "work_root": "work",
    "download_method": "iseq",
    "iseq_gzip": True,           # Download FASTQ as gzip
    "iseq_aspera": True,         # Enable Aspera acceleration
    "iseq_database": "ena",      # Select database: ena or sra
    "iseq_protocol": "ftp",      # Select protocol: ftp or https
    "iseq_parallel": 8,          # Parallel download count
    "iseq_threads": 16           # Processing threads
}

result = run_analysis("SRR123456", config=config)
```

#### iseq command-line examples
Command-line options supported by iseq that you can mirror in configuration:

```bash
# Basic download
iseq -i SRR123456

# Batch download with gzip compression
iseq -i SRR_Acc_List.txt -g

# Download gzip files with Aspera acceleration
iseq -i PRJNA211801 -a -g

# Specify database and protocol
iseq -i SRR123456 -d ena -r ftp -g

# Parallel downloads
iseq -i accession_list.txt -p 10 -g
```



### Tool Parameters

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

## 📁 Input Formats

### SRA Data
- Single SRR accession: "SRR123456"
- Multiple SRR accessions: ["SRR123456", "SRR789012"]
- GEO accession: "GSE123456"

### FASTQ Files
- Paired files: ["sample_R1.fastq.gz", "sample_R2.fastq.gz"]
- Multiple pairs: ["sample1_R1.fastq.gz", "sample1_R2.fastq.gz", "sample2_R1.fastq.gz", "sample2_R2.fastq.gz"]

## 🔍 Sample ID Handling

### Automatic Extraction
The system automatically derives sample IDs from file names:
- `sample1_R1.fastq.gz` -> `sample1`
- `Tumor_001_L001_R1_001.fastq.gz` -> `Tumor_001`
- `Sample_A_R1.fastq.gz` -> `Sample_A`


## 🔧 Tool Requirements

### Required software
- **sra-tools**: SRA data downloads
- **STAR**: RNA-seq alignment
- **fastp**: Quality control
- **featureCounts**: Gene quantification
- **samtools**: BAM file handling
- **entrez-direct**: Metadata retrieval

### Installation check
```python
from omicverse.bulk._alignment import check_all_tools

# Check tools
results = check_all_tools()
for tool, (available, path) in results.items():
    print(f"{tool}: {'✅' if available else '❌'}")
```

### Automatic installation
```python
# Automatically install missing tools (requires a conda environment)
results = check_all_tools(auto_install=True)
```

## 📊 Output Results

### Result structure
```python
result = {
    "type": "company",  # Input type
    "fastq_input": [(sample_id, fq1_path, fq2_path), ...],
    "fastq_qc": [(sample_id, clean_fq1, clean_fq2), ...],
    "bam": [(sample_id, bam_path, index_dir), ...],
    "counts": {
        "tables": [(sample_id, count_file), ...],
        "matrix": matrix_file_path
    }
}
```

### File organization
```
work/
├── meta/                    
│   ├── sample_metadata.csv  # Sample metadata
│   └── sample_id_mapping.json # ID mapping
├── prefetch/                    
│   ├── SRRID  # Sample metadata
│       └── SRRID.sra # Raw files
├── fasterq/                    
│   ├── SRRID  
│       └── SRRID_R1.fastq.gz # Raw files
│       └── SRRID_R2.fastq.gz # Raw files
├── index/                    
│   ├── _cache  # Automatically detected indices for downloaded data
├── fastp/                   # QC results
│   ├── Sample_001/
│   │   ├── Sample_001_clean_R1.fastq.gz
│   │   └── Sample_001_clean_R2.fastq.gz
│   └── fastp_reports/
├── star/                    # Alignment results
│   ├── Sample_001/
│   │   ├── Aligned.sortedByCoord.out.bam
│   │   └── Sample_001.sorted.bam
│   └── logs/
└── counts/                  # Quantification results
    ├── Sample_001/
    │   └── Sample_001.counts.txt
    └── matrix.auto.csv      # Combined matrix
```

## 🚨 Error Handling

### Common Errors

1. **File not found**
   ```
   Error: File not found: /path/to/sample.fastq.gz
   Solution: Verify the file path is correct
   ```

2. **Sample ID conflict**
   ```
   Error: Duplicate sample IDs detected
   Solution: Use a different sample_prefix or specify sample IDs manually
   ```

### Fault tolerance
- Continue processing remaining samples when some fail
- Configurable automatic retry mechanism
- Detailed error logs

## 🔬 Advanced Features

### Custom sample IDs
```python
# Manually specify sample ID mapping
fastq_pairs = [
    ("Patient_001_Tumor", "/path/to/tumor_R1.fastq.gz", "/path/to/tumor_R2.fastq.gz"),
    ("Patient_001_Normal", "/path/to/normal_R1.fastq.gz", "/path/to/normal_R2.fastq.gz")
]

result = pipeline.run_from_fastq(fastq_pairs)
```

### Batch processing
```python
# Batch process multiple directories
data_dirs = ["/path/to/batch1", "/path/to/batch2", "/path/to/batch3"]

for i, data_dir in enumerate(data_dirs):
    result = pipeline.run_pipeline(
        data_dir,
        input_type="company",
        sample_prefix=f"Batch{i+1}"
    )
```

### Quality control parameters
```python
config = AlignmentConfig(
    fastp_params={
        "qualified_quality_phred": 30,  # Increase quality threshold
        "length_required": 75,          # Increase minimum length
        "detect_adapter_for_pe": True   # Auto-detect adapters
    }
)
```

## 📈 Performance Optimization

### Parallel processing
```python
config = AlignmentConfig(
    threads=16,           # Increase thread count
    max_workers=4,        # Number of parallel samples
    continue_on_error=True # Continue after errors
)
```

### Memory management
```python
config = AlignmentConfig(
    memory="32G",         # Increase memory
    retry_attempts=3,     # Retry attempts
    retry_delay=10        # Retry delay
)
```

## 🔗 Related Files

- `alignment.py` - Core pipeline class
- `pipeline_config.py` - Configuration management
- `tools_check.py` - Tool checks

## 📞 Support

If issues arise:
1. Check that tools are installed correctly
2. Review log files for detailed error information
3. Ensure input data formats are correct
4. Refer to the sample code

## 📄 Changelog

### v2.0.0
- ✅ Added direct FASTQ file support
- ✅ Added automatic input type detection
- ✅ Enhanced tool checks and installation guidance
- ✅ Added unified configuration system
- ✅ Improved error handling and logging
- ✅ Added sample ID standardization mechanism
