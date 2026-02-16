## Environment setup

```bash
# Install bioinformatics tools (auto_install=True handles this, but manual setup is faster)
conda install -c bioconda -y sra-tools fastp star subread samtools
conda install -c conda-forge -y pigz

# For single-cell kb-python path
pip install kb-python
```

## Complete bulk RNA-seq pipeline

```python
import omicverse as ov

# ── Configuration ──
sra_ids = ['SRR1234567', 'SRR1234568']
genome_fasta = 'reference/genome.fa'
gtf_file = 'reference/genes.gtf'
genome_index_dir = 'reference/star_index'

# ── Step 1: Download SRA data ──
pre = ov.alignment.prefetch(sra_ids, output_dir='prefetch', jobs=4)
fq = ov.alignment.fqdump(sra_ids, output_dir='fastq', sra_dir='prefetch',
                          gzip=True, threads=8, jobs=4)

# ── Step 2: Build sample tuples from fqdump output ──
fq_list = fq if isinstance(fq, list) else [fq]
fastp_samples = [
    (r['srr'], r['fq1'], r['fq2'] if r['fq2'] else None)
    for r in fq_list
]

# ── Step 3: Quality control with fastp ──
clean = ov.alignment.fastp(fastp_samples, output_dir='fastp', threads=8, jobs=2)

# ── Step 4: Build STAR samples from fastp output ──
clean_list = clean if isinstance(clean, list) else [clean]
star_samples = [
    (r['sample'], r['clean1'], r['clean2'] if r['clean2'] else None)
    for r in clean_list
]

# ── Step 5: STAR alignment (auto-builds index if missing) ──
bams = ov.alignment.STAR(
    star_samples,
    genome_dir=genome_index_dir,
    output_dir='star_out',
    gtf=gtf_file,
    genome_fasta_files=[genome_fasta],
    auto_index=True,
    threads=8,
    memory='50G',
    jobs=2,
)

# ── Step 6: Gene quantification with featureCounts ──
bam_list = bams if isinstance(bams, list) else [bams]
bam_items = [
    (r['sample'], r['bam'])
    for r in bam_list if 'bam' in r
]

count_matrix = ov.alignment.featureCount(
    bam_items,
    gtf=gtf_file,
    output_dir='counts',
    gene_mapping=True,
    merge_matrix=True,
    threads=8,
)

# count_matrix is a pandas DataFrame (gene_name x samples)
print(count_matrix.shape)
print(count_matrix.head())
```

## Starting from local FASTQ files (skip download)

```python
import omicverse as ov

# Define samples directly
samples = [
    ('Control_1', 'data/ctrl1_R1.fq.gz', 'data/ctrl1_R2.fq.gz'),
    ('Control_2', 'data/ctrl2_R1.fq.gz', 'data/ctrl2_R2.fq.gz'),
    ('Treated_1', 'data/treat1_R1.fq.gz', 'data/treat1_R2.fq.gz'),
    ('Treated_2', 'data/treat2_R1.fq.gz', 'data/treat2_R2.fq.gz'),
]

# QC
clean = ov.alignment.fastp(samples, output_dir='fastp', threads=8, jobs=4)

# Build STAR input
clean_list = clean if isinstance(clean, list) else [clean]
star_samples = [
    (r['sample'], r['clean1'], r['clean2'] if r['clean2'] else None)
    for r in clean_list
]

# Align
bams = ov.alignment.STAR(
    star_samples,
    genome_dir='star_index',
    output_dir='star_out',
    gtf='genes.gtf',
    genome_fasta_files=['genome.fa'],
    threads=8,
)

# Quantify
bam_list = bams if isinstance(bams, list) else [bams]
bam_items = [(r['sample'], r['bam']) for r in bam_list if 'bam' in r]

counts = ov.alignment.featureCount(
    bam_items, gtf='genes.gtf', output_dir='counts',
    gene_mapping=True, merge_matrix=True,
)
```

## Single-cell 10x v3 pipeline (kb-python)

```python
import omicverse as ov

# Build kallisto index
ref_result = ov.alignment.ref(
    index_path='kb_ref/index.idx',
    t2g_path='kb_ref/t2g.txt',
    fasta_paths=['genome.fa'],
    gtf_paths=['genes.gtf'],
    threads=8,
)

# Quantify 10x Chromium v3 data
count_result = ov.alignment.count(
    index_path='kb_ref/index.idx',
    t2g_path='kb_ref/t2g.txt',
    technology='10XV3',
    fastq_paths=['sample_R1.fastq.gz', 'sample_R2.fastq.gz'],
    output_path='kb_out',
    h5ad=True,
    filter_barcodes=True,
    threads=8,
)

# Load result into AnnData
import scanpy as sc
adata = sc.read_h5ad('kb_out/adata.h5ad')
```

## Single-cell one-click analysis

```python
import omicverse as ov

# One-click 10x v3 analysis (downloads human reference automatically)
result = ov.alignment.single.analyze_10x_v3_data(
    fastq_files=['sample_R1.fastq.gz', 'sample_R2.fastq.gz'],
    reference_output_dir='reference',
    analysis_output_dir='analysis',
    threads_ref=8,
    threads_count=2,
    h5ad=True,
    filter_barcodes=True,
)
```

## Key function signatures

```python
# prefetch
ov.alignment.prefetch(sra_ids, output_dir='prefetch', threads=4, jobs=None,
                       retries=2, validate=True, transport=None, location=None,
                       link_mode='symlink', auto_install=True)

# fqdump
ov.alignment.fqdump(sra_ids, output_dir='fastq', threads=8, memory='4G',
                     gzip=False, library_layout='auto', jobs=None,
                     retries=2, sra_dir=None, auto_install=True, force=False)

# fastp
ov.alignment.fastp(samples, output_dir='fastp', threads=8, jobs=None,
                    output_gzip=None, extra_args=None, auto_install=True,
                    overwrite=False)

# STAR
ov.alignment.STAR(samples, genome_dir, output_dir='star', threads=8,
                   memory='50G', jobs=None, gtf=None, sjdb_overhang=None,
                   genome_fasta_files=None, auto_index=True, strict=False,
                   extra_args=None, auto_install=True, overwrite=False)

# featureCount
ov.alignment.featureCount(bam_items, gtf, output_dir='counts', threads=8,
                           jobs=None, simple=True, merge_matrix=True,
                           gene_mapping=False, gene_map=None,
                           overwrite=False, auto_install=True,
                           strict=False, auto_fix=True)

# ref (kallisto index)
ov.alignment.ref(index_path, t2g_path, fasta_paths=None, gtf_paths=None,
                  workflow='standard', threads=8, overwrite=False)

# count (single-cell quantification)
ov.alignment.count(index_path, t2g_path, technology, fastq_paths,
                    output_path='.', threads=8, workflow='standard',
                    h5ad=False, filter_barcodes=False, cellranger=False)
```
