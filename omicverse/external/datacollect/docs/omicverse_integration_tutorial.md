# DataCollect Integration Tutorial for OmicVerse

This tutorial shows how to use the DataCollect module within the OmicVerse ecosystem.

## Installation

The DataCollect module is included with OmicVerse as an external module:

```python
import omicverse as ov
```

## Quick Start

### Collecting Protein Data

```python
import omicverse as ov

# Collect protein data
protein_data = ov.datacollect.collect_protein_data("P04637")  # p53

# Convert to pandas DataFrame  
protein_df = ov.datacollect.to_pandas(protein_data, "protein")
print(protein_df.head())
```

### Collecting Expression Data

```python
# Collect GEO expression data
expression_data = ov.datacollect.collect_expression_data("GSE123456")

# Convert to AnnData for OmicVerse analysis
adata = ov.datacollect.to_anndata(expression_data)

# Use with OmicVerse workflows
ov.bulk.pyDEG(adata)
```

### Multi-Database Collection

```python
# Collect from multiple sources
protein_data = ov.datacollect.collect_protein_data("TP53")
pathway_data = ov.datacollect.collect_pathway_data("hsa04110")
expression_data = ov.datacollect.collect_expression_data("GSE123456")

# Integrate for multi-omics analysis
multi_data = {
    "protein": protein_data,
    "pathway": pathway_data,
    "expression": expression_data
}

mudata_obj = ov.datacollect.to_mudata(multi_data)
```

## Available APIs

DataCollect provides access to 29+ biological databases:

### Protein & Structure APIs
- UniProt: Protein sequences and annotations
- PDB: Protein structures
- AlphaFold: Structure predictions
- InterPro: Protein domains
- STRING: Protein interactions

### Genomics & Variant APIs  
- Ensembl: Gene information
- ClinVar: Clinical variants
- dbSNP: SNP data
- gnomAD: Population genetics
- UCSC: Genome browser data

### Expression & Regulatory APIs
- GEO: Gene expression datasets
- OpenTargets: Drug targets
- ReMap: Transcription factor binding
- ENCODE: Regulatory elements

### Pathway APIs
- KEGG: Biological pathways
- Reactome: Pathway analysis
- Guide to Pharmacology: Drug targets

## Advanced Usage

### Custom Data Processing

```python
from omicverse.external.datacollect.api.proteins import UniProtClient
from omicverse.external.datacollect.collectors import UniProtCollector

# Direct API usage
client = UniProtClient()
raw_data = client.get_protein_info("P04637")

# Advanced collection with custom processing
collector = UniProtCollector()
processed_data = collector.process_and_save("P04637", save_to_db=True)
```

### Batch Processing

```python
# Collect multiple proteins
protein_ids = ["P04637", "P53_HUMAN", "P21359"]
results = []

for protein_id in protein_ids:
    data = ov.datacollect.collect_protein_data(protein_id)
    results.append(data)

# Convert to combined DataFrame
import pandas as pd
protein_df = pd.concat([ov.datacollect.to_pandas(r, "protein") for r in results])
```

### Integration with OmicVerse Workflows

```python
# Complete workflow example
# 1. Collect expression data
adata = ov.datacollect.to_anndata(
    ov.datacollect.collect_expression_data("GSE123456")
)

# 2. Quality control
ov.pp.qc(adata)

# 3. Differential expression
ov.bulk.pyDEG(adata, condition='treatment')

# 4. Pathway analysis using collected pathway data
pathway_info = ov.datacollect.collect_pathway_data("hsa04110")
# Use pathway info for enrichment analysis...
```

## Configuration

Configure API keys and settings:

```python
from omicverse.external.datacollect.config import settings

# Set API keys
settings.ncbi_api_key = "your_ncbi_key_here"
settings.iucn_api_key = "your_iucn_key_here"

# Configure rate limits
settings.api_rate_limits = {
    "uniprot": 10,    # requests per second
    "ncbi": 3,
    "ensembl": 15
}
```

## Error Handling

DataCollect includes robust error handling:

```python
try:
    data = ov.datacollect.collect_protein_data("INVALID_ID")
except Exception as e:
    print(f"Collection failed: {e}")
    # Automatic retry logic will have already attempted recovery
```

## Data Validation

```python
from omicverse.external.datacollect.utils.validation import (
    SequenceValidator, IdentifierValidator
)

# Validate sequences
is_valid = SequenceValidator.is_valid_protein_sequence("MVLSPADKTNVKAAW")

# Validate identifiers
is_valid_uniprot = IdentifierValidator.validate("P04637", "uniprot")
```

## Support and Documentation

- Full API documentation: See `docs/` directory
- Issue reporting: GitHub issues
- Community support: OmicVerse community channels
