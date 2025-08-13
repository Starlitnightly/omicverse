# DataCollect API Reference for OmicVerse

Complete API reference for the DataCollect external module in OmicVerse.

## Table of Contents
1. [Main Collection Functions](#main-collection-functions)
2. [API Clients](#api-clients)
3. [Format Converters](#format-converters)
4. [Collectors](#collectors)
5. [Utilities](#utilities)

## Main Collection Functions

### `collect_protein_data(identifier, source='uniprot', to_format='pandas', **kwargs)`

Collect protein data from specified source and convert to desired format.

**Parameters:**
- `identifier` (str): Protein identifier (UniProt ID, gene name, PDB ID, etc.)
- `source` (str, default='uniprot'): Data source
  - `'uniprot'`: UniProt protein database
  - `'pdb'`: Protein Data Bank structures
  - `'alphafold'`: AlphaFold predicted structures
  - `'interpro'`: InterPro domains and families
  - `'string'`: STRING protein interactions
  - `'emdb'`: Electron Microscopy Data Bank
- `to_format` (str, default='pandas'): Output format
  - `'pandas'`: pandas DataFrame
  - `'anndata'`: AnnData object
  - `'mudata'`: MuData object
  - `'dict'`: Raw dictionary
- `**kwargs`: Additional parameters for specific clients

**Returns:**
- Data in specified format

**Examples:**
```python
import omicverse as ov

# Basic protein data collection
protein_df = ov.external.datacollect.collect_protein_data("P04637")

# AlphaFold structure data
structure_data = ov.external.datacollect.collect_protein_data(
    "P04637", 
    source="alphafold", 
    to_format="pandas"
)

# Protein interactions
interactions = ov.external.datacollect.collect_protein_data(
    "TP53", 
    source="string", 
    include_scores=True
)
```

### `collect_expression_data(identifier, source='geo', to_format='anndata', **kwargs)`

Collect gene expression data and convert to AnnData format (OmicVerse standard).

**Parameters:**
- `identifier` (str): Dataset identifier (GEO accession, etc.)
- `source` (str, default='geo'): Data source
  - `'geo'`: Gene Expression Omnibus
  - `'ccre'`: Candidate cis-Regulatory Elements
- `to_format` (str, default='anndata'): Output format
  - `'anndata'`: AnnData object (recommended for OmicVerse)
  - `'pandas'`: pandas DataFrame
  - `'mudata'`: MuData object
  - `'dict'`: Raw dictionary
- `**kwargs`: Additional parameters

**Returns:**
- Expression data in specified format (AnnData by default)

**Examples:**
```python
# Collect GEO dataset as AnnData (OmicVerse standard)
adata = ov.external.datacollect.collect_expression_data("GSE123456")

# Collect as pandas DataFrame
expression_df = ov.external.datacollect.collect_expression_data(
    "GSE123456", 
    to_format="pandas"
)

# Regulatory elements data
regulatory_data = ov.external.datacollect.collect_expression_data(
    "chr1:1000000-2000000", 
    source="ccre"
)
```

### `collect_pathway_data(identifier, source='kegg', to_format='pandas', **kwargs)`

Collect pathway data from specified source.

**Parameters:**
- `identifier` (str): Pathway identifier
- `source` (str, default='kegg'): Data source
  - `'kegg'`: KEGG pathways
  - `'reactome'`: Reactome pathways
  - `'gtopdb'`: Guide to Pharmacology
- `to_format` (str, default='pandas'): Output format
- `**kwargs`: Additional parameters

**Returns:**
- Pathway data in specified format

**Examples:**
```python
# KEGG pathway data
pathway_df = ov.external.datacollect.collect_pathway_data("hsa04110")

# Reactome pathway
reactome_data = ov.external.datacollect.collect_pathway_data(
    "R-HSA-68886", 
    source="reactome"
)
```

## API Clients

### Protein APIs

#### `UniProtClient`

Client for UniProt REST API.

```python
from omicverse.external.datacollect.api.proteins import UniProtClient

client = UniProtClient()

# Get protein data
protein_data = client.get_protein("P04637")

# Search proteins
results = client.search("kinase AND organism_id:9606", limit=50)

# Get protein features
features = client.get_features("P04637")
```

**Methods:**
- `get_protein(accession)`: Get single protein data
- `search(query, limit=100)`: Search proteins
- `get_features(accession)`: Get protein features
- `get_go_terms(accession)`: Get GO annotations

#### `PDBClient`

Client for Protein Data Bank API.

```python
from omicverse.external.datacollect.api.proteins import PDBClient

client = PDBClient()

# Get structure data
structure_data = client.get_structure("1TUP")

# Download structure file
structure_file = client.download_structure("1TUP", format="pdb")

# Search structures
results = client.search("p53", limit=10)
```

#### `AlphaFoldClient`

Client for AlphaFold Database API.

```python
from omicverse.external.datacollect.api.proteins import AlphaFoldClient

client = AlphaFoldClient()

# Get predicted structure
prediction = client.get_prediction("P04637")

# Download prediction file
pred_file = client.download_prediction("P04637", format="pdb")
```

### Genomics APIs

#### `EnsemblClient`

Client for Ensembl REST API.

```python
from omicverse.external.datacollect.api.genomics import EnsemblClient

client = EnsemblClient()

# Get gene information
gene_data = client.get_gene("ENSG00000141510")

# Get gene variants
variants = client.get_variants("ENSG00000141510")

# Sequence lookup
sequence = client.get_sequence("ENSG00000141510")
```

#### `ClinVarClient`

Client for ClinVar API.

```python
from omicverse.external.datacollect.api.genomics import ClinVarClient

client = ClinVarClient()

# Get clinical variants
variants = client.get_variants("BRCA1")

# Get variant by ID
variant_data = client.get_variant("VCV000001234")
```

### Expression APIs

#### `GEOClient`

Client for Gene Expression Omnibus API.

```python
from omicverse.external.datacollect.api.expression import GEOClient

client = GEOClient()

# Get dataset information
dataset_info = client.get_dataset_info("GSE123456")

# Download expression data
expression_data = client.get_expression_data("GSE123456")

# Get sample metadata
sample_metadata = client.get_sample_metadata("GSE123456")
```

#### `OpenTargetsClient`

Client for Open Targets API.

```python
from omicverse.external.datacollect.api.expression import OpenTargetsClient

client = OpenTargetsClient()

# Get target information
target_data = client.get_target("ENSG00000141510")

# Get drug associations
drug_associations = client.get_drug_associations("ENSG00000141510")

# Disease associations
disease_associations = client.get_disease_associations("ENSG00000141510")
```

### Pathway APIs

#### `KEGGClient`

Client for KEGG API.

```python
from omicverse.external.datacollect.api.pathways import KEGGClient

client = KEGGClient()

# Get pathway information
pathway_data = client.get_pathway("hsa04110")

# Get pathway genes
genes = client.get_pathway_genes("hsa04110")

# Find pathways
pathways = client.find_pathways("cell cycle")
```

#### `ReactomeClient`

Client for Reactome API.

```python
from omicverse.external.datacollect.api.pathways import ReactomeClient

client = ReactomeClient()

# Get pathway data
pathway_data = client.get_pathway("R-HSA-68886")

# Get pathway hierarchy
hierarchy = client.get_pathway_hierarchy("R-HSA-68886")
```

### Specialized APIs

#### `BLASTClient`

Client for BLAST sequence similarity searches.

```python
from omicverse.external.datacollect.api.specialized import BLASTClient

client = BLASTClient()

# BLAST search
results = client.blast_search(
    sequence="ATCGATCGATCG",
    program="blastn",
    database="nt"
)

# Protein BLAST
protein_results = client.blast_search(
    sequence="MVLSPADKTNVKAAW",
    program="blastp",
    database="nr"
)
```

## Format Converters

### `to_pandas(data, data_type='generic')`

Convert collected data to pandas DataFrame.

**Parameters:**
- `data` (dict): Raw data from API client
- `data_type` (str): Type of data for specialized conversion
  - `'protein'`: Protein data conversion
  - `'gene'`: Gene data conversion
  - `'expression'`: Expression data conversion
  - `'pathway'`: Pathway data conversion
  - `'variant'`: Variant data conversion
  - `'generic'`: Generic conversion

**Returns:**
- pandas DataFrame

**Example:**
```python
from omicverse.external.datacollect.utils.omicverse_adapters import to_pandas

# Convert protein data
protein_df = to_pandas(raw_protein_data, "protein")

# Convert gene data
gene_df = to_pandas(raw_gene_data, "gene")
```

### `to_anndata(data, obs_keys=None, var_keys=None)`

Convert collected data to AnnData format for OmicVerse compatibility.

**Parameters:**
- `data` (dict): Raw data from API client
- `obs_keys` (list, optional): Keys to use for observation metadata
- `var_keys` (list, optional): Keys to use for variable metadata

**Returns:**
- AnnData object

**Example:**
```python
from omicverse.external.datacollect.utils.omicverse_adapters import to_anndata

# Convert expression data to AnnData
adata = to_anndata(
    expression_data,
    obs_keys=['sample_id', 'condition', 'tissue'],
    var_keys=['gene_symbol', 'gene_id']
)
```

### `to_mudata(data)`

Convert collected data to MuData format for multi-omics analysis.

**Parameters:**
- `data` (dict): Raw data containing multiple modalities

**Returns:**
- MuData object

**Example:**
```python
from omicverse.external.datacollect.utils.omicverse_adapters import to_mudata

# Convert multi-omics data
mudata_obj = to_mudata({
    'rna': expression_data,
    'protein': protein_data,
    'pathway': pathway_data
})
```

## Collectors

### `BaseCollector`

Base class for all data collectors.

**Methods:**
- `collect_single(identifier, **kwargs)`: Collect data for one entity
- `collect_batch(identifiers, **kwargs)`: Collect data for multiple entities
- `save_to_database(data)`: Save collected data to database
- `process_and_save(identifier, **kwargs)`: Collect and save in one step

### `UniProtCollector`

Collector for UniProt protein data.

```python
from omicverse.external.datacollect.collectors import UniProtCollector

collector = UniProtCollector()

# Collect single protein
protein_data = collector.collect_single("P04637")

# Batch collection
proteins = collector.collect_batch(["P04637", "P21359", "P53_HUMAN"])

# Save to database
collector.save_to_database(protein_data)
```

### `GEOCollector`

Collector for GEO expression data.

```python
from omicverse.external.datacollect.collectors import GEOCollector

collector = GEOCollector()

# Collect expression dataset
expression_data = collector.collect_single("GSE123456")

# Process and save
collector.process_and_save("GSE123456", include_metadata=True)
```

### `BatchCollector`

Utility for batch data collection across multiple sources.

```python
from omicverse.external.datacollect.collectors import BatchCollector

collector = BatchCollector()

# Batch protein collection
proteins = collector.collect_proteins(
    ["P04637", "P21359"], 
    include_features=True,
    format="pandas"
)

# Batch expression collection
datasets = collector.collect_expression_data(
    ["GSE123456", "GSE789012"],
    format="anndata_dict"
)
```

## Utilities

### Data Validation

```python
from omicverse.external.datacollect.utils.validation import SequenceValidator, IdentifierValidator

# Validate protein sequence
is_valid = SequenceValidator.is_valid_protein_sequence("MVLSPADKTNVKAAW")

# Validate identifiers
is_valid_uniprot = IdentifierValidator.validate("P04637", "uniprot")
is_valid_pdb = IdentifierValidator.validate("1A3N", "pdb")

# Detect identifier type
id_type = IdentifierValidator.detect_type("NM_000546")  # Returns "refseq"
```

### Data Transformation

```python
from omicverse.external.datacollect.utils.transformers import SequenceTransformer, DataNormalizer

# Translate DNA to protein
protein_seq = SequenceTransformer.translate_dna_to_protein("ATGGCG...")

# Clean sequences
clean_seq = DataNormalizer.clean_sequence("ACGT-123", "dna")  # Returns "ACGT"

# Normalize gene names
normalized = DataNormalizer.normalize_gene_name("tp53")  # Returns "TP53"
```

### Configuration

```python
from omicverse.external.datacollect.config.config import settings

# Access API settings
uniprot_url = settings.api.uniprot_base_url
rate_limit = settings.api.default_rate_limit

# Database settings
db_url = settings.database_url
```

## Error Handling

All API clients and collectors implement comprehensive error handling:

```python
try:
    data = ov.external.datacollect.collect_protein_data("P04637")
except APIError as e:
    print(f"API error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration Options

### Environment Variables

```bash
# API Keys
UNIPROT_API_KEY=your_key
NCBI_API_KEY=your_key
EBI_API_KEY=your_key

# Rate Limits
DEFAULT_RATE_LIMIT=10
UNIPROT_RATE_LIMIT=10
GEO_RATE_LIMIT=3

# Database
DATABASE_URL=sqlite:///datacollect.db

# Cache
CACHE_DIR=./data/cache
ENABLE_CACHE=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=datacollect.log
```

### Configuration File

Create `~/.omicverse/datacollect.yaml`:

```yaml
api:
  uniprot:
    base_url: "https://rest.uniprot.org"
    rate_limit: 10
    timeout: 30
    
  geo:
    cache_dir: "./data/cache"
    max_retries: 3

formats:
  default_expression_format: "anndata"
  default_protein_format: "pandas"
  default_pathway_format: "pandas"

omicverse:
  auto_convert_formats: true
  cache_converted_data: true
```

## Examples Integration with OmicVerse

### Complete Analysis Workflow

```python
import omicverse as ov

# 1. Collect expression data
adata = ov.external.datacollect.collect_expression_data("GSE123456", format="anndata")

# 2. Collect pathway data
pathway_data = ov.external.datacollect.collect_pathway_data("hsa04110", format="pandas")

# 3. Collect protein information
proteins = ov.external.datacollect.collect_protein_data(["P04637", "P21359"], format="pandas")

# 4. OmicVerse analysis
ov.pp.preprocess(adata)
deg_results = ov.bulk.pyDEG(adata)

# 5. Integration analysis
# Use collected data in downstream analysis
```

This API reference provides comprehensive documentation for using DataCollect within the OmicVerse ecosystem, enabling seamless data collection and analysis workflows.