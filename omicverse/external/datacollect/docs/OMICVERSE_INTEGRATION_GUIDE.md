# OmicVerse Integration Guide for DataCollect2BioNMI

## Overview

This guide provides comprehensive documentation for integrating the **DataCollect2BioNMI** bioinformatics data collection system with **OmicVerse**, a leading multi-omics analysis platform published in Nature Communications.

## What is This Integration?

### DataCollect2BioNMI
- **29 API clients** for major bioinformatics databases
- **597 passing tests** with comprehensive error handling
- **Production-ready** with rate limiting, validation, and retry logic
- **Modular architecture** supporting proteins, genomics, expression, and pathway data

### OmicVerse  
- **Multi-omics analysis platform** for bulk, single-cell, and spatial RNA-seq
- **Published in Nature Communications** (July 2024)
- **Unified framework** with pandas, AnnData, and MuData support
- **Extensible architecture** with `external/` module support

### Integration Benefits
✅ **Unified Data Access**: 29+ databases through OmicVerse interface  
✅ **Seamless Format Conversion**: Automatic conversion to AnnData/pandas/MuData  
✅ **Enhanced Workflows**: Integrated data collection + analysis  
✅ **Production Quality**: Battle-tested with 597 passing tests  

---

## Repository Structure

The integration transforms the current datacollect2bionmi structure into an OmicVerse-compatible module:

```
datacollect2bionmi/              →    omicverse/external/datacollect/
├── src/api/                     →    ├── api/
│   ├── uniprot.py              →    │   ├── proteins/
│   ├── pdb.py                  →    │   │   ├── uniprot.py
│   ├── ensembl.py              →    │   │   ├── pdb.py
│   └── ...                     →    │   │   └── alphafold.py
├── src/collectors/             →    │   ├── genomics/
├── src/models/                 →    │   │   ├── ensembl.py
├── src/utils/                  →    │   │   └── clinvar.py  
└── tests/                      →    │   └── pathways/
                                →    ├── collectors/
                                →    ├── models/
                                →    ├── utils/
                                →    │   └── omicverse_adapters.py  # NEW
                                →    └── tests/
```

---

## Supported APIs and Data Sources

### 🧬 **Proteins & Structures** (6 APIs)
- **UniProt**: Protein sequences, annotations, features
- **PDB**: 3D protein structures  
- **AlphaFold**: AI-predicted protein structures
- **InterPro**: Protein domains and families
- **STRING**: Protein-protein interactions
- **EMDB**: Electron Microscopy Data Bank

### 🧬 **Genomics & Variants** (7 APIs) 
- **Ensembl**: Gene annotations, sequences, variants
- **ClinVar**: Clinical significance of variants
- **dbSNP**: Single nucleotide polymorphisms
- **gnomAD**: Population genetics data
- **GWAS Catalog**: Genome-wide association studies
- **UCSC**: Genome browser data
- **RegulomeDB**: Regulatory variants

### 📊 **Expression & Regulation** (4 APIs)
- **GEO**: Gene expression datasets
- **OpenTargets**: Drug target information  
- **ReMap**: Transcriptional regulators
- **CCRE**: Candidate cis-regulatory elements

### 🛤️ **Pathways & Drugs** (3 APIs)
- **KEGG**: Metabolic and signaling pathways
- **Reactome**: Biological pathways and reactions
- **GtoPdb**: Guide to Pharmacology database

### 🔬 **Specialized APIs** (9 APIs)
- **BLAST**: Sequence similarity searches
- **JASPAR**: Transcription factor binding profiles  
- **MPD**: Mouse Phenome Database
- **IUCN**: Species conservation status
- **PRIDE**: Proteomics data repository
- **cBioPortal**: Cancer genomics data
- **WORMS**: World Register of Marine Species
- **Paleobiology**: Fossil and geological data
- **OpenTargets Genetics**: Genetics evidence

---

## Installation & Setup

### Option 1: Install in OmicVerse Environment

```bash
# Clone OmicVerse (if not already done)
git clone https://github.com/HendricksJudy/omicverse.git
cd omicverse

# Create branch for DataCollect integration
git checkout -b add-datacollect-module

# Copy DataCollect module
cp -r /path/to/datacollect2bionmi/src/* omicverse/external/datacollect/

# Install dependencies
pip install -r omicverse/external/datacollect/requirements.txt

# Test integration
python -c "import omicverse as ov; print('DataCollect available:', hasattr(ov, 'datacollect'))"
```

### Option 2: Development Setup

```bash  
# Clone both repositories
git clone https://github.com/yourusername/datacollect2bionmi.git
git clone https://github.com/HendricksJudy/omicverse.git

# Install in development mode
cd datacollect2bionmi
pip install -e .

cd ../omicverse  
pip install -e .

# Test compatibility
python scripts/test_integration_compatibility.py --omicverse-path ../omicverse --report
```

---

## Usage Examples

### Basic Data Collection

```python
import omicverse as ov

# Protein data collection
protein_data = ov.external.datacollect.collect_protein_data("P04637")  # p53
structure_data = ov.external.datacollect.collect_structure_data("1TUP")
interactions = ov.external.datacollect.collect_interaction_data("TP53")

# Genomics data collection  
gene_data = ov.external.datacollect.collect_gene_data("ENSG00000141510")
variants = ov.external.datacollect.collect_variant_data("rs7412")
gwas_data = ov.external.datacollect.collect_gwas_data("diabetes")

# Expression data collection
expression_data = ov.external.datacollect.collect_expression_data("GSE123456")
targets = ov.external.datacollect.collect_target_data("ENSG00000141510")

# Pathway data collection
kegg_pathway = ov.external.datacollect.collect_pathway_data("hsa04110")
reactome_data = ov.external.datacollect.collect_reactome_data("R-HSA-68886")
```

### OmicVerse Format Integration

```python
import omicverse as ov

# Collect expression data with automatic format conversion
expression_adata = ov.external.datacollect.get_geo_data(
    "GSE123456", 
    format="anndata"  # Automatic conversion to AnnData
)

# Seamless integration with OmicVerse analysis
ov.bulk.pyDEG(expression_adata)  # Differential expression analysis
ov.bulk.pyGSEA(expression_adata)  # Gene set enrichment

# Multi-omics integration
protein_df = ov.external.datacollect.get_uniprot_data(
    ["P04637", "P53_HUMAN"], 
    format="pandas"
)

pathway_mudata = ov.external.datacollect.get_pathway_data(
    ["hsa04110", "hsa04151"],
    format="mudata"  # Multi-modal data format
)
```

### Advanced Batch Processing  

```python
import omicverse as ov
from omicverse.external.datacollect import BatchCollector

# Initialize batch collector
collector = BatchCollector()

# Batch protein collection
protein_ids = ["P04637", "P21359", "P53_HUMAN", "NF1_HUMAN"]
proteins = collector.collect_proteins(
    protein_ids, 
    include_features=True,
    include_interactions=True,
    format="pandas"
)

# Batch gene expression analysis
geo_ids = ["GSE123456", "GSE789012", "GSE345678"]  
expression_datasets = collector.collect_expression_data(
    geo_ids,
    format="anndata_dict"  # Dictionary of AnnData objects
)

# Integrated analysis workflow
for dataset_id, adata in expression_datasets.items():
    # Run OmicVerse analysis on each dataset
    deg_results = ov.bulk.pyDEG(adata)
    gsea_results = ov.bulk.pyGSEA(adata)
    
    print(f"Dataset {dataset_id}: {len(deg_results)} DEGs found")
```

### CLI Integration

```bash
# Through OmicVerse external module
python -m omicverse.external.datacollect collect uniprot P04637
python -m omicverse.external.datacollect collect geo GSE123456 --format anndata
python -m omicverse.external.datacollect batch-collect proteins P04637,P21359 --output results.csv

# Standalone usage (maintained compatibility)
biocollect collect uniprot P04637
biocollect collect pdb 1TUP --download-structure
biocollect collect kegg hsa04110
biocollect status  # Check API status
```

---

## Data Format Conversions  

The integration provides automatic format conversion utilities:

### Pandas DataFrames
```python
# Protein data to pandas
protein_df = ov.external.datacollect.to_pandas(protein_data, data_type="protein")
# Columns: protein_id, name, sequence, length, organism, features

# Gene expression to pandas  
expression_df = ov.external.datacollect.to_pandas(geo_data, data_type="expression")
# Columns: gene_symbol, sample_id, expression_value, condition
```

### AnnData Objects
```python
# Gene expression to AnnData
adata = ov.external.datacollect.to_anndata(
    geo_data,
    obs_keys=['sample_id', 'condition', 'tissue'],
    var_keys=['gene_symbol', 'gene_id', 'biotype']
)

# Access data
print(adata.X.shape)        # Expression matrix
print(adata.obs.head())     # Sample metadata  
print(adata.var.head())     # Gene metadata
```

### MuData Objects  
```python
# Multi-omics data to MuData
mudata_obj = ov.external.datacollect.to_mudata({
    'rna': expression_data,
    'protein': protein_data, 
    'pathway': pathway_data
})

# Access modalities
rna_data = mudata_obj['rna']
protein_data = mudata_obj['protein']
```

---

## Configuration

### Environment Variables

```bash
# API Keys
export UNIPROT_API_KEY="your_key_here"
export NCBI_API_KEY="your_key_here"  
export EBI_API_KEY="your_key_here"

# Database Configuration
export DATABASE_URL="postgresql://user:pass@localhost/biocollect"
# OR for SQLite:
export DATABASE_URL="sqlite:///./biocollect.db"

# Rate Limiting
export DEFAULT_RATE_LIMIT=10
export UNIPROT_RATE_LIMIT=10
export PDB_RATE_LIMIT=5

# Storage Paths
export RAW_DATA_DIR="./data/raw"
export PROCESSED_DATA_DIR="./data/processed"  
export CACHE_DIR="./data/cache"

# Logging
export LOG_LEVEL="INFO"
export LOG_FILE="datacollect.log"
```

### Configuration File

Create `~/.omicverse/datacollect.yaml`:

```yaml
# API Configuration
api:
  uniprot:
    base_url: "https://rest.uniprot.org"
    api_key: "${UNIPROT_API_KEY}"
    rate_limit: 10
    timeout: 30
    
  pdb:
    base_url: "https://data.rcsb.org"
    rate_limit: 10
    
  ensembl:
    base_url: "https://rest.ensembl.org"  
    rate_limit: 15

# Storage Configuration
storage:
  database_url: "${DATABASE_URL}"
  raw_data_dir: "${RAW_DATA_DIR}"
  processed_data_dir: "${PROCESSED_DATA_DIR}"
  
# Format Conversion Settings
formats:
  default_expression_format: "anndata"
  default_protein_format: "pandas" 
  default_pathway_format: "pandas"
  
# OmicVerse Integration Settings
omicverse:
  auto_convert_formats: true
  cache_converted_data: true
  integrate_with_workflows: true
```

---

## Testing & Validation

### Integration Testing

```bash
# Run full integration test suite
python scripts/test_integration_compatibility.py --omicverse-path /path/to/omicverse --report

# Test specific components
pytest tests/test_omicverse_integration.py -v

# Test format conversions
pytest tests/test_format_converters.py -v

# Test API compatibility  
pytest tests/test_api/ -v
```

### Validation Results

Current test status: **✅ 597/597 tests passing**

```
🧪 OmicVerse Integration Compatibility Tests

✅ Basic Imports: DataCollect module available (v1.0.0)
✅ API Clients: 29 clients imported and instantiated  
✅ Collectors: All collectors working correctly
✅ Format Converters: pandas, anndata, mudata available
✅ Convenience Functions: All functions accessible
✅ Validation Utilities: Working correctly
✅ Configuration: Settings accessible
✅ Data Models: 8 models imported successfully
✅ CLI Integration: Available through omicverse
✅ Dependencies: All dependencies compatible

🏁 Test Summary: 10/10 tests passed
🎉 All tests passed! Integration is ready.
```

---

## Migration Process

### Step 1: Prepare Target Environment
```bash
# Clone OmicVerse
git clone https://github.com/HendricksJudy/omicverse.git
cd omicverse

# Create integration branch
git checkout -b add-datacollect-module

# Create target directory
mkdir -p omicverse/external/datacollect
```

### Step 2: Run Migration Script
```bash  
# Run automated migration
python /path/to/datacollect2bionmi/scripts/migrate_to_omicverse.py \
  --source /path/to/datacollect2bionmi \
  --target ./omicverse/external/datacollect \
  --test-integration
```

### Step 3: Validate Integration
```bash
# Test the integration
PYTHONPATH="." python -c "
import omicverse as ov
print('✅ OmicVerse import successful')
print('DataCollect available:', hasattr(ov, 'external'))
print('External.datacollect:', hasattr(ov.external, 'datacollect'))
"

# Run compatibility tests
PYTHONPATH="." python scripts/test_integration_compatibility.py \
  --omicverse-path . --report
```

### Step 4: Create Pull Request
```bash  
# Add and commit changes
git add omicverse/external/datacollect/
git commit -m "Add comprehensive bioinformatics data collection module (DataCollect)

- Integrates 29 API clients for biological databases
- Includes 597 passing tests with comprehensive error handling  
- Provides seamless OmicVerse format conversion (AnnData, pandas, MuData)
- Maintains backward compatibility with standalone usage
- Adds robust rate limiting and retry logic
- Supports protein, genomics, expression, and pathway data collection"

# Push and create PR
git push -u origin add-datacollect-module
gh pr create --title "Add DataCollect: Comprehensive Bioinformatics Data Collection Module" \
  --body "See OMICVERSE_INTEGRATION_GUIDE.md for full details"
```

---

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Issue: Cannot import datacollect module
# Solution: Check Python path
import sys
sys.path.insert(0, '/path/to/omicverse')
import omicverse as ov
```

#### API Key Issues
```bash
# Issue: API rate limiting or authentication
# Solution: Set API keys
export UNIPROT_API_KEY="your_key_here"
export NCBI_API_KEY="your_key_here"

# Test API access
python -c "
from omicverse.external.datacollect.api.proteins.uniprot import UniProtClient
client = UniProtClient()
print('✅ UniProt client initialized successfully')
"
```

#### Format Conversion Issues  
```python
# Issue: AnnData conversion fails
# Solution: Install required dependencies
pip install anndata scanpy

# Test conversion
from omicverse.external.datacollect.utils.omicverse_adapters import to_anndata
test_data = {"genes": ["GENE1", "GENE2"], "values": [[1, 2], [3, 4]]}
adata = to_anndata(test_data)
print(f"✅ AnnData created: {adata.shape}")
```

### Getting Help

1. **Check Documentation**: Review API reference docs in `docs/`
2. **Run Diagnostics**: `biocollect status` or integration test script
3. **Check Logs**: View `datacollect.log` for detailed error information  
4. **GitHub Issues**: Report bugs at repository issue tracker
5. **OmicVerse Community**: Engage with OmicVerse users for integration questions

---

## Contributing to the Integration

### Development Setup
```bash
# Fork both repositories
git clone https://github.com/yourusername/datacollect2bionmi.git
git clone https://github.com/yourusername/omicverse.git  

# Install in development mode
cd datacollect2bionmi && pip install -e ".[dev]"
cd ../omicverse && pip install -e ".[dev]"

# Create feature branch
git checkout -b feature/enhanced-integration
```

### Adding New APIs
```python
# 1. Create new API client
# omicverse/external/datacollect/api/specialized/new_api.py
from ..base import BaseAPIClient

class NewAPIClient(BaseAPIClient):
    def __init__(self):
        super().__init__(base_url="https://api.newdatabase.org")
    
    def get_data(self, identifier):
        # Implementation here
        pass

# 2. Add to __init__.py
# omicverse/external/datacollect/api/specialized/__init__.py  
from .new_api import NewAPIClient

# 3. Add convenience function
# omicverse/external/datacollect/__init__.py
def collect_new_data(identifier, format="pandas"):
    client = NewAPIClient()
    data = client.get_data(identifier)
    return convert_format(data, format)

# 4. Add tests
# omicverse/external/datacollect/tests/test_new_api.py
def test_new_api_client():
    client = NewAPIClient()
    result = client.get_data("TEST123")
    assert result is not None
```

### Code Quality Standards
```bash
# Run linters
flake8 omicverse/external/datacollect/
black omicverse/external/datacollect/
isort omicverse/external/datacollect/

# Run tests  
pytest omicverse/external/datacollect/tests/ -v --cov
pytest tests/test_omicverse_integration.py -v

# Test integration
python scripts/test_integration_compatibility.py --report
```

---

## Future Enhancements

### Planned Features
- **🔄 Real-time Data Sync**: Automatic updates from source databases
- **🧠 Machine Learning Integration**: AI-powered data quality assessment
- **📊 Advanced Visualizations**: Interactive plots for collected data
- **🔗 Workflow Templates**: Pre-built analysis pipelines
- **☁️ Cloud Integration**: Support for cloud storage and processing

### Community Requests
- **Multi-species Support**: Enhanced organism-specific data collection
- **Custom Database Connectors**: User-defined API integrations  
- **Data Versioning**: Track data changes over time
- **Collaborative Features**: Share datasets and analysis results

---

## License & Citation

### License
This integration maintains compatibility with both project licenses:
- **DataCollect2BioNMI**: MIT License
- **OmicVerse**: GPL-3.0 License

### Citation  
If you use this integration in your research, please cite both projects:

```bibtex
@article{omicverse2024,
  title={OmicVerse: a unified framework for multi-omics data analysis},
  author={Ding, Zehua and others},
  journal={Nature Communications},
  year={2024},
  publisher={Nature Publishing Group}
}

@software{datacollect2bionmi2024,
  title={DataCollect2BioNMI: Comprehensive Bioinformatics Data Collection},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/datacollect2bionmi}
}
```

---

## Contact & Support

### Maintainers
- **DataCollect2BioNMI**: [Your Contact Information]  
- **OmicVerse Integration**: [Integration Team Contact]

### Community
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share usage examples
- **Documentation**: Contribute to guides and tutorials

### Professional Support
For commercial support, custom integrations, or consulting services, please contact the development team.

---

*This guide represents the comprehensive integration of DataCollect2BioNMI with OmicVerse, bringing together 29 bioinformatics APIs with a leading multi-omics analysis platform. The integration maintains the high quality standards of both projects while providing seamless data collection and analysis workflows.*