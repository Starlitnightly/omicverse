# DataCollect Documentation Index

Welcome to the comprehensive documentation for DataCollect, the bioinformatics data collection external module for OmicVerse.

## 📚 Documentation Overview

### Getting Started
| Document | Description | Best For |
|----------|-------------|----------|
| **[Main README](../README.md)** | Overview and quick start guide | First-time users |
| **[Quick Start Guide](QUICK_START.md)** | 5-minute setup and basic usage | Getting up and running quickly |
| **[OmicVerse Tutorial](OMICVERSE_TUTORIAL.md)** | Comprehensive tutorial with examples | Learning through practical examples |

### Reference Documentation
| Document | Description | Best For |
|----------|-------------|----------|
| **[API Reference](API_REFERENCE.md)** | Complete API documentation | Original datacollect2bionmi users |
| **[OmicVerse API Reference](OMICVERSE_API_REFERENCE.md)** | OmicVerse-specific API documentation | OmicVerse integration users |
| **[CLI Reference](CLI_REFERENCE.md)** | Command-line interface documentation | Command-line users |

### Configuration & Setup
| Document | Description | Best For |
|----------|-------------|----------|
| **[Configuration Guide](CONFIGURATION.md)** | Setup and configuration options | Production deployment |
| **[Database Guide](DATABASE.md)** | Database setup and management | Data persistence needs |

### Advanced Usage
| Document | Description | Best For |
|----------|-------------|----------|
| **[Workflows Guide](WORKFLOWS.md)** | Advanced analysis workflows | Complex analysis pipelines |
| **[Data Export Guide](DATA_EXPORT.md)** | Data export and format conversion | Data sharing and integration |
| **[Integration Guide](OMICVERSE_INTEGRATION_GUIDE.md)** | Detailed integration documentation | Developers and integrators |

### Support & Maintenance
| Document | Description | Best For |
|----------|-------------|----------|
| **[Troubleshooting Guide](TROUBLESHOOTING.md)** | Common issues and solutions | Problem solving |

## Supported Databases

### Proteins & Structures
- **UniProt**: Protein sequences and annotations
- **PDB**: 3D protein structures
- **AlphaFold**: AI-predicted protein structures
- **InterPro**: Protein families and domains
- **STRING**: Protein-protein interactions
- **EMDB**: Electron microscopy structures

### Genomics & Variants
- **Ensembl**: Gene information and annotations
- **ClinVar**: Clinical genetic variants
- **dbSNP**: Single nucleotide polymorphisms
- **gnomAD**: Population genetics data
- **GWAS Catalog**: Genome-wide association studies
- **UCSC Genome Browser**: Genomic annotations

### Expression & Regulation
- **GEO**: Gene expression datasets
- **OpenTargets**: Drug target information
- **ReMap**: Transcription factor binding sites
- **CCRE**: Candidate cis-regulatory elements

### Pathways & Drugs
- **KEGG**: Biological pathways and processes
- **Reactome**: Pathway analysis
- **Guide to Pharmacology**: Drug and target database

### Specialized Databases
- **BLAST**: Sequence similarity search
- **JASPAR**: Transcription factor binding profiles
- **MPD**: Mouse phenotype database
- **IUCN**: Species conservation status
- **PRIDE**: Proteomics data repository
- **cBioPortal**: Cancer genomics data
- **RegulomeDB**: Regulatory variants

## Quick Examples

### Command Line Usage
```bash
# Initialize database
biocollect init

# Collect protein data
biocollect collect uniprot P04637

# Collect structure data
biocollect collect pdb 1A3N

# Collect gene expression data
biocollect collect geo GSE123456

# Check collection status
biocollect status
```

### OmicVerse Integration Usage
```python
import omicverse as ov

# Collect protein data
protein_data = ov.external.datacollect.collect_protein_data("P04637")

# Collect expression data as AnnData (OmicVerse standard)
adata = ov.external.datacollect.collect_expression_data("GSE123456", format="anndata")

# Use with OmicVerse analysis
deg_results = ov.bulk.pyDEG(adata)

# Pathway data collection
pathway_data = ov.external.datacollect.collect_pathway_data("hsa04110")
```

## Key Features

- ✅ **29+ Database APIs**: Comprehensive coverage of major bioinformatics resources
- ✅ **OmicVerse Integration**: Seamless integration with OmicVerse multi-omics analysis
- ✅ **Format Conversion**: Automatic conversion to AnnData, pandas, MuData formats
- ✅ **Production Ready**: 597+ passing tests with robust error handling
- ✅ **Rate Limited**: Respectful API usage with built-in throttling
- ✅ **Data Validation**: Automatic validation of biological sequences and identifiers
- ✅ **Batch Processing**: Efficient collection of large datasets
- ✅ **CLI & Python API**: Flexible usage options for different workflows

## Support & Community

- **Documentation**: You're reading it! Check the guides above for detailed information
- **Issues**: Report bugs and request features on GitHub Issues
- **Questions**: Ask questions in GitHub Discussions

## License

This project is licensed under the MIT License - see the LICENSE file for details.