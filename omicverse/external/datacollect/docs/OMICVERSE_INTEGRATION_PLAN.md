# OmicVerse Integration Plan for DataCollect2BioNMI

## Overview

This document outlines the comprehensive plan to integrate the DataCollect2BioNMI bioinformatics data collection system into the OmicVerse ecosystem as a new `external` module.

## Target Repository
- **Repository**: https://github.com/HendricksJudy/omicverse
- **Target Path**: `omicverse/external/datacollect/`
- **Integration Type**: New external data collection module

## OmicVerse Context Analysis

### What is OmicVerse?
- **Purpose**: Comprehensive Python library for multi-omics data analysis
- **Focus**: Bulk, single-cell, and spatial RNA sequencing data
- **Goal**: "Solve all tasks in RNA-seq" - unified transcriptomics framework
- **Published**: Nature Communications (July 2024)
- **License**: GPL-3.0 (compatible with our project)

### Current OmicVerse Architecture
- **Core Infrastructure**: pandas, anndata, numpy, mudata
- **Modular Structure**: bulk/, single/, utils/, llm/, external/, etc.
- **Integration Ready**: `external/` directory exists for third-party integrations

## Integration Strategy

### Phase 1: Module Restructuring
Transform datacollect2bionmi into an OmicVerse-compatible module:

```
omicverse/external/datacollect/
├── __init__.py                 # Main module interface
├── api/                        # API clients (29 databases)
│   ├── __init__.py
│   ├── base.py                # Base API client
│   ├── proteins/              # Protein-related APIs
│   │   ├── __init__.py
│   │   ├── uniprot.py
│   │   ├── pdb.py
│   │   ├── alphafold.py
│   │   ├── interpro.py
│   │   └── string.py
│   ├── genomics/              # Genomics APIs
│   │   ├── __init__.py
│   │   ├── ensembl.py
│   │   ├── clinvar.py
│   │   ├── dbsnp.py
│   │   ├── gnomad.py
│   │   └── ucsc.py
│   ├── expression/            # Expression & regulatory APIs
│   │   ├── __init__.py
│   │   ├── geo.py
│   │   ├── opentargets.py
│   │   ├── remap.py
│   │   └── ccre.py
│   ├── pathways/              # Pathway APIs
│   │   ├── __init__.py
│   │   ├── kegg.py
│   │   ├── reactome.py
│   │   └── gtopdb.py
│   └── specialized/           # Specialized APIs
│       ├── __init__.py
│       ├── blast.py
│       ├── jaspar.py
│       ├── mpd.py
│       └── others...
├── collectors/                # Data collection logic
│   ├── __init__.py
│   ├── base.py
│   ├── batch.py              # Batch processing
│   └── integration.py        # OmicVerse data integration
├── models/                    # Data models (adapted for OmicVerse)
│   ├── __init__.py
│   ├── base.py
│   └── biological.py         # Unified biological entity models
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── validation.py
│   ├── transformers.py       # Convert to AnnData/pandas formats
│   └── omicverse_adapters.py # OmicVerse-specific adapters
├── config/                    # Configuration
│   ├── __init__.py
│   └── settings.py
└── tests/                     # Test suite (597 tests)
    ├── __init__.py
    └── [all test files...]
```

### Phase 2: OmicVerse Integration Points

#### 2.1 Data Format Compatibility
```python
# Transform collected data to OmicVerse formats
def to_anndata(gene_expression_data):
    """Convert GEO expression data to AnnData format"""
    pass

def to_pandas_df(tabular_data):
    """Convert API results to pandas DataFrame"""
    pass

def to_mudata(multi_omics_data):
    """Convert multi-omics data to MuData format"""
    pass
```

#### 2.2 Integration with OmicVerse Workflows
```python
import omicverse as ov

# Enhanced data collection capabilities
ov.external.datacollect.collect_protein_data("P04637")
ov.external.datacollect.collect_expression_data("GSE123456")
ov.external.datacollect.collect_pathway_data("hsa04110")

# Seamless integration with OmicVerse analysis
data = ov.external.datacollect.get_geo_data("GSE123456", format="anndata")
ov.bulk.pyDEG(data)  # Existing OmicVerse functionality
```

#### 2.3 CLI Integration
```bash
# Through OmicVerse
python -m omicverse.external.datacollect collect uniprot P04637

# Standalone (maintained compatibility)
biocollect collect uniprot P04637
```

### Phase 3: Documentation Integration

#### 3.1 Update OmicVerse Documentation
- Add datacollect module to OmicVerse docs
- Create integration tutorials
- Update API reference

#### 3.2 Maintain Standalone Documentation
- Keep comprehensive API documentation
- Maintain development guides
- Update installation instructions

### Phase 4: Testing Integration

#### 4.1 Maintain Current Test Suite
- All 597 tests remain functional
- Test both standalone and integrated modes
- Add OmicVerse-specific integration tests

#### 4.2 Add Integration Tests
```python
def test_omicverse_anndata_conversion():
    """Test conversion to AnnData format"""
    pass

def test_omicverse_workflow_integration():
    """Test integration with OmicVerse workflows"""
    pass
```

## Implementation Steps

### Step 1: Prepare Repository Structure
```bash
# 1. Clone the target repository
git clone https://github.com/HendricksJudy/omicverse.git
cd omicverse

# 2. Create branch for integration
git checkout -b add-datacollect-module

# 3. Create target directory structure
mkdir -p omicverse/external/datacollect
```

### Step 2: Module Migration
```bash
# 1. Copy and restructure source code
cp -r /path/to/datacollect2bionmi/src/* omicverse/external/datacollect/

# 2. Update import statements for OmicVerse
# 3. Create OmicVerse-compatible __init__.py files
# 4. Add format conversion utilities
```

### Step 3: Integration Code
```python
# omicverse/external/datacollect/__init__.py
"""
DataCollect module for OmicVerse - Comprehensive bioinformatics data collection

Provides access to 29+ biological databases with seamless OmicVerse integration.
"""

from .api import *
from .collectors import *
from .utils.omicverse_adapters import to_anndata, to_pandas, to_mudata

__version__ = "1.0.0"
__all__ = [
    # API clients
    "UniProtClient", "PDBClient", "AlphaFoldClient", "InterProClient",
    "EnsemblClient", "ClinVarClient", "dbSNPClient", "GEOClient",
    "KEGGClient", "ReactomeClient", "OpenTargetsClient",
    # Collectors
    "collect_protein_data", "collect_expression_data", "collect_pathway_data",
    # Format converters
    "to_anndata", "to_pandas", "to_mudata"
]
```

### Step 4: Update OmicVerse Main Module
```python
# omicverse/__init__.py - Add to existing imports
from .external import datacollect

# Make available at top level
__all__.extend(["datacollect"])
```

### Step 5: Documentation Updates

#### 5.1 Create Integration Tutorial
```markdown
# Using DataCollect with OmicVerse

## Quick Start
```python
import omicverse as ov

# Collect gene expression data
data = ov.datacollect.get_geo_data("GSE123456", format="anndata")

# Integrate with OmicVerse analysis
ov.bulk.pyDEG(data)
```

#### 5.2 Update README
Add datacollect to OmicVerse feature list and installation instructions.

### Step 6: Testing and Validation
```bash
# 1. Run existing test suite
cd omicverse/external/datacollect
python -m pytest tests/ -v

# 2. Run integration tests
python -m pytest tests/test_omicverse_integration.py -v

# 3. Test OmicVerse compatibility
python -c "import omicverse as ov; print(ov.datacollect.__version__)"
```

### Step 7: Submission Process
1. **Create Pull Request**
   - Branch: `add-datacollect-module`
   - Title: "Add comprehensive bioinformatics data collection module (DataCollect)"
   - Description: Detailed integration summary

2. **PR Content**
   - 29 API clients with 597 passing tests
   - Seamless OmicVerse integration
   - Maintained backward compatibility
   - Comprehensive documentation

## Benefits for OmicVerse Users

### 1. **Expanded Data Access**
- 29+ bioinformatics databases
- Protein, genomics, expression, pathway data
- Automated data collection and validation

### 2. **Enhanced Workflows**
```python
# Before: Manual data retrieval
# Now: Integrated data collection
expression_data = ov.datacollect.get_geo_data("GSE123456", format="anndata")
protein_data = ov.datacollect.get_uniprot_data("P04637")
pathway_data = ov.datacollect.get_kegg_pathway("hsa04110")

# Seamless analysis integration
ov.bulk.pyDEG(expression_data)
```

### 3. **Production-Ready Quality**
- 100% test coverage (597/597 tests passing)
- Advanced error handling and retry logic
- Rate limiting and API key management
- Comprehensive data validation

## Maintenance Strategy

### 1. **Dual Maintenance**
- Maintain both standalone and integrated versions
- Sync updates between repositories
- Ensure compatibility across versions

### 2. **Community Collaboration**
- Engage with OmicVerse maintainers
- Contribute to OmicVerse ecosystem
- Support user community

### 3. **Continuous Integration**
- Automated testing in both environments
- Version compatibility checks
- Documentation synchronization

## Timeline

### Week 1-2: Preparation
- Repository analysis and planning
- Code restructuring for OmicVerse compatibility
- Integration utilities development

### Week 3-4: Implementation
- Module migration and integration
- Testing and validation
- Documentation updates

### Week 5-6: Submission and Review
- Pull request submission
- Community feedback incorporation
- Final testing and deployment

## Success Metrics

1. **Integration Success**: Module successfully integrates with OmicVerse
2. **Test Coverage**: All 597 tests continue to pass
3. **Community Adoption**: Positive feedback from OmicVerse community
4. **Documentation**: Comprehensive integration documentation
5. **Compatibility**: Maintains backward compatibility with standalone version

## Risk Mitigation

### 1. **Dependency Conflicts**
- Careful analysis of OmicVerse dependencies
- Virtual environment testing
- Gradual migration approach

### 2. **Breaking Changes**
- Maintain backward compatibility
- Version pinning for critical dependencies
- Comprehensive testing strategy

### 3. **Maintenance Overhead**
- Automated testing pipelines
- Documentation synchronization tools
- Clear contribution guidelines

## Conclusion

This integration plan provides a comprehensive roadmap for incorporating DataCollect2BioNMI into the OmicVerse ecosystem, enhancing its data collection capabilities while maintaining the high quality and reliability standards of both projects.