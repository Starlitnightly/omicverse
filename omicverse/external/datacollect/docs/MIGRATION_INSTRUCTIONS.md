# OmicVerse Integration Migration Instructions

## Step-by-Step Guide to Upload DataCollect2BioNMI to OmicVerse

This document provides detailed instructions for migrating the DataCollect2BioNMI project to the OmicVerse repository.

## Prerequisites

### 1. Environment Setup
```bash
# Ensure Python 3.8+ is installed
python --version

# Install required packages
pip install click pathlib

# Install Git (if not already installed)
git --version
```

### 2. Repository Access
- Ensure you have access to fork/contribute to https://github.com/HendricksJudy/omicverse
- Have your GitHub credentials configured

## Migration Process

### Step 1: Clone the OmicVerse Repository

```bash
# Clone the OmicVerse repository
git clone https://github.com/HendricksJudy/omicverse.git
cd omicverse

# Create a new branch for the integration
git checkout -b add-datacollect-module

# Verify the repository structure
ls -la omicverse/
```

### Step 2: Run the Migration Script

```bash
# From your datacollect2bionmi directory
cd /path/to/datacollect2bionmi

# Make the migration script executable
chmod +x scripts/migrate_to_omicverse.py

# Run the migration
python scripts/migrate_to_omicverse.py \
    --source /path/to/datacollect2bionmi \
    --target /path/to/omicverse
```

Expected output:
```
🚀 Starting OmicVerse migration...
🔍 Validating environment...
✅ Environment validation complete
📁 Creating directory structure...
✅ Directory structure created
🔄 Migrating API clients...
  ✅ uniprot.py -> proteins/
  ✅ pdb.py -> proteins/
  [... more API migrations ...]
✅ API clients migrated
[... continues with all migration steps ...]
🎉 Migration completed successfully!
📁 Module created at: /path/to/omicverse/omicverse/external/datacollect
```

### Step 3: Update Import Statements

```bash
# Navigate to the OmicVerse directory
cd /path/to/omicverse

# Run the import updater script
python update_imports.py
```

### Step 4: Update OmicVerse Main Module

Edit `omicverse/__init__.py` to include the DataCollect module:

```python
# Add to existing imports
from .external import datacollect

# Add to __all__ list
__all__ = [
    # ... existing exports ...
    "datacollect",
]
```

### Step 5: Install Dependencies

Update the OmicVerse `requirements.txt` or `setup.py` to include DataCollect dependencies:

```bash
# Check DataCollect requirements
cat /path/to/datacollect2bionmi/requirements.txt

# Add these to OmicVerse requirements:
# requests>=2.25.1
# pandas>=1.3.0
# numpy>=1.21.0
# sqlalchemy>=1.4.0
# click>=8.0.0
# pydantic>=1.8.0
# rich>=10.0.0
# [... other dependencies ...]
```

### Step 6: Test the Integration

```bash
# Test basic import
python -c "import omicverse as ov; print(ov.datacollect.__version__)"

# Run the test suite
cd omicverse/external/datacollect
python -m pytest tests/ -v

# Test OmicVerse integration
python -c "
import omicverse as ov
data = ov.datacollect.collect_protein_data('P04637')
print('Integration successful!')
"
```

### Step 7: Create Pull Request

```bash
# Add all changes
git add .

# Commit with descriptive message
git commit -m "Add comprehensive bioinformatics data collection module (DataCollect)

- Integrates 29 API clients for biological databases
- Includes 597 passing tests with comprehensive error handling
- Provides seamless OmicVerse format conversion (AnnData, pandas, MuData)
- Maintains backward compatibility with standalone usage
- Adds robust rate limiting and retry logic
- Supports protein, genomics, expression, and pathway data collection

APIs included:
- Proteins & Structures: UniProt, PDB, AlphaFold, InterPro, STRING, EMDB
- Genomics & Variants: Ensembl, ClinVar, dbSNP, gnomAD, GWAS Catalog, UCSC
- Expression & Regulation: GEO, OpenTargets, ReMap, CCRE
- Pathways & Drugs: KEGG, Reactome, GtoPdb
- Specialized: BLAST, JASPAR, MPD, IUCN, PRIDE, cBioPortal, RegulomeDB

This module enhances OmicVerse's data acquisition capabilities while
maintaining the high quality standards of both projects."

# Push the branch
git push origin add-datacollect-module
```

### Step 8: Create the Pull Request on GitHub

1. Go to https://github.com/HendricksJudy/omicverse
2. Click "Compare & pull request" for your branch
3. Fill in the PR details:

**Title:** Add comprehensive bioinformatics data collection module (DataCollect)

**Description:**
```markdown
## Overview
This PR integrates the DataCollect2BioNMI project as a new external module in OmicVerse, providing comprehensive bioinformatics data collection capabilities.

## Features Added
- **29 API clients** for major biological databases
- **597 passing tests** with comprehensive error handling
- **OmicVerse integration** with AnnData, pandas, and MuData format conversion
- **Production-ready quality** with advanced rate limiting and retry logic
- **Backward compatibility** maintains standalone functionality

## API Coverage
### Proteins & Structures
- UniProt, PDB, AlphaFold, InterPro, STRING, EMDB

### Genomics & Variants  
- Ensembl, ClinVar, dbSNP, gnomAD, GWAS Catalog, UCSC

### Expression & Regulation
- GEO, OpenTargets, OpenTargets Genetics, ReMap, CCRE

### Pathways & Drugs
- KEGG, Reactome, Guide to Pharmacology (GtoPdb)

### Specialized Databases
- BLAST, JASPAR, MPD, IUCN, PRIDE, cBioPortal, RegulomeDB, WORMS, Paleobiology

## Integration Examples
```python
import omicverse as ov

# Collect and analyze expression data
adata = ov.datacollect.to_anndata(
    ov.datacollect.collect_expression_data("GSE123456")
)
ov.bulk.pyDEG(adata)

# Multi-omics data collection
protein_data = ov.datacollect.collect_protein_data("TP53")
pathway_data = ov.datacollect.collect_pathway_data("hsa04110")
```

## Quality Assurance
- ✅ All 597 tests passing
- ✅ Comprehensive error handling and retry logic
- ✅ Advanced rate limiting for API protection
- ✅ Data validation and integrity checks
- ✅ Extensive documentation and tutorials

## Files Added
- `omicverse/external/datacollect/` - Main module
- 29 API client implementations
- Comprehensive test suite
- OmicVerse format converters
- Integration tutorials and documentation

## Testing
```bash
# Test the integration
python -c "import omicverse as ov; print(ov.datacollect.__version__)"

# Run test suite
cd omicverse/external/datacollect && python -m pytest tests/ -v
```

## Benefits for OmicVerse Users
- **Expanded Data Access**: 29+ biological databases
- **Seamless Integration**: Native AnnData/pandas conversion
- **Enhanced Workflows**: Integrated data collection and analysis
- **Production Quality**: Robust error handling and validation

This module significantly enhances OmicVerse's data acquisition capabilities while maintaining the project's high quality standards.
```

4. Submit the pull request

## Post-Migration Tasks

### 1. Update Documentation

Create or update the following documentation files in the OmicVerse repository:

```bash
# Add to OmicVerse docs
docs/external/datacollect_tutorial.md
docs/external/datacollect_api_reference.md
docs/examples/datacollect_examples.ipynb
```

### 2. Update OmicVerse README

Add DataCollect to the main OmicVerse README.md:

```markdown
## Features

### Data Collection (NEW)
- **29 Biological Databases**: Comprehensive data access including proteins, genomics, expression, and pathways
- **OmicVerse Integration**: Seamless conversion to AnnData, pandas, and MuData formats
- **Production Quality**: 597 passing tests with robust error handling

### API Coverage
- Proteins & Structures: UniProt, PDB, AlphaFold, InterPro, STRING
- Genomics & Variants: Ensembl, ClinVar, dbSNP, gnomAD, UCSC
- Expression & Regulation: GEO, OpenTargets, ReMap, ENCODE
- Pathways & Drugs: KEGG, Reactome, Guide to Pharmacology

### Quick Example
```python
import omicverse as ov

# Collect expression data and analyze
adata = ov.datacollect.to_anndata(
    ov.datacollect.collect_expression_data("GSE123456")
)
ov.bulk.pyDEG(adata)
```

### 3. Community Engagement

- Announce the integration on relevant forums/communities
- Create example notebooks demonstrating the integration
- Respond to community feedback and questions

### 4. Maintenance Plan

- Set up automated testing for the integrated module
- Monitor for any integration issues
- Keep the module synchronized with upstream changes
- Maintain compatibility with OmicVerse updates

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, check Python path
export PYTHONPATH="/path/to/omicverse:$PYTHONPATH"

# Or install in development mode
cd /path/to/omicverse
pip install -e .
```

#### Test Failures
```bash
# Run tests with verbose output to identify issues
cd omicverse/external/datacollect
python -m pytest tests/ -v --tb=short

# Check specific test categories
python -m pytest tests/test_api/ -v
python -m pytest tests/test_collectors/ -v
```

#### Permission Issues
```bash
# Ensure scripts are executable
chmod +x scripts/*.py

# Check file permissions
ls -la omicverse/external/datacollect/
```

### Getting Help

1. **Check the logs**: Migration script provides detailed logging
2. **Review documentation**: See docs/OMICVERSE_INTEGRATION_PLAN.md
3. **Run diagnostics**: Test basic imports and functionality
4. **Community support**: Reach out to OmicVerse maintainers

## Success Verification

After completing the migration, verify success with these tests:

```bash
# 1. Basic import test
python -c "import omicverse as ov; print('✅ Basic import successful')"

# 2. Module availability test  
python -c "import omicverse as ov; print(f'✅ DataCollect version: {ov.datacollect.__version__}')"

# 3. API client test
python -c "
import omicverse as ov
from omicverse.external.datacollect.api.proteins import UniProtClient
client = UniProtClient()
print('✅ API clients accessible')
"

# 4. Format conversion test
python -c "
import omicverse as ov
import pandas as pd
test_data = {'test': [1, 2, 3]}
df = ov.datacollect.to_pandas(test_data)
print('✅ Format conversion working')
"

# 5. Full test suite
cd omicverse/external/datacollect
python -m pytest tests/ --tb=short -q
echo "✅ All tests should pass (597/597)"
```

If all tests pass, the migration is successful! 🎉

## Next Steps

1. **Community Review**: Wait for OmicVerse maintainer review
2. **Address Feedback**: Make any requested changes
3. **Documentation**: Enhance based on reviewer feedback  
4. **Announcement**: Share with the bioinformatics community
5. **Continuous Improvement**: Monitor usage and gather feedback

The integration brings together the comprehensive data collection capabilities of DataCollect2BioNMI with the powerful analysis framework of OmicVerse, creating a complete solution for omics data workflows.