# Quick Start Guide

Get up and running with BioinformaticsDataCollector in under 5 minutes.

## Prerequisites

- Python 3.9 or higher
- Internet connection for API access
- 100MB+ free disk space for database

## Installation

### Option 1: From Source (Recommended)
```bash
git clone https://github.com/yourusername/datacollect2bionmi.git
cd datacollect2bionmi
pip install -r requirements.txt
pip install -e .
```

### Option 2: Direct Install
```bash
pip install datacollect2bionmi
```

## Initial Setup

### 1. Initialize Database
```bash
biocollect init
```
This creates a local SQLite database with all necessary tables.

### 2. Verify Installation
```bash
biocollect status
```
Should show database connection and empty table statistics.

## Your First Data Collection

### Collect a Protein (UniProt)
```bash
# Collect p53 tumor suppressor protein
biocollect collect uniprot P04637
```

### Collect a Structure (PDB)
```bash
# Collect p53 DNA-binding domain structure
biocollect collect pdb 1TUP
```

### Check What You've Collected
```bash
biocollect status
```

## Common Collection Commands

### Proteins & Structures
```bash
# Protein information
biocollect collect uniprot P04637
biocollect collect uniprot BRCA1 --id-type gene

# Protein structures
biocollect collect pdb 1A3N
biocollect collect alphafold P04637

# Protein interactions
biocollect collect string TP53
biocollect collect interpro P04637
```

### Genomics & Variants
```bash
# Gene information
biocollect collect ensembl ENSG00000141510

# Genetic variants
biocollect collect clinvar BRCA1 --id-type gene
biocollect collect dbsnp rs7412
biocollect collect gwas-catalog diabetes
```

### Expression & Pathways
```bash
# Gene expression
biocollect collect geo GSE123456

# Biological pathways
biocollect collect kegg hsa04110
biocollect collect reactome R-HSA-68886
biocollect collect opentargets ENSG00000141510
```

### Specialized Data
```bash
# Sequence search
biocollect collect blast --sequence "ATCGTAGCTAGC" --program blastn

# Conservation status
biocollect collect iucn "Panthera leo"

# Proteomics data
biocollect collect pride PXD000001
```

## Batch Collection

### From File
Create a file `identifiers.txt`:
```
P04637
P53_HUMAN
TP53
```

Then collect all:
```bash
biocollect collect uniprot --batch-file identifiers.txt
```

### From Command Line
```bash
biocollect collect uniprot P04637,P53_HUMAN,Q04206 --batch
```

## Python API Usage

### Basic Collection
```python
from src.collectors.uniprot_collector import UniProtCollector

# Initialize collector
collector = UniProtCollector()

# Collect single protein
protein = collector.process_and_save("P04637")
print(f"Collected: {protein.protein_name}")

# Search and collect
proteins = collector.search_and_collect(
    query="kinase AND organism_id:9606",
    max_results=10
)
```

### Working with Collected Data
```python
from src.models.base import get_db
from src.models.protein import Protein

# Query database
with next(get_db()) as db:
    proteins = db.query(Protein).filter(
        Protein.organism == "Homo sapiens"
    ).all()
    
    for protein in proteins:
        print(f"{protein.accession}: {protein.protein_name}")
```

## Configuration (Optional)

### Environment Variables
Create `.env` file:
```env
# API Keys (optional but recommended)
UNIPROT_API_KEY=your_key_here
NCBI_API_KEY=your_key_here

# Database (default: SQLite)
DATABASE_URL=sqlite:///./biocollect.db

# Logging
LOG_LEVEL=INFO
```

### Custom Directories
```env
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
```

## Next Steps

Now that you have BioinformaticsDataCollector running:

1. **Explore More Databases**: Try different collectors like `ensembl`, `kegg`, `geo`
2. **Set Up API Keys**: Get better rate limits with proper API credentials
3. **Batch Processing**: Process larger datasets efficiently
4. **Data Export**: Export your collected data for analysis
5. **Python Integration**: Use the Python API in your research scripts

## Need Help?

- **Full Documentation**: [docs/README.md](README.md)
- **CLI Reference**: [docs/CLI_REFERENCE.md](CLI_REFERENCE.md)
- **Configuration Guide**: [docs/CONFIGURATION.md](CONFIGURATION.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Common Issues

### Database Issues
```bash
# Reset database if needed
rm biocollect.db
biocollect init
```

### Network/API Issues
```bash
# Check connectivity
biocollect collect uniprot P04637 --debug
```

### Permission Issues
```bash
# Make sure you have write permissions
chmod 755 .
```