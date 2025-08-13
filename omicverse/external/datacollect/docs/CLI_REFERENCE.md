# CLI Reference

Complete command-line interface documentation for BioinformaticsDataCollector.

## Installation & Setup

### Global Options
```bash
biocollect --help                    # Show help
biocollect --log-level DEBUG        # Set logging level (DEBUG, INFO, WARNING, ERROR)
biocollect --log-file path/to/log    # Specify log file
```

### Database Management

#### Initialize Database
```bash
biocollect init
```
Creates the SQLite database and all necessary tables. Must be run before first use.

#### Check Status
```bash
biocollect status
```
Shows database connection status and record counts for all tables.

## Data Collection Commands

All data collection uses the `collect` subcommand:
```bash
biocollect collect <database> <identifier> [options]
```

### Proteins & Structures

#### UniProt Proteins
```bash
# Collect single protein
biocollect collect uniprot P04637                    # p53 tumor suppressor
biocollect collect uniprot P04637 --no-features      # Skip protein features
biocollect collect uniprot P04637 --save-file        # Save JSON copy

# Search and collect proteins
biocollect collect uniprot-search "kinase" --limit 10
biocollect collect uniprot-search "kinase" --organism "Homo sapiens" --limit 20
biocollect collect uniprot-search "reviewed:yes AND organism_id:9606" --limit 50
```

#### PDB Structures
```bash
# Collect structure metadata
biocollect collect pdb 1A3N                          # Basic metadata
biocollect collect pdb 1A3N --download               # Download PDB file
biocollect collect pdb 1A3N --download --save-file   # Download + save JSON

# Search by sequence similarity
biocollect collect pdb-blast "MVLSPADKTNVKAAW" --e-value 0.01 --limit 5
```

#### AlphaFold Predictions
```bash
# Collect AlphaFold structure prediction
biocollect collect alphafold P04637                   # Metadata only
biocollect collect alphafold P04637 --download        # Download structure file
biocollect collect alphafold P04637 --download-pae    # Download confidence data
biocollect collect alphafold P04637 --download --download-pae --save-file
```

#### InterPro Domains
```bash
# Collect protein domain annotations
biocollect collect interpro P04637                    # Domain/family data
biocollect collect interpro P04637 --save-file        # Save JSON copy
```

#### STRING Interactions
```bash
# Collect protein interaction data
biocollect collect string TP53                       # Human protein (default)
biocollect collect string TP53 --species 10090       # Mouse protein
biocollect collect string TP53 --no-partners         # Skip interaction partners
biocollect collect string TP53 --partner-limit 50    # Collect up to 50 partners
```

### Genomics & Variants

#### Ensembl Genes
```bash
# Collect by Ensembl gene ID
biocollect collect ensembl ENSG00000141510            # TP53 gene
biocollect collect ensembl ENSG00000141510 --no-expand # Skip transcripts/proteins

# Collect by gene symbol
biocollect collect ensembl TP53 --id-type symbol --species human
biocollect collect ensembl Tp53 --id-type symbol --species mouse
```

#### ClinVar Clinical Variants
```bash
# Single variant by ClinVar ID
biocollect collect clinvar 12345 --save-file

# All variants for a gene
biocollect collect clinvar BRCA1 --id-type gene --limit 100
biocollect collect clinvar BRCA1 --id-type gene --pathogenic-only --limit 50

# Variants by disease
biocollect collect clinvar "breast cancer" --id-type disease --limit 200
```

#### dbSNP Variants
```bash
# Single SNP by rs ID
biocollect collect dbsnp rs7412 --save-file

# All variants for a gene
biocollect collect dbsnp-gene TP53 --limit 100
biocollect collect dbsnp-gene TP53 --organism mouse --limit 50

# Variants in genomic region
biocollect collect dbsnp-region 17 7565097 7590856   # TP53 locus
biocollect collect dbsnp-region X 123456 234567 --organism human
```

### Expression & Pathways

#### GEO Gene Expression
```bash
# Collect dataset by accession
biocollect collect geo GSE123456 --save-file         # Series
biocollect collect geo GSM123456 --save-file         # Sample
biocollect collect geo GDS123 --save-file            # Dataset

# Search for datasets containing a gene
biocollect collect geo-search TP53 --limit 20
biocollect collect geo-search TP53 --organism "Mus musculus" --limit 10
```

#### KEGG Pathways
```bash
# Collect pathway data
biocollect collect kegg hsa04110 --save-file         # Cell cycle pathway
biocollect collect kegg mmu04110 --save-file         # Mouse cell cycle
```

## Export Commands

### Database Export
```bash
# Export entire database (not yet implemented)
biocollect export database output.json --format json
biocollect export database output.csv --format csv --table proteins
biocollect export database sequences.fasta --format fasta --table proteins
```

## Common Patterns & Examples

### Collecting Complete Protein Information
```bash
# Collect all available data for p53
biocollect collect uniprot P04637 --save-file
biocollect collect pdb-blast "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD") --e-value 0.001 --limit 10
biocollect collect alphafold P04637 --download --download-pae
biocollect collect interpro P04637 --save-file
biocollect collect string TP53 --partner-limit 30 --save-file
```

### Collecting Gene Expression Data
```bash
# Find and collect datasets for a gene of interest
biocollect collect geo-search BRCA1 --limit 10 --save-file
# Then collect specific interesting datasets:
biocollect collect geo GSE123456 --save-file
biocollect collect geo GSE789012 --save-file
```

### Collecting Clinical Variants
```bash
# Comprehensive variant collection for a disease gene
biocollect collect clinvar BRCA1 --id-type gene --limit 500 --save-file
biocollect collect dbsnp-gene BRCA1 --limit 200 --save-file
biocollect collect ensembl BRCA1 --id-type symbol --expand --save-file
```

### Batch Processing Multiple Genes
```bash
# Create a shell script for multiple genes:
#!/bin/bash
genes=("TP53" "BRCA1" "BRCA2" "ATM" "CHEK2")
for gene in "${genes[@]}"; do
  echo "Processing $gene..."
  biocollect collect uniprot-search "$gene AND reviewed:yes AND organism_id:9606" --limit 1
  biocollect collect clinvar "$gene" --id-type gene --pathogenic-only --limit 100
  biocollect collect geo-search "$gene" --limit 5
done
```

## Advanced Usage

### Debugging & Troubleshooting
```bash
# Enable debug logging
biocollect --log-level DEBUG collect uniprot P04637

# Log to file
biocollect --log-file debug.log --log-level DEBUG collect pdb 1A3N --download

# Check what's in your database
biocollect status
```

### Working with Rate Limits
```bash
# If you're getting rate limited, the tool automatically handles retries
# But you can reduce the rate in your .env file:
echo "UNIPROT_RATE_LIMIT=5" >> .env
echo "NCBI_RATE_LIMIT=2" >> .env
```

### File Management
```bash
# By default, files are saved to:
# - Database: ./biocollect.db
# - Raw data: ./data/raw/
# - JSON exports: current directory

# Customize in .env file:
echo "DATABASE_URL=postgresql://user:pass@host/db" >> .env
echo "RAW_DATA_DIR=/path/to/raw/data" >> .env
echo "PROCESSED_DATA_DIR=/path/to/processed/data" >> .env
```

## Error Handling

### Common Errors

#### Database Not Initialized
```bash
# Error: Database connection failed. Run 'biocollect init' first.
# Solution:
biocollect init
```

#### Network/API Errors
```bash
# Error: Failed to connect to API
# Solutions:
# 1. Check internet connection
# 2. Try again (temporary API issues)
# 3. Check if API key is needed (NCBI, UniProt for high volume)
```

#### Rate Limiting
```bash
# Error: Too many requests (429)
# Solution: Tool automatically retries with backoff
# Or reduce rate limits in .env file
```

#### Invalid Identifiers
```bash
# Error: Identifier not found
# Solutions:
# 1. Check identifier format (P04637 not p04637)
# 2. Try alternative identifier types
# 3. Search first to find correct identifier
```

### Getting Help
```bash
biocollect --help                     # Main help
biocollect collect --help             # Collection commands
biocollect collect uniprot --help     # Specific command help
biocollect export --help              # Export commands
```

## Performance Tips

1. **Use API Keys**: Set up API keys for better rate limits
2. **Batch Operations**: Use search commands instead of individual collections when possible
3. **Selective Data**: Use `--no-features`, `--no-expand` flags to reduce data volume
4. **Local Caching**: Tool automatically caches API responses to avoid re-downloading
5. **Database Choice**: Use PostgreSQL instead of SQLite for large datasets

## Integration with Scripts

### Shell Scripts
```bash
#!/bin/bash
# Example: Collect data for a list of genes
while read gene; do
  biocollect collect uniprot-search "$gene AND reviewed:yes" --limit 1
done < genes.txt
```

### Python Integration
```bash
# After collecting data via CLI, access it programmatically:
python -c "
from src.models.protein import Protein
from src.models.base import get_db

with next(get_db()) as db:
    proteins = db.query(Protein).all()
    print(f'Collected {len(proteins)} proteins')
"
```