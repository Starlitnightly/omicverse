# Quick Reference

## CLI Commands

### Database Management
```bash
biocollect init                    # Initialize database
biocollect status                  # Check database status
```

### UniProt Collection
```bash
biocollect collect uniprot P04637                      # Basic collection
biocollect collect uniprot P04637 --features           # With features
biocollect collect uniprot P04637 --save-file          # Save to JSON

# Search and collect
biocollect collect uniprot-search "kinase AND organism_id:9606" --max-results 50
```

### PDB Collection
```bash
biocollect collect pdb 1A3N                            # Basic collection
biocollect collect pdb 1A3N --download                 # Download structure
biocollect collect pdb 1A3N --format cif               # Specific format

# Search by sequence
biocollect collect pdb-blast MVLSPADKTNVKAAW --e-value 0.001
```

### AlphaFold Collection
```bash
biocollect collect alphafold P04637                    # Basic prediction
biocollect collect alphafold P04637 --download         # Download structure
biocollect collect alphafold P04637 --download-pae     # Download PAE data
```

## Python API

### Quick Start
```python
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector
from src.collectors.alphafold_collector import AlphaFoldCollector

# Collect protein
uniprot = UniProtCollector()
protein = uniprot.process_and_save("P04637")

# Collect structure
pdb = PDBCollector()
structure = pdb.process_and_save("1A3N")

# Collect AlphaFold prediction
alphafold = AlphaFoldCollector()
prediction = alphafold.process_and_save("P04637")
```

### Batch Operations
```python
# Multiple proteins
proteins = uniprot.collect_batch(["P04637", "P38398", "P51587"])

# Search and collect
results = uniprot.search_and_collect(
    query="kinase AND reviewed:true",
    max_results=100
)
```

### Data Access
```python
from src.models.protein import Protein
from src.models.base import get_db

# Query database
db = next(get_db())
proteins = db.query(Protein).filter(
    Protein.organism == "Homo sapiens"
).all()

# Access relationships
for protein in proteins:
    print(f"{protein.accession}: {len(protein.features)} features")
    for go_term in protein.go_terms:
        print(f"  - {go_term.name}")
```

## Validation

### Sequences
```python
from src.utils.validation import SequenceValidator

# Validate protein
is_valid = SequenceValidator.is_valid_protein_sequence("MVLSPADKTNVKAAW")

# Validate DNA
is_valid = SequenceValidator.is_valid_dna_sequence("ATCGATCG")
```

### Identifiers
```python
from src.utils.validation import IdentifierValidator

# Validate specific type
is_valid = IdentifierValidator.validate("P04637", "uniprot")

# Detect type
id_type = IdentifierValidator.detect_type("NM_000546")  # "refseq"
```

## Transformation

### Sequences
```python
from src.utils.transformers import SequenceTransformer

# Translate DNA
protein = SequenceTransformer.translate_dna_to_protein("ATGGCG...")

# Reverse complement
rev_comp = SequenceTransformer.reverse_complement("ATCG")

# Find ORFs
orfs = SequenceTransformer.get_orfs(dna_sequence, min_length=100)
```

### Data Normalization
```python
from src.utils.transformers import DataNormalizer

# Clean sequence
clean = DataNormalizer.clean_sequence("ACGT-123", "dna")

# Normalize names
gene = DataNormalizer.normalize_gene_name("tp53")  # "TP53"
organism = DataNormalizer.normalize_organism_name("human")  # "Homo sapiens"
```

## Configuration

### Environment Variables
```bash
# Essential
DATABASE_URL=sqlite:///./biocollect.db
LOG_LEVEL=INFO

# API Keys
UNIPROT_API_KEY=your_key
NCBI_API_KEY=your_key

# Directories
RAW_DATA_DIR=./data/raw
CACHE_DIR=./cache
```

### Python Access
```python
from config import settings

print(settings.database_url)
print(settings.uniprot_rate_limit)
```

## Common Patterns

### Error Handling
```python
try:
    protein = collector.process_and_save(accession)
except HTTPError as e:
    if e.response.status_code == 404:
        print("Not found")
except Exception as e:
    logger.error(f"Failed: {e}")
```

### Progress Tracking
```python
from tqdm import tqdm

accessions = ["P04637", "P38398", "P51587"]
for acc in tqdm(accessions, desc="Collecting"):
    collector.process_and_save(acc)
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_protein_cached(accession):
    return collector.collect_single(accession)
```

## Database Queries

### Basic Queries
```python
# Get all proteins
proteins = db.query(Protein).all()

# Filter by organism
human_proteins = db.query(Protein).filter(
    Protein.organism == "Homo sapiens"
).all()

# With structure
with_structure = db.query(Protein).filter(
    Protein.has_3d_structure == "Y"
).all()
```

### Complex Queries
```python
# Join with features
from sqlalchemy.orm import joinedload

proteins = db.query(Protein).options(
    joinedload(Protein.features),
    joinedload(Protein.go_terms)
).filter(
    Protein.sequence_length > 500
).all()

# Aggregate
from sqlalchemy import func

avg_length = db.query(
    func.avg(Protein.sequence_length)
).scalar()
```

## Export Formats

### FASTA
```python
from src.utils.transformers import DataExporter

proteins = db.query(Protein).all()
DataExporter.proteins_to_fasta(proteins, "output.fasta")
```

### CSV
```python
import pandas as pd

data = [p.to_dict() for p in proteins]
df = pd.DataFrame(data)
df.to_csv("proteins.csv", index=False)
```

### JSON
```python
import json

data = [p.to_dict() for p in proteins]
with open("proteins.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Troubleshooting

### Check Installation
```bash
biocollect --version
biocollect status
```

### Enable Debug Mode
```bash
export LOG_LEVEL=DEBUG
biocollect collect uniprot P04637
```

### Test API Connection
```python
from src.api.uniprot import UniProtClient

client = UniProtClient()
response = client.get("/uniprotkb/P04637")
print(response.status_code)
```

### Reset Database
```bash
rm biocollect.db
biocollect init
```