# Tutorial: Getting Started with BioinformaticsDataCollector

This tutorial will walk you through common use cases and workflows for the BioinformaticsDataCollector.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Protein Collection](#basic-protein-collection)
3. [Working with Structures](#working-with-structures)
4. [Batch Processing](#batch-processing)
5. [Data Validation and Transformation](#data-validation-and-transformation)
6. [Advanced Queries](#advanced-queries)
7. [Exporting Data](#exporting-data)

## Installation and Setup

### 1. Create a Virtual Environment

```bash
# Using venv
python -m venv biocollect_env
source biocollect_env/bin/activate  # On Windows: biocollect_env\Scripts\activate

# Or using conda
conda create -n biocollect python=3.8
conda activate biocollect
```

### 2. Install the Package

```bash
cd datacollect2bionmi
pip install -r requirements.txt
pip install -e .
```

### 3. Initialize the Database

```bash
biocollect init
```

This creates a SQLite database and necessary tables.

### 4. Check Installation

```bash
biocollect status
```

You should see:
```
Database Status: Connected
Database Path: ./biocollect.db

Table Statistics:
  proteins: 0 records
  protein_features: 0 records
  go_terms: 0 records
  structures: 0 records
  chains: 0 records
  ligands: 0 records
```

## Basic Protein Collection

### Example 1: Collect a Single Protein

Let's collect data for human p53 tumor suppressor:

```bash
biocollect collect uniprot P04637
```

Output:
```
Collecting UniProt data for P04637...
Successfully collected: Cellular tumor antigen p53
  Organism: Homo sapiens (Human)
  Length: 393 aa
  PDB IDs: 1A1U,1AIE,1C26,1DT7...
```

### Example 2: Collect with Features

To include detailed protein features:

```bash
biocollect collect uniprot P04637 --features --save-file
```

This saves the data to both the database and a JSON file.

### Example 3: Using Python API

```python
from src.collectors.uniprot_collector import UniProtCollector

# Initialize collector
collector = UniProtCollector()

# Collect protein data
data = collector.collect_single("P04637", include_features=True)

# Examine the data
print(f"Protein: {data['protein_name']}")
print(f"Length: {data['sequence_length']} aa")
print(f"Number of features: {len(data.get('features', []))}")

# Save to database
protein = collector.save_to_database(data)
print(f"Saved with ID: {protein.id}")
```

## Working with Structures

### Example 1: Collect PDB Structure

```bash
# Basic collection
biocollect collect pdb 1TSR

# With structure file download
biocollect collect pdb 1TSR --download
```

### Example 2: Collect All Structures for a Protein

```python
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector

# Get protein
uniprot = UniProtCollector()
protein = uniprot.process_and_save("P04637")

# Get all associated structures
pdb = PDBCollector()
if protein.pdb_ids:
    pdb_ids = protein.pdb_ids.split(",")
    print(f"Found {len(pdb_ids)} structures")
    
    for pdb_id in pdb_ids[:5]:  # First 5 structures
        structure = pdb.process_and_save(pdb_id)
        print(f"- {pdb_id}: {structure.title}")
        print(f"  Resolution: {structure.resolution} Å")
```

## Batch Processing

### Example 1: Collect Multiple Proteins

```python
# List of cancer-related proteins
proteins = [
    "P04637",  # p53
    "P38398",  # BRCA1
    "P51587",  # BRCA2
    "P42345",  # mTOR
    "P04626",  # ERBB2
]

collector = UniProtCollector()
for accession in proteins:
    try:
        protein = collector.process_and_save(accession)
        print(f"✓ {protein.gene_name}: {protein.protein_name}")
    except Exception as e:
        print(f"✗ {accession}: {e}")
```

### Example 2: Search and Collect

```python
# Search for all human kinases
proteins = collector.search_and_collect(
    query="kinase AND organism_id:9606 AND reviewed:true",
    max_results=100,
    include_features=True
)

print(f"Collected {len(proteins)} kinases")

# Analyze results
kinase_types = {}
for protein in proteins:
    # Extract kinase type from protein name
    if "tyrosine" in protein.protein_name.lower():
        kinase_type = "Tyrosine kinase"
    elif "serine" in protein.protein_name.lower():
        kinase_type = "Serine/threonine kinase"
    else:
        kinase_type = "Other kinase"
    
    kinase_types[kinase_type] = kinase_types.get(kinase_type, 0) + 1

for ktype, count in kinase_types.items():
    print(f"{ktype}: {count}")
```

## Data Validation and Transformation

### Example 1: Validate Sequences

```python
from src.utils.validation import SequenceValidator

sequences = [
    "MVLSPADKTNVKAAW",  # Valid protein
    "ACGTACGTNN",       # DNA with ambiguous
    "AUGCAUGC",         # RNA
    "MVLSPA-DKTN",      # Invalid protein (contains gap)
]

for seq in sequences:
    if SequenceValidator.is_valid_protein_sequence(seq):
        print(f"✓ Protein: {seq}")
    elif SequenceValidator.is_valid_dna_sequence(seq, allow_ambiguous=True):
        print(f"✓ DNA: {seq}")
    elif SequenceValidator.is_valid_rna_sequence(seq):
        print(f"✓ RNA: {seq}")
    else:
        print(f"✗ Invalid: {seq}")
```

### Example 2: Transform Sequences

```python
from src.utils.transformers import SequenceTransformer

# DNA to protein translation
dna = "ATGGCCGATCACTAA"
protein = SequenceTransformer.translate_dna_to_protein(dna)
print(f"DNA: {dna}")
print(f"Protein: {protein}")

# Find ORFs
orfs = SequenceTransformer.get_orfs(dna, min_length=3)
for i, orf in enumerate(orfs):
    print(f"ORF {i+1}: {orf['sequence']} (frame {orf['frame']})")
```

### Example 3: Clean and Normalize Data

```python
from src.utils.transformers import DataNormalizer

# Clean sequences
dirty_seq = "ACGT-123 XYZ"
clean_dna = DataNormalizer.clean_sequence(dirty_seq, "dna")
print(f"Cleaned: {clean_dna}")  # ACGT

# Normalize gene names
gene_names = ["tp53", "Tp53", "TP53", "p53"]
normalized = [DataNormalizer.normalize_gene_name(g) for g in gene_names]
print(f"Normalized: {set(normalized)}")  # {'TP53'}

# Normalize organism names
organisms = ["H. sapiens", "Homo sapiens", "human", "HUMAN"]
normalized = [DataNormalizer.normalize_organism_name(o) for o in organisms]
print(f"Normalized: {set(normalized)}")  # {'Homo sapiens'}
```

## Advanced Queries

### Example 1: Complex UniProt Search

```python
# Find all human proteins involved in DNA repair with 3D structure
query = (
    "organism_id:9606 AND "  # Human
    "keyword:DNA repair AND "  # Function
    "database:pdb AND "       # Has structure
    "reviewed:true"           # SwissProt only
)

proteins = collector.search_and_collect(
    query=query,
    max_results=50,
    fields=["accession", "gene_names", "protein_name", "length", "database(PDB)"]
)

# Analyze results
total_length = sum(p.sequence_length for p in proteins)
avg_length = total_length / len(proteins)
print(f"Average protein length: {avg_length:.0f} aa")
```

### Example 2: Find Structures by Sequence

```python
# Search PDB by sequence similarity
sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"

pdb_collector = PDBCollector()
structures = pdb_collector.search_by_sequence(
    sequence=sequence,
    e_value=0.001,
    max_results=10
)

for struct in structures:
    print(f"{struct.structure_id}: {struct.title}")
    print(f"  Resolution: {struct.resolution} Å")
    print(f"  Organism: {struct.organism}")
```

## Exporting Data

### Example 1: Export to CSV

```python
from src.models.protein import Protein
from src.models.base import get_db
import pandas as pd

# Get all proteins from database
db = next(get_db())
proteins = db.query(Protein).all()

# Convert to DataFrame
data = []
for protein in proteins:
    data.append({
        "accession": protein.accession,
        "gene": protein.gene_name,
        "name": protein.protein_name,
        "organism": protein.organism,
        "length": protein.sequence_length,
        "has_structure": protein.has_3d_structure == "Y"
    })

df = pd.DataFrame(data)
df.to_csv("proteins.csv", index=False)
print(f"Exported {len(df)} proteins to proteins.csv")
```

### Example 2: Export Structures with Chains

```python
from src.models.structure import Structure

structures = db.query(Structure).all()

data = []
for struct in structures:
    for chain in struct.chains:
        data.append({
            "pdb_id": struct.structure_id,
            "title": struct.title,
            "chain_id": chain.chain_id,
            "length": chain.length,
            "uniprot": chain.uniprot_accession,
            "resolution": struct.resolution
        })

df = pd.DataFrame(data)
df.to_csv("structure_chains.csv", index=False)
```

### Example 3: Export for Analysis

```python
from src.utils.transformers import DataExporter

# Export proteins to various formats
proteins = db.query(Protein).filter(
    Protein.organism == "Homo sapiens"
).all()

# FASTA format
DataExporter.proteins_to_fasta(proteins, "human_proteins.fasta")

# GFF format with features
DataExporter.proteins_to_gff(proteins, "human_features.gff")

# Custom JSON with selected fields
import json

export_data = []
for protein in proteins:
    export_data.append({
        "accession": protein.accession,
        "sequence": protein.sequence,
        "features": [
            {
                "type": f.feature_type,
                "start": f.start_position,
                "end": f.end_position,
                "description": f.description
            }
            for f in protein.features
        ],
        "go_terms": [
            {
                "id": go.go_id,
                "name": go.name,
                "namespace": go.namespace
            }
            for go in protein.go_terms
        ]
    })

with open("proteins_detailed.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

## Tips and Best Practices

1. **Start Small**: Test with a few proteins before running large batch jobs
2. **Use Rate Limiting**: The built-in rate limiting prevents API bans
3. **Handle Errors**: Always use try-except blocks for API calls
4. **Save Progress**: For large jobs, save intermediate results
5. **Monitor Usage**: Check database size and API quotas regularly

```python
# Example: Robust batch processing
import time
from src.utils.logging import get_logger

logger = get_logger(__name__)

def safe_collect(accessions, checkpoint_file="progress.txt"):
    """Collect proteins with checkpointing."""
    
    # Load previous progress
    completed = set()
    try:
        with open(checkpoint_file) as f:
            completed = set(line.strip() for line in f)
    except FileNotFoundError:
        pass
    
    collector = UniProtCollector()
    
    for accession in accessions:
        if accession in completed:
            logger.info(f"Skipping {accession} (already done)")
            continue
        
        try:
            protein = collector.process_and_save(accession)
            logger.info(f"✓ Collected {accession}: {protein.protein_name}")
            
            # Save progress
            with open(checkpoint_file, "a") as f:
                f.write(f"{accession}\n")
                
        except Exception as e:
            logger.error(f"✗ Failed {accession}: {e}")
            
        # Be nice to the API
        time.sleep(0.5)
```

## Next Steps

- Explore the [API Documentation](API.md) for detailed method references
- Read the [Configuration Guide](CONFIGURATION.md) for advanced setup
- Check the [Developer Guide](DEVELOPER.md) to extend functionality
- Browse example scripts in the `examples/` directory