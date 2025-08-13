# Database Guide

Complete guide to working with collected biological data in BioinformaticsDataCollector.

## Database Overview

BioinformaticsDataCollector uses SQLAlchemy ORM with support for:
- **SQLite** (default) - Single-user, file-based database
- **PostgreSQL** (recommended for production) - Multi-user, server-based database

### Database Schema

The database is organized into logical groups:

#### Core Tables
- **proteins** - UniProt protein entries
- **structures** - PDB and AlphaFold structures
- **genes** - Genomic information from Ensembl
- **variants** - Genetic variants from ClinVar/dbSNP

#### Annotation Tables  
- **protein_features** - Protein domains, motifs, modifications
- **go_terms** - Gene Ontology annotations
- **interactions** - Protein-protein interactions
- **pathways** - Biological pathway memberships

#### Metadata Tables
- **collection_runs** - Track when data was collected
- **data_sources** - Source database information

## Getting Started

### Initialize Database
```python
from src.utils.database import initialize_database

# Create all tables
initialize_database()
```

### Database Connection
```python
from src.models.base import get_db

# Get database session
with next(get_db()) as db:
    # Your database operations here
    pass
```

## Basic Queries

### Working with Proteins

#### Find Specific Proteins
```python
from src.models.protein import Protein
from src.models.base import get_db

with next(get_db()) as db:
    # By UniProt accession
    p53 = db.query(Protein).filter(Protein.accession == "P04637").first()
    
    # By gene name
    tp53_proteins = db.query(Protein).filter(Protein.gene_name == "TP53").all()
    
    # By organism
    human_proteins = db.query(Protein).filter(
        Protein.organism == "Homo sapiens"
    ).limit(100).all()
```

#### Search by Name/Function
```python
# Proteins containing "kinase" in name
kinases = db.query(Protein).filter(
    Protein.protein_name.contains("kinase")
).all()

# Proteins with DNA binding function
dna_binding = db.query(Protein).filter(
    Protein.function_description.contains("DNA binding")
).all()

# Case-insensitive search
cancer_related = db.query(Protein).filter(
    Protein.protein_name.ilike("%tumor%")
).all()
```

#### Filter by Properties
```python
# Proteins with 3D structures
structured = db.query(Protein).filter(
    Protein.has_3d_structure == "Y"
).all()

# Large proteins (>1000 amino acids)
large_proteins = db.query(Protein).filter(
    Protein.sequence_length > 1000
).all()

# Proteins from specific taxonomy
mammals = db.query(Protein).filter(
    Protein.organism_id.in_([9606, 10090, 10116])  # human, mouse, rat
).all()
```

### Working with Structures

#### Find Structures
```python
from src.models.structure import Structure, Chain, Ligand

with next(get_db()) as db:
    # By PDB ID
    structure = db.query(Structure).filter(
        Structure.structure_id == "1A3N"
    ).first()
    
    # High-resolution structures
    high_res = db.query(Structure).filter(
        Structure.resolution < 2.0,
        Structure.resolution.isnot(None)
    ).all()
    
    # By experimental method
    xray_structures = db.query(Structure).filter(
        Structure.structure_type == "X-RAY DIFFRACTION"
    ).all()
```

#### Access Related Data
```python
# Get structure with chains and ligands
structure = db.query(Structure).filter(
    Structure.structure_id == "1A3N"
).first()

print(f"Title: {structure.title}")
print(f"Chains: {len(structure.chains)}")
print(f"Ligands: {len(structure.ligands)}")

# Chain information
for chain in structure.chains:
    print(f"Chain {chain.chain_id}: {chain.length} residues")

# Ligand information  
for ligand in structure.ligands:
    print(f"Ligand {ligand.ligand_id}: {ligand.name}")
```

### Working with Variants

#### Clinical Variants
```python
from src.models.genomic import Variant

with next(get_db()) as db:
    # Pathogenic variants
    pathogenic = db.query(Variant).filter(
        Variant.clinical_significance.contains("Pathogenic")
    ).all()
    
    # Variants in specific gene
    brca1_variants = db.query(Variant).filter(
        Variant.gene_symbol == "BRCA1"
    ).all()
    
    # Variants by chromosome
    chr17_variants = db.query(Variant).filter(
        Variant.chromosome == "17"
    ).all()
```

#### Population Variants
```python
# Common variants (MAF > 1%)
common_variants = db.query(Variant).filter(
    Variant.minor_allele_frequency > 0.01
).all()

# Rare variants (MAF < 0.1%)
rare_variants = db.query(Variant).filter(
    Variant.minor_allele_frequency < 0.001,
    Variant.minor_allele_frequency.isnot(None)
).all()
```

## Advanced Queries

### Joins and Relationships

#### Protein-Structure Relationships
```python
from sqlalchemy.orm import joinedload

# Load protein with related features
protein = db.query(Protein).options(
    joinedload(Protein.features),
    joinedload(Protein.go_terms)
).filter(Protein.accession == "P04637").first()

# Access related data without additional queries
for feature in protein.features:
    print(f"Feature: {feature.type} at {feature.start_position}-{feature.end_position}")

for go_term in protein.go_terms:
    print(f"GO: {go_term.go_id} - {go_term.name}")
```

#### Complex Joins
```python
from sqlalchemy import and_, or_
from src.models.protein import Protein
from src.models.structure import Structure

# Proteins with high-resolution structures
query = db.query(Protein, Structure).join(
    Structure, 
    Protein.accession == Structure.uniprot_accession
).filter(
    Structure.resolution < 2.0
)

for protein, structure in query.all():
    print(f"{protein.accession}: {structure.structure_id} ({structure.resolution}Å)")
```

### Aggregation Queries

#### Count Statistics
```python
from sqlalchemy import func, distinct

# Count proteins by organism
organism_counts = db.query(
    Protein.organism,
    func.count(Protein.id).label('count')
).group_by(Protein.organism).order_by(
    func.count(Protein.id).desc()
).limit(10).all()

for organism, count in organism_counts:
    print(f"{organism}: {count} proteins")

# Average protein length by organism
avg_lengths = db.query(
    Protein.organism,
    func.avg(Protein.sequence_length).label('avg_length')
).group_by(Protein.organism).all()
```

#### Statistical Analysis
```python
# Protein length distribution
from sqlalchemy import case

length_distribution = db.query(
    case([
        (Protein.sequence_length < 100, 'Small'),
        (Protein.sequence_length < 500, 'Medium'),
        (Protein.sequence_length < 1000, 'Large'),
    ], else_='Very Large').label('size_category'),
    func.count(Protein.id).label('count')
).group_by('size_category').all()

for category, count in length_distribution:
    print(f"{category}: {count}")
```

## Data Analysis Examples

### Protein Family Analysis
```python
def analyze_protein_family(gene_symbols):
    """Analyze a family of related proteins."""
    with next(get_db()) as db:
        proteins = db.query(Protein).filter(
            Protein.gene_name.in_(gene_symbols)
        ).all()
        
        analysis = {
            'count': len(proteins),
            'organisms': set(p.organism for p in proteins),
            'avg_length': sum(p.sequence_length for p in proteins) / len(proteins),
            'with_structures': sum(1 for p in proteins if p.has_3d_structure == 'Y'),
            'functions': [p.function_description for p in proteins if p.function_description]
        }
        
        return analysis

# Analyze tumor suppressors
tumor_suppressors = ['TP53', 'RB1', 'APC', 'BRCA1', 'BRCA2']
analysis = analyze_protein_family(tumor_suppressors)
print(f"Found {analysis['count']} proteins")
print(f"Structure coverage: {analysis['with_structures']/analysis['count']*100:.1f}%")
```

### Pathway Enrichment
```python
def find_pathway_proteins(pathway_keywords):
    """Find proteins associated with specific pathways."""
    with next(get_db()) as db:
        proteins = db.query(Protein).filter(
            or_(*[Protein.function_description.contains(keyword) 
                  for keyword in pathway_keywords])
        ).all()
        
        return proteins

# DNA repair proteins
dna_repair = find_pathway_proteins(['DNA repair', 'DNA damage', 'homologous recombination'])
print(f"Found {len(dna_repair)} DNA repair proteins")
```

### Cross-Database Analysis
```python
def structure_coverage_analysis():
    """Analyze protein structure coverage."""
    with next(get_db()) as db:
        # Proteins with and without structures
        total = db.query(Protein).count()
        with_pdb = db.query(Protein).filter(Protein.pdb_ids.isnot(None)).count()
        with_alphafold = db.query(Protein).filter(
            Protein.has_3d_structure == 'Y'
        ).count()
        
        # Structure quality distribution
        structures = db.query(Structure).filter(
            Structure.resolution.isnot(None)
        ).all()
        
        resolutions = [s.resolution for s in structures]
        avg_resolution = sum(resolutions) / len(resolutions)
        
        return {
            'total_proteins': total,
            'pdb_coverage': with_pdb / total * 100,
            'alphafold_coverage': with_alphafold / total * 100,
            'avg_resolution': avg_resolution,
            'total_structures': len(structures)
        }

stats = structure_coverage_analysis()
print(f"PDB coverage: {stats['pdb_coverage']:.1f}%")
print(f"AlphaFold coverage: {stats['alphafold_coverage']:.1f}%")
```

## Data Export

### Export to Pandas DataFrame
```python
import pandas as pd
from src.models.protein import Protein

def proteins_to_dataframe(organism=None, limit=None):
    """Export proteins to pandas DataFrame."""
    with next(get_db()) as db:
        query = db.query(Protein)
        
        if organism:
            query = query.filter(Protein.organism == organism)
        if limit:
            query = query.limit(limit)
            
        proteins = query.all()
        
        data = []
        for protein in proteins:
            data.append({
                'accession': protein.accession,
                'gene_name': protein.gene_name,
                'protein_name': protein.protein_name,
                'organism': protein.organism,
                'length': protein.sequence_length,
                'has_structure': protein.has_3d_structure == 'Y',
                'pdb_count': len(protein.pdb_ids.split(',')) if protein.pdb_ids else 0
            })
    
    return pd.DataFrame(data)

# Export human proteins
df = proteins_to_dataframe(organism="Homo sapiens", limit=1000)
print(df.head())
```

### Export Sequences
```python
def export_fasta(organism=None, output_file=None):
    """Export protein sequences in FASTA format."""
    with next(get_db()) as db:
        query = db.query(Protein)
        if organism:
            query = query.filter(Protein.organism == organism)
            
        proteins = query.all()
        
        fasta_content = []
        for protein in proteins:
            header = f">{protein.accession} {protein.protein_name}"
            fasta_content.append(header)
            # Split sequence into 80-character lines
            seq = protein.sequence
            for i in range(0, len(seq), 80):
                fasta_content.append(seq[i:i+80])
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(fasta_content))
        
        return '\n'.join(fasta_content)

# Export human protein sequences
export_fasta(organism="Homo sapiens", output_file="human_proteins.fasta")
```

## Database Maintenance

### Statistics and Monitoring
```python
def get_database_statistics():
    """Get comprehensive database statistics."""
    from src.utils.database import get_table_stats
    
    with next(get_db()) as db:
        stats = get_table_stats(db)
        
        # Additional analysis
        organism_counts = db.query(
            Protein.organism,
            func.count(Protein.id)
        ).group_by(Protein.organism).all()
        
        structure_types = db.query(
            Structure.structure_type,
            func.count(Structure.id)
        ).group_by(Structure.structure_type).all()
        
        return {
            'table_counts': stats,
            'top_organisms': organism_counts[:10],
            'structure_types': structure_types
        }

stats = get_database_statistics()
print("Table counts:", stats['table_counts'])
print("Top organisms:", stats['top_organisms'][:5])
```

### Data Cleanup
```python
def cleanup_duplicate_proteins():
    """Remove duplicate protein entries."""
    with next(get_db()) as db:
        # Find duplicates by accession
        duplicates = db.query(
            Protein.accession,
            func.count(Protein.id).label('count')
        ).group_by(Protein.accession).having(
            func.count(Protein.id) > 1
        ).all()
        
        removed_count = 0
        for accession, count in duplicates:
            # Keep the first entry, remove others
            proteins = db.query(Protein).filter(
                Protein.accession == accession
            ).all()
            
            for protein in proteins[1:]:  # Skip first
                db.delete(protein)
                removed_count += 1
        
        db.commit()
        return removed_count

# Run cleanup
removed = cleanup_duplicate_proteins()
print(f"Removed {removed} duplicate proteins")
```

### Backup and Recovery
```bash
# For SQLite
cp biocollect.db biocollect_backup.db

# For PostgreSQL
pg_dump biocollect > biocollect_backup.sql
```

## Performance Optimization

### Query Optimization
```python
# Use indexes for common queries
from sqlalchemy import Index
from src.models.protein import Protein

# Add custom indexes
Index('ix_protein_organism_gene', Protein.organism, Protein.gene_name)
Index('ix_protein_length', Protein.sequence_length)
```

### Bulk Operations
```python
def bulk_update_structure_flags():
    """Efficiently update structure flags for all proteins."""
    with next(get_db()) as db:
        # Update all proteins that have PDB IDs
        db.query(Protein).filter(
            Protein.pdb_ids.isnot(None)
        ).update({
            Protein.has_3d_structure: 'Y'
        })
        
        # Update all proteins without PDB IDs
        db.query(Protein).filter(
            Protein.pdb_ids.is_(None)
        ).update({
            Protein.has_3d_structure: 'N'
        })
        
        db.commit()

# Run bulk update
bulk_update_structure_flags()
```

### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimized connection for high-throughput applications
engine = create_engine(
    "postgresql://user:pass@host/db",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600
)
```

## Integration Examples

### With BioPython
```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def create_biopython_records():
    """Convert database proteins to BioPython SeqRecords."""
    with next(get_db()) as db:
        proteins = db.query(Protein).limit(10).all()
        
        records = []
        for protein in proteins:
            seq = Seq(protein.sequence)
            record = SeqRecord(
                seq,
                id=protein.accession,
                description=protein.protein_name or ""
            )
            records.append(record)
        
        return records

# Export to GenBank format
records = create_biopython_records()
SeqIO.write(records, "proteins.gb", "genbank")
```

### With Machine Learning
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_function_features():
    """Extract features from protein function descriptions."""
    with next(get_db()) as db:
        proteins = db.query(Protein).filter(
            Protein.function_description.isnot(None)
        ).all()
        
        descriptions = [p.function_description for p in proteins]
        accessions = [p.accession for p in proteins]
        
        # TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        features = vectorizer.fit_transform(descriptions)
        
        return features, accessions, vectorizer.get_feature_names_out()

# Get features for clustering or classification
features, accessions, feature_names = extract_function_features()
print(f"Extracted {features.shape[1]} features from {features.shape[0]} proteins")
```

## Troubleshooting

### Common Database Issues

#### Connection Errors
```python
# Test database connection
from src.utils.database import check_database_connection

if not check_database_connection():
    print("Database connection failed!")
    print("Try: biocollect init")
```

#### Lock Errors (SQLite)
```python
# Handle SQLite locking gracefully
import time
from sqlalchemy.exc import OperationalError

def safe_database_operation(operation, max_retries=3):
    """Retry database operations on lock errors."""
    for attempt in range(max_retries):
        try:
            return operation()
        except OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second
                continue
            raise
```

#### Memory Issues
```python
# Stream large result sets
def stream_large_query():
    """Process large result sets without loading all into memory."""
    with next(get_db()) as db:
        # Use yield_per to stream results
        for protein in db.query(Protein).yield_per(100):
            # Process one protein at a time
            process_protein(protein)
```

#### Query Performance
```python
# Profile slow queries
import time
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute") 
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    if total > 1.0:  # Log slow queries
        print(f"Slow query: {total:.2f}s - {statement[:100]}...")
```