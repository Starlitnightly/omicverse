# Data Export Guide

Complete guide to exporting and sharing data from BioinformaticsDataCollector.

## Table of Contents

1. [Export Formats](#export-formats)
2. [Command Line Exports](#command-line-exports)
3. [Python API Exports](#python-api-exports)
4. [Database Exports](#database-exports)
5. [Sharing and Collaboration](#sharing-and-collaboration)
6. [Integration with Analysis Tools](#integration-with-analysis-tools)

## Export Formats

BioinformaticsDataCollector supports multiple export formats for different use cases:

### Standard Formats
- **JSON** - Structured data with full metadata
- **CSV** - Tabular data for spreadsheet analysis
- **FASTA** - Sequence data for bioinformatics tools
- **TSV** - Tab-separated values for data processing
- **XML** - Structured markup for data exchange

### Specialized Formats
- **BioPython** - SeqRecord objects for Python analysis
- **Pandas DataFrame** - For data science workflows
- **BLAST database** - For sequence similarity searches
- **Phylip** - For phylogenetic analysis
- **Stockholm** - For multiple sequence alignments

## Command Line Exports

### Basic Export Commands

#### Export All Proteins
```bash
# Export to JSON
biocollect export database proteins.json --format json --table proteins

# Export to CSV
biocollect export database proteins.csv --format csv --table proteins

# Export sequences only
biocollect export database proteins.fasta --format fasta --table proteins
```

#### Export Specific Data
```bash
# Export human proteins only
biocollect export database human_proteins.json --format json --table proteins --filter organism="Homo sapiens"

# Export high-resolution structures
biocollect export database structures.csv --format csv --table structures --filter "resolution < 2.0"

# Export pathogenic variants
biocollect export database pathogenic.json --format json --table variants --filter "clinical_significance LIKE '%Pathogenic%'"
```

#### Export with Date Range
```bash
# Export recently collected data
biocollect export database recent.json --format json --date-from "2024-01-01" --date-to "2024-12-31"

# Export specific collection run
biocollect export database collection_123.json --format json --run-id 123
```

### Batch Export Scripts

#### Export by Gene List
```bash
#!/bin/bash
# export_gene_list.sh

gene_list=("TP53" "BRCA1" "BRCA2" "ATM" "CHEK2")
output_dir="gene_exports"
mkdir -p $output_dir

for gene in "${gene_list[@]}"; do
    echo "Exporting data for $gene..."
    
    # Export protein data
    biocollect export database "$output_dir/${gene}_proteins.json" \
        --format json --table proteins --filter gene_name="$gene"
    
    # Export variant data
    biocollect export database "$output_dir/${gene}_variants.csv" \
        --format csv --table variants --filter gene_symbol="$gene"
    
    # Export sequence
    biocollect export database "$output_dir/${gene}_sequence.fasta" \
        --format fasta --table proteins --filter gene_name="$gene"
done

echo "Export complete! Files saved to $output_dir/"
```

#### Export by Organism
```bash
#!/bin/bash
# export_by_organism.sh

organisms=("Homo sapiens" "Mus musculus" "Drosophila melanogaster")

for organism in "${organisms[@]}"; do
    # Clean organism name for filename
    clean_name=$(echo "$organism" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    
    echo "Exporting $organism data..."
    
    biocollect export database "${clean_name}_proteins.json" \
        --format json --table proteins --filter organism="$organism"
done
```

## Python API Exports

### Basic Data Export

#### Export Proteins to JSON
```python
import json
from src.models.protein import Protein
from src.models.base import get_db

def export_proteins_json(output_file, organism=None, limit=None):
    """Export proteins to JSON format."""
    with next(get_db()) as db:
        query = db.query(Protein)
        
        if organism:
            query = query.filter(Protein.organism == organism)
        if limit:
            query = query.limit(limit)
        
        proteins = query.all()
        
        # Convert to serializable format
        export_data = []
        for protein in proteins:
            protein_data = {
                'accession': protein.accession,
                'gene_name': protein.gene_name,
                'protein_name': protein.protein_name,
                'organism': protein.organism,
                'sequence': protein.sequence,
                'sequence_length': protein.sequence_length,
                'molecular_weight': protein.molecular_weight,
                'function_description': protein.function_description,
                'has_3d_structure': protein.has_3d_structure == 'Y',
                'pdb_ids': protein.pdb_ids.split(',') if protein.pdb_ids else []
            }
            export_data.append(protein_data)
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return len(export_data)

# Export human proteins
count = export_proteins_json('human_proteins.json', organism='Homo sapiens')
print(f"Exported {count} human proteins")
```

#### Export to Pandas DataFrame
```python
import pandas as pd
from src.models.protein import Protein
from src.models.genomic import Variant
from src.models.base import get_db

def export_to_dataframe(table_name, **filters):
    """Export database table to pandas DataFrame."""
    with next(get_db()) as db:
        if table_name == 'proteins':
            query = db.query(Protein)
            if 'organism' in filters:
                query = query.filter(Protein.organism == filters['organism'])
            
            data = []
            for protein in query.all():
                data.append({
                    'accession': protein.accession,
                    'gene_name': protein.gene_name,
                    'protein_name': protein.protein_name,
                    'organism': protein.organism,
                    'length': protein.sequence_length,
                    'has_structure': protein.has_3d_structure == 'Y'
                })
            
        elif table_name == 'variants':
            query = db.query(Variant)
            if 'gene_symbol' in filters:
                query = query.filter(Variant.gene_symbol == filters['gene_symbol'])
            
            data = []
            for variant in query.all():
                data.append({
                    'rsid': variant.rsid,
                    'gene_symbol': variant.gene_symbol,
                    'chromosome': variant.chromosome,
                    'position': variant.position,
                    'clinical_significance': variant.clinical_significance
                })
        
        return pd.DataFrame(data)

# Usage
df_proteins = export_to_dataframe('proteins', organism='Homo sapiens')
df_variants = export_to_dataframe('variants', gene_symbol='BRCA1')

# Save to CSV
df_proteins.to_csv('human_proteins.csv', index=False)
df_variants.to_csv('brca1_variants.csv', index=False)
```

#### Export Sequences to FASTA
```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from src.models.protein import Protein
from src.models.base import get_db

def export_fasta(output_file, organism=None, gene_list=None, limit=None):
    """Export protein sequences to FASTA format."""
    with next(get_db()) as db:
        query = db.query(Protein)
        
        if organism:
            query = query.filter(Protein.organism == organism)
        if gene_list:
            query = query.filter(Protein.gene_name.in_(gene_list))
        if limit:
            query = query.limit(limit)
        
        proteins = query.all()
        
        # Create SeqRecord objects
        records = []
        for protein in proteins:
            record = SeqRecord(
                Seq(protein.sequence),
                id=protein.accession,
                description=f"{protein.protein_name} [{protein.organism}]"
            )
            records.append(record)
        
        # Write to FASTA file
        SeqIO.write(records, output_file, "fasta")
        
        return len(records)

# Export cancer gene sequences
cancer_genes = ['TP53', 'BRCA1', 'BRCA2', 'RB1', 'APC']
count = export_fasta('cancer_genes.fasta', 
                     organism='Homo sapiens', 
                     gene_list=cancer_genes)
print(f"Exported {count} sequences")
```

### Advanced Export Functions

#### Export with Relationships
```python
def export_protein_with_relationships(accession, output_file):
    """Export protein with all related data."""
    from src.models.protein import Protein, ProteinFeature, GOTerm
    from src.models.structure import Structure
    from src.models.genomic import Variant
    
    with next(get_db()) as db:
        protein = db.query(Protein).filter(
            Protein.accession == accession
        ).first()
        
        if not protein:
            return None
        
        # Collect related data
        export_data = {
            'protein': {
                'accession': protein.accession,
                'gene_name': protein.gene_name,
                'protein_name': protein.protein_name,
                'organism': protein.organism,
                'sequence': protein.sequence,
                'function_description': protein.function_description
            },
            'features': [],
            'go_terms': [],
            'structures': [],
            'variants': []
        }
        
        # Protein features
        features = db.query(ProteinFeature).filter(
            ProteinFeature.protein_id == protein.id
        ).all()
        
        for feature in features:
            export_data['features'].append({
                'type': feature.type,
                'start': feature.start_position,
                'end': feature.end_position,
                'description': feature.description
            })
        
        # GO terms
        for go_term in protein.go_terms:
            export_data['go_terms'].append({
                'id': go_term.go_id,
                'name': go_term.name,
                'category': go_term.category
            })
        
        # Structures
        if protein.pdb_ids:
            pdb_ids = protein.pdb_ids.split(',')
            structures = db.query(Structure).filter(
                Structure.structure_id.in_(pdb_ids)
            ).all()
            
            for structure in structures:
                export_data['structures'].append({
                    'pdb_id': structure.structure_id,
                    'title': structure.title,
                    'method': structure.structure_type,
                    'resolution': structure.resolution
                })
        
        # Related variants
        if protein.gene_name:
            variants = db.query(Variant).filter(
                Variant.gene_symbol == protein.gene_name
            ).limit(50).all()
            
            for variant in variants:
                export_data['variants'].append({
                    'rsid': variant.rsid,
                    'clinical_significance': variant.clinical_significance,
                    'variant_type': variant.variant_type
                })
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data

# Export comprehensive p53 data
data = export_protein_with_relationships('P04637', 'p53_complete.json')
if data:
    print(f"Exported complete data for {data['protein']['protein_name']}")
    print(f"  Features: {len(data['features'])}")
    print(f"  GO terms: {len(data['go_terms'])}")
    print(f"  Structures: {len(data['structures'])}")
    print(f"  Variants: {len(data['variants'])}")
```

#### Export for Specific Analysis Types
```python
def export_for_phylogenetics(gene_name, output_file):
    """Export sequences formatted for phylogenetic analysis."""
    from src.models.protein import Protein
    
    with next(get_db()) as db:
        # Get orthologs across species
        proteins = db.query(Protein).filter(
            Protein.gene_name == gene_name
        ).all()
        
        # Group by species
        species_sequences = {}
        for protein in proteins:
            if protein.organism not in species_sequences:
                species_sequences[protein.organism] = protein
            # Keep the longest sequence if multiple
            elif len(protein.sequence) > len(species_sequences[protein.organism].sequence):
                species_sequences[protein.organism] = protein
        
        # Create records with species names in IDs
        records = []
        for organism, protein in species_sequences.items():
            # Clean organism name for phylip format
            species_code = organism.replace(' ', '_')[:10]  # Phylip limit
            
            record = SeqRecord(
                Seq(protein.sequence),
                id=species_code,
                description=f"{organism} {gene_name}"
            )
            records.append(record)
        
        # Write in Phylip format
        SeqIO.write(records, output_file, "phylip")
        
        return len(records)

# Export TP53 orthologs for phylogenetic analysis
count = export_for_phylogenetics('TP53', 'tp53_orthologs.phy')
print(f"Exported {count} sequences for phylogenetic analysis")
```

## Database Exports

### Full Database Backup
```python
def backup_database(output_dir):
    """Create complete database backup with all tables."""
    import os
    import json
    from datetime import datetime
    from src.models import protein, structure, genomic
    from src.models.base import get_db
    
    os.makedirs(output_dir, exist_ok=True)
    
    backup_info = {
        'backup_date': datetime.now().isoformat(),
        'tables': {}
    }
    
    with next(get_db()) as db:
        # Backup proteins
        proteins = db.query(protein.Protein).all()
        protein_data = []
        for p in proteins:
            protein_data.append({
                'accession': p.accession,
                'gene_name': p.gene_name,
                'protein_name': p.protein_name,
                'organism': p.organism,
                'sequence': p.sequence,
                # ... include all relevant fields
            })
        
        with open(f"{output_dir}/proteins.json", 'w') as f:
            json.dump(protein_data, f, indent=2)
        backup_info['tables']['proteins'] = len(protein_data)
        
        # Backup structures
        structures = db.query(structure.Structure).all()
        structure_data = []
        for s in structures:
            structure_data.append({
                'structure_id': s.structure_id,
                'title': s.title,
                'structure_type': s.structure_type,
                'resolution': s.resolution,
                # ... include all relevant fields
            })
        
        with open(f"{output_dir}/structures.json", 'w') as f:
            json.dump(structure_data, f, indent=2)
        backup_info['tables']['structures'] = len(structure_data)
        
        # Backup variants
        variants = db.query(genomic.Variant).all()
        variant_data = []
        for v in variants:
            variant_data.append({
                'rsid': v.rsid,
                'gene_symbol': v.gene_symbol,
                'clinical_significance': v.clinical_significance,
                # ... include all relevant fields
            })
        
        with open(f"{output_dir}/variants.json", 'w') as f:
            json.dump(variant_data, f, indent=2)
        backup_info['tables']['variants'] = len(variant_data)
    
    # Save backup info
    with open(f"{output_dir}/backup_info.json", 'w') as f:
        json.dump(backup_info, f, indent=2)
    
    return backup_info

# Create full backup
backup_info = backup_database('database_backup_2024')
print("Database backup completed:")
for table, count in backup_info['tables'].items():
    print(f"  {table}: {count} records")
```

### Selective Export
```python
def export_project_data(project_name, gene_list, output_dir):
    """Export data for specific research project."""
    import os
    import json
    from src.models.protein import Protein
    from src.models.genomic import Variant
    from src.models.structure import Structure
    
    os.makedirs(output_dir, exist_ok=True)
    
    project_data = {
        'project_name': project_name,
        'genes': gene_list,
        'export_date': datetime.now().isoformat(),
        'data': {
            'proteins': [],
            'variants': [],
            'structures': []
        }
    }
    
    with next(get_db()) as db:
        # Export proteins
        proteins = db.query(Protein).filter(
            Protein.gene_name.in_(gene_list)
        ).all()
        
        for protein in proteins:
            project_data['data']['proteins'].append({
                'accession': protein.accession,
                'gene_name': protein.gene_name,
                'protein_name': protein.protein_name,
                'organism': protein.organism,
                'sequence_length': protein.sequence_length,
                'function_description': protein.function_description
            })
        
        # Export variants
        variants = db.query(Variant).filter(
            Variant.gene_symbol.in_(gene_list)
        ).all()
        
        for variant in variants:
            project_data['data']['variants'].append({
                'rsid': variant.rsid,
                'gene_symbol': variant.gene_symbol,
                'clinical_significance': variant.clinical_significance,
                'variant_type': variant.variant_type
            })
        
        # Export relevant structures
        pdb_ids = []
        for protein in proteins:
            if protein.pdb_ids:
                pdb_ids.extend(protein.pdb_ids.split(','))
        
        if pdb_ids:
            structures = db.query(Structure).filter(
                Structure.structure_id.in_(pdb_ids)
            ).all()
            
            for structure in structures:
                project_data['data']['structures'].append({
                    'structure_id': structure.structure_id,
                    'title': structure.title,
                    'resolution': structure.resolution,
                    'structure_type': structure.structure_type
                })
    
    # Save project data
    output_file = f"{output_dir}/{project_name}_data.json"
    with open(output_file, 'w') as f:
        json.dump(project_data, f, indent=2)
    
    return project_data

# Export cancer research project data
cancer_genes = ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'CHEK2']
data = export_project_data('cancer_study_2024', cancer_genes, 'cancer_project')
print(f"Exported project data:")
print(f"  Proteins: {len(data['data']['proteins'])}")
print(f"  Variants: {len(data['data']['variants'])}")
print(f"  Structures: {len(data['data']['structures'])}")
```

## Sharing and Collaboration

### Create Shareable Dataset
```python
def create_shareable_dataset(name, description, gene_list, include_sequences=False):
    """Create a dataset package for sharing with collaborators."""
    import zipfile
    import tempfile
    import shutil
    from datetime import datetime
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = f"{temp_dir}/{name}"
        os.makedirs(dataset_dir)
        
        # Export data
        project_data = export_project_data(name, gene_list, dataset_dir)
        
        # Export sequences if requested
        if include_sequences:
            count = export_fasta(
                f"{dataset_dir}/sequences.fasta",
                gene_list=gene_list
            )
            print(f"Included {count} sequences")
        
        # Create README
        readme_content = f"""# {name}

## Description
{description}

## Contents
- Proteins: {len(project_data['data']['proteins'])}
- Variants: {len(project_data['data']['variants'])}
- Structures: {len(project_data['data']['structures'])}

## Genes Included
{', '.join(gene_list)}

## Files
- `{name}_data.json`: Complete dataset in JSON format
- `sequences.fasta`: Protein sequences (if included)
- `README.md`: This file

## Data Sources
Data collected from:
- UniProt: Protein sequences and annotations
- ClinVar: Clinical variant interpretations
- PDB: Protein structure information

## Usage
```python
import json
with open('{name}_data.json', 'r') as f:
    data = json.load(f)

# Access proteins
proteins = data['data']['proteins']
for protein in proteins:
    print(protein['gene_name'], protein['protein_name'])
```

## Citation
When using this dataset, please cite BioinformaticsDataCollector and the original data sources.

Generated: {project_data['export_date']}
"""
        
        with open(f"{dataset_dir}/README.md", 'w') as f:
            f.write(readme_content)
        
        # Create ZIP package
        zip_filename = f"{name}_dataset.zip"
        shutil.make_archive(name + "_dataset", 'zip', temp_dir, name)
        
        return zip_filename

# Create shareable cancer dataset
zip_file = create_shareable_dataset(
    name="cancer_genes_2024",
    description="Comprehensive data for major cancer-associated genes",
    gene_list=['TP53', 'BRCA1', 'BRCA2', 'ATM', 'CHEK2'],
    include_sequences=True
)
print(f"Created shareable dataset: {zip_file}")
```

### Version Control for Datasets
```python
def version_dataset(dataset_name, version, changes_description):
    """Create versioned dataset with change tracking."""
    import hashlib
    import json
    from datetime import datetime
    
    # Export current data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = f"datasets/{dataset_name}/v{version}_{timestamp}"
    
    # Create version info
    version_info = {
        'dataset_name': dataset_name,
        'version': version,
        'timestamp': timestamp,
        'changes': changes_description,
        'checksums': {}
    }
    
    # Export data and calculate checksums
    export_files = ['proteins.json', 'variants.json', 'structures.json']
    
    for filename in export_files:
        filepath = f"{version_dir}/{filename}"
        # Export data logic here...
        
        # Calculate checksum
        with open(filepath, 'rb') as f:
            content = f.read()
            checksum = hashlib.sha256(content).hexdigest()
            version_info['checksums'][filename] = checksum
    
    # Save version info
    with open(f"{version_dir}/version_info.json", 'w') as f:
        json.dump(version_info, f, indent=2)
    
    return version_dir

# Create versioned dataset
version_dir = version_dataset(
    "cancer_genes",
    "2.1",
    "Added CHEK2 gene data and updated variant classifications"
)
print(f"Created dataset version: {version_dir}")
```

## Integration with Analysis Tools

### Export for R Analysis
```python
def export_for_r(output_file, table_name='proteins'):
    """Export data in R-friendly CSV format."""
    df = export_to_dataframe(table_name)
    
    # Clean column names for R (no spaces, special characters)
    df.columns = [col.replace(' ', '_').replace('-', '_').lower() 
                  for col in df.columns]
    
    # Save with proper encoding
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Create R script template
    r_script = f"""# R Analysis Script for {table_name}
# Load data
data <- read.csv("{output_file}", stringsAsFactors = FALSE)

# Basic exploration
str(data)
summary(data)

# Example analysis
if ("sequence_length" %in% colnames(data)) {{
    hist(data$sequence_length, 
         main = "Protein Length Distribution",
         xlab = "Sequence Length (amino acids)")
}}

# Save results
# write.csv(results, "analysis_results.csv", row.names = FALSE)
"""
    
    with open(output_file.replace('.csv', '.R'), 'w') as f:
        f.write(r_script)
    
    print(f"Created R-ready dataset: {output_file}")
    print(f"Created R script template: {output_file.replace('.csv', '.R')}")

# Export for R
export_for_r('proteins_for_r.csv', 'proteins')
```

### Export for Cytoscape (Network Analysis)
```python
def export_for_cytoscape(gene_list, output_prefix):
    """Export protein interaction network for Cytoscape."""
    from src.models.protein import Protein
    from src.models.base import get_db
    import pandas as pd
    
    # Create nodes file
    nodes = []
    edges = []
    
    with next(get_db()) as db:
        proteins = db.query(Protein).filter(
            Protein.gene_name.in_(gene_list)
        ).all()
        
        for protein in proteins:
            # Node data
            nodes.append({
                'id': protein.gene_name,
                'accession': protein.accession,
                'protein_name': protein.protein_name,
                'sequence_length': protein.sequence_length,
                'has_structure': protein.has_3d_structure == 'Y'
            })
            
            # Edge data (if interaction data exists)
            # This would require interaction data in database
            # For now, create example edges between cancer genes
            if protein.gene_name in ['TP53', 'BRCA1', 'BRCA2']:
                for other_gene in ['TP53', 'BRCA1', 'BRCA2']:
                    if protein.gene_name != other_gene:
                        edges.append({
                            'source': protein.gene_name,
                            'target': other_gene,
                            'interaction_type': 'protein-protein'
                        })
    
    # Save nodes file
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f"{output_prefix}_nodes.csv", index=False)
    
    # Save edges file
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(f"{output_prefix}_edges.csv", index=False)
    
    print(f"Created Cytoscape files:")
    print(f"  Nodes: {output_prefix}_nodes.csv ({len(nodes)} nodes)")
    print(f"  Edges: {output_prefix}_edges.csv ({len(edges)} edges)")

# Export cancer gene network
cancer_genes = ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'CHEK2']
export_for_cytoscape(cancer_genes, 'cancer_network')
```

### Export for Machine Learning
```python
def export_for_ml(output_file, feature_type='sequence'):
    """Export data formatted for machine learning."""
    from sklearn.preprocessing import LabelEncoder
    from src.models.protein import Protein
    
    with next(get_db()) as db:
        proteins = db.query(Protein).all()
        
        if feature_type == 'sequence':
            # Sequence-based features
            data = []
            for protein in proteins:
                # Basic sequence features
                seq = protein.sequence
                features = {
                    'accession': protein.accession,
                    'length': len(seq),
                    'molecular_weight': protein.molecular_weight or 0,
                    'hydrophobic_ratio': seq.count('A') + seq.count('I') + seq.count('L') + seq.count('V') / len(seq),
                    'charged_ratio': seq.count('K') + seq.count('R') + seq.count('D') + seq.count('E') / len(seq),
                    'aromatic_ratio': seq.count('F') + seq.count('W') + seq.count('Y') / len(seq),
                    'has_structure': 1 if protein.has_3d_structure == 'Y' else 0,
                    'organism': protein.organism
                }
                data.append(features)
            
        # Convert to DataFrame and encode categorical variables
        df = pd.DataFrame(data)
        
        # Encode organism labels
        le = LabelEncoder()
        df['organism_encoded'] = le.fit_transform(df['organism'])
        
        # Save data and label encoder
        df.to_csv(output_file, index=False)
        
        # Save label encoder
        import pickle
        with open(output_file.replace('.csv', '_label_encoder.pkl'), 'wb') as f:
            pickle.dump(le, f)
        
        print(f"Created ML dataset: {output_file}")
        print(f"Features: {list(df.columns)}")
        print(f"Samples: {len(df)}")

# Export for machine learning
export_for_ml('protein_features_ml.csv', 'sequence')
```

This comprehensive export guide provides users with multiple options for getting their data out of BioinformaticsDataCollector and into their preferred analysis tools and workflows.