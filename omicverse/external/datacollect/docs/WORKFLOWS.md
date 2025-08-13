# Common Workflows

Real-world usage examples and workflows for BioinformaticsDataCollector.

## Table of Contents

1. [Research Project Workflows](#research-project-workflows)
2. [Data Integration Workflows](#data-integration-workflows) 
3. [Analysis Workflows](#analysis-workflows)
4. [Automation Workflows](#automation-workflows)
5. [Collaboration Workflows](#collaboration-workflows)

## Research Project Workflows

### 1. Cancer Gene Analysis Project

**Scenario**: You're studying tumor suppressor genes and need comprehensive data about TP53, BRCA1, BRCA2, and their interactions.

#### Step 1: Collect Core Protein Data
```bash
# Collect primary protein information
biocollect collect uniprot P04637 --save-file  # TP53
biocollect collect uniprot P38398 --save-file  # BRCA1  
biocollect collect uniprot P51587 --save-file  # BRCA2

# Alternative: Search by gene symbol
biocollect collect uniprot-search "TP53 AND reviewed:yes AND organism_id:9606" --limit 1
biocollect collect uniprot-search "BRCA1 AND reviewed:yes AND organism_id:9606" --limit 1
biocollect collect uniprot-search "BRCA2 AND reviewed:yes AND organism_id:9606" --limit 1
```

#### Step 2: Collect Structural Information
```bash
# Find and collect available structures
biocollect collect pdb-blast "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD" --e-value 0.001

# Collect AlphaFold predictions
biocollect collect alphafold P04637 --download --download-pae
biocollect collect alphafold P38398 --download --download-pae  
biocollect collect alphafold P51587 --download --download-pae
```

#### Step 3: Collect Interaction Networks
```bash
# Protein-protein interactions
biocollect collect string TP53 --partner-limit 50 --save-file
biocollect collect string BRCA1 --partner-limit 50 --save-file
biocollect collect string BRCA2 --partner-limit 50 --save-file
```

#### Step 4: Collect Variant Information
```bash
# Clinical variants
biocollect collect clinvar TP53 --id-type gene --pathogenic-only --limit 200 --save-file
biocollect collect clinvar BRCA1 --id-type gene --pathogenic-only --limit 200 --save-file
biocollect collect clinvar BRCA2 --id-type gene --pathogenic-only --limit 200 --save-file

# Population variants
biocollect collect dbsnp-gene TP53 --limit 100 --save-file
biocollect collect dbsnp-gene BRCA1 --limit 100 --save-file  
biocollect collect dbsnp-gene BRCA2 --limit 100 --save-file
```

#### Step 5: Collect Expression Data
```bash
# Find relevant expression datasets
biocollect collect geo-search TP53 --organism "Homo sapiens" --limit 10 --save-file
biocollect collect geo-search BRCA1 --organism "Homo sapiens" --limit 10 --save-file
biocollect collect geo-search BRCA2 --organism "Homo sapiens" --limit 10 --save-file

# Collect specific datasets of interest
biocollect collect geo GSE123456 --save-file  # Replace with actual GSE numbers
```

#### Python Analysis Script
```python
# analyze_cancer_genes.py
from src.models.protein import Protein
from src.models.genomic import Variant
from src.models.base import get_db
import pandas as pd

def analyze_cancer_genes():
    """Analyze collected cancer gene data."""
    cancer_genes = ['TP53', 'BRCA1', 'BRCA2']
    
    with next(get_db()) as db:
        # Get protein information
        proteins = db.query(Protein).filter(
            Protein.gene_name.in_(cancer_genes)
        ).all()
        
        # Get variants
        pathogenic_variants = db.query(Variant).filter(
            Variant.gene_symbol.in_(cancer_genes),
            Variant.clinical_significance.contains('Pathogenic')
        ).all()
        
        # Analysis
        results = {}
        for gene in cancer_genes:
            gene_protein = next((p for p in proteins if p.gene_name == gene), None)
            gene_variants = [v for v in pathogenic_variants if v.gene_symbol == gene]
            
            results[gene] = {
                'protein_length': gene_protein.sequence_length if gene_protein else None,
                'has_structure': gene_protein.has_3d_structure == 'Y' if gene_protein else False,
                'pathogenic_variants': len(gene_variants),
                'function': gene_protein.function_description[:100] if gene_protein else None
            }
        
        return results

# Run analysis
results = analyze_cancer_genes()
for gene, data in results.items():
    print(f"\n{gene}:")
    print(f"  Length: {data['protein_length']} aa")
    print(f"  Structure: {data['has_structure']}")
    print(f"  Pathogenic variants: {data['pathogenic_variants']}")
```

### 2. Drug Target Discovery Project

**Scenario**: You're identifying potential drug targets in a metabolic pathway.

#### Workflow Script
```bash
#!/bin/bash
# drug_target_workflow.sh

# Define pathway of interest (e.g., glycolysis)
pathway="hsa00010"  # KEGG glycolysis pathway

echo "Collecting pathway data..."
biocollect collect kegg $pathway --save-file

# Key enzymes in glycolysis
enzymes=("HK1" "GPI" "PFKL" "ALDOA" "TPI1" "GAPDH" "PGK1" "PGAM1" "ENO1" "PKM")

echo "Collecting enzyme data..."
for enzyme in "${enzymes[@]}"; do
    echo "Processing $enzyme..."
    
    # Protein data
    biocollect collect uniprot-search "$enzyme AND reviewed:yes AND organism_id:9606" --limit 1
    
    # Structure data (for druggability assessment)
    biocollect collect alphafold $enzyme --download
    
    # Expression data (tissue specificity)
    biocollect collect geo-search $enzyme --limit 5
    
    # Known drug interactions
    biocollect collect string $enzyme --partner-limit 20
done

echo "Workflow complete!"
```

### 3. Comparative Genomics Project

**Scenario**: Compare orthologous proteins across species.

#### Multi-Species Collection
```python
# comparative_genomics.py
from src.collectors.uniprot_collector import UniProtCollector

def collect_orthologs(gene_name, species_ids):
    """Collect orthologous proteins across species."""
    collector = UniProtCollector()
    orthologs = []
    
    for species_id in species_ids:
        try:
            proteins = collector.search_and_collect(
                f"{gene_name} AND organism_id:{species_id}",
                max_results=1
            )
            if proteins:
                orthologs.append(proteins[0])
        except Exception as e:
            print(f"Failed to collect {gene_name} for species {species_id}: {e}")
    
    return orthologs

# Collect TP53 orthologs across mammals
species = {
    9606: "Human",
    10090: "Mouse", 
    10116: "Rat",
    9913: "Cattle",
    9615: "Dog",
    9544: "Macaque"
}

tp53_orthologs = collect_orthologs("TP53", species.keys())

# Analyze sequence conservation
for protein in tp53_orthologs:
    print(f"{protein.organism}: {protein.sequence_length} aa")
```

## Data Integration Workflows

### 1. Multi-Database Integration

**Scenario**: Integrate data from multiple databases for a comprehensive protein profile.

```python
# protein_integration.py
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector
from src.collectors.string_collector import STRINGCollector
from src.collectors.clinvar_collector import ClinVarCollector

def comprehensive_protein_profile(uniprot_id):
    """Create comprehensive protein profile from multiple databases."""
    
    # Initialize collectors
    uniprot_collector = UniProtCollector()
    pdb_collector = PDBCollector()
    string_collector = STRINGCollector()
    clinvar_collector = ClinVarCollector()
    
    profile = {}
    
    # UniProt data (primary)
    print(f"Collecting UniProt data for {uniprot_id}...")
    protein = uniprot_collector.process_and_save(uniprot_id)
    profile['protein'] = protein
    
    # PDB structures
    if protein.pdb_ids:
        print("Collecting PDB structures...")
        pdb_ids = protein.pdb_ids.split(',')[:3]  # First 3 structures
        structures = []
        for pdb_id in pdb_ids:
            structure = pdb_collector.process_and_save(
                pdb_id.strip(),
                download_structure=True
            )
            structures.append(structure)
        profile['structures'] = structures
    
    # Protein interactions
    if protein.gene_name:
        print("Collecting protein interactions...")
        string_data = string_collector.collect_single(protein.gene_name, 9606)
        interactions = string_collector.save_to_database(string_data)
        profile['interactions'] = interactions
    
    # Clinical variants
    if protein.gene_name:
        print("Collecting clinical variants...")
        variants_data = clinvar_collector.collect_by_gene(
            protein.gene_name, 
            pathogenic_only=True,
            limit=50
        )
        variants = clinvar_collector.save_variants_to_database(variants_data)
        profile['variants'] = variants
    
    return profile

# Usage
profile = comprehensive_protein_profile("P04637")  # TP53
print(f"Collected data for {profile['protein'].protein_name}")
print(f"  Structures: {len(profile.get('structures', []))}")
print(f"  Interactions: {len(profile.get('interactions', []))}")
print(f"  Variants: {len(profile.get('variants', []))}")
```

### 2. Cross-Reference Validation

**Scenario**: Validate data consistency across databases.

```python
# validation_workflow.py
def validate_cross_references(uniprot_id):
    """Validate cross-references between databases."""
    from src.models.protein import Protein
    from src.models.structure import Structure
    from src.models.base import get_db
    
    with next(get_db()) as db:
        protein = db.query(Protein).filter(
            Protein.accession == uniprot_id
        ).first()
        
        if not protein:
            return {"error": "Protein not found"}
        
        validation = {
            "uniprot_id": protein.accession,
            "gene_name": protein.gene_name,
            "pdb_ids_claimed": protein.pdb_ids.split(',') if protein.pdb_ids else [],
            "pdb_ids_found": [],
            "discrepancies": []
        }
        
        # Check if claimed PDB structures exist in database
        for pdb_id in validation["pdb_ids_claimed"]:
            structure = db.query(Structure).filter(
                Structure.structure_id == pdb_id.strip()
            ).first()
            
            if structure:
                validation["pdb_ids_found"].append(pdb_id.strip())
            else:
                validation["discrepancies"].append(
                    f"PDB {pdb_id.strip()} not found in structures table"
                )
        
        return validation

# Validate multiple proteins
proteins_to_validate = ["P04637", "P38398", "P51587"]
for uniprot_id in proteins_to_validate:
    result = validate_cross_references(uniprot_id)
    print(f"\n{uniprot_id} validation:")
    print(f"  Found {len(result.get('pdb_ids_found', []))} of {len(result.get('pdb_ids_claimed', []))} claimed structures")
    if result.get('discrepancies'):
        print(f"  Discrepancies: {result['discrepancies']}")
```

## Analysis Workflows

### 1. Phylogenetic Analysis Workflow

```python
# phylogenetic_workflow.py
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from src.models.protein import Protein
from src.models.base import get_db

def prepare_phylogenetic_analysis(gene_name, species_list):
    """Prepare sequences for phylogenetic analysis."""
    
    with next(get_db()) as db:
        # Get orthologs for specified species
        orthologs = db.query(Protein).filter(
            Protein.gene_name == gene_name,
            Protein.organism.in_(species_list)
        ).all()
        
        # Create SeqRecord objects
        sequences = []
        for protein in orthologs:
            record = SeqRecord(
                Seq(protein.sequence),
                id=protein.accession,
                description=f"{protein.organism}_{protein.gene_name}"
            )
            sequences.append(record)
        
        # Save to FASTA file
        output_file = f"{gene_name}_orthologs.fasta"
        SeqIO.write(sequences, output_file, "fasta")
        
        return sequences, output_file

# Usage
species = [
    "Homo sapiens",
    "Mus musculus", 
    "Rattus norvegicus",
    "Macaca mulatta",
    "Pan troglodytes"
]

sequences, fasta_file = prepare_phylogenetic_analysis("TP53", species)
print(f"Created {fasta_file} with {len(sequences)} sequences")
print("Next steps:")
print("1. Run multiple sequence alignment (e.g., ClustalW, MUSCLE)")
print("2. Build phylogenetic tree (e.g., MEGA, RAxML)")
print("3. Visualize and analyze tree topology")
```

### 2. Functional Domain Analysis

```python
# domain_analysis.py
def analyze_protein_domains(gene_list):
    """Analyze domain composition across protein family."""
    from src.models.protein import Protein, ProteinFeature
    from src.models.base import get_db
    from collections import Counter
    
    with next(get_db()) as db:
        # Get proteins and their features
        proteins = db.query(Protein).filter(
            Protein.gene_name.in_(gene_list)
        ).all()
        
        domain_analysis = {}
        all_domains = []
        
        for protein in proteins:
            features = db.query(ProteinFeature).filter(
                ProteinFeature.protein_id == protein.id,
                ProteinFeature.type == 'domain'
            ).all()
            
            domains = [f.description for f in features]
            domain_analysis[protein.gene_name] = {
                'protein_length': protein.sequence_length,
                'domain_count': len(domains),
                'domains': domains
            }
            all_domains.extend(domains)
        
        # Domain frequency analysis
        domain_freq = Counter(all_domains)
        
        return {
            'per_protein': domain_analysis,
            'domain_frequency': domain_freq.most_common(10),
            'total_proteins': len(proteins),
            'unique_domains': len(set(all_domains))
        }

# Analyze kinase family domains
kinases = ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'MAPK1', 'MAPK3']
analysis = analyze_protein_domains(kinases)

print("Domain Analysis Results:")
print(f"Total proteins: {analysis['total_proteins']}")
print(f"Unique domains: {analysis['unique_domains']}")
print("\nMost common domains:")
for domain, count in analysis['domain_frequency']:
    print(f"  {domain}: {count}")
```

## Automation Workflows

### 1. Scheduled Data Updates

```python
# scheduled_updates.py
import schedule
import time
from datetime import datetime

def update_protein_data():
    """Update protein data for key proteins."""
    key_proteins = ['P04637', 'P38398', 'P51587']  # TP53, BRCA1, BRCA2
    
    print(f"Starting protein update at {datetime.now()}")
    
    from src.collectors.uniprot_collector import UniProtCollector
    collector = UniProtCollector()
    
    for accession in key_proteins:
        try:
            protein = collector.process_and_save(accession)
            print(f"Updated {accession}: {protein.protein_name}")
        except Exception as e:
            print(f"Failed to update {accession}: {e}")
    
    print(f"Update completed at {datetime.now()}")

def update_variant_data():
    """Update clinical variant data."""
    genes = ['TP53', 'BRCA1', 'BRCA2']
    
    print(f"Starting variant update at {datetime.now()}")
    
    from src.collectors.clinvar_collector import ClinVarCollector
    collector = ClinVarCollector()
    
    for gene in genes:
        try:
            variants_data = collector.collect_by_gene(gene, limit=100)
            variants = collector.save_variants_to_database(variants_data)
            print(f"Updated {len(variants)} variants for {gene}")
        except Exception as e:
            print(f"Failed to update variants for {gene}: {e}")
    
    print(f"Variant update completed at {datetime.now()}")

# Schedule updates
schedule.every().monday.at("02:00").do(update_protein_data)
schedule.every().wednesday.at("03:00").do(update_variant_data)

if __name__ == "__main__":
    print("Starting scheduled update service...")
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour
```

### 2. Batch Processing Pipeline

```bash
#!/bin/bash
# batch_pipeline.sh

# Configuration
GENE_LIST_FILE="genes_of_interest.txt"
OUTPUT_DIR="batch_results"
LOG_FILE="batch_pipeline.log"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "$(date): Starting batch pipeline" >> $LOG_FILE

# Process each gene in the list
while IFS= read -r gene; do
    echo "$(date): Processing $gene" >> $LOG_FILE
    
    # Create gene-specific directory
    gene_dir="$OUTPUT_DIR/$gene"
    mkdir -p "$gene_dir"
    
    # Collect protein data
    biocollect collect uniprot-search "$gene AND reviewed:yes AND organism_id:9606" \
        --limit 1 --save-file > "$gene_dir/uniprot.json" 2>> $LOG_FILE
    
    # Collect structure data
    biocollect collect alphafold "$gene" --download \
        --save-file > "$gene_dir/alphafold.json" 2>> $LOG_FILE
    
    # Collect interaction data
    biocollect collect string "$gene" --partner-limit 20 \
        --save-file > "$gene_dir/string.json" 2>> $LOG_FILE
    
    # Collect variant data
    biocollect collect clinvar "$gene" --id-type gene --pathogenic-only \
        --limit 50 --save-file > "$gene_dir/clinvar.json" 2>> $LOG_FILE
    
    echo "$(date): Completed $gene" >> $LOG_FILE
    sleep 5  # Rate limiting
    
done < "$GENE_LIST_FILE"

echo "$(date): Batch pipeline completed" >> $LOG_FILE

# Generate summary report
python generate_batch_report.py $OUTPUT_DIR > batch_summary.html
```

## Collaboration Workflows

### 1. Data Sharing Workflow

```python
# data_sharing.py
def create_research_dataset(project_name, gene_list):
    """Create a shareable dataset for research collaboration."""
    import json
    import os
    from datetime import datetime
    from src.models.protein import Protein
    from src.models.genomic import Variant
    from src.models.base import get_db
    
    # Create project directory
    project_dir = f"datasets/{project_name}"
    os.makedirs(project_dir, exist_ok=True)
    
    dataset = {
        'project_name': project_name,
        'created_date': datetime.now().isoformat(),
        'genes': gene_list,
        'proteins': [],
        'variants': []
    }
    
    with next(get_db()) as db:
        # Collect protein data
        proteins = db.query(Protein).filter(
            Protein.gene_name.in_(gene_list)
        ).all()
        
        for protein in proteins:
            protein_data = {
                'accession': protein.accession,
                'gene_name': protein.gene_name,
                'protein_name': protein.protein_name,
                'organism': protein.organism,
                'sequence_length': protein.sequence_length,
                'has_structure': protein.has_3d_structure == 'Y',
                'function': protein.function_description
            }
            dataset['proteins'].append(protein_data)
        
        # Collect variant data
        variants = db.query(Variant).filter(
            Variant.gene_symbol.in_(gene_list),
            Variant.clinical_significance.contains('Pathogenic')
        ).all()
        
        for variant in variants:
            variant_data = {
                'rsid': variant.rsid,
                'gene_symbol': variant.gene_symbol,
                'clinical_significance': variant.clinical_significance,
                'variant_type': variant.variant_type,
                'chromosome': variant.chromosome,
                'position': variant.position
            }
            dataset['variants'].append(variant_data)
    
    # Save dataset
    dataset_file = f"{project_dir}/{project_name}_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Create README
    readme_content = f"""# {project_name} Dataset

Created: {dataset['created_date']}
Genes: {', '.join(gene_list)}

## Contents
- Proteins: {len(dataset['proteins'])}
- Pathogenic variants: {len(dataset['variants'])}

## Files
- `{project_name}_dataset.json`: Complete dataset in JSON format
- `README.md`: This file

## Usage
Load the dataset in Python:
```python
import json
with open('{project_name}_dataset.json', 'r') as f:
    data = json.load(f)
```

## Citation
Please cite BioinformaticsDataCollector when using this dataset.
"""
    
    with open(f"{project_dir}/README.md", 'w') as f:
        f.write(readme_content)
    
    return dataset_file, project_dir

# Create dataset for cancer research collaboration
genes = ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'RB1']
dataset_file, project_dir = create_research_dataset("cancer_genes_2024", genes)
print(f"Created research dataset: {dataset_file}")
print(f"Project directory: {project_dir}")
```

### 2. Quality Control Workflow

```python
# quality_control.py
def run_quality_checks():
    """Run comprehensive quality checks on collected data."""
    from src.models.protein import Protein
    from src.models.structure import Structure
    from src.models.genomic import Variant
    from src.models.base import get_db
    from src.utils.validation import SequenceValidator, IdentifierValidator
    
    qc_report = {
        'timestamp': datetime.now().isoformat(),
        'proteins': {'total': 0, 'issues': []},
        'structures': {'total': 0, 'issues': []},
        'variants': {'total': 0, 'issues': []}
    }
    
    with next(get_db()) as db:
        # Check proteins
        proteins = db.query(Protein).all()
        qc_report['proteins']['total'] = len(proteins)
        
        for protein in proteins:
            # Validate sequence
            if not SequenceValidator.is_valid_protein_sequence(protein.sequence):
                qc_report['proteins']['issues'].append(
                    f"{protein.accession}: Invalid protein sequence"
                )
            
            # Check sequence length consistency
            if len(protein.sequence) != protein.sequence_length:
                qc_report['proteins']['issues'].append(
                    f"{protein.accession}: Sequence length mismatch"
                )
            
            # Validate UniProt accession format
            if not IdentifierValidator.validate(protein.accession, 'uniprot'):
                qc_report['proteins']['issues'].append(
                    f"{protein.accession}: Invalid UniProt accession format"
                )
        
        # Check structures
        structures = db.query(Structure).all()
        qc_report['structures']['total'] = len(structures)
        
        for structure in structures:
            # Check resolution values
            if structure.resolution and (structure.resolution < 0 or structure.resolution > 10):
                qc_report['structures']['issues'].append(
                    f"{structure.structure_id}: Suspicious resolution value: {structure.resolution}"
                )
            
            # Validate PDB ID format
            if not IdentifierValidator.validate(structure.structure_id, 'pdb'):
                qc_report['structures']['issues'].append(
                    f"{structure.structure_id}: Invalid PDB ID format"
                )
        
        # Check variants
        variants = db.query(Variant).all()
        qc_report['variants']['total'] = len(variants)
        
        for variant in variants:
            # Check chromosome values
            valid_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
            if variant.chromosome and variant.chromosome not in valid_chromosomes:
                qc_report['variants']['issues'].append(
                    f"{variant.rsid}: Invalid chromosome: {variant.chromosome}"
                )
            
            # Check position values
            if variant.position and variant.position <= 0:
                qc_report['variants']['issues'].append(
                    f"{variant.rsid}: Invalid position: {variant.position}"
                )
    
    return qc_report

# Run quality control
qc_results = run_quality_checks()
print(f"Quality Control Report - {qc_results['timestamp']}")
print(f"Proteins: {qc_results['proteins']['total']} total, {len(qc_results['proteins']['issues'])} issues")
print(f"Structures: {qc_results['structures']['total']} total, {len(qc_results['structures']['issues'])} issues")
print(f"Variants: {qc_results['variants']['total']} total, {len(qc_results['variants']['issues'])} issues")

if any(qc_results[category]['issues'] for category in ['proteins', 'structures', 'variants']):
    print("\nIssues found:")
    for category in ['proteins', 'structures', 'variants']:
        for issue in qc_results[category]['issues'][:5]:  # Show first 5 issues
            print(f"  {issue}")
```

## Performance Optimization Workflows

### 1. Large-Scale Data Collection

```python
# large_scale_collection.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.collectors.uniprot_collector import UniProtCollector

def collect_protein_batch(accession_batch):
    """Collect a batch of proteins efficiently."""
    collector = UniProtCollector()
    results = []
    
    for accession in accession_batch:
        try:
            protein = collector.process_and_save(accession)
            results.append((accession, protein))
        except Exception as e:
            results.append((accession, f"Error: {e}"))
    
    return results

def large_scale_collection(accession_list, batch_size=50, max_workers=4):
    """Collect large numbers of proteins efficiently."""
    
    # Split into batches
    batches = [
        accession_list[i:i+batch_size] 
        for i in range(0, len(accession_list), batch_size)
    ]
    
    all_results = []
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_protein_batch, batch) 
            for batch in batches
        ]
        
        for i, future in enumerate(futures):
            print(f"Processing batch {i+1}/{len(batches)}...")
            results = future.result()
            all_results.extend(results)
    
    return all_results

# Example: Collect 1000 human proteins
with open('human_protein_accessions.txt', 'r') as f:
    accessions = [line.strip() for line in f if line.strip()]

results = large_scale_collection(accessions[:1000])
print(f"Collected {len([r for r in results if not isinstance(r[1], str)])} proteins successfully")
```

These workflows demonstrate real-world usage patterns and can be adapted for specific research needs. Each workflow includes both command-line and Python approaches, allowing users to choose the most appropriate method for their requirements.