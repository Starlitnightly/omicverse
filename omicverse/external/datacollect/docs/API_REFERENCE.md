# Python API Reference

Complete reference for using BioinformaticsDataCollector programmatically in Python.

## Installation & Setup

```python
# Install the package
pip install -e .

# Import required modules
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector
from src.models.protein import Protein
from src.models.base import get_db
```

## Core Architecture

### Base Classes

#### BaseCollector
All collectors inherit from `BaseCollector` which provides common functionality:

```python
from src.collectors.base import BaseCollector

class MyCollector(BaseCollector):
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Implement data collection for single identifier."""
        pass
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Implement data collection for multiple identifiers.""" 
        pass
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save collected data to database."""
        pass
```

#### BaseModel
All database models inherit from `BaseModel`:

```python
from src.models.base import BaseModel

class MyModel(BaseModel):
    __tablename__ = "my_table"
    # SQLAlchemy column definitions
```

## Collectors

### UniProt Collector

#### Basic Usage
```python
from src.collectors.uniprot_collector import UniProtCollector

# Initialize collector
collector = UniProtCollector()

# Collect single protein
protein_data = collector.collect_single("P04637")
print(protein_data["protein_name"])  # "Cellular tumor antigen p53"

# Save to database
protein = collector.save_to_database(protein_data)
print(protein.accession)  # "P04637"

# One-step collect and save
protein = collector.process_and_save("P04637")
```

#### Advanced UniProt Usage
```python
# Collect with options
data = collector.collect_single("P04637", include_features=True)

# Search and collect multiple proteins
proteins = collector.search_and_collect(
    query="kinase AND organism_id:9606",
    max_results=10
)

# Batch collection
protein_data_list = collector.collect_batch(["P04637", "P53_HUMAN", "Q04206"])

# Custom database session
from src.models.base import get_db
with next(get_db()) as db:
    collector = UniProtCollector(db_session=db)
    protein = collector.process_and_save("P04637")
    # protein is already committed to the database
```

#### UniProt Data Structure
```python
# Collected data dictionary structure
protein_data = {
    "accession": "P04637",
    "entry_name": "P53_HUMAN", 
    "protein_name": "Cellular tumor antigen p53",
    "gene_name": "TP53",
    "organism": "Homo sapiens",
    "organism_id": 9606,
    "sequence": "MEEPQSDPSVEPPLSQETFSD...",
    "sequence_length": 393,
    "molecular_weight": 43653,
    "function_description": "Acts as a tumor suppressor...",
    "pdb_ids": "1A1U,1AIE,1C26...",
    "features": [
        {
            "type": "domain",
            "location": {"start": 94, "end": 292},
            "description": "DNA-binding domain"
        }
    ],
    "go_terms": [
        {
            "id": "GO:0003677",
            "name": "DNA binding",
            "category": "molecular_function"
        }
    ]
}
```

### PDB Collector

#### Basic Usage
```python
from src.collectors.pdb_collector import PDBCollector

collector = PDBCollector()

# Collect structure metadata
structure_data = collector.collect_single("1A3N")

# Collect with structure file download
structure_data = collector.collect_single("1A3N", download_structure=True)

# Save to database
structure = collector.save_to_database(structure_data)
print(f"Resolution: {structure.resolution} Å")
```

#### Advanced PDB Usage
```python
# Search by sequence similarity
structures = collector.search_by_sequence(
    sequence="MVLSPADKTNVKAAW",
    e_value=0.01,
    max_results=5
)

# Batch collection
structure_data_list = collector.collect_batch(
    ["1A3N", "1TUP", "2AC0"],
    download_structure=True
)
```

### AlphaFold Collector

```python
from src.collectors.alphafold_collector import AlphaFoldCollector

collector = AlphaFoldCollector()

# Collect AlphaFold prediction
structure_data = collector.collect_single("P04637")

# Download structure and confidence files
structure_data = collector.collect_single(
    "P04637", 
    download_structure=True,
    download_pae=True
)

structure = collector.save_to_database(structure_data)
print(f"Mean pLDDT: {structure.r_factor}")  # Confidence score
```

### Other Collectors

#### InterPro Collector
```python
from src.collectors.interpro_collector import InterProCollector

collector = InterProCollector()
domain_data = collector.collect_single("P04637")
domains = collector.save_to_database(domain_data)

for domain in domains:
    print(f"{domain.interpro_id}: {domain.name} ({domain.type})")
```

#### STRING Collector  
```python
from src.collectors.string_collector import STRINGCollector

collector = STRINGCollector()
interaction_data = collector.collect_single("TP53", species=9606)
interactions = collector.save_to_database(interaction_data)
```

#### ClinVar Collector
```python
from src.collectors.clinvar_collector import ClinVarCollector

collector = ClinVarCollector()

# Single variant
variant_data = collector.collect_single("12345")
variant = collector.save_to_database(variant_data)

# Variants by gene
variants_data = collector.collect_by_gene("BRCA1", pathogenic_only=True)
variants = collector.save_variants_to_database(variants_data)
```

## Database Models

### Protein Model
```python
from src.models.protein import Protein, ProteinFeature, GOTerm
from src.models.base import get_db

# Query proteins
with next(get_db()) as db:
    # Find specific protein
    protein = db.query(Protein).filter(Protein.accession == "P04637").first()
    
    # Find proteins by organism
    human_proteins = db.query(Protein).filter(
        Protein.organism == "Homo sapiens"
    ).all()
    
    # Find proteins with structures
    structured_proteins = db.query(Protein).filter(
        Protein.has_3d_structure == "Y"
    ).all()
    
    # Complex queries
    kinases = db.query(Protein).filter(
        Protein.protein_name.contains("kinase")
    ).limit(10).all()
```

### Structure Model
```python
from src.models.structure import Structure, Chain, Ligand

with next(get_db()) as db:
    # Find structure
    structure = db.query(Structure).filter(
        Structure.structure_id == "1A3N"
    ).first()
    
    # Access related data
    chains = structure.chains
    ligands = structure.ligands
    
    # Query by resolution
    high_res = db.query(Structure).filter(
        Structure.resolution < 2.0
    ).all()
```

### Genomic Models
```python
from src.models.genomic import Gene, Variant

with next(get_db()) as db:
    # Find gene
    gene = db.query(Gene).filter(Gene.symbol == "TP53").first()
    
    # Find variants
    pathogenic = db.query(Variant).filter(
        Variant.clinical_significance.contains("Pathogenic")
    ).all()
```

## Utility Functions

### Data Validation
```python
from src.utils.validation import SequenceValidator, IdentifierValidator

# Validate sequences
is_valid = SequenceValidator.is_valid_protein_sequence("MVLSPADKTNVKAAW")
is_dna = SequenceValidator.is_dna_sequence("ATGCGTACG")

# Validate identifiers
is_uniprot = IdentifierValidator.validate("P04637", "uniprot")
id_type = IdentifierValidator.detect_type("NM_000546")  # Returns "refseq"

# Validate biological data
from src.utils.validation import BiologicalValidator
is_valid_taxon = BiologicalValidator.is_valid_taxonomy_id(9606)
```

### Data Transformation
```python
from src.utils.transformers import SequenceTransformer, DataNormalizer

# Sequence operations
protein_seq = SequenceTransformer.translate_dna_to_protein("ATGGCGTAG")
reverse_comp = SequenceTransformer.reverse_complement("ATGC")

# Data normalization  
clean_seq = DataNormalizer.clean_sequence("ACGT-123-N", "dna")
normalized_name = DataNormalizer.normalize_gene_name("tp53")  # Returns "TP53"
```

### Database Utilities
```python
from src.utils.database import initialize_database, get_table_stats

# Initialize database
initialize_database()

# Get statistics
with next(get_db()) as db:
    stats = get_table_stats(db)
    print(f"Proteins: {stats['proteins']}")
```

## Advanced Usage Patterns

### Custom Data Processing Pipeline
```python
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector
from src.models.base import get_db

def process_protein_family(gene_symbols):
    """Process a family of related proteins."""
    uniprot_collector = UniProtCollector()
    pdb_collector = PDBCollector()
    
    results = []
    
    for symbol in gene_symbols:
        # Search for human protein
        proteins = uniprot_collector.search_and_collect(
            f"{symbol} AND reviewed:yes AND organism_id:9606",
            max_results=1
        )
        
        if proteins:
            protein = proteins[0]
            
            # Collect structures if available
            if protein.pdb_ids:
                pdb_ids = protein.pdb_ids.split(",")[:3]  # First 3 structures
                for pdb_id in pdb_ids:
                    structure = pdb_collector.process_and_save(
                        pdb_id.strip(),
                        download_structure=True
                    )
            
            results.append(protein)
    
    return results

# Usage
cancer_genes = ["TP53", "BRCA1", "BRCA2", "ATM"]
proteins = process_protein_family(cancer_genes)
```

### Batch Data Analysis
```python
from src.models.protein import Protein
from src.models.structure import Structure
from sqlalchemy import func

def analyze_protein_structures():
    """Analyze protein structure coverage."""
    with next(get_db()) as db:
        # Count proteins with/without structures
        total_proteins = db.query(Protein).count()
        structured_proteins = db.query(Protein).filter(
            Protein.has_3d_structure == "Y"
        ).count()
        
        # Average resolution by method
        avg_resolution = db.query(
            Structure.structure_type,
            func.avg(Structure.resolution).label('avg_res')
        ).group_by(Structure.structure_type).all()
        
        return {
            "total_proteins": total_proteins,
            "structured_proteins": structured_proteins,
            "coverage": structured_proteins / total_proteins * 100,
            "avg_resolution_by_method": dict(avg_resolution)
        }

stats = analyze_protein_structures()
print(f"Structure coverage: {stats['coverage']:.1f}%")
```

### Error Handling
```python
import logging
from src.collectors.uniprot_collector import UniProtCollector

logger = logging.getLogger(__name__)

def robust_protein_collection(accessions):
    """Collect proteins with error handling."""
    collector = UniProtCollector()
    results = []
    errors = []
    
    for accession in accessions:
        try:
            protein = collector.process_and_save(accession)
            results.append(protein)
            logger.info(f"Successfully collected {accession}")
        except Exception as e:
            logger.error(f"Failed to collect {accession}: {e}")
            errors.append((accession, str(e)))
    
    return results, errors

# Usage
accessions = ["P04637", "INVALID", "P53_HUMAN", "Q04206"]
proteins, errors = robust_protein_collection(accessions)
print(f"Collected: {len(proteins)}, Errors: {len(errors)}")
```

## Configuration in Code

### Custom API Clients
```python
from src.api.uniprot import UniProtClient

# Custom configuration
client = UniProtClient(
    rate_limit=5,  # 5 requests per second
    timeout=60,    # 60 second timeout
    max_retries=5  # 5 retry attempts
)

collector = UniProtCollector()
collector.api_client = client  # Use custom client
```

### Custom Database Sessions
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.base import Base

# Custom database
engine = create_engine("postgresql://user:pass@host/db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Use with collectors
with Session() as db_session:
    collector = UniProtCollector(db_session=db_session)
    protein = collector.process_and_save("P04637")
    db_session.commit()
```

## Integration Examples

### With Pandas
```python
import pandas as pd
from src.models.protein import Protein
from src.models.base import get_db

def proteins_to_dataframe():
    """Convert proteins to pandas DataFrame."""
    with next(get_db()) as db:
        proteins = db.query(Protein).all()
        
        data = []
        for protein in proteins:
            data.append({
                'accession': protein.accession,
                'name': protein.protein_name,
                'organism': protein.organism,
                'length': protein.sequence_length,
                'has_structure': protein.has_3d_structure == 'Y'
            })
    
    return pd.DataFrame(data)

df = proteins_to_dataframe()
print(df.head())
```

### With Jupyter Notebooks
```python
# Cell 1: Setup
%matplotlib inline
from src.collectors.uniprot_collector import UniProtCollector
from src.models.protein import Protein
import matplotlib.pyplot as plt

# Cell 2: Collect data
collector = UniProtCollector()
proteins = collector.search_and_collect("kinase AND organism_id:9606", max_results=50)

# Cell 3: Visualize
lengths = [p.sequence_length for p in proteins]
plt.hist(lengths, bins=20)
plt.xlabel('Protein Length (aa)')
plt.ylabel('Count')
plt.title('Human Kinase Length Distribution')
```

## Performance Optimization

### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimized database connection
engine = create_engine(
    "postgresql://user:pass@host/db",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### Bulk Operations
```python
def bulk_save_proteins(protein_data_list):
    """Efficiently save many proteins."""
    from src.models.base import get_db
    
    with next(get_db()) as db:
        proteins = []
        for data in protein_data_list:
            protein = Protein(**data)
            proteins.append(protein)
        
        db.bulk_save_objects(proteins)
        db.commit()
        
    return len(proteins)
```

### Async Operations
```python
import asyncio
from src.api.base import BaseAPIClient

async def async_collect_multiple(accessions):
    """Collect multiple proteins asynchronously."""
    # Implementation would use async API clients
    # This is a framework for future async support
    pass
```