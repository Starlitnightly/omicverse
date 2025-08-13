# API Implementation Plan

## Overview

This document outlines the implementation plan for adding 28 missing bioinformatics APIs to the BioinformaticsDataCollector project.

## Current Status

### ✅ Implemented (2/30)
- **UniProt** - Protein data
- **PDB** - Protein structure data

### ❌ Not Implemented (28/30)

## Implementation Priority

### Priority 1: Core Structural Biology APIs
These complement existing UniProt/PDB functionality:

1. **AlphaFold** (query_alphafold)
   - Protein structure predictions
   - Complements PDB experimental structures
   - REST API: https://alphafold.ebi.ac.uk/api/docs

2. **InterPro** (query_interpro)
   - Protein families and domains
   - Functional analysis
   - REST API: https://www.ebi.ac.uk/interpro/api/

3. **STRING** (query_stringdb)
   - Protein-protein interactions
   - Network analysis
   - REST API: https://string-db.org/help/api/

### Priority 2: Genomics APIs
Core genomic data sources:

4. **Ensembl** (query_ensembl)
   - Gene annotations
   - Genomic sequences
   - REST API: https://rest.ensembl.org/

5. **NCBI Suite**
   - **ClinVar** (query_clinvar) - Clinical variants
   - **dbSNP** (query_dbsnp) - SNP database
   - **GEO** (query_geo) - Gene expression
   - **BLAST** (blast_sequence) - Sequence similarity

6. **UCSC Genome Browser** (query_ucsc)
   - Genome annotations
   - REST API: https://api.genome.ucsc.edu/

### Priority 3: Pathway & Function APIs

7. **KEGG** (query_kegg)
   - Metabolic pathways
   - REST API: https://www.kegg.jp/kegg/rest/

8. **Reactome** (query_reactome)
   - Biological pathways
   - REST API: https://reactome.org/ContentService/

### Priority 4: Disease & Phenotype APIs

9. **OpenTargets Platform** (query_opentarget)
   - Drug targets
   - GraphQL API

10. **OpenTargets Genetics** (query_opentarget_genetics)
    - Genetic associations
    - GraphQL API

11. **GWAS Catalog** (query_gwas_catalog)
    - GWAS studies
    - REST API

12. **gnomAD** (query_gnomad)
    - Population genetics
    - GraphQL API

13. **cBioPortal** (query_cbioportal)
    - Cancer genomics
    - REST API

### Priority 5: Specialized Databases

14. **JASPAR** (query_jaspar)
    - Transcription factor binding
    - REST API

15. **RegulomeDB** (query_regulomedb)
    - Regulatory variants
    - REST API

16. **SCREEN** (region_to_ccre_screen, get_genes_near_ccre)
    - cis-regulatory elements
    - REST API

17. **ReMap** (query_remap)
    - Regulatory elements
    - REST API

### Priority 6: Additional Databases

18. **PRIDE** (query_pride)
    - Proteomics data
    - REST API

19. **GtoPdb** (query_gtopdb)
    - Pharmacology database
    - REST API

20. **EMDB** (query_emdb)
    - Electron microscopy
    - REST API

21. **MPD** (query_mpd)
    - Mouse phenotypes
    - REST API

22. **WoRMS** (query_worms)
    - Marine species
    - REST API

23. **IUCN** (query_iucn)
    - Conservation status
    - REST API

24. **Paleobiology** (query_paleobiology)
    - Fossil data
    - REST API

## Implementation Template

For each API, implement:

### 1. API Client (`src/api/{name}.py`)
```python
from typing import Dict, Any, List, Optional
from .base import BaseAPIClient
from config import settings

class {Name}Client(BaseAPIClient):
    """Client for {Full Name} API."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        base_url = settings.api.{name}_base_url
        super().__init__(
            base_url=base_url,
            rate_limit=settings.api.{name}_rate_limit,
            **kwargs
        )
        self.api_key = api_key or settings.api.{name}_api_key
    
    def get_default_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    # Implement specific methods
```

### 2. Collector (`src/collectors/{name}_collector.py`)
```python
from typing import Dict, Any, List
from src.api.{name} import {Name}Client
from src.models.{model} import {Model}
from .base import BaseCollector

class {Name}Collector(BaseCollector):
    """Collector for {Full Name} data."""
    
    def __init__(self, db_session=None, **kwargs):
        api_client = {Name}Client(**kwargs)
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        # Implementation
        pass
    
    def save_to_database(self, data: Dict[str, Any]) -> {Model}:
        # Implementation
        pass
```

### 3. Tests (`tests/test_{name}.py`)
```python
import pytest
from unittest.mock import Mock, patch
from src.api.{name} import {Name}Client
from src.collectors.{name}_collector import {Name}Collector

class Test{Name}Client:
    def test_initialization(self):
        # Test client setup
        pass
    
    def test_api_methods(self):
        # Test each API method
        pass

class Test{Name}Collector:
    def test_collect_single(self):
        # Test data collection
        pass
    
    def test_save_to_database(self):
        # Test database operations
        pass
```

### 4. CLI Command (`src/cli.py`)
```python
@collect.command()
@click.argument("identifier")
@click.option("--option", help="Description")
def {name}(identifier: str, option: str):
    """Collect data from {Full Name}."""
    from src.collectors.{name}_collector import {Name}Collector
    
    collector = {Name}Collector()
    # Implementation
```

### 5. Configuration (`config/config.py`)
```python
# API settings
{name}_base_url: str = Field(
    default="https://api.{name}.org",
    env="{NAME}_BASE_URL"
)
{name}_api_key: Optional[str] = Field(
    default=None,
    env="{NAME}_API_KEY"
)
{name}_rate_limit: int = Field(
    default=10,
    env="{NAME}_RATE_LIMIT"
)
```

## Implementation Schedule

### Phase 1 (Week 1-2): Core APIs
- AlphaFold
- InterPro
- STRING
- Ensembl
- KEGG

### Phase 2 (Week 3-4): NCBI Suite
- ClinVar
- dbSNP
- GEO
- BLAST

### Phase 3 (Week 5-6): Disease/Phenotype APIs
- OpenTargets (Platform & Genetics)
- GWAS Catalog
- gnomAD
- cBioPortal

### Phase 4 (Week 7-8): Specialized Databases
- JASPAR
- RegulomeDB
- SCREEN
- ReMap
- Reactome

### Phase 5 (Week 9-10): Additional APIs
- PRIDE
- GtoPdb
- EMDB
- MPD
- WoRMS
- IUCN
- Paleobiology

## Testing Strategy

1. **Unit Tests**: Each API client and collector
2. **Integration Tests**: API interaction with real endpoints
3. **Mock Tests**: For rate-limited or authenticated APIs
4. **End-to-End Tests**: Complete data flow from API to database

## Documentation Updates

For each API, add:
1. API reference in `docs/API.md`
2. Usage examples in `docs/TUTORIAL.md`
3. Configuration options in `docs/CONFIGURATION.md`
4. Quick reference in `docs/QUICK_REFERENCE.md`

## Notes

- Some APIs require authentication (API keys)
- GraphQL APIs (OpenTargets, gnomAD) need special handling
- Rate limits vary significantly between APIs
- Some APIs have complex query languages (KEGG, ClinVar)
- Consider implementing caching for expensive queries
- Add progress bars for batch operations
- Implement proper error handling for each API's specific errors