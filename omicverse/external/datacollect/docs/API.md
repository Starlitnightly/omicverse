# API Documentation

## Overview

The BioinformaticsDataCollector provides a unified API for accessing multiple bioinformatics databases. All API clients inherit from a base class that provides common functionality like rate limiting, retry logic, and error handling.

## Base API Client

### `BaseAPIClient`

The foundation for all API clients, providing:

- **Rate limiting**: Prevents exceeding API limits
- **Automatic retries**: Handles temporary failures
- **Session management**: Reuses HTTP connections
- **Timeout handling**: Configurable request timeouts

```python
from src.api.base import BaseAPIClient

class MyAPIClient(BaseAPIClient):
    def get_default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "MyClient/1.0",
            "Accept": "application/json"
        }
```

#### Key Methods

- `get(endpoint, **kwargs)`: Make GET request
- `post(endpoint, **kwargs)`: Make POST request
- `build_url(endpoint)`: Build full URL from endpoint
- `request(method, endpoint, **kwargs)`: Generic request method

#### Configuration

- `base_url`: API base URL
- `rate_limit`: Requests per second (default: 10)
- `timeout`: Request timeout in seconds (default: 30)
- `max_retries`: Maximum retry attempts (default: 3)

## UniProt API Client

### `UniProtClient`

Access the UniProt REST API for protein data.

```python
from src.api.uniprot import UniProtClient

client = UniProtClient()

# Get single entry
entry = client.get_entry("P04637")

# Search proteins
results = client.search(
    query="kinase AND organism_id:9606",
    fields=["accession", "protein_name", "length"],
    size=100
)

# Batch retrieve
entries = client.batch_retrieve(["P04637", "P53_HUMAN", "Q9Y6K9"])

# ID mapping
mappings = client.id_mapping(
    from_db="UniProtKB_AC-ID",
    to_db="RefSeq_Protein",
    ids=["P04637", "Q9Y6K9"]
)
```

#### Methods

##### `get_entry(accession: str) -> Dict`
Retrieve complete entry data for a UniProt accession.

**Parameters:**
- `accession`: UniProt accession (e.g., "P04637")

**Returns:** Dictionary with protein data including sequence, features, annotations

##### `search(query: str, fields: List[str], size: int) -> Dict`
Search UniProt with query syntax.

**Parameters:**
- `query`: UniProt query syntax (e.g., "kinase AND organism_id:9606")
- `fields`: Fields to retrieve (default: basic fields)
- `size`: Maximum results (max: 500)

**Returns:** Search results with requested fields

##### `batch_retrieve(accessions: List[str]) -> Dict`
Retrieve multiple entries in one request.

**Parameters:**
- `accessions`: List of UniProt accessions

**Returns:** Dictionary mapping accessions to entry data

## PDB API Client

### `SimplePDBClient`

Access PDB structure data using REST endpoints.

```python
from src.api.pdb_simple import SimplePDBClient

client = SimplePDBClient()

# Get entry summary
entry = client.get_entry("1A3N")

# Download structure file
pdb_content = client.get_structure("1A3N", format="pdb")
cif_content = client.get_structure("1A3N", format="cif")

# Get polymer information
polymer_info = client.get_polymer_info("1A3N")
```

#### Methods

##### `get_entry(pdb_id: str) -> Dict`
Get PDB entry summary data.

**Parameters:**
- `pdb_id`: 4-character PDB ID

**Returns:** Entry data including title, resolution, experimental method

##### `get_structure(pdb_id: str, format: str) -> str`
Download structure file content.

**Parameters:**
- `pdb_id`: 4-character PDB ID
- `format`: File format ("pdb" or "cif")

**Returns:** Structure file content as string

## Collectors

### `BaseCollector`

Abstract base class for all data collectors.

```python
from src.collectors.base import BaseCollector

class MyCollector(BaseCollector):
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        # Implement data collection logic
        pass
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        # Implement database saving logic
        pass
```

#### Key Methods

##### `collect_single(identifier: str, **kwargs) -> Dict`
Collect data for a single entity.

##### `collect_batch(identifiers: List[str], **kwargs) -> List[Dict]`
Collect data for multiple entities.

##### `save_to_database(data: Dict) -> Model`
Save collected data to database.

##### `process_and_save(identifier: str, **kwargs) -> Model`
Collect and save in one operation.

##### `save_to_file(data: Any, filename: str, format: str)`
Save data to file (JSON or CSV).

### `UniProtCollector`

Collector for UniProt protein data.

```python
from src.collectors.uniprot_collector import UniProtCollector

collector = UniProtCollector()

# Collect single protein
data = collector.collect_single("P04637", include_features=True)

# Save to database
protein = collector.save_to_database(data)

# Search and collect
proteins = collector.search_and_collect(
    query="kinase AND reviewed:true",
    max_results=50,
    include_features=True
)
```

#### Methods

##### `collect_single(accession: str, include_features: bool) -> Dict`
Collect data for one protein.

**Parameters:**
- `accession`: UniProt accession
- `include_features`: Include protein features (default: True)

**Returns:** Protein data dictionary

##### `search_and_collect(query: str, max_results: int, **kwargs) -> List[Protein]`
Search UniProt and collect all results.

**Parameters:**
- `query`: UniProt search query
- `max_results`: Maximum proteins to collect
- `**kwargs`: Additional parameters for collection

**Returns:** List of saved Protein objects

### `PDBCollector`

Collector for PDB structure data.

```python
from src.collectors.pdb_collector import PDBCollector

collector = PDBCollector()

# Collect structure
data = collector.collect_single("1A3N", download_structure=True)

# Save to database
structure = collector.save_to_database(data)

# Search by sequence similarity
structures = collector.search_by_sequence(
    sequence="MVLSPADKTNVKAAW...",
    e_value=0.001,
    max_results=20
)
```

#### Methods

##### `collect_single(pdb_id: str, download_structure: bool) -> Dict`
Collect data for one structure.

**Parameters:**
- `pdb_id`: 4-character PDB ID
- `download_structure`: Download structure file (default: False)

**Returns:** Structure data dictionary

## Error Handling

All API clients handle common errors:

```python
try:
    data = client.get_entry("INVALID_ID")
except HTTPError as e:
    if e.response.status_code == 404:
        print("Entry not found")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
except Timeout:
    print("Request timed out")
except ConnectionError:
    print("Network error")
```

## Rate Limiting

Rate limiting is automatic but can be configured:

```python
# Create client with custom rate limit
client = UniProtClient(rate_limit=5)  # 5 requests per second

# Or configure globally
from config import settings
settings.api.uniprot_rate_limit = 5
```

## Caching

API responses can be cached:

```python
# Enable caching for session
client = UniProtClient(cache_enabled=True, cache_ttl=3600)

# Or use manual caching
from functools import lru_cache

@lru_cache(maxsize=100)
def get_protein(accession):
    return collector.collect_single(accession)
```

## Async Support

For high-throughput operations:

```python
import asyncio
from src.api.base import BaseAPIClient

async def fetch_many(accessions):
    tasks = []
    for acc in accessions:
        task = client.get_async(f"/uniprotkb/{acc}")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## Best Practices

1. **Use batch operations** when possible to reduce API calls
2. **Handle errors gracefully** - APIs may be temporarily unavailable
3. **Respect rate limits** - Use built-in rate limiting
4. **Cache responses** when appropriate to avoid repeated calls
5. **Use appropriate timeouts** for long-running requests
6. **Log API interactions** for debugging

## Example: Complete Workflow

```python
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector
from src.utils.logging import setup_logging

# Setup
setup_logging("INFO")
uniprot = UniProtCollector()
pdb = PDBCollector()

# Collect protein
protein = uniprot.process_and_save("P04637")
print(f"Collected: {protein.protein_name}")

# Collect associated structures
if protein.pdb_ids:
    pdb_ids = protein.pdb_ids.split(",")
    for pdb_id in pdb_ids:
        structure = pdb.process_and_save(
            pdb_id, 
            download_structure=True
        )
        print(f"  Structure: {structure.title}")
```