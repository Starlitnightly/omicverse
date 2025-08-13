# Developer Guide

This guide covers how to extend and contribute to the BioinformaticsDataCollector project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Architecture](#project-architecture)
3. [Adding New Data Sources](#adding-new-data-sources)
4. [Creating New Models](#creating-new-models)
5. [Writing Tests](#writing-tests)
6. [Code Style](#code-style)
7. [Contributing](#contributing)

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/datacollect2bionmi.git
cd datacollect2bionmi
git remote add upstream https://github.com/ORIGINAL_OWNER/datacollect2bionmi.git
```

### 2. Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/test_collectors.py::TestUniProtCollector -v
```

## Project Architecture

### Directory Structure

```
src/
├── api/              # External API clients
├── collectors/       # Data collection logic
├── models/          # Database models
├── utils/           # Utilities and helpers
└── cli.py           # CLI entry point
```

### Key Design Patterns

1. **Abstract Base Classes**: `BaseAPIClient`, `BaseCollector`, `BaseModel`
2. **Dependency Injection**: Collectors receive API clients
3. **Strategy Pattern**: Different validation strategies
4. **Factory Pattern**: Model creation from API data

### Data Flow

```
API Client → Collector → Validator → Transformer → Model → Database
     ↓           ↓           ↓           ↓           ↓
   HTTP      Extract    Validate    Transform     Save
  Request      Data       Data        Data       to DB
```

## Adding New Data Sources

### Step 1: Create API Client

```python
# src/api/ncbi.py
from typing import Dict, Any
from .base import BaseAPIClient

class NCBIClient(BaseAPIClient):
    """Client for NCBI E-utilities API."""
    
    def __init__(self, api_key: str = None, **kwargs):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        super().__init__(base_url=base_url, **kwargs)
        self.api_key = api_key
    
    def get_default_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["api_key"] = self.api_key
        return headers
    
    def get_gene(self, gene_id: int) -> Dict[str, Any]:
        """Get gene information."""
        endpoint = f"/esummary.fcgi"
        params = {
            "db": "gene",
            "id": gene_id,
            "retmode": "json"
        }
        response = self.get(endpoint, params=params)
        return response.json()
```

### Step 2: Create Collector

```python
# src/collectors/ncbi_collector.py
from typing import Dict, Any
from src.api.ncbi import NCBIClient
from src.models.genomic import Gene
from .base import BaseCollector

class NCBICollector(BaseCollector):
    """Collector for NCBI gene data."""
    
    def __init__(self, db_session=None, api_key=None):
        api_client = NCBIClient(api_key=api_key)
        super().__init__(api_client, db_session)
    
    def collect_single(self, gene_id: int, **kwargs) -> Dict[str, Any]:
        """Collect data for a single gene."""
        # Get gene summary
        gene_data = self.api_client.get_gene(gene_id)
        
        # Extract relevant fields
        summary = gene_data["result"][str(gene_id)]
        
        return {
            "gene_id": gene_id,
            "symbol": summary.get("name"),
            "description": summary.get("description"),
            "chromosome": summary.get("chromosome"),
            "map_location": summary.get("maplocation"),
            "organism": summary.get("organism", {}).get("scientificname"),
            "organism_id": summary.get("organism", {}).get("taxid"),
        }
    
    def save_to_database(self, data: Dict[str, Any]) -> Gene:
        """Save gene data to database."""
        gene = self.db_session.query(Gene).filter_by(
            gene_id=data["gene_id"]
        ).first()
        
        if not gene:
            gene = Gene(
                id=self.generate_id("ncbi_gene", data["gene_id"]),
                source="NCBI"
            )
        
        # Update fields
        for field in ["gene_id", "symbol", "description", 
                     "chromosome", "map_location", "organism"]:
            if field in data:
                setattr(gene, field, data[field])
        
        if not gene in self.db_session:
            self.db_session.add(gene)
        
        self.db_session.commit()
        return gene
```

### Step 3: Add CLI Command

```python
# In src/cli.py
@collect.command()
@click.argument("gene_id", type=int)
@click.option("--api-key", envvar="NCBI_API_KEY", help="NCBI API key")
def ncbi(gene_id: int, api_key: str):
    """Collect gene data from NCBI."""
    from src.collectors.ncbi_collector import NCBICollector
    
    console.print(f"[blue]Collecting NCBI gene {gene_id}...[/blue]")
    
    collector = NCBICollector(api_key=api_key)
    
    try:
        gene = collector.process_and_save(gene_id)
        console.print(f"[green]Successfully collected: {gene.symbol}[/green]")
        console.print(f"  Description: {gene.description}")
        console.print(f"  Organism: {gene.organism}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
```

### Step 4: Write Tests

```python
# tests/test_ncbi.py
import pytest
from unittest.mock import Mock, patch
from src.collectors.ncbi_collector import NCBICollector

class TestNCBICollector:
    def test_collect_single(self):
        collector = NCBICollector()
        
        with patch.object(collector.api_client, 'get_gene') as mock_get:
            mock_get.return_value = {
                "result": {
                    "672": {
                        "name": "BRCA1",
                        "description": "BRCA1 DNA repair associated",
                        "organism": {
                            "scientificname": "Homo sapiens",
                            "taxid": 9606
                        }
                    }
                }
            }
            
            data = collector.collect_single(672)
            
            assert data["symbol"] == "BRCA1"
            assert data["organism"] == "Homo sapiens"
```

## Creating New Models

### Step 1: Define Model

```python
# src/models/pathway.py
from sqlalchemy import Column, String, Text, Table, ForeignKey
from sqlalchemy.orm import relationship
from .base import BaseModel, Base

# Many-to-many association
pathway_proteins = Table(
    'pathway_proteins',
    Base.metadata,
    Column('pathway_id', String, ForeignKey('pathways.id')),
    Column('protein_id', String, ForeignKey('proteins.id'))
)

class Pathway(BaseModel):
    """Model for biological pathways."""
    
    __tablename__ = 'pathways'
    
    pathway_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    database = Column(String)  # KEGG, Reactome, etc.
    
    # Relationships
    proteins = relationship(
        "Protein",
        secondary=pathway_proteins,
        back_populates="pathways"
    )
    
    def __repr__(self):
        return f"<Pathway(pathway_id='{self.pathway_id}', name='{self.name}')>"
```

### Step 2: Add Relationship to Existing Models

```python
# In src/models/protein.py
from .pathway import pathway_proteins

class Protein(BaseModel):
    # ... existing fields ...
    
    # Add relationship
    pathways = relationship(
        "Pathway",
        secondary=pathway_proteins,
        back_populates="proteins"
    )
```

### Step 3: Create Migration

```python
# migrations/add_pathways.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'pathways',
        sa.Column('id', sa.String, primary_key=True),
        sa.Column('source', sa.String),
        sa.Column('created_at', sa.DateTime),
        sa.Column('updated_at', sa.DateTime),
        sa.Column('pathway_id', sa.String, unique=True),
        sa.Column('name', sa.String),
        sa.Column('description', sa.Text),
        sa.Column('database', sa.String)
    )
    
    op.create_table(
        'pathway_proteins',
        sa.Column('pathway_id', sa.String, sa.ForeignKey('pathways.id')),
        sa.Column('protein_id', sa.String, sa.ForeignKey('proteins.id'))
    )
```

## Writing Tests

### Test Structure

```python
# tests/test_feature.py
import pytest
from unittest.mock import Mock, patch, MagicMock

class TestFeature:
    """Test new feature."""
    
    @pytest.fixture
    def mock_data(self):
        """Sample data for testing."""
        return {
            "id": "123",
            "name": "Test",
            "value": 42
        }
    
    def test_basic_functionality(self, mock_data):
        """Test basic feature behavior."""
        # Arrange
        instance = MyClass()
        
        # Act
        result = instance.process(mock_data)
        
        # Assert
        assert result.name == "Test"
        assert result.value == 42
    
    @patch('src.module.external_function')
    def test_with_mock(self, mock_func):
        """Test with external dependency mocked."""
        mock_func.return_value = {"status": "success"}
        
        result = my_function()
        
        mock_func.assert_called_once()
        assert result["status"] == "success"
    
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError) as exc_info:
            invalid_function("bad input")
        
        assert "Invalid input" in str(exc_info.value)
```

### Testing Best Practices

1. **Use Fixtures**: Share test data and setup
2. **Mock External Services**: Don't make real API calls
3. **Test Edge Cases**: Empty data, None values, errors
4. **Test Integration**: How components work together
5. **Maintain Coverage**: Aim for >80% coverage

## Code Style

### Python Style Guide

We follow PEP 8 with these additions:

1. **Line Length**: 88 characters (Black default)
2. **Imports**: Group and sort with isort
3. **Docstrings**: Google style
4. **Type Hints**: Use for public APIs

### Example

```python
"""Module docstring explaining purpose."""

from typing import Dict, List, Optional, Union

from external_package import something
from another_package import another_thing

from src.models import MyModel
from src.utils import utility_function


class ExampleClass:
    """Class for demonstrating style.
    
    Attributes:
        name: The name of the instance
        value: The numeric value
    """
    
    def __init__(self, name: str, value: int = 0) -> None:
        """Initialize ExampleClass.
        
        Args:
            name: The name to use
            value: Optional initial value (default: 0)
        """
        self.name = name
        self.value = value
    
    def process(self, data: Dict[str, Any]) -> Optional[MyModel]:
        """Process data and return model.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed model instance or None if invalid
            
        Raises:
            ValueError: If data is invalid
        """
        if not self._validate(data):
            raise ValueError(f"Invalid data: {data}")
        
        return MyModel(**data)
    
    def _validate(self, data: Dict[str, Any]) -> bool:
        """Validate input data (private method)."""
        return "required_field" in data
```

### Tools

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/

# All checks
make lint
```

## Contributing

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following style guide
- Add tests for new functionality
- Update documentation
- Add docstrings

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check style
make lint

# Test CLI
python -m src.cli --help
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add NCBI gene collector

- Add NCBIClient for E-utilities API
- Implement NCBICollector with gene support  
- Add CLI command for gene collection
- Add comprehensive tests"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what and why
3. **Tests**: All tests must pass
4. **Coverage**: Don't decrease coverage
5. **Documentation**: Update if needed

### Commit Message Format

```
type: subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Refactoring
- `test`: Tests
- `chore`: Maintenance

## Advanced Topics

### Adding Async Support

```python
# src/api/async_client.py
import asyncio
import aiohttp
from typing import List, Dict, Any

class AsyncAPIClient:
    """Async API client for high-throughput operations."""
    
    def __init__(self, base_url: str, rate_limit: int = 10):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
    
    async def fetch_many(self, endpoints: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple endpoints concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_one(session, endpoint) 
                for endpoint in endpoints
            ]
            return await asyncio.gather(*tasks)
    
    async def _fetch_one(self, session, endpoint):
        """Fetch single endpoint with rate limiting."""
        async with self._semaphore:
            url = f"{self.base_url}{endpoint}"
            async with session.get(url) as response:
                return await response.json()
```

### Custom Validators

```python
# src/utils/custom_validators.py
from typing import Any
from pydantic import field_validator
from src.utils.validation import BaseModel

class CustomData(BaseModel):
    """Model with custom validation."""
    
    sequence: str
    score: float
    
    @field_validator("sequence")
    def validate_custom_sequence(cls, v):
        """Custom sequence validation logic."""
        if not v or len(v) < 10:
            raise ValueError("Sequence too short")
        
        # Custom validation
        if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in v.upper()):
            raise ValueError("Invalid amino acids")
        
        return v.upper()
    
    @field_validator("score")
    def validate_score(cls, v):
        """Validate score range."""
        if not 0 <= v <= 1:
            raise ValueError("Score must be between 0 and 1")
        return v
```

### Performance Optimization

```python
# Bulk operations
from sqlalchemy import bindparam
from sqlalchemy.dialects.sqlite import insert

def bulk_insert_proteins(proteins: List[Dict], db_session):
    """Efficiently insert multiple proteins."""
    
    # Prepare insert statement
    stmt = insert(Protein).values(
        id=bindparam("id"),
        accession=bindparam("accession"),
        protein_name=bindparam("protein_name"),
        # ... other fields
    )
    
    # Update on conflict
    stmt = stmt.on_conflict_do_update(
        index_elements=["accession"],
        set_={
            "protein_name": stmt.excluded.protein_name,
            "updated_at": datetime.utcnow()
        }
    )
    
    # Execute bulk operation
    db_session.execute(stmt, proteins)
    db_session.commit()
```

## Resources

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)