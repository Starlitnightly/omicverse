"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from omicverse.external.datacollect.models.base import Base, init_db
from omicverse.external.datacollect.models.protein import Protein, GOTerm
from omicverse.external.datacollect.models.structure import Structure, Chain, Ligand
from omicverse.external.datacollect.models.genomic import Gene, Variant


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def test_db():
    """Create a test database."""
    # Use in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def sample_protein_data():
    """Sample protein data for testing."""
    return {
        "accession": "P12345",
        "entry_name": "TEST_HUMAN",
        "protein_name": "Test protein",
        "gene_name": "TEST",
        "organism": "Homo sapiens",
        "organism_id": 9606,
        "sequence": "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
        "sequence_length": 45,
        "molecular_weight": 5000.0,
        "function_description": "Test function",
        "go_terms": [
            {"id": "GO:0005737", "properties": {"GoTerm": "cytoplasm", "GoAspect": "C"}},
            {"id": "GO:0008270", "properties": {"GoTerm": "zinc ion binding", "GoAspect": "F"}}
        ],
        "pdb_ids": ["1ABC", "2DEF"],
        "features": [
            {
                "type": "domain",
                "location": {"start": {"value": 10}, "end": {"value": 30}},
                "description": "Test domain"
            }
        ]
    }


@pytest.fixture
def sample_structure_data():
    """Sample structure data for testing."""
    return {
        "pdb_id": "1ABC",
        "title": "Test structure",
        "structure_type": "X-RAY DIFFRACTION",
        "resolution": 2.5,
        "r_factor": 0.2,
        "deposition_date": "2023-01-01T00:00:00Z",
        "release_date": "2023-01-15T00:00:00Z",
        "organism": "Homo sapiens",
        "chains": [
            {
                "chain_id": "A",
                "sequence": "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF",
                "molecule_type": "protein",
                "length": 45,
                "uniprot_accession": "P12345"
            }
        ],
        "ligands": [
            {
                "ligand_id": "ZN",
                "name": "ZINC ION",
                "formula": "Zn",
                "molecular_weight": 65.38
            }
        ]
    }


@pytest.fixture
def sample_variant_data():
    """Sample variant data for testing."""
    return {
        "variant_id": "test_variant_1",
        "rsid": "rs12345",
        "chromosome": "1",
        "position": 100000,
        "reference_allele": "A",
        "alternate_allele": "G",
        "variant_type": "SNP",
        "consequence": "missense",
        "gene_symbol": "TEST",
        "protein_change": "p.Val10Met",
        "global_maf": 0.05,
        "clinical_significance": "benign"
    }


@pytest.fixture
def mock_api_response():
    """Mock API response data."""
    def _mock_response(data, status_code=200):
        class MockResponse:
            def __init__(self, data, status_code):
                self.data = data
                self.status_code = status_code
                self.headers = {"Content-Type": "application/json"}
            
            def json(self):
                return self.data
            
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")
            
            @property
            def text(self):
                return str(self.data)
        
        return MockResponse(data, status_code)
    
    return _mock_response