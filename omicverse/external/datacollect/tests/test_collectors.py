"""Tests for data collector classes."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path

from omicverse.external.datacollect.collectors.base import BaseCollector
from omicverse.external.datacollect.collectors.uniprot_collector import UniProtCollector
from omicverse.external.datacollect.collectors.pdb_collector import PDBCollector
from omicverse.external.datacollect.models.protein import Protein, GOTerm
from omicverse.external.datacollect.models.structure import Structure, Chain


class MockCollector(BaseCollector):
    """Mock collector for testing base functionality."""
    
    def collect_single(self, identifier, **kwargs):
        return {"id": identifier, "data": "test"}
    
    def collect_batch(self, identifiers, **kwargs):
        return [self.collect_single(id) for id in identifiers]
    
    def save_to_database(self, data):
        return Mock(id=data["id"])


class TestBaseCollector:
    """Test base collector functionality."""
    
    def test_generate_id(self):
        """Test ID generation."""
        collector = MockCollector(Mock(), Mock())
        
        id1 = collector.generate_id("test", "123")
        id2 = collector.generate_id("test", "123")
        id3 = collector.generate_id("test", "456")
        
        assert id1 == id2  # Same input = same ID
        assert id1 != id3  # Different input = different ID
        assert len(id1) == 16  # SHA256 truncated to 16 chars
    
    def test_save_to_file_json(self, temp_dir):
        """Test saving to JSON file."""
        with patch('src.collectors.base.settings.storage.processed_data_dir', temp_dir):
            collector = MockCollector(Mock(), Mock())
            data = {"test": "data"}
            
            path = collector.save_to_file(data, "test.json", format="json")
            
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == data
    
    def test_save_to_file_csv(self, temp_dir):
        """Test saving to CSV file."""
        with patch('src.collectors.base.settings.storage.processed_data_dir', temp_dir):
            collector = MockCollector(Mock(), Mock())
            data = [{"col1": "val1", "col2": "val2"}]
            
            path = collector.save_to_file(data, "test.csv", format="csv")
            
            assert path.exists()
            import pandas as pd
            df = pd.read_csv(path)
            assert len(df) == 1
            assert df.iloc[0]["col1"] == "val1"
    
    def test_process_and_save(self):
        """Test complete processing pipeline."""
        collector = MockCollector(Mock(), Mock())
        
        with patch.object(collector, 'save_to_database') as mock_save:
            mock_save.return_value = Mock(id="test123")
            
            result = collector.process_and_save("test123")
            
            assert result is not None
            mock_save.assert_called_once()


class TestUniProtCollector:
    """Test UniProt collector."""
    
    @patch('src.collectors.uniprot_collector.UniProtClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = UniProtCollector()
        mock_client_class.assert_called_once()
    
    def test_collect_single(self, sample_protein_data):
        """Test collecting single protein."""
        collector = UniProtCollector()
        
        with patch.object(collector.api_client, 'get_entry') as mock_get:
            mock_get.return_value = {
                "primaryAccession": "P12345",
                "uniProtkbId": "TEST_HUMAN",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Test protein"}
                    }
                },
                "organism": {
                    "scientificName": "Homo sapiens",
                    "taxonId": 9606
                },
                "sequence": {
                    "value": sample_protein_data["sequence"],
                    "length": 45,
                    "molWeight": 5000
                },
                "genes": [{"geneName": {"value": "TEST"}}],
                "features": []
            }
            
            result = collector.collect_single("P12345")
            
            assert result["accession"] == "P12345"
            assert result["protein_name"] == "Test protein"
            assert result["organism"] == "Homo sapiens"
            assert result["sequence"] == sample_protein_data["sequence"]
    
    def test_save_to_database(self, test_db, sample_protein_data):
        """Test saving protein to database."""
        collector = UniProtCollector(db_session=test_db)
        
        # Save new protein
        protein = collector.save_to_database(sample_protein_data)
        
        assert isinstance(protein, Protein)
        assert protein.accession == "P12345"
        assert protein.protein_name == "Test protein"
        assert protein.organism == "Homo sapiens"
        
        # Check it's in database
        saved = test_db.query(Protein).filter_by(accession="P12345").first()
        assert saved is not None
        assert saved.id == protein.id
    
    def test_extract_go_terms(self, test_db):
        """Test GO term extraction and saving."""
        collector = UniProtCollector(db_session=test_db)
        
        data = {
            "accession": "P12345",
            "sequence": "MVLSEGEWQLVLHVWAK",
            "go_terms": [
                {"id": "GO:0005737", "properties": {"GoTerm": "cytoplasm", "GoAspect": "C"}},
                {"id": "GO:0008270", "properties": {"GoTerm": "zinc ion binding", "GoAspect": "F"}}
            ]
        }
        
        protein = collector.save_to_database(data)
        
        assert len(protein.go_terms) == 2
        go_ids = [term.go_id for term in protein.go_terms]
        assert "GO:0005737" in go_ids
        assert "GO:0008270" in go_ids
    
    @patch.object(UniProtCollector, 'process_and_save')
    def test_search_and_collect(self, mock_process):
        """Test search and collect functionality."""
        collector = UniProtCollector()
        mock_process.return_value = Mock(accession="P12345")
        
        with patch.object(collector.api_client, 'search') as mock_search:
            mock_search.return_value = {
                "results": [
                    {"primaryAccession": "P12345"},
                    {"primaryAccession": "P67890"}
                ]
            }
            
            results = collector.search_and_collect("test query", max_results=2)
            
            assert len(results) == 2
            assert mock_process.call_count == 2


class TestPDBCollector:
    """Test PDB collector."""
    
    @patch('src.collectors.pdb_collector.SimplePDBClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = PDBCollector()
        mock_client_class.assert_called_once()
    
    def test_collect_single(self, sample_structure_data):
        """Test collecting single structure."""
        collector = PDBCollector()
        
        with patch.object(collector.api_client, 'get_entry') as mock_entry:
            with patch.object(collector.api_client, 'get_polymer_info') as mock_polymer:
                mock_entry.return_value = {
                    "struct": {"title": "Test structure"},
                    "exptl": [{"method": "X-RAY DIFFRACTION"}],
                    "reflns": [{"d_resolution_high": 2.5}],
                    "refine": [{"ls_R_factor_R_work": 0.2}],
                    "rcsb_accession_info": {
                        "deposit_date": "2023-01-01T00:00:00Z",
                        "initial_release_date": "2023-01-15T00:00:00Z"
                    },
                    "rcsb_entry_info": {
                        "polymer_entity_ids": ["1"]
                    }
                }
                
                mock_polymer.return_value = {
                    "rcsb_entity_source_organism": [{
                        "ncbi_scientific_name": "Homo sapiens"
                    }]
                }
                
                result = collector.collect_single("1ABC")
                
                assert result["pdb_id"] == "1ABC"
                assert result["title"] == "Test structure"
                assert result["structure_type"] == "X-RAY DIFFRACTION"
                assert result["resolution"] == 2.5
                assert len(result["chains"]) >= 1
                assert isinstance(result["ligands"], list)
    
    def test_save_to_database(self, test_db, sample_structure_data):
        """Test saving structure to database."""
        collector = PDBCollector(db_session=test_db)
        
        # Save new structure
        structure = collector.save_to_database(sample_structure_data)
        
        assert isinstance(structure, Structure)
        assert structure.structure_id == "1ABC"
        assert structure.title == "Test structure"
        assert structure.resolution == 2.5
        assert len(structure.chains) == 1
        assert len(structure.ligands) == 1
        
        # Check relationships
        chain = structure.chains[0]
        assert chain.chain_id == "A"
        assert chain.uniprot_accession == "P12345"
        
        ligand = structure.ligands[0]
        assert ligand.ligand_id == "ZN"
        assert ligand.name == "ZINC ION"
    
    def test_download_structure_file(self, temp_dir):
        """Test structure file download."""
        collector = PDBCollector()
        
        with patch('src.collectors.pdb_collector.settings.storage.raw_data_dir', temp_dir):
            with patch.object(collector.api_client, 'get_structure') as mock_get:
                mock_get.return_value = "HEADER    TEST STRUCTURE\nATOM      1  N   MET A   1"
                
                path = collector._download_structure_file("1ABC")
                
                assert path.exists()
                assert path.name == "1ABC.pdb"
                content = path.read_text()
                assert "HEADER" in content