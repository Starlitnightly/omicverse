"""Tests for AlphaFold API client and collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date

from omicverse.external.datacollect.api.alphafold import AlphaFoldClient
from omicverse.external.datacollect.collectors.alphafold_collector import AlphaFoldCollector
from omicverse.external.datacollect.models.structure import Structure


@pytest.fixture
def mock_alphafold_response():
    """Sample AlphaFold API response."""
    return [{
        "entryId": "AF-P04637-F1",
        "gene": "TP53",
        "uniprotAccession": "P04637",
        "uniprotId": "P53_HUMAN",
        "uniprotDescription": "Cellular tumor antigen p53",
        "organismScientificName": "Homo sapiens",
        "taxId": 9606,
        "sequenceLength": 393,
        "modelCreatedDate": "2022-06-01",
        "latestVersion": 4,
        "allVersions": [1, 2, 3, 4],
        "isReviewed": True,
        "isReferenceProteome": True,
        "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v4.cif",
        "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v4.bcif",
        "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v4.pdb",
        "paeImageUrl": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-predicted_aligned_error_v4.png",
        "paeDocUrl": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-predicted_aligned_error_v4.json",
        "confidenceVersion": "v4",
        "confidenceMean": 72.59,
        "coverage": 100.0
    }]


class TestAlphaFoldClient:
    """Test AlphaFold API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = AlphaFoldClient()
        assert "alphafold.ebi.ac.uk" in client.base_url
        assert client.rate_limit == 10
    
    @patch.object(AlphaFoldClient, 'get')
    def test_get_prediction_by_uniprot(self, mock_get, mock_alphafold_response):
        """Test getting prediction by UniProt accession."""
        mock_response = Mock()
        mock_response.json.return_value = mock_alphafold_response
        mock_get.return_value = mock_response
        
        client = AlphaFoldClient()
        result = client.get_prediction_by_uniprot("P04637")
        
        mock_get.assert_called_once_with("/prediction/P04637")
        assert result[0]["entryId"] == "AF-P04637-F1"
        assert result[0]["gene"] == "TP53"
    
    @patch.object(AlphaFoldClient, 'get')
    def test_search_by_gene(self, mock_get):
        """Test searching by gene name."""
        mock_response = Mock()
        mock_response.json.return_value = [{"gene": "EGFR"}]
        mock_get.return_value = mock_response
        
        client = AlphaFoldClient()
        result = client.search_by_gene("EGFR", organism="Homo sapiens")
        
        mock_get.assert_called_once_with(
            "/search",
            params={"gene": "EGFR", "organism": "Homo sapiens"}
        )
    
    @patch('requests.Session.get')
    def test_get_structure_file(self, mock_session_get):
        """Test downloading structure file."""
        mock_response = Mock()
        mock_response.text = "ATOM  1  N   MET A   1..."
        mock_response.raise_for_status = Mock()
        mock_session_get.return_value = mock_response
        
        client = AlphaFoldClient()
        result = client.get_structure_file("P04637", format="pdb")
        
        assert "ATOM" in result
        assert mock_session_get.called


class TestAlphaFoldCollector:
    """Test AlphaFold collector."""
    
    @patch('src.collectors.alphafold_collector.AlphaFoldClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = AlphaFoldCollector()
        mock_client_class.assert_called_once()
    
    def test_collect_single(self, mock_alphafold_response):
        """Test collecting single prediction."""
        collector = AlphaFoldCollector()
        
        with patch.object(collector.api_client, 'get_prediction_by_uniprot') as mock_get:
            mock_get.return_value = mock_alphafold_response
            
            result = collector.collect_single("P04637")
            
            assert result["alphafold_id"] == "AF-P04637-F1"
            assert result["uniprot_accession"] == "P04637"
            assert result["organism"] == "Homo sapiens"
            assert result["gene_name"] == "TP53"
            assert result["mean_plddt"] == 72.59
    
    def test_save_to_database(self, test_db, mock_alphafold_response):
        """Test saving to database."""
        collector = AlphaFoldCollector(db_session=test_db)
        
        data = {
            "alphafold_id": "AF-P04637-F1",
            "uniprot_accession": "P04637",
            "organism": "Homo sapiens",
            "gene_name": "TP53",
            "protein_name": "Cellular tumor antigen p53",
            "sequence_length": 393,
            "model_created_date": "2022-06-01",
            "mean_plddt": 72.59,
        }
        
        structure = collector.save_to_database(data)
        
        assert isinstance(structure, Structure)
        assert structure.structure_id == "AF-P04637-F1"
        assert structure.source == "AlphaFold"
        assert structure.structure_type == "PREDICTED"
        assert structure.organism == "Homo sapiens"
        assert structure.r_factor == 72.59  # pLDDT stored here
        assert len(structure.chains) == 1
        assert structure.chains[0].uniprot_accession == "P04637"
    
    @patch.object(AlphaFoldCollector, 'collect_single')
    @patch.object(AlphaFoldCollector, 'save_to_database')
    def test_process_and_save(self, mock_save, mock_collect):
        """Test process and save workflow."""
        collector = AlphaFoldCollector()
        
        mock_data = {"alphafold_id": "AF-P04637-F1"}
        mock_structure = Mock()
        
        mock_collect.return_value = mock_data
        mock_save.return_value = mock_structure
        
        result = collector.process_and_save("P04637", download_structure=True)
        
        mock_collect.assert_called_once_with(
            "P04637",
            download_structure=True
        )
        mock_save.assert_called_once_with(mock_data)
        assert result == mock_structure