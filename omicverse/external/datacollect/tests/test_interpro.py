"""Tests for InterPro API client and collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.api.interpro import InterProClient
from omicverse.external.datacollect.collectors.interpro_collector import InterProCollector
from omicverse.external.datacollect.models.interpro import Domain, DomainLocation
from omicverse.external.datacollect.models.protein import Protein


@pytest.fixture
def mock_interpro_entry():
    """Sample InterPro entry response."""
    return {
        "metadata": {
            "accession": "IPR000001",
            "name": "Kringle",
            "type": "domain",
            "description": "Kringle domains are found in many proteins",
            "member_databases": {
                "pfam": {"PF00024": {"name": "Kringle", "description": "Kringle domain"}}
            },
            "go_terms": [
                {"id": "GO:0005509", "name": "calcium ion binding", "category": "molecular_function"}
            ]
        }
    }


@pytest.fixture
def mock_protein_entries():
    """Sample protein entries response."""
    return {
        "results": [
            {
                "metadata": {
                    "accession": "IPR000001",
                    "name": "Kringle",
                    "type": "domain"
                },
                "proteins": [
                    {
                        "accession": "P04637",
                        "name": "P53_HUMAN",
                        "protein_length": 393,
                        "entry_protein_locations": [
                            {
                                "fragments": [
                                    {"start": 50, "end": 120}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }


class TestInterProClient:
    """Test InterPro API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = InterProClient()
        assert "www.ebi.ac.uk/interpro/api" in client.base_url
        assert client.rate_limit == 20
    
    @patch.object(InterProClient, 'get')
    def test_get_entry(self, mock_get, mock_interpro_entry):
        """Test getting InterPro entry."""
        mock_response = Mock()
        mock_response.json.return_value = mock_interpro_entry
        mock_get.return_value = mock_response
        
        client = InterProClient()
        result = client.get_entry("IPR000001")
        
        mock_get.assert_called_once_with("/entry/interpro/IPR000001")
        assert result["metadata"]["accession"] == "IPR000001"
        assert result["metadata"]["name"] == "Kringle"
    
    @patch.object(InterProClient, 'get')
    def test_get_protein_entries(self, mock_get, mock_protein_entries):
        """Test getting protein entries."""
        mock_response = Mock()
        mock_response.json.return_value = mock_protein_entries
        mock_get.return_value = mock_response
        
        client = InterProClient()
        result = client.get_protein_entries("P04637")
        
        mock_get.assert_called_once_with("/protein/UniProt/P04637/entry/interpro")
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["accession"] == "IPR000001"
    
    @patch.object(InterProClient, 'get_protein_entries')
    def test_get_domains_for_protein(self, mock_get_protein_entries):
        """Test getting domains for protein."""
        mock_get_protein_entries.return_value = {
            "results": [
                {
                    "metadata": {
                        "accession": "IPR000001",
                        "name": "Kringle domain",
                        "type": "domain"
                    },
                    "proteins": [
                        {
                            "accession": "P04637",
                            "entry_protein_locations": [
                                {"fragments": [{"start": 50, "end": 120}]}
                            ]
                        }
                    ]
                }
            ]
        }
        
        client = InterProClient()
        result = client.get_domains_for_protein("P04637")
        
        assert len(result) == 1
        assert result[0]["interpro_id"] == "IPR000001"
        assert result[0]["name"] == "Kringle domain"
        assert result[0]["entry_type"] == "domain"
    
    @patch.object(InterProClient, 'get')
    def test_search(self, mock_get):
        """Test search functionality."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"metadata": {"accession": "IPR000001", "name": "Kringle"}}
            ]
        }
        mock_get.return_value = mock_response
        
        client = InterProClient()
        result = client.search("kinase", entry_type="domain")
        
        mock_get.assert_called_once_with(
            "/search/entry",
            params={"search": "kinase", "page_size": 20, "type": "domain"}
        )


class TestInterProCollector:
    """Test InterPro collector."""
    
    @patch('src.collectors.interpro_collector.InterProClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = InterProCollector()
        mock_client_class.assert_called_once()
    
    def test_collect_single(self, mock_protein_entries):
        """Test collecting single protein annotations."""
        collector = InterProCollector()
        
        with patch.object(collector.api_client, 'get_protein_entries') as mock_entries:
            with patch.object(collector.api_client, 'get_domains_for_protein') as mock_domains:
                mock_entries.return_value = mock_protein_entries["results"]
                mock_domains.return_value = [
                    {
                        "interpro_id": "IPR000001",
                        "name": "Kringle",
                        "type": "domain",
                        "locations": [{"fragments": [{"start": 50, "end": 120}]}]
                    }
                ]
                
                result = collector.collect_single("P04637")
                
                assert result["uniprot_accession"] == "P04637"
                assert result["total_entries"] == 1
                assert result["domain_count"] == 1
                assert result["family_count"] == 0
    
    def test_save_to_database(self, test_db):
        """Test saving to database."""
        collector = InterProCollector(db_session=test_db)
        
        # Create a test protein
        protein = Protein(
            id="test_protein_1",
            accession="P04637",
            protein_name="Test protein",
            sequence="MVLSEGEWQLVLHVWAK",
            sequence_length=17,
            source="test"
        )
        test_db.add(protein)
        test_db.commit()
        
        data = {
            "uniprot_accession": "P04637",
            "domains": [
                {
                    "interpro_id": "IPR000001",
                    "name": "Kringle",
                    "type": "domain",
                    "locations": [{"fragments": [{"start": 50, "end": 120}]}]
                }
            ]
        }
        
        with patch.object(collector.api_client, 'get_entry') as mock_entry:
            mock_entry.return_value = {
                "description": "Test domain",
                "member_databases": {"pfam": {"PF00024": {}}}
            }
            
            domains = collector.save_to_database(data)
            
            assert len(domains) == 1
            assert isinstance(domains[0], Domain)
            assert domains[0].interpro_id == "IPR000001"
            assert domains[0].name == "Kringle"
            assert domains[0].type == "domain"
            
            # Check domain location
            locations = test_db.query(DomainLocation).filter_by(
                domain_id=domains[0].id
            ).all()
            assert len(locations) == 1
            assert locations[0].start_position == 50
            assert locations[0].end_position == 120
    
    @patch.object(InterProCollector, 'collect_single')
    @patch.object(InterProCollector, 'save_to_database')
    def test_collect_for_proteins(self, mock_save, mock_collect):
        """Test collecting for multiple proteins."""
        collector = InterProCollector()
        
        # Create mock proteins
        proteins = [
            Mock(accession="P04637"),
            Mock(accession="P53039")
        ]
        
        mock_collect.side_effect = [
            {"uniprot_accession": "P04637", "domains": []},
            {"uniprot_accession": "P53039", "domains": []}
        ]
        mock_save.side_effect = [[], []]
        
        results = collector.collect_for_proteins(proteins)
        
        assert len(results) == 2
        assert "P04637" in results
        assert "P53039" in results
        assert mock_collect.call_count == 2
        assert mock_save.call_count == 2