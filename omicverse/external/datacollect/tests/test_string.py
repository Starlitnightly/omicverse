"""Tests for STRING API client and collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.api.string import STRINGClient
from omicverse.external.datacollect.collectors.string_collector import STRINGCollector
from omicverse.external.datacollect.models.protein import Protein
from omicverse.external.datacollect.models.interaction import ProteinInteraction


@pytest.fixture
def mock_string_id_response():
    """Sample STRING ID mapping response."""
    return [
        {
            "queryItem": "TP53",
            "queryIndex": 0,
            "stringId": "9606.ENSP00000269305",
            "ncbiTaxonId": 9606,
            "taxonName": "Homo sapiens",
            "preferredName": "TP53",
            "annotation": "tumor protein p53"
        }
    ]


@pytest.fixture
def mock_string_network_response():
    """Sample STRING network response."""
    return [
        {
            "stringId_A": "9606.ENSP00000269305",
            "stringId_B": "9606.ENSP00000353099",
            "preferredName_A": "TP53",
            "preferredName_B": "MDM2",
            "ncbiTaxonId": 9606,
            "score": 0.999,
            "nscore": 0.0,
            "fscore": 0.0,
            "pscore": 0.0,
            "ascore": 0.538,
            "escore": 0.999,
            "dscore": 0.999,
            "tscore": 0.999
        }
    ]


@pytest.fixture
def mock_string_enrichment_response():
    """Sample STRING enrichment response."""
    return [
        {
            "term": "GO:0006915",
            "description": "apoptotic process",
            "number_of_genes": 5,
            "number_of_genes_in_background": 1000,
            "ncbiTaxonId": 9606,
            "inputGenes": ["TP53", "MDM2"],
            "preferredNames": ["TP53", "MDM2"],
            "p_value": 0.00001,
            "fdr": 0.0001
        }
    ]


class TestSTRINGClient:
    """Test STRING API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = STRINGClient()
        assert "string-db.org/api" in client.base_url
        assert client.rate_limit == 10
        assert client.api_version == "v11"
    
    @patch.object(STRINGClient, 'post')
    def test_get_string_ids(self, mock_post, mock_string_id_response):
        """Test mapping to STRING IDs."""
        mock_response = Mock()
        mock_response.json.return_value = mock_string_id_response
        mock_post.return_value = mock_response
        
        client = STRINGClient()
        result = client.get_string_ids(["TP53"], 9606)
        
        mock_post.assert_called_once_with(
            "/v11/json/get_string_ids",
            data={
                "identifiers": "TP53",
                "species": 9606,
                "echo_query": 1,
                "format": "json"
            }
        )
        assert len(result) == 1
        assert result[0]["stringId"] == "9606.ENSP00000269305"
        assert result[0]["preferredName"] == "TP53"
    
    @patch.object(STRINGClient, 'get')
    def test_get_network(self, mock_get, mock_string_network_response):
        """Test getting network."""
        mock_response = Mock()
        mock_response.json.return_value = mock_string_network_response
        mock_get.return_value = mock_response
        
        client = STRINGClient()
        result = client.get_network(["9606.ENSP00000269305"], 9606)
        
        mock_get.assert_called_once_with(
            "/v11/json/network",
            params={
                "identifiers": "9606.ENSP00000269305",
                "species": 9606,
                "required_score": 400,
                "network_type": "functional",
                "add_nodes": 0
            }
        )
        assert len(result) == 1
        assert result[0]["preferredName_A"] == "TP53"
        assert result[0]["preferredName_B"] == "MDM2"
        assert result[0]["score"] == 0.999
    
    @patch.object(STRINGClient, 'get')
    def test_get_interaction_partners(self, mock_get):
        """Test getting interaction partners."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "stringId_A": "9606.ENSP00000269305",
                "stringId_B": "9606.ENSP00000353099",
                "preferredName_A": "TP53",
                "preferredName_B": "MDM2",
                "score": 0.999
            }
        ]
        mock_get.return_value = mock_response
        
        client = STRINGClient()
        result = client.get_interaction_partners(["9606.ENSP00000269305"], 9606)
        
        mock_get.assert_called_once_with(
            "/v11/json/interaction_partners",
            params={
                "identifiers": "9606.ENSP00000269305",
                "species": 9606,
                "required_score": 400,
                "limit": 10
            }
        )
    
    @patch.object(STRINGClient, 'get')
    def test_get_enrichment(self, mock_get, mock_string_enrichment_response):
        """Test functional enrichment."""
        mock_response = Mock()
        mock_response.json.return_value = mock_string_enrichment_response
        mock_get.return_value = mock_response
        
        client = STRINGClient()
        result = client.get_enrichment(["9606.ENSP00000269305"], 9606)
        
        assert len(result) == 1
        assert result[0]["term"] == "GO:0006915"
        assert result[0]["p_value"] == 0.00001
    
    @patch.object(STRINGClient, 'get')
    def test_get_actions(self, mock_get):
        """Test getting interaction actions."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "stringId_A": "9606.ENSP00000269305",
                "stringId_B": "9606.ENSP00000353099",
                "mode": "inhibition",
                "a_is_acting": True,
                "score": 0.999
            }
        ]
        mock_get.return_value = mock_response
        
        client = STRINGClient()
        result = client.get_actions(["9606.ENSP00000269305"], 9606)
        
        assert len(result) == 1
        assert result[0]["mode"] == "inhibition"


class TestSTRINGCollector:
    """Test STRING collector."""
    
    @patch('src.collectors.string_collector.STRINGClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = STRINGCollector()
        mock_client_class.assert_called_once()
        assert collector.default_species == 9606
        assert collector.default_score_threshold == 400
    
    def test_collect_single(self, mock_string_id_response, mock_string_network_response):
        """Test collecting single protein."""
        collector = STRINGCollector()
        
        with patch.object(collector.api_client, 'get_string_ids') as mock_ids:
            with patch.object(collector.api_client, 'get_interaction_partners') as mock_partners:
                with patch.object(collector.api_client, 'get_network') as mock_network:
                    with patch.object(collector.api_client, 'get_actions') as mock_actions:
                        with patch.object(collector.api_client, 'get_ppi_enrichment') as mock_enrich:
                            mock_ids.return_value = mock_string_id_response
                            mock_partners.return_value = [
                                {
                                    "stringId_B": "9606.ENSP00000353099",
                                    "preferredName_B": "MDM2",
                                    "score": 0.999
                                }
                            ]
                            mock_network.return_value = mock_string_network_response
                            mock_actions.return_value = []
                            mock_enrich.return_value = {"p_value": 0.001}
                            
                            result = collector.collect_single("TP53")
                            
                            assert result["identifier"] == "TP53"
                            assert result["string_id"] == "9606.ENSP00000269305"
                            assert result["preferred_name"] == "TP53"
                            assert result["interaction_count"] == 1
                            assert result["enrichment"]["p_value"] == 0.001
    
    def test_save_to_database(self, test_db):
        """Test saving to database."""
        collector = STRINGCollector(db_session=test_db)
        
        # Create test proteins
        protein1 = Protein(
            id="test_protein_1",
            accession="P04637",
            protein_name="TP53",
            gene_name="TP53",
            sequence="MVLSEGEWQLVLHVWAK",
            sequence_length=17,
            organism_id=9606,
            source="test"
        )
        protein2 = Protein(
            id="test_protein_2",
            accession="Q00987",
            protein_name="MDM2",
            gene_name="MDM2",
            sequence="MVLSEGEWQLVLHVWAK",
            sequence_length=17,
            organism_id=9606,
            source="test"
        )
        test_db.add(protein1)
        test_db.add(protein2)
        test_db.commit()
        
        data = {
            "identifier": "TP53",
            "string_id": "9606.ENSP00000269305",
            "preferred_name": "TP53",
            "species": 9606,
            "interactions": [
                {
                    "stringId_B": "9606.ENSP00000353099",
                    "preferredName_B": "MDM2",
                    "score": 0.999,
                    "nscore": 0.0,
                    "escore": 0.999
                }
            ],
            "actions": []
        }
        
        interactions = collector.save_to_database(data)
        
        assert len(interactions) == 1
        assert isinstance(interactions[0], ProteinInteraction)
        assert interactions[0].confidence_score == 0.999
        assert interactions[0].source == "STRING"
        
        # Check that STRING ID was updated
        test_db.refresh(protein1)
        assert protein1.string_id == "9606.ENSP00000269305"
    
    def test_collect_network(self, mock_string_id_response):
        """Test collecting network for multiple proteins."""
        collector = STRINGCollector()
        
        with patch.object(collector.api_client, 'get_string_ids') as mock_ids:
            with patch.object(collector.api_client, 'get_network') as mock_network:
                with patch.object(collector.api_client, 'get_enrichment') as mock_enrich:
                    with patch.object(collector.api_client, 'get_ppi_enrichment') as mock_ppi:
                        mock_ids.return_value = mock_string_id_response * 2
                        mock_network.return_value = [
                            {
                                "stringId_A": "9606.ENSP00000269305",
                                "stringId_B": "9606.ENSP00000353099"
                            }
                        ]
                        mock_enrich.return_value = []
                        mock_ppi.return_value = {"p_value": 0.001}
                        
                        result = collector.collect_network(["TP53", "MDM2"])
                        
                        assert result["node_count"] == 2
                        assert result["edge_count"] == 1
                        assert result["ppi_enrichment"]["p_value"] == 0.001
    
    @patch.object(STRINGCollector, '_get_protein_by_identifier')
    def test_create_or_update_interaction(self, mock_get_protein, test_db):
        """Test creating/updating interactions."""
        collector = STRINGCollector(db_session=test_db)
        
        # Create test proteins
        protein1 = Mock(id="p1", string_id="9606.ENSP00000269305")
        protein2 = Mock(id="p2", string_id="9606.ENSP00000353099")
        
        interaction_data = {
            "score": 0.999,
            "nscore": 0.1,
            "escore": 0.9
        }
        
        # Test creating new interaction
        interaction = collector._create_or_update_interaction(
            protein1, protein2, interaction_data
        )
        
        assert interaction.protein1_id == "p1"
        assert interaction.protein2_id == "p2"
        assert interaction.confidence_score == 0.999
        assert interaction.source == "STRING"
        assert "experimental" in interaction.evidence