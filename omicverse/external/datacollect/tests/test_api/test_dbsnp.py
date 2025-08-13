"""Tests for dbSNP API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.api.dbsnp import dbSNPClient


class TestdbSNPClient:
    """Test cases for dbSNP API client."""
    
    @pytest.fixture
    def client(self):
        """Create a dbSNP client instance."""
        return dbSNPClient()
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = Mock()
        response.json.return_value = {"test": "data"}
        response.raise_for_status = Mock()
        return response
    
    @pytest.fixture
    def variant_data(self):
        """Sample variant data structure."""
        return {
            "refsnp_id": "7412",
            "primary_snapshot_data": {
                "variant_type": "snv",
                "placements_with_allele": [{
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [{
                            "is_chromosome": True,
                            "sequence_name": "NC_000019.10",
                            "traits": [{"trait_name": "Chr19"}]
                        }]
                    },
                    "alleles": [{
                        "location": {"position": 44908684},
                        "allele": {
                            "spdi": {
                                "deleted_sequence": "C",
                                "inserted_sequence": "T"
                            }
                        }
                    }]
                }],
                "allele_annotations": [{
                    "frequency": [{
                        "study_name": "1000Genomes",
                        "populations": [{
                            "population": "Total",
                            "allele_count": 100,
                            "total_count": 5008
                        }]
                    }]
                }]
            }
        }
    
    def test_get_variant_by_rsid(self, client, mock_response, variant_data):
        """Test getting variant by RS ID."""
        mock_response.json.return_value = variant_data
        
        with patch.object(client, 'get', return_value=mock_response):
            result = client.get_variant_by_rsid("rs7412")
            
            assert result["refsnp_id"] == "7412"
            assert "primary_snapshot_data" in result
            
            # Check that rs prefix was stripped
            client.get.assert_called_once_with("/beta/refsnp/7412")
    
    def test_get_variant_allele_annotations(self, client, mock_response):
        """Test getting allele annotations."""
        mock_response.json.return_value = {
            "alleles": [
                {"allele_id": "1", "annotation": "test1"},
                {"allele_id": "2", "annotation": "test2"}
            ]
        }
        
        with patch.object(client, 'get', return_value=mock_response):
            result = client.get_variant_allele_annotations("7412")
            
            assert len(result) == 2
            assert result[0]["allele_id"] == "1"
    
    def test_search_by_gene(self, client):
        """Test searching by gene."""
        # Mock esearch response
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {
                "idlist": ["123", "456"]
            }
        }
        search_response.raise_for_status = Mock()
        
        # Mock esummary response
        summary_response = Mock()
        summary_response.json.return_value = {
            "result": {
                "uids": ["123", "456"],
                "123": {"snp_id": "7412"},
                "456": {"snp_id": "429358"}
            }
        }
        summary_response.raise_for_status = Mock()
        
        with patch.object(client.session, 'get', side_effect=[search_response, summary_response]):
            results = client.search_by_gene("APOE", "human", 10)
            
            assert len(results) == 2
            assert "rs7412" in results
            assert "rs429358" in results
    
    def test_search_by_position(self, client):
        """Test searching by genomic position."""
        # Mock search response
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {
                "idlist": ["789", "012"]
            }
        }
        search_response.raise_for_status = Mock()
        
        with patch.object(client.session, 'get', return_value=search_response):
            with patch.object(client, '_convert_ids_to_rsids', return_value=["rs789", "rs012"]):
                results = client.search_by_position("19", 44908684, 44909393, "human")
                
                assert len(results) == 2
                assert "rs789" in results
    
    def test_get_clinical_significance(self, client, variant_data):
        """Test getting clinical significance."""
        # Add clinical data to variant
        variant_data["primary_snapshot_data"]["allele_annotations"] = [{
            "assembly_annotation": [{
                "genes": [{
                    "name": "APOE",
                    "clinical_significance": "pathogenic",
                    "review_status": "criteria provided"
                }]
            }]
        }]
        
        with patch.object(client, 'get_variant_by_rsid', return_value=variant_data):
            result = client.get_clinical_significance("7412")
            
            assert len(result) == 1
            assert result[0]["gene"] == "APOE"
            assert result[0]["clinical_significance"] == "pathogenic"
    
    def test_get_population_frequency(self, client, variant_data):
        """Test getting population frequency."""
        with patch.object(client, 'get_variant_by_rsid', return_value=variant_data):
            result = client.get_population_frequency("7412")
            
            assert result["rsid"] == "rs7412"
            assert result["global_frequency"] == pytest.approx(100/5008)
            assert "Total" in result["population_frequencies"]
    
    def test_get_variant_consequences(self, client, variant_data):
        """Test getting variant consequences."""
        # Add gene consequence data
        variant_data["primary_snapshot_data"]["placements_with_allele"][0]["placement_annot"][
            "seq_id_traits_by_assembly"][0]["genes"] = [{
                "name": "APOE",
                "id": "348",
                "consequence_type": "missense_variant"
            }]
        variant_data["primary_snapshot_data"]["placements_with_allele"][0]["alleles"][0]["hgvs"] = "NP_000032.1:p.Arg176Cys"
        
        with patch.object(client, 'get_variant_by_rsid', return_value=variant_data):
            result = client.get_variant_consequences("7412")
            
            assert len(result) == 1
            assert result[0]["gene_name"] == "APOE"
            assert result[0]["consequence_type"] == "missense_variant"
            assert result[0]["hgvs"] == "NP_000032.1:p.Arg176Cys"
    
    def test_convert_ids_to_rsids(self, client):
        """Test converting internal IDs to RS IDs."""
        summary_response = Mock()
        summary_response.json.return_value = {
            "result": {
                "uids": ["123", "456"],
                "123": {"snp_id": "7412"},
                "456": {"snp_id": "429358"}
            }
        }
        
        with patch.object(client.session, 'get', return_value=summary_response):
            result = client._convert_ids_to_rsids(["123", "456"])
            
            assert len(result) == 2
            assert "rs7412" in result
            assert "rs429358" in result
    
    def test_headers_with_api_key(self, client):
        """Test that API key is included in headers."""
        with patch.object(client, 'api_key', 'test_key_123'):
            headers = client.get_default_headers()
            
            assert headers["api-key"] == "test_key_123"
            assert "User-Agent" in headers
    
    def test_error_handling(self, client, mock_response):
        """Test error handling."""
        mock_response.json.side_effect = Exception("JSON Error")
        
        with patch.object(client, 'get', return_value=mock_response):
            with pytest.raises(Exception, match="JSON Error"):
                client.get_variant_by_rsid("rs7412")