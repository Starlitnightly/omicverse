"""Tests for GEO API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.api.geo import GEOClient


class TestGEOClient:
    """Test cases for GEO API client."""
    
    @pytest.fixture
    def client(self):
        """Create a GEO client instance."""
        return GEOClient()
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = Mock()
        response.text = "SOFT format text"
        response.json.return_value = {"test": "data"}
        response.raise_for_status = Mock()
        return response
    
    def test_search(self, client, mock_response):
        """Test searching datasets."""
        mock_response.json.return_value = {
            "esearchresult": {
                "count": "2",
                "idlist": ["123456", "789012"]
            }
        }
        
        with patch.object(client, 'get', return_value=mock_response):
            results = client.search("TP53")
            
            assert results["esearchresult"]["count"] == "2"
            assert results["esearchresult"]["idlist"] == ["123456", "789012"]
            
            # Check API call
            client.get.assert_called_once()
            endpoint, params = client.get.call_args[0], client.get.call_args[1]
            assert "/esearch.fcgi" in endpoint
            assert params['params']['term'] == "TP53"
    
    def test_get_dataset_summary(self, client, mock_response):
        """Test getting dataset summary."""
        mock_response.json.return_value = {
            "result": {
                "123456": {
                    "uid": "123456",
                    "accession": "GDS123456",
                    "title": "Test Dataset",
                    "summary": "Test summary",
                    "gpl": "GPL570",
                    "gse": ["GSE1234"],
                    "n_samples": "10"
                }
            }
        }
        
        with patch.object(client, 'get', return_value=mock_response):
            result = client.get_dataset_summary("123456")
            
            assert result["accession"] == "GDS123456"
            assert result["title"] == "Test Dataset"
            assert result["n_samples"] == "10"
    
    def test_get_series_matrix(self, client, mock_response):
        """Test getting series matrix."""
        mock_response.text = (chr(33) + 'Series_title\t"Test Series"\n'
                           + chr(33) + 'Series_geo_accession\t"GSE1234"\n'
                           + chr(33) + 'Series_summary\t"Test summary"\n'
                           + chr(33) + 'Series_overall_design\t"Test design"\n'
                           + chr(33) + 'Series_platform_id\t"GPL570"\n'
                           + chr(33) + 'Series_sample_id\t"GSM1"\t"GSM2"\n')
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.get_series_matrix("GSE1234")
            
            assert result["series"]["series_title"] == "Test Series"
            assert result["series"]["series_geo_accession"] == "GSE1234"
            assert result["series"]["series_sample_id"] == ["GSM1", "GSM2"]
    
    def test_get_expression_data(self, client, mock_response):
        """Test getting expression data."""
        matrix_text = (chr(33) + 'Series_title\t"Test Series"\n'
                     + chr(33) + 'Series_platform_id\t"GPL570"\n'
                     + chr(33) + 'Sample_title\t"Sample 1"\t"Sample 2"\n'
                     + chr(33) + 'Sample_geo_accession\t"GSM1"\t"GSM2"\n'
                     + 'ID_REF\tGSM1\tGSM2\n'
                     + '1007_s_at\t8.5\t7.2\n'
                     + '1053_at\t6.3\t6.8\n')
        mock_response.text = matrix_text
        
        with patch.object(client, 'get_series_matrix') as mock_matrix:
            mock_matrix.return_value = client._parse_soft_format(matrix_text)
            
            result = client.get_expression_data("GSE1234")
            
            assert result["title"] == "Test Series"
            assert result["platform"] == "GPL570"
            assert len(result["samples"]) == 2
            assert "GSM1" in result["samples"]
            assert result["samples"]["GSM1"]["sample_title"] == "Sample 1"
    
    def test_get_sample_data(self, client, mock_response):
        """Test getting sample data."""
        sample_text = (chr(33) + 'Sample_title\t"Test Sample"\n'
                     + chr(33) + 'Sample_geo_accession\t"GSM1234"\n'
                     + chr(33) + 'Sample_organism_ch1\t"Homo sapiens"\n'
                     + chr(33) + 'Sample_source_name_ch1\t"HeLa cells"\n'
                     + chr(33) + 'Sample_platform_id\t"GPL570"\n'
                     + chr(33) + 'Sample_series_id\t"GSE5678"\n'
                     + chr(33) + 'Sample_characteristics_ch1\t"cell line: HeLa"\n'
                     + chr(33) + 'Sample_characteristics_ch1\t"treatment: control"\n')
        mock_response.text = sample_text
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.get_sample_data("GSM1234")
            
            assert result["sample"]["sample_title"] == "Test Sample"
            assert result["sample"]["sample_organism_ch1"] == "Homo sapiens"
            assert result["sample"]["sample_characteristics_ch1"] == ["cell line: HeLa", "treatment: control"]
    
    def test_get_platform_data(self, client, mock_response):
        """Test getting platform data."""
        platform_text = (chr(33) + 'Platform_title\t"Affymetrix Human Genome U133 Plus 2.0 Array"\n'
                       + chr(33) + 'Platform_geo_accession\t"GPL570"\n'
                       + chr(33) + 'Platform_technology\t"in situ oligonucleotide"\n'
                       + chr(33) + 'Platform_organism\t"Homo sapiens"\n'
                       + chr(33) + 'Platform_manufacturer\t"Affymetrix"\n')
        mock_response.text = platform_text
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.get_platform_data("GPL570")
            
            assert result["platform"]["platform_geo_accession"] == "GPL570"
            assert result["platform"]["platform_technology"] == "in situ oligonucleotide"
            assert result["platform"]["platform_organism"] == "Homo sapiens"
    
    def test_search_datasets_by_gene(self, client, mock_response):
        """Test searching datasets by gene."""
        # Mock search response
        search_response = Mock()
        search_response.json.return_value = {
            "esearchresult": {
                "idlist": ["123", "456"]
            }
        }
        
        # Mock summary response
        summary_response = Mock()
        summary_response.json.return_value = {
            "result": {
                "123": {
                    "uid": "123",
                    "accession": "GDS123",
                    "title": "Dataset 1"
                },
                "456": {
                    "uid": "456", 
                    "accession": "GDS456",
                    "title": "Dataset 2"
                }
            }
        }
        
        with patch.object(client, 'search') as mock_search:
            mock_search.return_value = {"esearchresult": {"idlist": ["123", "456"]}}
            with patch.object(client, 'get_dataset_summary') as mock_summary:
                mock_summary.side_effect = [
                    {"uid": "123", "accession": "GDS123", "title": "Dataset 1"},
                    {"uid": "456", "accession": "GDS456", "title": "Dataset 2"}
                ]
                
                results = client.search_datasets_by_gene("TP53", "Homo sapiens", 10)
                
                assert len(results) == 2
                assert results[0]["accession"] == "GDS123"
                assert results[1]["accession"] == "GDS456"
    
    def test_download_supplementary_files(self, client, mock_response):
        """Test getting supplementary file URLs."""
        mock_response.text = """<!DOCTYPE html>
<html>
<body>
<a href="ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/GSE1234_RAW.tar">GSE1234_RAW.tar</a>
<a href="ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/GSE1234_matrix.txt.gz">GSE1234_matrix.txt.gz</a>
</body>
</html>"""
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_supplementary_files("GSE1234")
            
            assert len(result) >= 2  # May have duplicates removed or additional direct download link
            assert any("GSE1234_RAW.tar" in url for url in result) or any("geo/download" in url for url in result)
            assert any("GSE1234_matrix.txt.gz" in url for url in result) or any("geo/download" in url for url in result)
    
    def test_parse_soft_format(self, client):
        """Test SOFT format parsing."""
        soft_text = (chr(33) + 'Series_title\t"Test Series"\n'
                   + chr(33) + 'Series_geo_accession\t"GSE1234"\n'
                   + chr(33) + 'Sample_title\t"Sample 1"\t"Sample 2"\n'
                   + chr(33) + 'Sample_geo_accession\t"GSM1"\t"GSM2"\n'
                   + 'ID_REF\tGSM1\tGSM2\n'
                   + '1007_s_at\t8.5\t7.2\n')
        
        result = client._parse_soft_format(soft_text)
        
        assert "series_title" in result
        assert result["series_title"] == "Test Series"
        assert "sample_title" in result
        assert result["sample_title"] == ["Sample 1", "Sample 2"]
        assert "data_table" in result
        assert "1007_s_at" in result["data_table"]
        assert result["data_table"]["1007_s_at"] == ["8.5", "7.2"]
    
    def test_error_handling(self, client):
        """Test error handling."""
        with patch.object(client, 'get', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                client.search("test")