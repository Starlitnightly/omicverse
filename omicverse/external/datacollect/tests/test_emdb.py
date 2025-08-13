"""Tests for EMDB API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.emdb import EMDBClient


class TestEMDBClient:
    """Test EMDB API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = EMDBClient()
        assert client.base_url == "https://www.ebi.ac.uk/emdb"
        assert client.rate_limit == 1.0

    def test_get_default_headers(self):
        """Test default headers."""
        client = EMDBClient()
        headers = client.get_default_headers()
        
        assert headers["User-Agent"] == "BioinformaticsCollector/1.0"
        assert headers["Accept"] == "application/json"

    @patch.object(EMDBClient, 'get')
    def test_query_emdb_basic(self, mock_get):
        """Test basic EMDB query."""
        mock_response = {
            "total_count": 2,
            "entries": [
                {
                    "emdb_id": "EMD-1234",
                    "title": "Structure of human ribosome",
                    "resolution": 3.2,
                    "method": "single particle",
                    "organism": "Homo sapiens",
                    "authors": ["Smith, J.", "Doe, A."],
                    "deposit_date": "2020-01-15",
                    "release_date": "2020-04-15"
                },
                {
                    "emdb_id": "EMD-5678",
                    "title": "Bacterial flagellar motor",
                    "resolution": 4.1,
                    "method": "tomography",
                    "organism": "Escherichia coli",
                    "authors": ["Brown, K.", "Wilson, L."],
                    "deposit_date": "2020-02-20",
                    "release_date": "2020-05-20"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.query_emdb(keyword="ribosome")
        
        mock_get.assert_called_once_with("/api/search", params={"q": "ribosome"})
        assert result["total_count"] == 2
        assert len(result["entries"]) == 2
        assert result["entries"][0]["emdb_id"] == "EMD-1234"

    @patch.object(EMDBClient, 'get')
    def test_query_emdb_with_filters(self, mock_get):
        """Test EMDB query with multiple filters."""
        mock_response = {
            "total_count": 1,
            "entries": [
                {
                    "emdb_id": "EMD-9999",
                    "title": "High-resolution structure",
                    "resolution": 2.5,
                    "method": "single particle",
                    "organism": "Homo sapiens"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.query_emdb(
            keyword="ribosome",
            resolution_min=2.0,
            resolution_max=3.0,
            organism="Homo sapiens"
        )
        
        expected_params = {
            "q": "ribosome",
            "resolution_from": 2.0,
            "resolution_to": 3.0,
            "organism": "Homo sapiens"
        }
        mock_get.assert_called_once_with("/api/search", params=expected_params)
        assert result["entries"][0]["resolution"] == 2.5

    @patch.object(EMDBClient, 'get')
    def test_get_entry(self, mock_get):
        """Test getting detailed EMDB entry information."""
        mock_response = {
            "emdb_id": "EMD-1234",
            "title": "Crystal structure of human insulin receptor",
            "authors": [
                {
                    "name": "Smith, J.A.",
                    "orcid": "0000-0000-0000-0000"
                }
            ],
            "sample": {
                "name": "Human insulin receptor",
                "organism": {
                    "scientific_name": "Homo sapiens",
                    "taxonomy_id": 9606
                },
                "molecular_weight": 156000,
                "oligomeric_state": "monomer"
            },
            "experiment": {
                "method": "single particle",
                "resolution": 3.8,
                "voltage": 300,
                "microscope": "FEI Titan Krios",
                "detector": "Gatan K2",
                "magnification": 130000,
                "pixel_size": 1.06
            },
            "processing": {
                "software": "RELION",
                "particles_picked": 1500000,
                "particles_final": 250000,
                "symmetry": "C1"
            },
            "deposition": {
                "deposit_date": "2020-01-15",
                "release_date": "2020-04-15",
                "revision_date": "2020-06-01"
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_entry("EMD-1234")
        
        mock_get.assert_called_once_with("/api/entry/EMD-1234")
        assert result["emdb_id"] == "EMD-1234"
        assert result["sample"]["organism"]["scientific_name"] == "Homo sapiens"
        assert result["experiment"]["resolution"] == 3.8

    @patch.object(EMDBClient, 'get')
    def test_search_by_author(self, mock_get):
        """Test searching entries by author."""
        mock_response = {
            "entries": [
                {
                    "emdb_id": "EMD-1111",
                    "title": "Structure A",
                    "authors": ["Smith, J.", "Brown, A."],
                    "resolution": 3.5
                },
                {
                    "emdb_id": "EMD-2222",
                    "title": "Structure B", 
                    "authors": ["Smith, J.", "Wilson, K."],
                    "resolution": 4.2
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.search_by_author("Smith, J.", limit=50)
        
        expected_params = {"author": "Smith, J.", "limit": 50}
        mock_get.assert_called_once_with("/api/search/author", params=expected_params)
        assert len(result) == 2
        assert result[0]["emdb_id"] == "EMD-1111"

    @patch.object(EMDBClient, 'get')
    def test_search_by_method(self, mock_get):
        """Test searching by reconstruction method."""
        mock_response = {
            "entries": [
                {
                    "emdb_id": "EMD-3333",
                    "title": "Single particle structure",
                    "method": "single particle",
                    "resolution": 2.8,
                    "year": 2021
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.search_by_method("single particle", year_from=2020)
        
        expected_params = {"method": "single particle", "year_from": 2020}
        mock_get.assert_called_once_with("/api/search/method", params=expected_params)
        assert len(result) == 1
        assert result[0]["method"] == "single particle"

    @patch.object(EMDBClient, 'get')
    def test_get_sample_info(self, mock_get):
        """Test getting sample information."""
        mock_response = {
            "sample_id": "EMD-1234-sample",
            "name": "Human ribosome 80S",
            "organism": {
                "scientific_name": "Homo sapiens",
                "common_name": "human",
                "taxonomy_id": 9606,
                "strain": "HeLa",
                "expression_system": "native"
            },
            "molecular_weight": 4200000,
            "oligomeric_state": "complex",
            "natural_source": {
                "tissue": "cervical carcinoma",
                "cell_line": "HeLa",
                "cellular_location": "cytoplasm"
            },
            "buffer": {
                "ph": 7.4,
                "details": "20 mM HEPES, 150 mM KCl, 5 mM MgCl2"
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_sample_info("EMD-1234")
        
        mock_get.assert_called_once_with("/api/entry/EMD-1234/sample")
        assert result["name"] == "Human ribosome 80S"
        assert result["organism"]["taxonomy_id"] == 9606
        assert result["molecular_weight"] == 4200000

    @patch.object(EMDBClient, 'get')
    def test_get_experiment_info(self, mock_get):
        """Test getting experimental details."""
        mock_response = {
            "experiment_id": "EMD-1234-exp",
            "method": "single particle",
            "microscopy": {
                "microscope": "FEI Titan Krios",
                "voltage": 300,
                "illumination_mode": "flood beam",
                "imaging_mode": "bright field",
                "electron_source": "field emission gun"
            },
            "detector": {
                "name": "Gatan K2 Summit",
                "mode": "counting",
                "dimensions": "4096 x 4096"
            },
            "acquisition": {
                "magnification": 165000,
                "pixel_size": 0.83,
                "defocus_min": -1.5,
                "defocus_max": -3.5,
                "total_dose": 80,
                "dose_rate": 8.0,
                "exposure_time": 10.0
            },
            "processing": {
                "software": ["RELION", "cisTEM"],
                "particles_picked": 2000000,
                "particles_final": 180000,
                "resolution_method": "FSC 0.143",
                "resolution": 3.2,
                "symmetry": "C1"
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_experiment_info("EMD-1234")
        
        mock_get.assert_called_once_with("/api/entry/EMD-1234/experiment")
        assert result["method"] == "single particle"
        assert result["microscopy"]["voltage"] == 300
        assert result["processing"]["resolution"] == 3.2

    @patch.object(EMDBClient, 'get')
    def test_get_map_statistics(self, mock_get):
        """Test getting map statistics."""
        mock_response = {
            "map_id": "EMD-1234-map",
            "file_info": {
                "format": "MRC",
                "size_mb": 125.6,
                "dimensions": "320 x 320 x 320",
                "voxel_size": 1.06
            },
            "statistics": {
                "minimum": -0.234,
                "maximum": 0.456,
                "mean": 0.001,
                "standard_deviation": 0.089,
                "recommended_contour": 0.045
            },
            "symmetry": {
                "point_group": "C1",
                "space_group": "P1"
            },
            "processing_details": {
                "reconstruction_method": "single particle",
                "applied_symmetry": "C1",
                "resolution_method": "FSC 0.143",
                "resolution": 3.2
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_map_statistics("EMD-1234")
        
        mock_get.assert_called_once_with("/api/entry/EMD-1234/map_stats")
        assert result["file_info"]["format"] == "MRC"
        assert result["statistics"]["recommended_contour"] == 0.045
        assert result["processing_details"]["resolution"] == 3.2

    @patch.object(EMDBClient, 'get')
    def test_search_by_resolution(self, mock_get):
        """Test searching by resolution cutoff."""
        mock_response = {
            "entries": [
                {
                    "emdb_id": "EMD-9001",
                    "title": "High-resolution structure 1",
                    "resolution": 2.1,
                    "method": "single particle"
                },
                {
                    "emdb_id": "EMD-9002",
                    "title": "High-resolution structure 2", 
                    "resolution": 2.5,
                    "method": "tomography"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.search_by_resolution(max_resolution=3.0, min_entries=20)
        
        expected_params = {"resolution_to": 3.0, "limit": 20}
        mock_get.assert_called_once_with("/api/search/resolution", params=expected_params)
        assert len(result) == 2
        assert all(entry["resolution"] <= 3.0 for entry in result)

    @patch.object(EMDBClient, 'get')
    def test_get_related_pdb(self, mock_get):
        """Test getting related PDB entries."""
        mock_response = {
            "pdb_ids": ["7ABC", "8XYZ", "9DEF"],
            "relationships": [
                {
                    "pdb_id": "7ABC",
                    "relationship_type": "associated structure",
                    "details": "Atomic model fitted into EM map"
                },
                {
                    "pdb_id": "8XYZ",
                    "relationship_type": "related structure",
                    "details": "Same sample, different conditions"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_related_pdb("EMD-1234")
        
        mock_get.assert_called_once_with("/api/entry/EMD-1234/pdb")
        assert result == ["7ABC", "8XYZ", "9DEF"]

    @patch.object(EMDBClient, 'get')
    def test_get_citations(self, mock_get):
        """Test getting citations for an entry."""
        mock_response = {
            "citations": [
                {
                    "citation_id": "primary",
                    "title": "High-resolution structure of human ribosome",
                    "authors": ["Smith, J.A.", "Brown, K.L.", "Wilson, M.R."],
                    "journal": "Nature",
                    "volume": "585",
                    "pages": "123-128",
                    "year": 2020,
                    "doi": "10.1038/s41586-020-2345-1",
                    "pmid": "32612345"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_citations("EMD-1234")
        
        mock_get.assert_called_once_with("/api/entry/EMD-1234/citations")
        assert len(result) == 1
        assert result[0]["journal"] == "Nature"
        assert result[0]["doi"] == "10.1038/s41586-020-2345-1"

    @patch.object(EMDBClient, 'get')
    def test_search_complexes(self, mock_get):
        """Test searching for protein complexes."""
        mock_response = {
            "entries": [
                {
                    "emdb_id": "EMD-4444",
                    "title": "Human ribosome complex",
                    "complex_name": "ribosome",
                    "species": "Homo sapiens",
                    "resolution": 2.9,
                    "subunits": ["60S", "40S"]
                },
                {
                    "emdb_id": "EMD-5555",
                    "title": "Bacterial ribosome complex",
                    "complex_name": "ribosome", 
                    "species": "Escherichia coli",
                    "resolution": 3.1,
                    "subunits": ["50S", "30S"]
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.search_complexes("ribosome", species="Homo sapiens")
        
        expected_params = {"complex": "ribosome", "species": "Homo sapiens"}
        mock_get.assert_called_once_with("/api/search/complex", params=expected_params)
        assert len(result) == 2
        assert result[0]["complex_name"] == "ribosome"

    @patch.object(EMDBClient, 'get')
    def test_get_statistics(self, mock_get):
        """Test getting EMDB database statistics."""
        mock_response = {
            "total_entries": 15000,
            "released_entries": 14500,
            "on_hold_entries": 500,
            "methods": {
                "single_particle": 12000,
                "tomography": 2500,
                "subtomogram_averaging": 400,
                "helical": 100
            },
            "resolution_distribution": {
                "sub_2A": 150,
                "2_3A": 2500,
                "3_4A": 4500,
                "4_5A": 3500,
                "above_5A": 3850
            },
            "organism_distribution": {
                "homo_sapiens": 4500,
                "mus_musculus": 1200,
                "escherichia_coli": 2800,
                "other": 6500
            },
            "yearly_deposits": {
                "2020": 1800,
                "2021": 2200,
                "2022": 2500,
                "2023": 2100
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_statistics()
        
        mock_get.assert_called_once_with("/api/statistics")
        assert result["total_entries"] == 15000
        assert result["methods"]["single_particle"] == 12000
        assert result["resolution_distribution"]["sub_2A"] == 150

    # Error handling tests
    @patch.object(EMDBClient, 'get')
    def test_query_emdb_no_results(self, mock_get):
        """Test handling of empty search results."""
        mock_response = {
            "total_count": 0,
            "entries": []
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.query_emdb(keyword="nonexistent")
        
        assert result["total_count"] == 0
        assert len(result["entries"]) == 0

    @patch.object(EMDBClient, 'get')
    def test_search_by_author_no_results(self, mock_get):
        """Test search by author with no results."""
        mock_response = {"entries": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.search_by_author("Nonexistent Author")
        
        assert result == []

    @patch.object(EMDBClient, 'get')
    def test_get_related_pdb_no_results(self, mock_get):
        """Test getting related PDB with no results."""
        mock_response = {"pdb_ids": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_related_pdb("EMD-9999")
        
        assert result == []

    @patch.object(EMDBClient, 'get')
    def test_get_citations_no_results(self, mock_get):
        """Test getting citations with no results."""
        mock_response = {"citations": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = EMDBClient()
        result = client.get_citations("EMD-9999")
        
        assert result == []

    @patch.object(EMDBClient, 'get')
    def test_get_entry_error_handling(self, mock_get):
        """Test error handling for get_entry."""
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        client = EMDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_entry("INVALID")

    @patch.object(EMDBClient, 'get')
    def test_query_emdb_error_handling(self, mock_get):
        """Test error handling for query_emdb."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        client = EMDBClient()
        
        with pytest.raises(requests.exceptions.RequestException):
            client.query_emdb(keyword="test")

    @responses.activate
    def test_real_http_request_query(self):
        """Test actual HTTP request for EMDB query."""
        responses.add(
            responses.GET,
            "https://www.ebi.ac.uk/emdb/api/search",
            json={
                "total_count": 1,
                "entries": [
                    {
                        "emdb_id": "EMD-1234",
                        "title": "Test structure",
                        "resolution": 3.5
                    }
                ]
            },
            status=200
        )
        
        client = EMDBClient()
        result = client.query_emdb(keyword="test")
        
        assert result["total_count"] == 1
        assert len(responses.calls) == 1
        
        # Check the request
        request = responses.calls[0].request
        assert request.headers["Accept"] == "application/json"
        assert "q=test" in request.url

    @responses.activate
    def test_real_http_request_entry(self):
        """Test actual HTTP request for entry details."""
        responses.add(
            responses.GET,
            "https://www.ebi.ac.uk/emdb/api/entry/EMD-1234",
            json={
                "emdb_id": "EMD-1234",
                "title": "Test structure",
                "experiment": {
                    "resolution": 3.5,
                    "method": "single particle"
                }
            },
            status=200
        )
        
        client = EMDBClient()
        result = client.get_entry("EMD-1234")
        
        assert result["emdb_id"] == "EMD-1234"
        assert len(responses.calls) == 1

    @responses.activate
    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        responses.add(
            responses.GET,
            "https://www.ebi.ac.uk/emdb/api/entry/INVALID",
            status=404
        )
        
        client = EMDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_entry("INVALID")

    def test_parameter_validation(self):
        """Test parameter validation and defaults."""
        client = EMDBClient()
        
        # Test query without parameters returns empty dict
        with patch.object(EMDBClient, 'get') as mock_get:
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = {}
            mock_get.return_value = mock_response_obj
            result = client.query_emdb()
            mock_get.assert_called_once_with("/api/search", params={})

    def test_search_methods_default_params(self):
        """Test search methods with default parameters."""
        client = EMDBClient()
        
        with patch.object(EMDBClient, 'get') as mock_get:
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = {"entries": []}
            mock_get.return_value = mock_response_obj
            
            # Test search_by_author with default limit
            client.search_by_author("Test Author")
            expected_params = {"author": "Test Author", "limit": 100}
            mock_get.assert_called_with("/api/search/author", params=expected_params)
            
            # Test search_by_method without year filter
            mock_get.reset_mock()
            client.search_by_method("single particle")
            expected_params = {"method": "single particle"}
            mock_get.assert_called_with("/api/search/method", params=expected_params)