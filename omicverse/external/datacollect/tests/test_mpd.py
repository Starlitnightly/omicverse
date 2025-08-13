"""Tests for Mouse Phenome Database (MPD) API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.mpd import MPDClient


class TestMPDClient:
    """Test MPD API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = MPDClient()
        assert client.base_url == "https://phenome.jax.org"
        assert client.rate_limit == 1.0

    def test_get_default_headers(self):
        """Test default headers."""
        client = MPDClient()
        headers = client.get_default_headers()
        
        assert headers["User-Agent"] == "BioinformaticsCollector/1.0"
        assert headers["Accept"] == "application/json"

    @patch.object(MPDClient, 'get')
    def test_query_mpd_all_params(self, mock_get):
        """Test querying MPD with all parameters."""
        mock_response = {
            "data": [
                {
                    "measure_id": 12345,
                    "measure_name": "Body weight",
                    "strain": "C57BL/6J",
                    "dataset_id": 678,
                    "value": 25.3,
                    "units": "g",
                    "sex": "F",
                    "age": "8 weeks"
                }
            ],
            "total": 1
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.query_mpd(measure_id=12345, strain="C57BL/6J", dataset_id=678)
        
        mock_get.assert_called_once_with("/api/pheno/retrieve", params={
            "measnum": 12345,
            "strain": "C57BL/6J",
            "dataset": 678
        })
        assert result == mock_response

    @patch.object(MPDClient, 'get')
    def test_query_mpd_partial_params(self, mock_get):
        """Test querying MPD with partial parameters."""
        mock_response = {"data": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.query_mpd(strain="BALB/cJ")
        
        mock_get.assert_called_once_with("/api/pheno/retrieve", params={
            "strain": "BALB/cJ"
        })
        assert result == mock_response

    @patch.object(MPDClient, 'get')
    def test_query_mpd_no_params(self, mock_get):
        """Test querying MPD with no parameters."""
        mock_response = {"data": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.query_mpd()
        
        mock_get.assert_called_once_with("/api/pheno/retrieve", params={})
        assert result == mock_response

    @patch.object(MPDClient, 'get')
    def test_search_strains(self, mock_get):
        """Test searching for strains."""
        mock_response = {
            "strains": [
                {
                    "strain_name": "C57BL/6J",
                    "panel": "inbred",
                    "strain_id": 1,
                    "description": "C57 black 6 inbred strain"
                },
                {
                    "strain_name": "BALB/cJ",
                    "panel": "inbred", 
                    "strain_id": 2,
                    "description": "BALB c inbred strain"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.search_strains(keyword="C57", panel="inbred")
        
        mock_get.assert_called_once_with("/api/strains/search", params={
            "q": "C57",
            "panel": "inbred"
        })
        assert len(result) == 2
        assert result[0]["strain_name"] == "C57BL/6J"

    @patch.object(MPDClient, 'get')
    def test_search_strains_no_results(self, mock_get):
        """Test searching strains with no results."""
        mock_response = {"strains": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.search_strains(keyword="NONEXISTENT")
        
        assert result == []

    @patch.object(MPDClient, 'get')
    def test_search_strains_missing_strains_key(self, mock_get):
        """Test searching strains when strains key is missing."""
        mock_response = {"other_data": "value"}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.search_strains(keyword="test")
        
        assert result == []

    @patch.object(MPDClient, 'get')
    def test_get_strain_info(self, mock_get):
        """Test getting strain information."""
        mock_response = {
            "strain_name": "C57BL/6J",
            "strain_id": 1,
            "panel": "inbred",
            "description": "C57 black 6 inbred strain",
            "synonyms": ["B6", "C57BL/6"],
            "origin": "The Jackson Laboratory",
            "phenotypes": ["behavior", "metabolism"]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_strain_info("C57BL/6J")
        
        mock_get.assert_called_once_with("/api/strains/info", params={"strain": "C57BL/6J"})
        assert result["strain_name"] == "C57BL/6J"
        assert result["strain_id"] == 1

    @patch.object(MPDClient, 'get')
    def test_search_phenotypes(self, mock_get):
        """Test searching for phenotypes."""
        mock_response = {
            "measures": [
                {
                    "measure_id": 12345,
                    "measure_name": "Body weight",
                    "category": "morphology",
                    "description": "Total body weight in grams",
                    "units": "g"
                },
                {
                    "measure_id": 12346,
                    "measure_name": "Bone density",
                    "category": "morphology",
                    "description": "Bone mineral density",
                    "units": "g/cm2"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.search_phenotypes("weight", category="morphology")
        
        mock_get.assert_called_once_with("/api/pheno/search", params={
            "q": "weight",
            "category": "morphology"
        })
        assert len(result) == 2
        assert result[0]["measure_name"] == "Body weight"

    @patch.object(MPDClient, 'get')
    def test_search_phenotypes_no_category(self, mock_get):
        """Test searching phenotypes without category filter."""
        mock_response = {"measures": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.search_phenotypes("behavior")
        
        mock_get.assert_called_once_with("/api/pheno/search", params={"q": "behavior"})
        assert result == []

    @patch.object(MPDClient, 'get')
    def test_get_measure_data(self, mock_get):
        """Test getting measure data."""
        mock_response = {
            "measure_id": 12345,
            "measure_name": "Body weight",
            "data": [
                {
                    "strain": "C57BL/6J",
                    "value": 25.3,
                    "sex": "F",
                    "age": "8 weeks",
                    "n": 10
                }
            ],
            "statistics": {
                "mean": 25.3,
                "std": 2.1,
                "min": 22.1,
                "max": 28.5
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_measure_data(12345, sex="F")
        
        mock_get.assert_called_once_with("/api/pheno/measure", params={
            "measnum": 12345,
            "sex": "F"
        })
        assert result["measure_id"] == 12345
        assert len(result["data"]) == 1

    @patch.object(MPDClient, 'get')
    def test_get_measure_data_no_sex(self, mock_get):
        """Test getting measure data without sex filter."""
        mock_response = {"measure_id": 12345, "data": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_measure_data(12345)
        
        mock_get.assert_called_once_with("/api/pheno/measure", params={"measnum": 12345})
        assert result["measure_id"] == 12345

    @patch.object(MPDClient, 'get')
    def test_get_dataset_info(self, mock_get):
        """Test getting dataset information."""
        mock_response = {
            "dataset_id": 678,
            "dataset_name": "Metabolic phenotypes in inbred strains",
            "description": "Comprehensive metabolic phenotyping study",
            "investigators": ["Dr. Smith", "Dr. Jones"],
            "publication": "PMC12345678",
            "strains": ["C57BL/6J", "BALB/cJ", "DBA/2J"],
            "measures": [12345, 12346, 12347]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_dataset_info(678)
        
        mock_get.assert_called_once_with("/api/datasets/info", params={"dataset": 678})
        assert result["dataset_id"] == 678
        assert len(result["strains"]) == 3

    @patch.object(MPDClient, 'get')
    def test_get_qtl_data(self, mock_get):
        """Test getting QTL data."""
        mock_response = {
            "qtls": [
                {
                    "qtl_id": "QTL001",
                    "chromosome": "1",
                    "start_mb": 10.5,
                    "end_mb": 15.2,
                    "peak_mb": 12.8,
                    "lod_score": 4.2,
                    "trait": "body weight",
                    "strain_cross": "B6xD2"
                },
                {
                    "qtl_id": "QTL002",
                    "chromosome": "1",
                    "start_mb": 20.1,
                    "end_mb": 25.8,
                    "peak_mb": 23.4,
                    "lod_score": 3.8,
                    "trait": "glucose tolerance",
                    "strain_cross": "B6xD2"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_qtl_data(chromosome="1", start_mb=10.0, end_mb=30.0)
        
        mock_get.assert_called_once_with("/api/qtl/retrieve", params={
            "chr": "1",
            "start": 10.0,
            "end": 30.0
        })
        assert len(result) == 2
        assert result[0]["qtl_id"] == "QTL001"

    @patch.object(MPDClient, 'get')
    def test_get_qtl_data_no_params(self, mock_get):
        """Test getting QTL data without parameters."""
        mock_response = {"qtls": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_qtl_data()
        
        mock_get.assert_called_once_with("/api/qtl/retrieve", params={})
        assert result == []

    @patch.object(MPDClient, 'get')
    def test_get_gene_expression(self, mock_get):
        """Test getting gene expression data."""
        mock_response = {
            "gene_symbol": "Actb",
            "gene_id": "ENSMUSG00000029580",
            "expression_data": [
                {
                    "strain": "C57BL/6J",
                    "tissue": "liver",
                    "expression_level": 1234.5,
                    "units": "TPM",
                    "age": "8 weeks",
                    "sex": "M"
                }
            ],
            "statistics": {
                "mean": 1234.5,
                "std": 145.2
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_gene_expression("Actb", tissue="liver")
        
        mock_get.assert_called_once_with("/api/expression/gene", params={
            "gene": "Actb",
            "tissue": "liver"
        })
        assert result["gene_symbol"] == "Actb"
        assert len(result["expression_data"]) == 1

    @patch.object(MPDClient, 'get')
    def test_get_gene_expression_no_tissue(self, mock_get):
        """Test getting gene expression without tissue filter."""
        mock_response = {"gene_symbol": "Actb", "expression_data": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_gene_expression("Actb")
        
        mock_get.assert_called_once_with("/api/expression/gene", params={"gene": "Actb"})
        assert result["gene_symbol"] == "Actb"

    @patch.object(MPDClient, 'get')
    def test_compare_strains(self, mock_get):
        """Test comparing strains."""
        mock_response = {
            "comparison": {
                "strain1": "C57BL/6J",
                "strain2": "BALB/cJ",
                "measure_id": 12345,
                "measure_name": "Body weight",
                "strain1_stats": {
                    "mean": 25.3,
                    "std": 2.1,
                    "n": 10
                },
                "strain2_stats": {
                    "mean": 23.8,
                    "std": 1.9,
                    "n": 12
                },
                "statistics": {
                    "t_test_p_value": 0.035,
                    "effect_size": 0.76,
                    "significant": True
                }
            }
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.compare_strains("C57BL/6J", "BALB/cJ", measure_id=12345)
        
        mock_get.assert_called_once_with("/api/strains/compare", params={
            "strain1": "C57BL/6J",
            "strain2": "BALB/cJ",
            "measnum": 12345
        })
        assert result["comparison"]["strain1"] == "C57BL/6J"

    @patch.object(MPDClient, 'get')
    def test_compare_strains_no_measure(self, mock_get):
        """Test comparing strains without specific measure."""
        mock_response = {"comparison": {}}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.compare_strains("C57BL/6J", "BALB/cJ")
        
        mock_get.assert_called_once_with("/api/strains/compare", params={
            "strain1": "C57BL/6J",
            "strain2": "BALB/cJ"
        })

    @patch.object(MPDClient, 'get')
    def test_get_correlations(self, mock_get):
        """Test getting correlations."""
        mock_response = {
            "correlations": [
                {
                    "measure_id": 12346,
                    "measure_name": "Bone density",
                    "correlation": 0.82,
                    "p_value": 0.001,
                    "n_strains": 15
                },
                {
                    "measure_id": 12347,
                    "measure_name": "Muscle mass",
                    "correlation": 0.75,
                    "p_value": 0.003,
                    "n_strains": 15
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_correlations(12345, min_correlation=0.7)
        
        mock_get.assert_called_once_with("/api/pheno/correlations", params={
            "measnum": 12345,
            "min_r": 0.7
        })
        assert len(result) == 2
        assert result[0]["correlation"] == 0.82

    @patch.object(MPDClient, 'get')
    def test_get_correlations_default_threshold(self, mock_get):
        """Test getting correlations with default threshold."""
        mock_response = {"correlations": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.get_correlations(12345)
        
        mock_get.assert_called_once_with("/api/pheno/correlations", params={
            "measnum": 12345,
            "min_r": 0.7
        })
        assert result == []

    @responses.activate
    def test_http_request_success(self):
        """Test successful HTTP request."""
        responses.add(
            responses.GET,
            "https://phenome.jax.org/api/strains/search",
            json={"strains": [{"strain_name": "C57BL/6J"}]},
            status=200
        )
        
        client = MPDClient()
        result = client.search_strains(keyword="C57")
        
        assert len(result) == 1
        assert result[0]["strain_name"] == "C57BL/6J"

    @responses.activate
    def test_http_request_error(self):
        """Test HTTP request error handling."""
        responses.add(
            responses.GET,
            "https://phenome.jax.org/api/strains/search",
            status=500
        )
        
        client = MPDClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.search_strains(keyword="C57")

    @responses.activate
    def test_http_request_timeout(self):
        """Test HTTP request timeout handling."""
        responses.add(
            responses.GET,
            "https://phenome.jax.org/api/strains/search",
            body=requests.exceptions.Timeout()
        )
        
        client = MPDClient()
        
        with pytest.raises(requests.exceptions.Timeout):
            client.search_strains(keyword="C57")

    @patch.object(MPDClient, 'get')
    def test_edge_case_empty_response(self, mock_get):
        """Test handling of completely empty response."""
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = {}
        mock_get.return_value = mock_response_obj
        
        client = MPDClient()
        result = client.search_strains(keyword="test")
        
        assert result == []

    @patch.object(MPDClient, 'get')
    def test_edge_case_none_response(self, mock_get):
        """Test handling of None response."""
        mock_get.return_value = None
        
        client = MPDClient()
        
        with pytest.raises(AttributeError):
            client.search_strains(keyword="test")