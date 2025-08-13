"""Tests for ENCODE cCRE (candidate cis-Regulatory Elements) API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.ccre import CCREClient


class TestCCREClient:
    """Test CCRE API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = CCREClient()
        assert client.base_url == "https://screen.encodeproject.org"
        assert client.rate_limit == 1.0

    def test_get_default_headers(self):
        """Test default headers."""
        client = CCREClient()
        headers = client.get_default_headers()
        
        assert headers["User-Agent"] == "BioinformaticsCollector/1.0"
        assert headers["Accept"] == "application/json"

    @patch.object(CCREClient, 'get')
    def test_region_to_ccre_screen(self, mock_get):
        """Test finding cCREs in a genomic region."""
        mock_response = {
            "ccres": [
                {
                    "accession": "EH37E0002345",
                    "chromosome": "chr1",
                    "start": 1000000,
                    "end": 1002000,
                    "type": "promoter-like",
                    "biosample_summaries": [
                        {
                            "biosample": "K562",
                            "dnase_signal": 15.2,
                            "h3k4me3_signal": 12.8,
                            "h3k27ac_signal": 8.9
                        }
                    ],
                    "gene_distance": 500,
                    "linked_genes": ["ENSG00000001234"]
                }
            ],
            "total": 1
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.region_to_ccre_screen("chr1", 1000000, 1002000, "GRCh38")
        
        mock_get.assert_called_once_with("/api/v1/ccres", params={
            "chromosome": "chr1",
            "start": 1000000,
            "end": 1002000,
            "genome": "GRCh38"
        })
        assert result == mock_response

    @patch.object(CCREClient, 'get')
    def test_region_to_ccre_screen_mouse(self, mock_get):
        """Test finding cCREs in mouse genome."""
        mock_response = {"ccres": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.region_to_ccre_screen("chr2", 5000000, 5010000, "mm10")
        
        mock_get.assert_called_once_with("/api/v1/ccres", params={
            "chromosome": "chr2",
            "start": 5000000,
            "end": 5010000,
            "genome": "mm10"
        })
        assert result == mock_response

    @patch.object(CCREClient, 'get')
    def test_get_ccre_details(self, mock_get):
        """Test getting detailed cCRE information."""
        mock_response = {
            "accession": "EH37E0002345",
            "chromosome": "chr1",
            "start": 1000000,
            "end": 1002000,
            "type": "promoter-like",
            "description": "Active promoter region",
            "biosample_data": [
                {
                    "biosample": "K562",
                    "cell_type": "lymphoblastoid",
                    "dnase_signal": 15.2,
                    "h3k4me3_signal": 12.8,
                    "h3k27ac_signal": 8.9,
                    "ctcf_signal": 3.2
                },
                {
                    "biosample": "HepG2", 
                    "cell_type": "hepatocellular",
                    "dnase_signal": 8.4,
                    "h3k4me3_signal": 6.1,
                    "h3k27ac_signal": 4.2,
                    "ctcf_signal": 1.8
                }
            ],
            "conservation_score": 0.85,
            "regulatory_annotations": ["enhancer", "promoter"]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_ccre_details("EH37E0002345")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345")
        assert result["accession"] == "EH37E0002345"
        assert len(result["biosample_data"]) == 2

    @patch.object(CCREClient, 'get')
    def test_get_genes_near_ccre(self, mock_get):
        """Test finding genes near a cCRE."""
        mock_response = {
            "genes": [
                {
                    "gene_id": "ENSG00000001234",
                    "gene_symbol": "EXAMPLE1",
                    "distance": 1500,
                    "chromosome": "chr1",
                    "start": 998500,
                    "end": 1010000,
                    "strand": "+",
                    "biotype": "protein_coding"
                },
                {
                    "gene_id": "ENSG00000005678",
                    "gene_symbol": "EXAMPLE2", 
                    "distance": 25000,
                    "chromosome": "chr1",
                    "start": 1025000,
                    "end": 1035000,
                    "strand": "-",
                    "biotype": "protein_coding"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_genes_near_ccre("EH37E0002345", distance=50000, genome="GRCh38")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345/genes", params={
            "distance": 50000,
            "genome": "GRCh38"
        })
        assert len(result) == 2
        assert result[0]["gene_symbol"] == "EXAMPLE1"

    @patch.object(CCREClient, 'get')
    def test_get_genes_near_ccre_default_params(self, mock_get):
        """Test finding genes near cCRE with default parameters."""
        mock_response = {"genes": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_genes_near_ccre("EH37E0002345")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345/genes", params={
            "distance": 100000,
            "genome": "GRCh38"
        })
        assert result == []

    @patch.object(CCREClient, 'get')
    def test_search_ccres(self, mock_get):
        """Test searching for cCREs."""
        mock_response = {
            "ccres": [
                {
                    "accession": "EH37E0002345",
                    "chromosome": "chr1",
                    "start": 1000000,
                    "end": 1002000,
                    "type": "promoter-like",
                    "cell_type": "K562"
                },
                {
                    "accession": "EH37E0002346",
                    "chromosome": "chr1",
                    "start": 2000000,
                    "end": 2001500,
                    "type": "enhancer-like", 
                    "cell_type": "K562"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.search_ccres(cell_type="K562", ccre_type="promoter", genome="GRCh38", limit=50)
        
        mock_get.assert_called_once_with("/api/v1/ccres/search", params={
            "genome": "GRCh38",
            "limit": 50,
            "cell_type": "K562",
            "type": "promoter"
        })
        assert len(result) == 2

    @patch.object(CCREClient, 'get')
    def test_search_ccres_default_params(self, mock_get):
        """Test searching cCREs with default parameters."""
        mock_response = {"ccres": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.search_ccres()
        
        mock_get.assert_called_once_with("/api/v1/ccres/search", params={
            "genome": "GRCh38",
            "limit": 100
        })
        assert result == []

    @patch.object(CCREClient, 'get')
    def test_search_ccres_missing_ccres_key(self, mock_get):
        """Test searching cCREs when ccres key is missing."""
        mock_response = {"other_data": "value"}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.search_ccres(cell_type="HepG2")
        
        assert result == []

    @patch.object(CCREClient, 'get')
    def test_get_ccre_expression(self, mock_get):
        """Test getting cCRE expression data."""
        mock_response = {
            "accession": "EH37E0002345",
            "expression_data": [
                {
                    "biosample": "K562",
                    "dnase_signal": 15.2,
                    "h3k4me3_signal": 12.8,
                    "h3k27ac_signal": 8.9,
                    "h3k4me1_signal": 5.4,
                    "ctcf_signal": 3.2
                },
                {
                    "biosample": "GM12878",
                    "dnase_signal": 11.8,
                    "h3k4me3_signal": 9.2,
                    "h3k27ac_signal": 6.7,
                    "h3k4me1_signal": 4.1,
                    "ctcf_signal": 2.8
                }
            ],
            "tissue_specificity": 0.65
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_ccre_expression("EH37E0002345", cell_type="K562")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345/expression", params={
            "cell_type": "K562"
        })
        assert result["accession"] == "EH37E0002345"
        assert len(result["expression_data"]) == 2

    @patch.object(CCREClient, 'get')
    def test_get_ccre_expression_no_cell_type(self, mock_get):
        """Test getting cCRE expression without cell type filter."""
        mock_response = {"accession": "EH37E0002345", "expression_data": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_ccre_expression("EH37E0002345")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345/expression", params={})
        assert result["accession"] == "EH37E0002345"

    @patch.object(CCREClient, 'get')
    def test_get_ccre_chromatin_state(self, mock_get):
        """Test getting cCRE chromatin state data."""
        mock_response = {
            "accession": "EH37E0002345",
            "chromatin_states": [
                {
                    "biosample": "K562",
                    "assay": "ChIP-seq",
                    "target": "H3K4me3",
                    "signal": 12.8,
                    "peak_called": True,
                    "enrichment": 3.2
                },
                {
                    "biosample": "K562",
                    "assay": "ChIP-seq", 
                    "target": "H3K27ac",
                    "signal": 8.9,
                    "peak_called": True,
                    "enrichment": 2.8
                }
            ],
            "predicted_state": "active_promoter"
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_ccre_chromatin_state("EH37E0002345", assay_type="ChIP-seq")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345/chromatin", params={
            "assay": "ChIP-seq"
        })
        assert result["accession"] == "EH37E0002345"
        assert len(result["chromatin_states"]) == 2

    @patch.object(CCREClient, 'get')
    def test_get_ccre_chromatin_state_no_assay(self, mock_get):
        """Test getting chromatin state without assay filter."""
        mock_response = {"accession": "EH37E0002345", "chromatin_states": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.get_ccre_chromatin_state("EH37E0002345")
        
        mock_get.assert_called_once_with("/api/v1/ccre/EH37E0002345/chromatin", params={})
        assert result["accession"] == "EH37E0002345"

    @patch.object(CCREClient, 'region_to_ccre_screen')
    def test_batch_query_ccres(self, mock_region_query):
        """Test batch querying multiple regions."""
        def mock_region_response(chr, start, end, genome):
            if chr == "chr1":
                return {"ccres": [{"accession": "EH37E0001"}]}
            else:
                return {"ccres": [{"accession": "EH37E0002"}]}
        
        mock_region_query.side_effect = mock_region_response
        
        regions = [
            {"chr": "chr1", "start": 1000000, "end": 1002000},
            {"chr": "chr2", "start": 2000000, "end": 2002000}
        ]
        
        client = CCREClient()
        results = client.batch_query_ccres(regions, genome="GRCh38")
        
        assert len(results) == 2
        assert results[0]["region"]["chr"] == "chr1"
        assert len(results[0]["ccres"]) == 1
        assert results[1]["region"]["chr"] == "chr2"
        assert len(results[1]["ccres"]) == 1

    @patch.object(CCREClient, 'region_to_ccre_screen')
    def test_batch_query_ccres_with_error(self, mock_region_query):
        """Test batch querying with error handling."""
        def mock_region_response(chr, start, end, genome):
            if chr == "chr1":
                return {"ccres": [{"accession": "EH37E0001"}]}
            else:
                raise requests.exceptions.HTTPError("API Error")
        
        mock_region_query.side_effect = mock_region_response
        
        regions = [
            {"chr": "chr1", "start": 1000000, "end": 1002000},
            {"chr": "chr2", "start": 2000000, "end": 2002000}
        ]
        
        client = CCREClient()
        results = client.batch_query_ccres(regions)
        
        assert len(results) == 2
        assert "ccres" in results[0]
        assert "error" in results[1]
        assert "API Error" in results[1]["error"]

    @patch.object(CCREClient, 'region_to_ccre_screen')
    def test_batch_query_ccres_empty_regions(self, mock_region_query):
        """Test batch querying with empty regions list."""
        client = CCREClient()
        results = client.batch_query_ccres([])
        
        assert results == []
        mock_region_query.assert_not_called()

    @responses.activate
    def test_http_request_success(self):
        """Test successful HTTP request."""
        responses.add(
            responses.GET,
            "https://screen.encodeproject.org/api/v1/ccres",
            json={
                "ccres": [{"accession": "EH37E0002345", "type": "promoter-like"}],
                "total": 1
            },
            status=200
        )
        
        client = CCREClient()
        result = client.region_to_ccre_screen("chr1", 1000000, 1002000)
        
        assert result["total"] == 1
        assert len(result["ccres"]) == 1

    @responses.activate
    def test_http_request_error(self):
        """Test HTTP request error handling."""
        responses.add(
            responses.GET,
            "https://screen.encodeproject.org/api/v1/ccres",
            status=500
        )
        
        client = CCREClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.region_to_ccre_screen("chr1", 1000000, 1002000)

    @responses.activate
    def test_http_request_timeout(self):
        """Test HTTP request timeout handling."""
        responses.add(
            responses.GET,
            "https://screen.encodeproject.org/api/v1/ccres",
            body=requests.exceptions.Timeout()
        )
        
        client = CCREClient()
        
        with pytest.raises(requests.exceptions.Timeout):
            client.region_to_ccre_screen("chr1", 1000000, 1002000)

    @patch.object(CCREClient, 'get')
    def test_edge_case_empty_response(self, mock_get):
        """Test handling of completely empty response."""
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = {}
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.search_ccres(cell_type="K562")
        
        assert result == []

    @patch.object(CCREClient, 'get')
    def test_edge_case_none_response(self, mock_get):
        """Test handling of None response."""
        mock_get.return_value = None
        
        client = CCREClient()
        
        with pytest.raises(AttributeError):
            client.search_ccres(cell_type="K562")

    @patch.object(CCREClient, 'get')
    def test_large_genomic_region(self, mock_get):
        """Test querying a large genomic region."""
        mock_response = {
            "ccres": [{"accession": f"EH37E000{i}"} for i in range(1000)],
            "total": 1000
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.region_to_ccre_screen("chr1", 1000000, 10000000)
        
        assert result["total"] == 1000
        assert len(result["ccres"]) == 1000

    @patch.object(CCREClient, 'get')
    def test_invalid_chromosome_format(self, mock_get):
        """Test with different chromosome format."""
        mock_response = {"ccres": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = CCREClient()
        result = client.region_to_ccre_screen("1", 1000000, 1002000)  # Without 'chr' prefix
        
        mock_get.assert_called_once_with("/api/v1/ccres", params={
            "chromosome": "1",
            "start": 1000000,
            "end": 1002000,
            "genome": "GRCh38"
        })
        assert result["total"] == 0