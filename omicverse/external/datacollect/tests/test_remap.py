"""Tests for ReMap Transcription Factor Binding Database API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.remap import ReMapClient


class TestReMapClient:
    """Test ReMap API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = ReMapClient()
        assert client.base_url == "https://remap.univ-amu.fr"
        assert client.rate_limit == 1.0

    def test_get_default_headers(self):
        """Test default headers."""
        client = ReMapClient()
        headers = client.get_default_headers()
        
        assert headers["User-Agent"] == "BioinformaticsCollector/1.0"
        assert headers["Accept"] == "application/json"

    @patch.object(ReMapClient, 'get')
    def test_query_remap(self, mock_get):
        """Test querying ReMap for TF binding sites."""
        mock_response = {
            "peaks": [
                {
                    "peak_id": "REMAP001",
                    "chromosome": "chr1",
                    "start": 1000000,
                    "end": 1001000,
                    "tf_name": "CTCF",
                    "cell_type": "K562",
                    "experiment": "ChIP-seq",
                    "score": 850.5,
                    "pvalue": 1.2e-8,
                    "source": "ENCODE"
                },
                {
                    "peak_id": "REMAP002",
                    "chromosome": "chr1",
                    "start": 1005000,
                    "end": 1006000,
                    "tf_name": "POLR2A",
                    "cell_type": "K562",
                    "experiment": "ChIP-seq",
                    "score": 720.3,
                    "pvalue": 3.4e-7,
                    "source": "ENCODE"
                }
            ],
            "total": 2
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.query_remap("chr1", 1000000, 1010000, "human", "hg38")
        
        mock_get.assert_called_once_with("/api/v1/peaks", params={
            "chr": "chr1",
            "start": 1000000,
            "end": 1010000,
            "species": "human",
            "assembly": "hg38"
        })
        assert result == mock_response

    @patch.object(ReMapClient, 'get')
    def test_query_remap_default_params(self, mock_get):
        """Test querying ReMap with default parameters."""
        mock_response = {"peaks": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.query_remap("chr2", 2000000, 2010000)
        
        mock_get.assert_called_once_with("/api/v1/peaks", params={
            "chr": "chr2",
            "start": 2000000,
            "end": 2010000,
            "species": "human",
            "assembly": "hg38"
        })
        assert result == mock_response

    @patch.object(ReMapClient, 'get')
    def test_query_remap_mouse(self, mock_get):
        """Test querying ReMap for mouse data."""
        mock_response = {"peaks": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.query_remap("chr3", 3000000, 3010000, "mouse", "mm10")
        
        mock_get.assert_called_once_with("/api/v1/peaks", params={
            "chr": "chr3",
            "start": 3000000,
            "end": 3010000,
            "species": "mouse",
            "assembly": "mm10"
        })

    @patch.object(ReMapClient, 'get')
    def test_search_by_tf(self, mock_get):
        """Test searching for TF binding sites."""
        mock_response = {
            "peaks": [
                {
                    "peak_id": "REMAP001",
                    "chromosome": "chr1",
                    "start": 1000000,
                    "end": 1001000,
                    "tf_name": "CTCF",
                    "cell_type": "K562",
                    "experiment": "ChIP-seq",
                    "score": 850.5,
                    "assembly": "hg38"
                },
                {
                    "peak_id": "REMAP002",
                    "chromosome": "chr5",
                    "start": 5000000,
                    "end": 5001000,
                    "tf_name": "CTCF",
                    "cell_type": "HepG2",
                    "experiment": "ChIP-seq", 
                    "score": 920.1,
                    "assembly": "hg38"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_tf("CTCF", "human", "K562", 500)
        
        mock_get.assert_called_once_with("/api/v1/search/tf", params={
            "tf": "CTCF",
            "species": "human",
            "limit": 500,
            "cell_type": "K562"
        })
        assert len(result) == 2
        assert result[0]["tf_name"] == "CTCF"

    @patch.object(ReMapClient, 'get')
    def test_search_by_tf_default_params(self, mock_get):
        """Test searching TF with default parameters."""
        mock_response = {"peaks": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_tf("TP53")
        
        mock_get.assert_called_once_with("/api/v1/search/tf", params={
            "tf": "TP53",
            "species": "human",
            "limit": 1000
        })
        assert result == []

    @patch.object(ReMapClient, 'get')
    def test_search_by_tf_missing_peaks_key(self, mock_get):
        """Test searching TF when peaks key is missing."""
        mock_response = {"other_data": "value"}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_tf("CTCF")
        
        assert result == []

    @patch.object(ReMapClient, 'get')
    def test_search_by_gene(self, mock_get):
        """Test finding TF binding sites near a gene."""
        mock_response = {
            "peaks": [
                {
                    "peak_id": "REMAP001",
                    "chromosome": "chr17",
                    "start": 7673534,
                    "end": 7674534,
                    "tf_name": "TP53",
                    "cell_type": "K562",
                    "distance_to_tss": 500,
                    "gene_symbol": "TP53"
                },
                {
                    "peak_id": "REMAP002",
                    "chromosome": "chr17",
                    "start": 7680000,
                    "end": 7681000,
                    "tf_name": "CTCF",
                    "cell_type": "K562",
                    "distance_to_tss": 7000,
                    "gene_symbol": "TP53"
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_gene("TP53", window=10000, species="human")
        
        mock_get.assert_called_once_with("/api/v1/search/gene", params={
            "gene": "TP53",
            "window": 10000,
            "species": "human"
        })
        assert len(result) == 2
        assert result[0]["gene_symbol"] == "TP53"

    @patch.object(ReMapClient, 'get')
    def test_search_by_gene_default_params(self, mock_get):
        """Test searching by gene with default parameters."""
        mock_response = {"peaks": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_gene("BRCA1")
        
        mock_get.assert_called_once_with("/api/v1/search/gene", params={
            "gene": "BRCA1",
            "window": 10000,
            "species": "human"
        })
        assert result == []

    @patch.object(ReMapClient, 'get')
    def test_get_tf_list(self, mock_get):
        """Test getting list of available TFs."""
        mock_response = {
            "tfs": [
                "CTCF",
                "TP53",
                "POLR2A",
                "MYC",
                "JUN",
                "FOS",
                "CEBPB",
                "NFE2L2"
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_tf_list("human", "K562")
        
        mock_get.assert_called_once_with("/api/v1/tfs", params={
            "species": "human",
            "cell_type": "K562"
        })
        assert len(result) == 8
        assert "CTCF" in result
        assert "TP53" in result

    @patch.object(ReMapClient, 'get')
    def test_get_tf_list_default_params(self, mock_get):
        """Test getting TF list with default parameters."""
        mock_response = {"tfs": ["CTCF", "TP53"]}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_tf_list()
        
        mock_get.assert_called_once_with("/api/v1/tfs", params={"species": "human"})
        assert len(result) == 2

    @patch.object(ReMapClient, 'get')
    def test_get_tf_list_missing_tfs_key(self, mock_get):
        """Test getting TF list when tfs key is missing."""
        mock_response = {"other_data": "value"}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_tf_list()
        
        assert result == []

    @patch.object(ReMapClient, 'get')
    def test_get_cell_types(self, mock_get):
        """Test getting list of available cell types."""
        mock_response = {
            "cell_types": [
                "K562",
                "HepG2",
                "MCF-7",
                "A549",
                "GM12878",
                "HeLa-S3",
                "HUVEC",
                "IMR-90"
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_cell_types("human")
        
        mock_get.assert_called_once_with("/api/v1/cell_types", params={"species": "human"})
        assert len(result) == 8
        assert "K562" in result
        assert "HepG2" in result

    @patch.object(ReMapClient, 'get')
    def test_get_cell_types_mouse(self, mock_get):
        """Test getting cell types for mouse."""
        mock_response = {"cell_types": ["MEF", "CH12", "3T3-L1"]}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_cell_types("mouse")
        
        mock_get.assert_called_once_with("/api/v1/cell_types", params={"species": "mouse"})
        assert len(result) == 3

    @patch.object(ReMapClient, 'get')
    def test_get_cell_types_missing_key(self, mock_get):
        """Test getting cell types when cell_types key is missing."""
        mock_response = {"other_data": "value"}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_cell_types()
        
        assert result == []

    @patch.object(ReMapClient, 'get')
    def test_get_peak_details(self, mock_get):
        """Test getting detailed peak information."""
        mock_response = {
            "peak_id": "REMAP001",
            "chromosome": "chr1",
            "start": 1000000,
            "end": 1001000,
            "tf_name": "CTCF",
            "cell_type": "K562",
            "experiment": "ChIP-seq",
            "score": 850.5,
            "pvalue": 1.2e-8,
            "source": "ENCODE",
            "assembly": "hg38",
            "nearby_genes": ["GENE1", "GENE2"],
            "conservation_score": 0.85,
            "motif_match": True
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_peak_details("REMAP001")
        
        mock_get.assert_called_once_with("/api/v1/peak/REMAP001")
        assert result["peak_id"] == "REMAP001"
        assert result["tf_name"] == "CTCF"
        assert result["score"] == 850.5

    @patch.object(ReMapClient, 'get')
    def test_get_enrichment(self, mock_get):
        """Test getting TF enrichment statistics."""
        mock_response = {
            "region": {
                "chromosome": "chr1",
                "start": 1000000,
                "end": 1010000
            },
            "enrichment_stats": [
                {
                    "tf_name": "CTCF",
                    "peak_count": 15,
                    "expected_count": 5.2,
                    "enrichment_ratio": 2.88,
                    "pvalue": 0.001,
                    "adjusted_pvalue": 0.05
                },
                {
                    "tf_name": "POLR2A",
                    "peak_count": 8,
                    "expected_count": 3.1,
                    "enrichment_ratio": 2.58,
                    "pvalue": 0.005,
                    "adjusted_pvalue": 0.08
                }
            ],
            "background_model": "genome_wide"
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_enrichment("chr1", 1000000, 1010000, "human")
        
        mock_get.assert_called_once_with("/api/v1/enrichment", params={
            "chr": "chr1",
            "start": 1000000,
            "end": 1010000,
            "species": "human"
        })
        assert result["region"]["chromosome"] == "chr1"
        assert len(result["enrichment_stats"]) == 2

    @patch.object(ReMapClient, 'query_remap')
    def test_batch_query_regions(self, mock_query):
        """Test batch querying multiple regions."""
        def mock_query_response(chr, start, end, species):
            if chr == "chr1":
                return {"peaks": [{"peak_id": "REMAP001", "tf_name": "CTCF"}]}
            else:
                return {"peaks": [{"peak_id": "REMAP002", "tf_name": "TP53"}]}
        
        mock_query.side_effect = mock_query_response
        
        regions = [
            {"chr": "chr1", "start": 1000000, "end": 1010000},
            {"chr": "chr2", "start": 2000000, "end": 2010000}
        ]
        
        client = ReMapClient()
        results = client.batch_query_regions(regions, "human")
        
        assert len(results) == 2
        assert results[0]["region"]["chr"] == "chr1"
        assert len(results[0]["peaks"]) == 1
        assert results[1]["region"]["chr"] == "chr2"
        assert len(results[1]["peaks"]) == 1

    @patch.object(ReMapClient, 'query_remap')
    def test_batch_query_regions_with_error(self, mock_query):
        """Test batch querying with error handling."""
        def mock_query_response(chr, start, end, species):
            if chr == "chr1":
                return {"peaks": [{"peak_id": "REMAP001"}]}
            else:
                raise requests.exceptions.HTTPError("API Error")
        
        mock_query.side_effect = mock_query_response
        
        regions = [
            {"chr": "chr1", "start": 1000000, "end": 1010000},
            {"chr": "chr2", "start": 2000000, "end": 2010000}
        ]
        
        client = ReMapClient()
        results = client.batch_query_regions(regions)
        
        assert len(results) == 2
        assert "peaks" in results[0]
        assert "error" in results[1]
        assert "API Error" in results[1]["error"]

    @patch.object(ReMapClient, 'get')
    def test_get_colocalization(self, mock_get):
        """Test getting TF co-localization data."""
        mock_response = {
            "tf1": "CTCF",
            "tf2": "POLR2A",
            "colocalization_stats": {
                "total_tf1_peaks": 1500,
                "total_tf2_peaks": 2300,
                "overlapping_peaks": 450,
                "overlap_percentage_tf1": 30.0,
                "overlap_percentage_tf2": 19.6,
                "jaccard_index": 0.15,
                "pvalue": 1.2e-15
            },
            "overlapping_regions": [
                {
                    "chromosome": "chr1",
                    "start": 1000000,
                    "end": 1001000,
                    "tf1_score": 850.5,
                    "tf2_score": 720.3
                }
            ]
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.get_colocalization("CTCF", "POLR2A", "human")
        
        mock_get.assert_called_once_with("/api/v1/colocalization", params={
            "tf1": "CTCF",
            "tf2": "POLR2A",
            "species": "human"
        })
        assert result["tf1"] == "CTCF"
        assert result["tf2"] == "POLR2A"
        assert result["colocalization_stats"]["overlapping_peaks"] == 450

    @responses.activate
    def test_http_request_success(self):
        """Test successful HTTP request."""
        responses.add(
            responses.GET,
            "https://remap.univ-amu.fr/api/v1/peaks",
            json={
                "peaks": [{"peak_id": "REMAP001", "tf_name": "CTCF"}],
                "total": 1
            },
            status=200
        )
        
        client = ReMapClient()
        result = client.query_remap("chr1", 1000000, 1010000)
        
        assert result["total"] == 1
        assert len(result["peaks"]) == 1

    @responses.activate
    def test_http_request_error(self):
        """Test HTTP request error handling."""
        responses.add(
            responses.GET,
            "https://remap.univ-amu.fr/api/v1/peaks",
            status=500
        )
        
        client = ReMapClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.query_remap("chr1", 1000000, 1010000)

    @responses.activate
    def test_http_request_timeout(self):
        """Test HTTP request timeout handling."""
        responses.add(
            responses.GET,
            "https://remap.univ-amu.fr/api/v1/search/tf",
            body=requests.exceptions.Timeout()
        )
        
        client = ReMapClient()
        
        with pytest.raises(requests.exceptions.Timeout):
            client.search_by_tf("CTCF")

    @patch.object(ReMapClient, 'get')
    def test_edge_case_empty_response(self, mock_get):
        """Test handling of completely empty response."""
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = {}
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_tf("CTCF")
        
        assert result == []

    @patch.object(ReMapClient, 'get')
    def test_edge_case_none_response(self, mock_get):
        """Test handling of None response."""
        mock_get.return_value = None
        
        client = ReMapClient()
        
        with pytest.raises(AttributeError):
            client.search_by_tf("CTCF")

    @patch.object(ReMapClient, 'get')
    def test_large_genomic_region(self, mock_get):
        """Test querying a large genomic region."""
        mock_response = {
            "peaks": [{"peak_id": f"REMAP{i:06d}"} for i in range(1000)],
            "total": 1000
        }
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.query_remap("chr1", 1000000, 50000000)  # 50Mb region
        
        assert result["total"] == 1000
        assert len(result["peaks"]) == 1000

    @patch.object(ReMapClient, 'get')
    def test_non_standard_species(self, mock_get):
        """Test querying with non-standard species."""
        mock_response = {"peaks": [], "total": 0}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.query_remap("chr1", 1000000, 1010000, "fly", "dm6")
        
        mock_get.assert_called_once_with("/api/v1/peaks", params={
            "chr": "chr1",
            "start": 1000000,
            "end": 1010000,
            "species": "fly",
            "assembly": "dm6"
        })

    @patch.object(ReMapClient, 'get')
    def test_search_by_tf_case_sensitivity(self, mock_get):
        """Test TF search case sensitivity."""
        mock_response = {"peaks": []}
        mock_response_obj = Mock()
        mock_response_obj.json.return_value = mock_response
        mock_get.return_value = mock_response_obj
        
        client = ReMapClient()
        result = client.search_by_tf("ctcf")  # lowercase
        
        mock_get.assert_called_once_with("/api/v1/search/tf", params={
            "tf": "ctcf",
            "species": "human",
            "limit": 1000
        })
        assert result == []