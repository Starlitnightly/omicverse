"""Tests for UCSC Genome Browser API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests
import json

from omicverse.external.datacollect.api.ucsc import UCSCClient


class TestUCSCClient:
    """Test UCSC API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = UCSCClient()
        assert client.base_url == "https://api.genome.ucsc.edu"
        assert client.rate_limit == 10

    def test_initialization_custom_params(self):
        """Test client initialization with custom parameters."""
        client = UCSCClient(
            base_url="https://custom.ucsc.edu/api",
            rate_limit=5
        )
        assert client.base_url == "https://custom.ucsc.edu/api"
        assert client.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = UCSCClient()
        headers = client.get_default_headers()
        
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    @patch.object(UCSCClient, 'get')
    def test_list_genomes(self, mock_get):
        """Test listing available genomes."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "ucscGenomes": {
                "hg38": {
                    "organism": "Human",
                    "description": "Dec. 2013 (GRCh38/hg38)",
                    "scientificName": "Homo sapiens",
                    "htmlPath": "/cgi-bin/hgGateway?genome=hg38"
                },
                "mm10": {
                    "organism": "Mouse",
                    "description": "Dec. 2011 (GRCm38/mm10)",
                    "scientificName": "Mus musculus",
                    "htmlPath": "/cgi-bin/hgGateway?genome=mm10"
                },
                "dm6": {
                    "organism": "D. melanogaster",
                    "description": "Aug. 2014 (BDGP Release 6 + ISO1 MT/dm6)",
                    "scientificName": "Drosophila melanogaster",
                    "htmlPath": "/cgi-bin/hgGateway?genome=dm6"
                }
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.list_genomes()
        
        mock_get.assert_called_once_with("/list/ucscGenomes")
        assert "ucscGenomes" in result
        assert "hg38" in result["ucscGenomes"]
        assert "mm10" in result["ucscGenomes"]
        assert result["ucscGenomes"]["hg38"]["organism"] == "Human"

    @patch.object(UCSCClient, 'get')
    def test_get_chromosomes(self, mock_get):
        """Test getting chromosome information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hg38": {
                "chr1": {"size": 248956422},
                "chr2": {"size": 242193529},
                "chr3": {"size": 198295559},
                "chrX": {"size": 156040895},
                "chrY": {"size": 57227415},
                "chrM": {"size": 16569}
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_chromosomes("hg38")
        
        mock_get.assert_called_once_with("/list/chromosomes", params={"genome": "hg38"})
        assert "hg38" in result
        assert "chr1" in result["hg38"]
        assert result["hg38"]["chr1"]["size"] == 248956422

    @patch.object(UCSCClient, 'get')
    def test_get_tracks(self, mock_get):
        """Test getting track information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hg38": {
                "ncbiRefSeqCurated": {
                    "type": "genePred",
                    "longLabel": "NCBI RefSeq genes, curated subset",
                    "shortLabel": "RefSeq Curated"
                },
                "knownGene": {
                    "type": "genePred", 
                    "longLabel": "UCSC Genes",
                    "shortLabel": "UCSC Genes"
                }
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_tracks("hg38")
        
        mock_get.assert_called_once_with("/list/tracks", params={"genome": "hg38"})
        assert "hg38" in result
        assert "ncbiRefSeqCurated" in result["hg38"]

    @patch.object(UCSCClient, 'get')
    def test_get_tracks_specific_track(self, mock_get):
        """Test getting specific track information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hg38": {
                "knownGene": {
                    "type": "genePred",
                    "longLabel": "UCSC Genes",
                    "shortLabel": "UCSC Genes"
                }
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_tracks("hg38", track="knownGene")
        
        mock_get.assert_called_once_with("/list/tracks", params={
            "genome": "hg38",
            "track": "knownGene"
        })

    @patch.object(UCSCClient, 'get')
    def test_get_data(self, mock_get):
        """Test getting track data for a genomic region."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hg38": {
                "chr1": [
                    {
                        "name": "NM_001005484",
                        "chrom": "chr1",
                        "strand": "+",
                        "txStart": 11873,
                        "txEnd": 14409,
                        "cdsStart": 12189,
                        "cdsEnd": 13639,
                        "exonCount": 3,
                        "exonStarts": "11873,12612,13220",
                        "exonEnds": "12227,12721,14409",
                        "name2": "DDX11L1"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_data("hg38", "ncbiRefSeqCurated", "chr1", 10000, 20000)
        
        mock_get.assert_called_once_with("/getData/track", params={
            "genome": "hg38",
            "track": "ncbiRefSeqCurated",
            "chrom": "chr1",
            "start": 10000,
            "end": 20000
        })
        assert "hg38" in result
        assert "chr1" in result["hg38"]
        assert len(result["hg38"]["chr1"]) == 1

    @patch.object(UCSCClient, 'get')
    def test_get_sequence(self, mock_get):
        """Test getting DNA sequence."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "dna": "AATGCGATCGATCGATCGATCGATCGATCGATC"
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_sequence("hg38", "chr1", 1000000, 1000032)
        
        mock_get.assert_called_once_with("/getData/sequence", params={
            "genome": "hg38",
            "chrom": "chr1",
            "start": 1000000,
            "end": 1000032
        })
        assert "dna" in result
        assert len(result["dna"]) == 33

    @patch.object(UCSCClient, 'get')
    def test_search(self, mock_get):
        """Test searching for genomic features."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "matches": [
                {
                    "name": "TP53",
                    "description": "tumor protein p53",
                    "position": "chr17:7661779-7687550",
                    "type": "gene"
                },
                {
                    "name": "TP53-AS1",
                    "description": "TP53 antisense RNA 1",
                    "position": "chr17:7687379-7687550", 
                    "type": "gene"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.search("hg38", "TP53", type="gene")
        
        mock_get.assert_called_once_with("/search", params={
            "genome": "hg38",
            "query": "TP53",
            "type": "gene"
        })
        assert "matches" in result
        assert len(result["matches"]) == 2
        assert result["matches"][0]["name"] == "TP53"

    @patch.object(UCSCClient, 'get')
    def test_search_no_type(self, mock_get):
        """Test searching without type filter."""
        mock_response = Mock()
        mock_response.json.return_value = {"matches": []}
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.search("hg38", "chr1:1000-2000")
        
        mock_get.assert_called_once_with("/search", params={
            "genome": "hg38",
            "query": "chr1:1000-2000"
        })

    @patch.object(UCSCClient, 'search')
    @patch.object(UCSCClient, 'get_data')
    def test_get_gene_predictions(self, mock_get_data, mock_search):
        """Test getting gene predictions."""
        mock_search.return_value = {
            "matches": [
                {
                    "name": "TP53",
                    "position": "chr17:7661779-7687550",
                    "type": "gene"
                }
            ]
        }
        
        mock_get_data.return_value = {
            "gene_data": [
                {
                    "name": "NM_000546",
                    "chrom": "chr17",
                    "strand": "-",
                    "txStart": 7661779,
                    "txEnd": 7687550,
                    "name2": "TP53"
                }
            ]
        }
        
        client = UCSCClient()
        result = client.get_gene_predictions("hg38", "TP53")
        
        mock_search.assert_called_once_with("hg38", "TP53", type="gene")
        mock_get_data.assert_called_once_with("hg38", "ncbiRefSeqCurated", "chr17", 7661779, 7687550)
        
        assert result["gene"] == "TP53"
        assert result["genome"] == "hg38"
        assert len(result["predictions"]) == 1

    @patch.object(UCSCClient, 'search')
    def test_get_gene_predictions_not_found(self, mock_search):
        """Test getting gene predictions when gene is not found."""
        mock_search.return_value = {"matches": []}
        
        client = UCSCClient()
        result = client.get_gene_predictions("hg38", "NONEXISTENT")
        
        assert "error" in result
        assert "not found" in result["error"]

    @patch.object(UCSCClient, 'get_data')
    def test_get_snp_data(self, mock_get_data):
        """Test getting SNP data."""
        mock_get_data.return_value = {
            "snp_data": [
                {
                    "name": "rs123456",
                    "chrom": "chr1",
                    "chromStart": 1000100,
                    "chromEnd": 1000101,
                    "strand": "+",
                    "refNCBI": "A",
                    "refUCSC": "A",
                    "observed": "A/G",
                    "molType": "genomic",
                    "class": "single"
                }
            ]
        }
        
        client = UCSCClient()
        result = client.get_snp_data("hg38", "chr1", 1000100, window=50)
        
        mock_get_data.assert_called_once_with("hg38", "snp151", "chr1", 1000050, 1000150)
        assert "snp_data" in result

    @patch.object(UCSCClient, 'get_data')
    def test_get_snp_data_no_data(self, mock_get_data):
        """Test getting SNP data when no data is available."""
        mock_get_data.side_effect = [
            Exception("Track not found"),
            Exception("Track not found"),
            Exception("Track not found"),
            Exception("Track not found")
        ]
        
        client = UCSCClient()
        result = client.get_snp_data("hg38", "chr1", 1000100)
        
        assert "error" in result
        assert "No SNP data available" in result["error"]

    @patch.object(UCSCClient, 'get_data')
    def test_get_conservation_scores(self, mock_get_data):
        """Test getting conservation scores."""
        def mock_data_response(genome, track, chrom, start, end):
            if "phyloP" in track:
                return {"phyloP_scores": [0.85, 0.92, 0.78]}
            elif "phastCons" in track:
                return {"phastCons_scores": [0.95, 0.88, 0.91]}
            else:
                raise Exception("Track not found")
        
        mock_get_data.side_effect = mock_data_response
        
        client = UCSCClient()
        result = client.get_conservation_scores("hg38", "chr1", 1000000, 1000003)
        
        assert "phyloP" in result
        assert "phastCons" in result
        assert result["phyloP"]["phyloP_scores"] == [0.85, 0.92, 0.78]

    @patch.object(UCSCClient, 'get_data')
    def test_get_conservation_scores_no_data(self, mock_get_data):
        """Test getting conservation scores when no data is available."""
        mock_get_data.side_effect = Exception("Track not found")
        
        client = UCSCClient()
        result = client.get_conservation_scores("hg38", "chr1", 1000000, 1000003)
        
        assert result == {}

    @patch.object(UCSCClient, 'get_data')
    def test_get_regulatory_elements(self, mock_get_data):
        """Test getting regulatory elements."""
        def mock_data_response(genome, track, chrom, start, end):
            if "encRegTfbsClustered" in track:
                return {"tfbs_data": [{"name": "CTCF", "score": 850}]}
            elif "encRegDnaseClustered" in track:
                return {"dnase_data": [{"signal": 15.2}]}
            else:
                raise Exception("Track not found")
        
        mock_get_data.side_effect = mock_data_response
        
        client = UCSCClient()
        result = client.get_regulatory_elements("hg38", "chr1", 1000000, 1001000)
        
        # Should try to get multiple tracks
        assert mock_get_data.call_count >= 2
        
        # Check that successful tracks are included
        track_names = [track for track in result.keys()]
        assert any("encRegTfbsClustered" in name for name in track_names)

    @patch.object(UCSCClient, 'get_data')
    def test_get_regulatory_elements_no_data(self, mock_get_data):
        """Test getting regulatory elements when no data is available."""
        mock_get_data.side_effect = Exception("Track not found")
        
        client = UCSCClient()
        result = client.get_regulatory_elements("hg38", "chr1", 1000000, 1001000)
        
        assert result == {}

    @responses.activate
    def test_http_request_success(self):
        """Test successful HTTP request."""
        responses.add(
            responses.GET,
            "https://api.genome.ucsc.edu/list/ucscGenomes",
            json={"ucscGenomes": {"hg38": {"organism": "Human"}}},
            status=200
        )
        
        client = UCSCClient()
        result = client.list_genomes()
        
        assert "ucscGenomes" in result
        assert "hg38" in result["ucscGenomes"]

    @responses.activate
    def test_http_request_error(self):
        """Test HTTP request error handling."""
        responses.add(
            responses.GET,
            "https://api.genome.ucsc.edu/list/ucscGenomes",
            status=500
        )
        
        client = UCSCClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.list_genomes()

    @responses.activate
    def test_http_request_timeout(self):
        """Test HTTP request timeout handling."""
        responses.add(
            responses.GET,
            "https://api.genome.ucsc.edu/list/chromosomes",
            body=requests.exceptions.Timeout()
        )
        
        client = UCSCClient()
        
        with pytest.raises(requests.exceptions.Timeout):
            client.get_chromosomes("hg38")

    @patch.object(UCSCClient, 'get')
    def test_edge_case_empty_response(self, mock_get):
        """Test handling of completely empty response."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.list_genomes()
        
        assert result == {}

    @patch.object(UCSCClient, 'get')
    def test_edge_case_none_response(self, mock_get):
        """Test handling of None response."""
        mock_get.return_value = None
        
        client = UCSCClient()
        
        with pytest.raises(AttributeError):
            client.list_genomes()

    @patch.object(UCSCClient, 'get')
    def test_large_genomic_region(self, mock_get):
        """Test querying a large genomic region."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hg38": {
                "chr1": [{"name": f"gene_{i}"} for i in range(100)]
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_data("hg38", "knownGene", "chr1", 1000000, 50000000)  # 50Mb region
        
        assert len(result["hg38"]["chr1"]) == 100

    @patch.object(UCSCClient, 'get')
    def test_mouse_genome(self, mock_get):
        """Test querying mouse genome."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "mm10": {
                "chr1": {"size": 195471971},
                "chr2": {"size": 182113224}
            }
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_chromosomes("mm10")
        
        mock_get.assert_called_once_with("/list/chromosomes", params={"genome": "mm10"})
        assert "mm10" in result

    @patch.object(UCSCClient, 'get')
    def test_sequence_case_handling(self, mock_get):
        """Test DNA sequence case handling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "dna": "ATGCatgcNNNNatgc"  # Mixed case with Ns
        }
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        result = client.get_sequence("hg38", "chr1", 1000000, 1000016)
        
        assert "dna" in result
        assert "N" in result["dna"]  # Should handle ambiguous bases

    @patch.object(UCSCClient, 'get')
    def test_search_position_format(self, mock_get):
        """Test search with different position formats."""
        mock_response = Mock()
        mock_response.json.return_value = {"matches": []}
        mock_get.return_value = mock_response
        
        client = UCSCClient()
        
        # Test different position formats
        position_formats = [
            "chr1:1000-2000",
            "1:1000-2000",
            "chr1:1,000-2,000"
        ]
        
        for pos in position_formats:
            result = client.search("hg38", pos)
            assert "matches" in result

    @patch.object(UCSCClient, 'get_data')
    def test_snp_data_edge_position(self, mock_get_data):
        """Test SNP data at chromosome edge."""
        mock_get_data.return_value = {"snp_data": []}
        
        client = UCSCClient()
        result = client.get_snp_data("hg38", "chr1", 0, window=100)  # Position at start
        
        # Should handle negative start position
        mock_get_data.assert_called_once_with("hg38", "snp151", "chr1", 0, 100)

    @patch.object(UCSCClient, 'search')
    def test_get_gene_predictions_malformed_position(self, mock_search):
        """Test gene predictions with malformed position data."""
        mock_search.return_value = {
            "matches": [
                {
                    "name": "TESTGENE",
                    "position": "malformed_position",  # Bad position format
                    "type": "gene"
                }
            ]
        }
        
        client = UCSCClient()
        result = client.get_gene_predictions("hg38", "TESTGENE")
        
        # Should handle malformed position gracefully
        assert result["gene"] == "TESTGENE"
        assert result["predictions"] == []