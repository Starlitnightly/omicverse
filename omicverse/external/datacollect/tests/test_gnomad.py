"""Tests for gnomAD API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.gnomad import GnomADClient


class TestGnomADClient:
    """Test GnomAD API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = GnomADClient()
        assert client.base_url == "https://gnomad.broadinstitute.org/api"
        assert client.rate_limit == 10
        
        # Test custom initialization
        client_custom = GnomADClient(
            base_url="https://custom.gnomad.org/api",
            rate_limit=5
        )
        assert client_custom.base_url == "https://custom.gnomad.org/api"
        assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = GnomADClient()
        headers = client.get_default_headers()
        
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    @patch.object(GnomADClient, 'query')
    def test_get_gene(self, mock_query):
        """Test getting gene information."""
        mock_response = {
            "data": {
                "gene": {
                    "gene_id": "ENSG00000141510",
                    "gene_symbol": "TP53",
                    "name": "tumor protein p53",
                    "canonical_transcript_id": "ENST00000269305",
                    "chrom": "17",
                    "start": 7661779,
                    "stop": 7687550,
                    "strand": "-",
                    "variants": [
                        {
                            "variant_id": "17-7674221-G-A",
                            "pos": 7674221,
                            "ref": "G",
                            "alt": "A",
                            "rsid": "rs28934571",
                            "consequence": "missense_variant",
                            "hgvs": "p.Arg273His",
                            "lof": None,
                            "genome": {
                                "ac": 123,
                                "an": 251456,
                                "af": 0.000489,
                                "homozygote_count": 0
                            },
                            "exome": {
                                "ac": 89,
                                "an": 125748,
                                "af": 0.000708,
                                "homozygote_count": 0
                            },
                            "populations": [
                                {
                                    "id": "afr",
                                    "ac": 12,
                                    "an": 24568,
                                    "homozygote_count": 0
                                }
                            ]
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.get_gene("TP53")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "GeneInfo" in args[0]
        assert args[1]["geneSymbol"] == "TP53"
        assert args[1]["dataset"] == "gnomad_r3"
        
        assert result["gene_symbol"] == "TP53"
        assert result["gene_id"] == "ENSG00000141510"
        assert len(result["variants"]) == 1
        assert result["variants"][0]["rsid"] == "rs28934571"

    @patch.object(GnomADClient, 'query')
    def test_get_gene_custom_dataset(self, mock_query):
        """Test getting gene information with custom dataset."""
        mock_query.return_value = {"data": {"gene": {"gene_symbol": "TP53"}}}
        
        client = GnomADClient()
        client.get_gene("TP53", dataset="gnomad_r2_1")
        
        args, kwargs = mock_query.call_args
        assert args[1]["dataset"] == "gnomad_r2_1"

    @patch.object(GnomADClient, 'query')
    def test_get_variant(self, mock_query):
        """Test getting variant information."""
        mock_response = {
            "data": {
                "variant": {
                    "variant_id": "17-7674221-G-A",
                    "chrom": "17",
                    "pos": 7674221,
                    "ref": "G",
                    "alt": "A",
                    "rsid": "rs28934571",
                    "reference_genome": "GRCh38",
                    "quality_metrics": {
                        "allele_balance": {
                            "alt": {
                                "bin_edges": [0.0, 0.1, 0.2],
                                "bin_freq": [100, 200, 150],
                                "n_smaller": 10,
                                "n_larger": 5
                            }
                        }
                    },
                    "genome": {
                        "ac": 123,
                        "an": 251456,
                        "af": 0.000489,
                        "homozygote_count": 0,
                        "hemizygote_count": 0,
                        "filters": ["PASS"],
                        "populations": [
                            {
                                "id": "afr",
                                "ac": 12,
                                "an": 24568,
                                "homozygote_count": 0,
                                "hemizygote_count": 0
                            }
                        ]
                    },
                    "transcript_consequences": [
                        {
                            "gene_id": "ENSG00000141510",
                            "gene_symbol": "TP53",
                            "transcript_id": "ENST00000269305",
                            "consequence": "missense_variant",
                            "hgvsc": "c.818G>A",
                            "hgvsp": "p.Arg273His",
                            "lof": None,
                            "polyphen_prediction": "probably_damaging",
                            "sift_prediction": "deleterious"
                        }
                    ],
                    "in_silico_predictors": {
                        "cadd": {
                            "phred": 25.3,
                            "raw": 4.123
                        },
                        "revel": {
                            "score": 0.78
                        }
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.get_variant("17-7674221-G-A")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "VariantInfo" in args[0]
        assert args[1]["variantId"] == "17-7674221-G-A"
        assert args[1]["dataset"] == "gnomad_r3"
        
        assert result["variant_id"] == "17-7674221-G-A"
        assert result["rsid"] == "rs28934571"
        assert result["genome"]["af"] == 0.000489
        assert len(result["transcript_consequences"]) == 1
        assert result["in_silico_predictors"]["cadd"]["phred"] == 25.3

    @patch.object(GnomADClient, 'query')
    def test_search_variants_by_rsid(self, mock_query):
        """Test searching variants by rsID."""
        mock_response = {
            "data": {
                "searchVariants": [
                    {
                        "variant_id": "17-7674221-G-A",
                        "chrom": "17",
                        "pos": 7674221,
                        "ref": "G",
                        "alt": "A",
                        "rsid": "rs28934571",
                        "genome": {"af": 0.000489},
                        "exome": {"af": 0.000708}
                    }
                ]
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.search_variants_by_rsid("rs28934571")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "SearchByRsid" in args[0]
        assert args[1]["rsid"] == "rs28934571"
        
        assert len(result) == 1
        assert result[0]["rsid"] == "rs28934571"
        assert result[0]["genome"]["af"] == 0.000489

    @patch.object(GnomADClient, 'query')
    def test_get_region_variants(self, mock_query):
        """Test getting variants in a genomic region."""
        mock_response = {
            "data": {
                "region": {
                    "variants": [
                        {
                            "variant_id": "17-7674221-G-A",
                            "pos": 7674221,
                            "ref": "G",
                            "alt": "A",
                            "rsid": "rs28934571",
                            "consequence": "missense_variant",
                            "genome": {
                                "ac": 123,
                                "an": 251456,
                                "af": 0.000489
                            },
                            "exome": {
                                "ac": 89,
                                "an": 125748,
                                "af": 0.000708
                            }
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.get_region_variants("17", 7670000, 7680000)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "RegionVariants" in args[0]
        assert args[1]["chrom"] == "17"
        assert args[1]["start"] == 7670000
        assert args[1]["stop"] == 7680000
        
        assert len(result) == 1
        assert result[0]["variant_id"] == "17-7674221-G-A"

    @patch.object(GnomADClient, 'query')
    def test_get_transcript(self, mock_query):
        """Test getting transcript information."""
        mock_response = {
            "data": {
                "transcript": {
                    "transcript_id": "ENST00000269305",
                    "gene_id": "ENSG00000141510",
                    "gene_symbol": "TP53",
                    "chrom": "17",
                    "start": 7661779,
                    "stop": 7687550,
                    "strand": "-",
                    "exons": [
                        {
                            "feature_type": "CDS",
                            "start": 7673534,
                            "stop": 7673700
                        }
                    ],
                    "variants": [
                        {
                            "variant_id": "17-7674221-G-A",
                            "pos": 7674221,
                            "ref": "G",
                            "alt": "A",
                            "rsid": "rs28934571",
                            "consequence": "missense_variant",
                            "hgvsc": "c.818G>A",
                            "hgvsp": "p.Arg273His",
                            "lof": None,
                            "genome": {"af": 0.000489},
                            "exome": {"af": 0.000708}
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.get_transcript("ENST00000269305")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "TranscriptInfo" in args[0]
        assert args[1]["transcriptId"] == "ENST00000269305"
        
        assert result["transcript_id"] == "ENST00000269305"
        assert result["gene_symbol"] == "TP53"
        assert len(result["exons"]) == 1
        assert len(result["variants"]) == 1

    @patch.object(GnomADClient, 'query')
    def test_get_constraint_scores(self, mock_query):
        """Test getting gene constraint scores."""
        mock_response = {
            "data": {
                "gene": {
                    "gene_id": "ENSG00000141510",
                    "gene_symbol": "TP53",
                    "constraint": {
                        "exp_lof": 35.2,
                        "exp_mis": 934.2,
                        "exp_syn": 414.6,
                        "obs_lof": 12,
                        "obs_mis": 701,
                        "obs_syn": 393,
                        "oe_lof": 0.34,
                        "oe_lof_lower": 0.18,
                        "oe_lof_upper": 0.57,
                        "oe_mis": 0.75,
                        "oe_mis_lower": 0.69,
                        "oe_mis_upper": 0.81,
                        "oe_syn": 0.95,
                        "oe_syn_lower": 0.86,
                        "oe_syn_upper": 1.04,
                        "lof_z": 5.89,
                        "mis_z": 8.91,
                        "syn_z": -1.13,
                        "pLI": 1.0,
                        "pRec": 0.0,
                        "pNull": 0.0
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.get_constraint_scores("TP53")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "GeneConstraint" in args[0]
        assert args[1]["geneSymbol"] == "TP53"
        
        assert result["pLI"] == 1.0
        assert result["oe_lof"] == 0.34
        assert result["lof_z"] == 5.89

    @patch.object(GnomADClient, 'query')
    def test_get_coverage(self, mock_query):
        """Test getting coverage statistics."""
        mock_response = {
            "data": {
                "region": {
                    "coverage": [
                        {
                            "pos": 7674221,
                            "mean": 32.1,
                            "median": 31.0,
                            "over_1": 0.99,
                            "over_5": 0.97,
                            "over_10": 0.94,
                            "over_15": 0.89,
                            "over_20": 0.82,
                            "over_25": 0.73,
                            "over_30": 0.61,
                            "over_50": 0.34,
                            "over_100": 0.02
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = GnomADClient()
        result = client.get_coverage("17", 7674220, 7674222, data_type="genome")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "Coverage" in args[0]
        assert args[1]["chrom"] == "17"
        assert args[1]["start"] == 7674220
        assert args[1]["stop"] == 7674222
        assert args[1]["dataType"] == "genome"
        
        assert len(result) == 1
        assert result[0]["pos"] == 7674221
        assert result[0]["mean"] == 32.1

    @patch.object(GnomADClient, 'query')
    def test_query_with_variables(self, mock_query):
        """Test GraphQL query with variables."""
        mock_query.return_value = {"data": {"test": "result"}}
        
        client = GnomADClient()
        query = "query Test($var: String!) { test(input: $var) }"
        variables = {"var": "value"}
        
        result = client.query(query, variables)
        
        mock_query.assert_called_once_with(query, variables)

    @patch.object(GnomADClient, 'query')
    def test_query_without_variables(self, mock_query):
        """Test GraphQL query without variables."""
        mock_query.return_value = {"data": {"test": "result"}}
        
        client = GnomADClient()
        query = "query { test }"
        
        result = client.query(query)
        
        mock_query.assert_called_once_with(query)

    @patch.object(GnomADClient, 'query')
    def test_empty_results(self, mock_query):
        """Test handling of empty results."""
        mock_query.return_value = {"data": {"gene": None}}
        
        client = GnomADClient()
        result = client.get_gene("NONEXISTENT")
        
        assert result == {}

    @patch.object(GnomADClient, 'query')
    def test_missing_data_field(self, mock_query):
        """Test handling of missing data field."""
        mock_query.return_value = {"errors": ["Some error"]}
        
        client = GnomADClient()
        result = client.get_gene("TP53")
        
        assert result == {}

    @responses.activate
    def test_query_http_request(self):
        """Test actual HTTP request for GraphQL query."""
        # Mock the HTTP response
        responses.add(
            responses.POST,
            "https://gnomad.broadinstitute.org/api",
            json={"data": {"test": "result"}},
            status=200
        )
        
        client = GnomADClient()
        query = "query { test }"
        result = client.query(query)
        
        assert result["data"]["test"] == "result"
        assert len(responses.calls) == 1
        
        # Check the request payload
        request = responses.calls[0].request
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["Accept"] == "application/json"
        
        import json
        payload = json.loads(request.body)
        assert payload["query"] == query

    @responses.activate 
    def test_query_http_request_with_variables(self):
        """Test HTTP request with GraphQL variables."""
        responses.add(
            responses.POST,
            "https://gnomad.broadinstitute.org/api",
            json={"data": {"gene": {"gene_symbol": "TP53"}}},
            status=200
        )
        
        client = GnomADClient()
        query = "query Test($gene: String!) { gene(symbol: $gene) { gene_symbol } }"
        variables = {"gene": "TP53"}
        
        result = client.query(query, variables)
        
        assert result["data"]["gene"]["gene_symbol"] == "TP53"
        
        # Check the request payload
        request = responses.calls[0].request
        import json
        payload = json.loads(request.body)
        assert payload["query"] == query
        assert payload["variables"] == variables

    @responses.activate
    def test_query_http_error(self):
        """Test handling of HTTP errors."""
        responses.add(
            responses.POST,
            "https://gnomad.broadinstitute.org/api",
            status=500
        )
        
        client = GnomADClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.query("query { test }")