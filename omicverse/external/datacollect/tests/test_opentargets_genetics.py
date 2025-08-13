"""Tests for OpenTargets Genetics API client."""

import pytest
from unittest.mock import Mock, patch
import responses
import requests

from omicverse.external.datacollect.api.opentargets_genetics import OpenTargetsGeneticsClient


class TestOpenTargetsGeneticsClient:
    """Test OpenTargets Genetics API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = OpenTargetsGeneticsClient()
        assert client.base_url == "https://api.genetics.opentargets.org/graphql"
        assert client.rate_limit == 10
        
        # Test custom initialization
        client_custom = OpenTargetsGeneticsClient(
            base_url="https://custom.genetics.org/graphql",
            rate_limit=5
        )
        assert client_custom.base_url == "https://custom.genetics.org/graphql"
        assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = OpenTargetsGeneticsClient()
        headers = client.get_default_headers()
        
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_gene_info(self, mock_query):
        """Test getting gene information."""
        mock_response = {
            "data": {
                "geneInfo": {
                    "id": "ENSG00000141510",
                    "symbol": "TP53",
                    "description": "tumor protein p53",
                    "chromosome": "17",
                    "start": 7661779,
                    "end": 7687550,
                    "bioType": "protein_coding",
                    "tss": 7687550,
                    "strand": "-"
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_gene_info("ENSG00000141510")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "GeneInfo" in args[0]
        assert args[1]["geneId"] == "ENSG00000141510"
        
        assert result["id"] == "ENSG00000141510"
        assert result["symbol"] == "TP53"
        assert result["chromosome"] == "17"

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_variant_info(self, mock_query):
        """Test getting variant information."""
        mock_response = {
            "data": {
                "variantInfo": {
                    "id": "17_7674221_G_A",
                    "rsId": "rs28934571",
                    "chromosome": "17",
                    "position": 7674221,
                    "refAllele": "G",
                    "altAllele": "A",
                    "nearestGene": {
                        "id": "ENSG00000141510",
                        "symbol": "TP53",
                        "distance": 100
                    },
                    "nearestGeneDistance": 100,
                    "mostSevereConsequence": "missense_variant",
                    "caddPhred": 25.3,
                    "gnomadAFR": 0.001,
                    "gnomadAMR": 0.002,
                    "gnomadEAS": 0.0005,
                    "gnomadEUR": 0.0008,
                    "gnomadNFE": 0.0009,
                    "gnomadSAS": 0.0012
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_variant_info("17_7674221_G_A")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "VariantInfo" in args[0]
        assert args[1]["variantId"] == "17_7674221_G_A"
        
        assert result["id"] == "17_7674221_G_A"
        assert result["rsId"] == "rs28934571"
        assert result["chromosome"] == "17"
        assert result["nearestGene"]["symbol"] == "TP53"

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_study_info(self, mock_query):
        """Test getting study information."""
        mock_response = {
            "data": {
                "studyInfo": {
                    "studyId": "GCST000001",
                    "traitReported": "Type 2 diabetes",
                    "traitCategory": "metabolic",
                    "pubAuthor": "Scott LJ",
                    "pubDate": "2007-04-26",
                    "pubJournal": "Science",
                    "pmid": "17463246",
                    "nCases": 1924,
                    "nTotal": 4549,
                    "nInitial": 1464,
                    "nReplication": 1132,
                    "hasSumstats": True
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_study_info("GCST000001")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "StudyInfo" in args[0]
        assert args[1]["studyId"] == "GCST000001"
        
        assert result["studyId"] == "GCST000001"
        assert result["traitReported"] == "Type 2 diabetes"
        assert result["pmid"] == "17463246"
        assert result["hasSumstats"] == True

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_associations_for_gene(self, mock_query):
        """Test getting associations for a gene."""
        mock_response = {
            "data": {
                "geneInfo": {
                    "id": "ENSG00000141510",
                    "symbol": "TP53"
                },
                "associatedStudiesForGene": [
                    {
                        "study": {
                            "studyId": "GCST000001",
                            "traitReported": "Type 2 diabetes",
                            "pmid": "17463246",
                            "pubAuthor": "Scott LJ",
                            "pubDate": "2007-04-26"
                        },
                        "variant": {
                            "id": "17_7674221_G_A",
                            "rsId": "rs28934571"
                        },
                        "pval": 2.3e-11,
                        "beta": 0.166,
                        "oddsRatio": 1.18,
                        "ci95Lower": 1.09,
                        "ci95Upper": 1.28
                    }
                ]
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_associations_for_gene("ENSG00000141510", page_size=25)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "GeneAssociations" in args[0]
        assert args[1]["geneId"] == "ENSG00000141510"
        assert args[1]["pageSize"] == 25
        
        assert result["geneInfo"]["symbol"] == "TP53"
        assert len(result["associatedStudiesForGene"]) == 1
        assert result["associatedStudiesForGene"][0]["pval"] == 2.3e-11

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_associations_for_variant(self, mock_query):
        """Test getting associations for a variant."""
        mock_response = {
            "data": {
                "variantInfo": {
                    "id": "17_7674221_G_A",
                    "rsId": "rs28934571"
                },
                "associatedStudiesForVariant": [
                    {
                        "study": {
                            "studyId": "GCST000001",
                            "traitReported": "Type 2 diabetes",
                            "pmid": "17463246",
                            "pubAuthor": "Scott LJ"
                        },
                        "pval": 2.3e-11,
                        "beta": 0.166,
                        "oddsRatio": 1.18,
                        "ci95Lower": 1.09,
                        "ci95Upper": 1.28
                    }
                ]
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_associations_for_variant("17_7674221_G_A", page_size=75)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "VariantAssociations" in args[0]
        assert args[1]["variantId"] == "17_7674221_G_A"
        assert args[1]["pageSize"] == 75
        
        assert result["variantInfo"]["rsId"] == "rs28934571"
        assert len(result["associatedStudiesForVariant"]) == 1
        assert result["associatedStudiesForVariant"][0]["oddsRatio"] == 1.18

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_colocalization(self, mock_query):
        """Test getting colocalization data."""
        mock_response = {
            "data": {
                "colocalisationForGene": [
                    {
                        "leftStudy": {
                            "studyId": "eQTL_1",
                            "traitReported": "Gene expression"
                        },
                        "rightStudy": {
                            "studyId": "GCST000001",
                            "traitReported": "Type 2 diabetes"
                        },
                        "h0": 0.1,
                        "h1": 0.2,
                        "h2": 0.15,
                        "h3": 0.25,
                        "h4": 0.3,
                        "log2h4h3": 0.26
                    }
                ]
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_colocalization("ENSG00000141510", "GCST000001")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "Colocalization" in args[0]
        assert args[1]["geneId"] == "ENSG00000141510"
        assert args[1]["studyId"] == "GCST000001"
        
        assert len(result) == 1
        assert result[0]["h4"] == 0.3
        assert result[0]["leftStudy"]["studyId"] == "eQTL_1"

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_pheWAS(self, mock_query):
        """Test getting PheWAS data."""
        mock_response = {
            "data": {
                "pheWAS": [
                    {
                        "study": {
                            "studyId": "GCST000001",
                            "traitReported": "Type 2 diabetes",
                            "traitCategory": "metabolic"
                        },
                        "pval": 2.3e-11,
                        "beta": 0.166,
                        "oddsRatio": 1.18
                    },
                    {
                        "study": {
                            "studyId": "GCST000002",
                            "traitReported": "Obesity",
                            "traitCategory": "metabolic"
                        },
                        "pval": 5.7e-8,
                        "beta": 0.089,
                        "oddsRatio": 1.09
                    }
                ]
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_pheWAS("17_7674221_G_A", page_size=200)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "PheWAS" in args[0]
        assert args[1]["variantId"] == "17_7674221_G_A"
        assert args[1]["pageSize"] == 200
        
        assert len(result) == 2
        assert result[0]["study"]["traitReported"] == "Type 2 diabetes"
        assert result[1]["oddsRatio"] == 1.09

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_credible_sets(self, mock_query):
        """Test getting credible sets."""
        mock_response = {
            "data": {
                "credibleSets": [
                    {
                        "variant": {
                            "id": "17_7674221_G_A",
                            "rsId": "rs28934571"
                        },
                        "posteriorProbability": 0.85,
                        "pval": 2.3e-11,
                        "beta": 0.166,
                        "standardError": 0.0305
                    },
                    {
                        "variant": {
                            "id": "17_7674222_C_T",
                            "rsId": "rs123456"
                        },
                        "posteriorProbability": 0.15,
                        "pval": 1.2e-8,
                        "beta": 0.142,
                        "standardError": 0.0275
                    }
                ]
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_credible_sets("GCST000001", "17_7674221_G_A")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "CredibleSets" in args[0]
        assert args[1]["studyId"] == "GCST000001"
        assert args[1]["variantId"] == "17_7674221_G_A"
        
        assert len(result) == 2
        assert result[0]["posteriorProbability"] == 0.85
        assert result[1]["variant"]["rsId"] == "rs123456"

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_search_studies(self, mock_query):
        """Test searching studies."""
        mock_response = {
            "data": {
                "search": {
                    "studies": [
                        {
                            "studyId": "GCST000001",
                            "traitReported": "Type 2 diabetes",
                            "pubAuthor": "Scott LJ",
                            "pmid": "17463246"
                        },
                        {
                            "studyId": "GCST000002",
                            "traitReported": "Type 1 diabetes",
                            "pubAuthor": "Barrett JC",
                            "pmid": "19430480"
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.search_studies("diabetes", page_size=20)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "SearchStudies" in args[0]
        assert args[1]["queryString"] == "diabetes"
        assert args[1]["pageSize"] == 20
        
        assert len(result) == 2
        assert result[0]["traitReported"] == "Type 2 diabetes"
        assert result[1]["studyId"] == "GCST000002"

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_manhattan_plot_data(self, mock_query):
        """Test getting Manhattan plot data."""
        mock_response = {
            "data": {
                "manhattan": {
                    "associations": [
                        {
                            "variant": {
                                "id": "17_7674221_G_A",
                                "chromosome": "17",
                                "position": 7674221
                            },
                            "pval": 2.3e-11
                        },
                        {
                            "variant": {
                                "id": "10_114758349_C_T",
                                "chromosome": "10",
                                "position": 114758349
                            },
                            "pval": 5.7e-8
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsGeneticsClient()
        result = client.get_manhattan_plot_data("GCST000001")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "ManhattanPlot" in args[0]
        assert args[1]["studyId"] == "GCST000001"
        
        assert len(result["associations"]) == 2
        assert result["associations"][0]["variant"]["chromosome"] == "17"
        assert result["associations"][1]["pval"] == 5.7e-8

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_query_with_variables(self, mock_query):
        """Test GraphQL query with variables."""
        mock_query.return_value = {"data": {"test": "result"}}
        
        client = OpenTargetsGeneticsClient()
        query = "query Test($var: String!) { test(input: $var) }"
        variables = {"var": "value"}
        
        result = client.query(query, variables)
        
        mock_query.assert_called_once_with(query, variables)

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_query_without_variables(self, mock_query):
        """Test GraphQL query without variables."""
        mock_query.return_value = {"data": {"test": "result"}}
        
        client = OpenTargetsGeneticsClient()
        query = "query { test }"
        
        result = client.query(query)
        
        mock_query.assert_called_once_with(query)

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_empty_results(self, mock_query):
        """Test handling of empty results."""
        mock_query.return_value = {"data": {"geneInfo": None}}
        
        client = OpenTargetsGeneticsClient()
        result = client.get_gene_info("NONEXISTENT")
        
        assert result == {}

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_missing_data_field(self, mock_query):
        """Test handling of missing data field."""
        mock_query.return_value = {"errors": ["Some error"]}
        
        client = OpenTargetsGeneticsClient()
        result = client.get_gene_info("ENSG00000141510")
        
        assert result == {}

    @responses.activate
    def test_query_http_request(self):
        """Test actual HTTP request for GraphQL query."""
        # Mock the HTTP response
        responses.add(
            responses.POST,
            "https://api.genetics.opentargets.org/graphql",
            json={"data": {"test": "result"}},
            status=200
        )
        
        client = OpenTargetsGeneticsClient()
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
            "https://api.genetics.opentargets.org/graphql",
            json={"data": {"geneInfo": {"symbol": "TP53"}}},
            status=200
        )
        
        client = OpenTargetsGeneticsClient()
        query = "query Test($geneId: String!) { geneInfo(geneId: $geneId) { symbol } }"
        variables = {"geneId": "ENSG00000141510"}
        
        result = client.query(query, variables)
        
        assert result["data"]["geneInfo"]["symbol"] == "TP53"
        
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
            "https://api.genetics.opentargets.org/graphql",
            status=500
        )
        
        client = OpenTargetsGeneticsClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.query("query { test }")

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_associations_for_gene_default_page_size(self, mock_query):
        """Test getting associations for gene with default page size."""
        mock_query.return_value = {"data": {}}
        
        client = OpenTargetsGeneticsClient()
        client.get_associations_for_gene("ENSG00000141510")
        
        args, kwargs = mock_query.call_args
        assert args[1]["pageSize"] == 50

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_get_pheWAS_default_page_size(self, mock_query):
        """Test getting PheWAS with default page size."""
        mock_query.return_value = {"data": {"pheWAS": []}}
        
        client = OpenTargetsGeneticsClient()
        client.get_pheWAS("17_7674221_G_A")
        
        args, kwargs = mock_query.call_args
        assert args[1]["pageSize"] == 100

    @patch.object(OpenTargetsGeneticsClient, 'query')
    def test_search_studies_default_page_size(self, mock_query):
        """Test searching studies with default page size."""
        mock_query.return_value = {"data": {"search": {"studies": []}}}
        
        client = OpenTargetsGeneticsClient()
        client.search_studies("diabetes")
        
        args, kwargs = mock_query.call_args
        assert args[1]["pageSize"] == 50