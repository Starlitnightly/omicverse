"""Tests for OpenTargets Platform API client."""

import pytest
from unittest.mock import Mock, patch
import responses
import requests

from omicverse.external.datacollect.api.opentargets import OpenTargetsClient


class TestOpenTargetsClient:
    """Test OpenTargets Platform API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = OpenTargetsClient()
        assert client.base_url == "https://api.platform.opentargets.org/api/v4/graphql"
        assert client.rate_limit == 10
        
        # Test custom initialization
        client_custom = OpenTargetsClient(
            base_url="https://custom.platform.org/graphql",
            rate_limit=5
        )
        assert client_custom.base_url == "https://custom.platform.org/graphql"
        assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = OpenTargetsClient()
        headers = client.get_default_headers()
        
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    @patch.object(OpenTargetsClient, 'query')
    def test_get_target(self, mock_query):
        """Test getting target information."""
        mock_response = {
            "data": {
                "target": {
                    "id": "ENSG00000141510",
                    "approvedSymbol": "TP53",
                    "approvedName": "tumor protein p53",
                    "bioType": "protein_coding",
                    "hgncId": "HGNC:11998",
                    "uniprotIds": ["P04637"],
                    "genomicLocation": {
                        "chromosome": "17",
                        "start": 7661779,
                        "end": 7687550,
                        "strand": "-"
                    },
                    "alternativeGenes": ["TP53"],
                    "pathways": [
                        {
                            "pathway": "p53 signaling pathway",
                            "pathwayId": "R-HSA-69481"
                        }
                    ],
                    "proteinAnnotations": [
                        {
                            "id": "P04637",
                            "functions": ["DNA binding", "transcription factor"]
                        }
                    ],
                    "tractability": [
                        {
                            "id": "clinical_precedence",
                            "modality": "Small molecule",
                            "value": True
                        }
                    ],
                    "safety": {
                        "adverseEvents": [
                            {
                                "count": 5,
                                "criticalValue": 2.5
                            }
                        ],
                        "safetyLiabilities": [
                            {
                                "event": "hepatotoxicity",
                                "eventId": "EFO_0004611"
                            }
                        ]
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_target("ENSG00000141510")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "TargetInfo" in args[0]
        assert args[1]["ensemblId"] == "ENSG00000141510"
        
        assert result["id"] == "ENSG00000141510"
        assert result["approvedSymbol"] == "TP53"
        assert result["genomicLocation"]["chromosome"] == "17"
        assert len(result["pathways"]) == 1

    @patch.object(OpenTargetsClient, 'query')
    def test_get_disease(self, mock_query):
        """Test getting disease information."""
        mock_response = {
            "data": {
                "disease": {
                    "id": "EFO_0001360",
                    "name": "type II diabetes mellitus",
                    "description": "A form of diabetes mellitus characterized by insulin resistance",
                    "synonyms": ["Type 2 diabetes", "T2D", "NIDDM"],
                    "therapeuticAreas": [
                        {
                            "id": "EFO_0000408",
                            "name": "endocrine system disease"
                        }
                    ],
                    "parents": [
                        {
                            "id": "EFO_0000400",
                            "name": "diabetes mellitus"
                        }
                    ],
                    "children": [
                        {
                            "id": "EFO_0007061",
                            "name": "maturity-onset diabetes of the young"
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_disease("EFO_0001360")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "DiseaseInfo" in args[0]
        assert args[1]["efoId"] == "EFO_0001360"
        
        assert result["id"] == "EFO_0001360"
        assert result["name"] == "type II diabetes mellitus"
        assert len(result["synonyms"]) == 3
        assert len(result["therapeuticAreas"]) == 1

    @patch.object(OpenTargetsClient, 'query')
    def test_get_associations_with_disease_filter(self, mock_query):
        """Test getting associations with disease filter."""
        mock_response = {
            "data": {
                "target": {
                    "id": "ENSG00000141510",
                    "approvedSymbol": "TP53",
                    "associatedDiseases": {
                        "score": 0.85,
                        "datatypeScores": [
                            {
                                "id": "genetic_association",
                                "score": 0.9
                            },
                            {
                                "id": "literature",
                                "score": 0.8
                            }
                        ],
                        "disease": {
                            "id": "EFO_0001360",
                            "name": "type II diabetes mellitus"
                        }
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_associations("ENSG00000141510", disease_id="EFO_0001360")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "AssociationScore" in args[0]
        assert args[1]["targetId"] == "ENSG00000141510"
        assert args[1]["diseaseId"] == "EFO_0001360"
        
        assert result["approvedSymbol"] == "TP53"
        assert result["associatedDiseases"]["score"] == 0.85
        assert len(result["associatedDiseases"]["datatypeScores"]) == 2

    @patch.object(OpenTargetsClient, 'query')
    def test_get_associations_without_disease_filter(self, mock_query):
        """Test getting associations without disease filter."""
        mock_response = {
            "data": {
                "target": {
                    "id": "ENSG00000141510",
                    "approvedSymbol": "TP53",
                    "associatedDiseases": {
                        "rows": [
                            {
                                "score": 0.85,
                                "datatypeScores": [
                                    {
                                        "id": "genetic_association",
                                        "score": 0.9
                                    }
                                ],
                                "disease": {
                                    "id": "EFO_0001360",
                                    "name": "type II diabetes mellitus"
                                }
                            }
                        ],
                        "count": 1
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_associations("ENSG00000141510", size=100)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "TargetAssociations" in args[0]
        assert args[1]["targetId"] == "ENSG00000141510"
        assert args[1]["size"] == 100
        
        assert result["approvedSymbol"] == "TP53"
        assert result["associatedDiseases"]["count"] == 1
        assert len(result["associatedDiseases"]["rows"]) == 1

    @patch.object(OpenTargetsClient, 'query')
    def test_get_evidence(self, mock_query):
        """Test getting evidence for target-disease association."""
        mock_response = {
            "data": {
                "evidences": {
                    "rows": [
                        {
                            "id": "evidence_1",
                            "score": 0.9,
                            "datasourceId": "gwas_catalog",
                            "datatypeId": "genetic_association",
                            "targetFromSourceId": "ENSG00000141510",
                            "diseaseFromSourceMappedId": "EFO_0001360",
                            "publicationYear": 2007,
                            "publicationFirstAuthor": "Scott LJ",
                            "literature": ["17463246"]
                        }
                    ],
                    "count": 1
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_evidence("ENSG00000141510", "EFO_0001360", 
                                   datasource="gwas_catalog", size=25)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "Evidence" in args[0]
        assert args[1]["targetId"] == "ENSG00000141510"
        assert args[1]["diseaseId"] == "EFO_0001360"
        assert args[1]["datasource"] == "[gwas_catalog]"
        assert args[1]["size"] == 25
        
        assert result["count"] == 1
        assert result["rows"][0]["score"] == 0.9
        assert result["rows"][0]["datasourceId"] == "gwas_catalog"

    @patch.object(OpenTargetsClient, 'query')
    def test_get_evidence_without_datasource(self, mock_query):
        """Test getting evidence without datasource filter."""
        mock_response = {"data": {"evidences": {"rows": [], "count": 0}}}
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_evidence("ENSG00000141510", "EFO_0001360")
        
        args, kwargs = mock_query.call_args
        assert "datasource" not in args[1]

    @patch.object(OpenTargetsClient, 'query')
    def test_get_drug_info(self, mock_query):
        """Test getting drug information."""
        mock_response = {
            "data": {
                "drug": {
                    "id": "CHEMBL25",
                    "name": "aspirin",
                    "description": "Acetylsalicylic acid",
                    "drugType": "Small molecule",
                    "maximumClinicalTrialPhase": 4,
                    "synonyms": ["acetylsalicylic acid", "ASA"],
                    "tradeNames": ["Aspirin", "Bayer Aspirin"],
                    "yearOfFirstApproval": 1897,
                    "mechanismsOfAction": {
                        "rows": [
                            {
                                "mechanismOfAction": "Cyclooxygenase inhibitor",
                                "targetName": "Prostaglandin-endoperoxide synthase 1",
                                "targets": [
                                    {
                                        "id": "ENSG00000095303",
                                        "approvedSymbol": "PTGS1"
                                    }
                                ]
                            }
                        ]
                    },
                    "indications": {
                        "rows": [
                            {
                                "disease": {
                                    "id": "EFO_0000270",
                                    "name": "cardiovascular disease"
                                },
                                "maxPhaseForIndication": 4
                            }
                        ]
                    },
                    "adverseEvents": {
                        "count": 15,
                        "criticalValue": 3.2
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_drug_info("CHEMBL25")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "DrugInfo" in args[0]
        assert args[1]["chemblId"] == "CHEMBL25"
        
        assert result["id"] == "CHEMBL25"
        assert result["name"] == "aspirin"
        assert result["maximumClinicalTrialPhase"] == 4
        assert len(result["synonyms"]) == 2

    @patch.object(OpenTargetsClient, 'query')
    def test_search_targets(self, mock_query):
        """Test searching for targets."""
        mock_response = {
            "data": {
                "search": {
                    "hits": [
                        {
                            "id": "ENSG00000141510",
                            "entity": "target",
                            "name": "TP53",
                            "description": "tumor protein p53",
                            "score": 0.95
                        },
                        {
                            "id": "ENSG00000077097",
                            "entity": "target",
                            "name": "TOP2A",
                            "description": "topoisomerase II alpha",
                            "score": 0.82
                        }
                    ],
                    "total": 2
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.search_targets("tumor protein", size=25)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "SearchTargets" in args[0]
        assert args[1]["queryString"] == "tumor protein"
        assert args[1]["size"] == 25
        
        assert len(result) == 2
        assert result[0]["name"] == "TP53"
        assert result[1]["score"] == 0.82

    @patch.object(OpenTargetsClient, 'query')
    def test_search_diseases(self, mock_query):
        """Test searching for diseases."""
        mock_response = {
            "data": {
                "search": {
                    "hits": [
                        {
                            "id": "EFO_0001360",
                            "entity": "disease",
                            "name": "type II diabetes mellitus",
                            "description": "A form of diabetes mellitus",
                            "score": 0.98
                        }
                    ],
                    "total": 1
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.search_diseases("diabetes", size=10)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "SearchDiseases" in args[0]
        assert args[1]["queryString"] == "diabetes"
        assert args[1]["size"] == 10
        
        assert len(result) == 1
        assert result[0]["entity"] == "disease"
        assert result[0]["score"] == 0.98

    @patch.object(OpenTargetsClient, 'query')
    def test_search_drugs(self, mock_query):
        """Test searching for drugs."""
        mock_response = {
            "data": {
                "search": {
                    "hits": [
                        {
                            "id": "CHEMBL25",
                            "entity": "drug",
                            "name": "aspirin",
                            "description": "Acetylsalicylic acid",
                            "score": 0.95
                        },
                        {
                            "id": "CHEMBL88",
                            "entity": "drug",
                            "name": "ibuprofen",
                            "description": "Non-steroidal anti-inflammatory drug",
                            "score": 0.88
                        }
                    ],
                    "total": 2
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.search_drugs("anti-inflammatory", size=15)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "SearchDrugs" in args[0]
        assert args[1]["queryString"] == "anti-inflammatory"
        assert args[1]["size"] == 15
        
        assert len(result) == 2
        assert result[0]["name"] == "aspirin"
        assert result[1]["id"] == "CHEMBL88"

    @patch.object(OpenTargetsClient, 'query')
    def test_get_known_drugs_for_target(self, mock_query):
        """Test getting known drugs for a target."""
        mock_response = {
            "data": {
                "target": {
                    "knownDrugs": {
                        "uniqueDrugs": 5,
                        "uniqueDiseases": 3,
                        "uniqueTargets": 1,
                        "count": 8,
                        "rows": [
                            {
                                "drug": {
                                    "id": "CHEMBL25",
                                    "name": "aspirin",
                                    "drugType": "Small molecule",
                                    "maximumClinicalTrialPhase": 4
                                },
                                "disease": {
                                    "id": "EFO_0000270",
                                    "name": "cardiovascular disease"
                                },
                                "phase": 4,
                                "status": "Approved",
                                "mechanismOfAction": "Cyclooxygenase inhibitor",
                                "references": [
                                    {
                                        "source": "ChEMBL",
                                        "ids": ["CHEMBL25"]
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_known_drugs_for_target("ENSG00000095303")
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "KnownDrugs" in args[0]
        assert args[1]["targetId"] == "ENSG00000095303"
        
        assert result["uniqueDrugs"] == 5
        assert result["count"] == 8
        assert len(result["rows"]) == 1
        assert result["rows"][0]["drug"]["name"] == "aspirin"

    @patch.object(OpenTargetsClient, 'query')
    def test_get_similar_targets(self, mock_query):
        """Test getting similar targets."""
        mock_response = {
            "data": {
                "target": {
                    "id": "ENSG00000141510",
                    "similarEntities": [
                        {
                            "score": 0.85,
                            "target": {
                                "id": "ENSG00000077097",
                                "approvedSymbol": "TOP2A",
                                "approvedName": "topoisomerase II alpha"
                            }
                        },
                        {
                            "score": 0.78,
                            "target": {
                                "id": "ENSG00000064666",
                                "approvedSymbol": "BRCA1",
                                "approvedName": "BRCA1 DNA repair associated"
                            }
                        }
                    ]
                }
            }
        }
        mock_query.return_value = mock_response
        
        client = OpenTargetsClient()
        result = client.get_similar_targets("ENSG00000141510", threshold=0.7, size=10)
        
        mock_query.assert_called_once()
        args, kwargs = mock_query.call_args
        assert "SimilarTargets" in args[0]
        assert args[1]["targetId"] == "ENSG00000141510"
        assert args[1]["threshold"] == 0.7
        assert args[1]["size"] == 10
        
        assert len(result) == 2
        assert result[0]["score"] == 0.85
        assert result[1]["target"]["approvedSymbol"] == "BRCA1"

    @patch.object(OpenTargetsClient, 'query')
    def test_query_with_variables(self, mock_query):
        """Test GraphQL query with variables."""
        mock_query.return_value = {"data": {"test": "result"}}
        
        client = OpenTargetsClient()
        query = "query Test($var: String!) { test(input: $var) }"
        variables = {"var": "value"}
        
        result = client.query(query, variables)
        
        mock_query.assert_called_once_with(query, variables)

    @patch.object(OpenTargetsClient, 'query')
    def test_query_without_variables(self, mock_query):
        """Test GraphQL query without variables."""
        mock_query.return_value = {"data": {"test": "result"}}
        
        client = OpenTargetsClient()
        query = "query { test }"
        
        result = client.query(query)
        
        mock_query.assert_called_once_with(query)

    @patch.object(OpenTargetsClient, 'query')
    def test_empty_results(self, mock_query):
        """Test handling of empty results."""
        mock_query.return_value = {"data": {"target": None}}
        
        client = OpenTargetsClient()
        result = client.get_target("NONEXISTENT")
        
        assert result == {}

    @patch.object(OpenTargetsClient, 'query')
    def test_missing_data_field(self, mock_query):
        """Test handling of missing data field."""
        mock_query.return_value = {"errors": ["Some error"]}
        
        client = OpenTargetsClient()
        result = client.get_target("ENSG00000141510")
        
        assert result == {}

    @responses.activate
    def test_query_http_request(self):
        """Test actual HTTP request for GraphQL query."""
        # Mock the HTTP response
        responses.add(
            responses.POST,
            "https://api.platform.opentargets.org/api/v4/graphql",
            json={"data": {"test": "result"}},
            status=200
        )
        
        client = OpenTargetsClient()
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
            "https://api.platform.opentargets.org/api/v4/graphql",
            json={"data": {"target": {"approvedSymbol": "TP53"}}},
            status=200
        )
        
        client = OpenTargetsClient()
        query = "query Test($ensemblId: String!) { target(ensemblId: $ensemblId) { approvedSymbol } }"
        variables = {"ensemblId": "ENSG00000141510"}
        
        result = client.query(query, variables)
        
        assert result["data"]["target"]["approvedSymbol"] == "TP53"
        
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
            "https://api.platform.opentargets.org/api/v4/graphql",
            status=500
        )
        
        client = OpenTargetsClient()
        
        with pytest.raises((requests.exceptions.HTTPError, requests.exceptions.RetryError)):
            client.query("query { test }")

    @patch.object(OpenTargetsClient, 'query')
    def test_get_associations_default_size(self, mock_query):
        """Test getting associations with default size."""
        mock_query.return_value = {"data": {"target": {}}}
        
        client = OpenTargetsClient()
        client.get_associations("ENSG00000141510")
        
        args, kwargs = mock_query.call_args
        assert args[1]["size"] == 50

    @patch.object(OpenTargetsClient, 'query')
    def test_search_targets_default_size(self, mock_query):
        """Test searching targets with default size."""
        mock_query.return_value = {"data": {"search": {"hits": []}}}
        
        client = OpenTargetsClient()
        client.search_targets("test")
        
        args, kwargs = mock_query.call_args
        assert args[1]["size"] == 50

    @patch.object(OpenTargetsClient, 'query')
    def test_get_similar_targets_default_params(self, mock_query):
        """Test getting similar targets with default parameters."""
        mock_query.return_value = {"data": {"target": {"similarEntities": []}}}
        
        client = OpenTargetsClient()
        client.get_similar_targets("ENSG00000141510")
        
        args, kwargs = mock_query.call_args
        assert args[1]["threshold"] == 0.5
        assert args[1]["size"] == 20

    @patch.object(OpenTargetsClient, 'query')
    def test_get_evidence_default_size(self, mock_query):
        """Test getting evidence with default size."""
        mock_query.return_value = {"data": {"evidences": {"rows": [], "count": 0}}}
        
        client = OpenTargetsClient()
        client.get_evidence("ENSG00000141510", "EFO_0001360")
        
        args, kwargs = mock_query.call_args
        assert args[1]["size"] == 50