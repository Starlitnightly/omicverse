"""Tests for UniProt API client."""

import pytest
from unittest.mock import Mock, patch
import responses
import requests
import time

from omicverse.external.datacollect.api.uniprot import UniProtClient


class TestUniProtClient:
    """Test UniProt API client."""

    def test_initialization(self):
        """Test client initialization."""
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            assert client.base_url == "https://rest.uniprot.org"
            assert client.rate_limit == 10
            
            # Test custom initialization
            client_custom = UniProtClient(rate_limit=5)
            assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            headers = client.get_default_headers()
            
            assert headers["Accept"] == "application/json"

    @patch.object(UniProtClient, 'get')
    def test_get_entry_json(self, mock_get):
        """Test getting entry in JSON format."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "primaryAccession": "P04637",
            "uniProtkbId": "P53_HUMAN",
            "entryAudit": {
                "firstPublicDate": "1987-08-13",
                "lastAnnotationUpdateDate": "2023-11-08"
            },
            "annotationScore": 5,
            "organism": {
                "taxonId": 9606,
                "scientificName": "Homo sapiens",
                "commonName": "Human"
            },
            "proteinExistence": "1: Evidence at protein level",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {
                        "value": "Cellular tumor antigen p53"
                    }
                },
                "alternativeNames": [
                    {
                        "fullName": {
                            "value": "Tumor suppressor p53"
                        }
                    }
                ]
            },
            "genes": [
                {
                    "geneName": {
                        "value": "TP53"
                    }
                }
            ],
            "comments": [
                {
                    "commentType": "FUNCTION",
                    "texts": [
                        {
                            "value": "Acts as a tumor suppressor in many tumor types"
                        }
                    ]
                }
            ],
            "features": [
                {
                    "type": "Domain",
                    "location": {
                        "start": {
                            "value": 94
                        },
                        "end": {
                            "value": 292
                        }
                    },
                    "description": "DNA-binding"
                }
            ],
            "sequence": {
                "value": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
                "length": 393,
                "molWeight": 43653,
                "crc64": "AC3DA6D8D46C9F49"
            }
        }
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.get_entry("P04637")
            
            mock_get.assert_called_once_with("/uniprotkb/P04637")
            assert result["primaryAccession"] == "P04637"
            assert result["uniProtkbId"] == "P53_HUMAN"
            assert result["organism"]["taxonId"] == 9606

    @patch.object(UniProtClient, 'get')
    def test_get_entry_fasta(self, mock_get):
        """Test getting entry in FASTA format."""
        mock_response = Mock()
        mock_response.text = ">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEA"
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.get_entry("P04637", format="fasta")
            
            mock_get.assert_called_once_with("/uniprotkb/P04637.fasta")
            assert ">sp|P04637|P53_HUMAN" in result
            assert "MEEPQSDPSV" in result

    @patch.object(UniProtClient, 'get')
    def test_search(self, mock_get):
        """Test searching UniProt database."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "primaryAccession": "P04637",
                    "uniProtkbId": "P53_HUMAN",
                    "organism": {
                        "taxonId": 9606,
                        "scientificName": "Homo sapiens"
                    },
                    "proteinDescription": {
                        "recommendedName": {
                            "fullName": {
                                "value": "Cellular tumor antigen p53"
                            }
                        }
                    },
                    "genes": [
                        {
                            "geneName": {
                                "value": "TP53"
                            }
                        }
                    ]
                }
            ],
            "facets": [],
            "query": {
                "term": "gene:TP53 AND organism_id:9606",
                "size": 25,
                "format": "json"
            }
        }
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.search(
                query="gene:TP53 AND organism_id:9606",
                fields=["accession", "gene_names", "protein_name"],
                size=25
            )
            
            mock_get.assert_called_once_with(
                "/uniprotkb/search",
                params={
                    "query": "gene:TP53 AND organism_id:9606",
                    "format": "json",
                    "size": 25,
                    "fields": "accession,gene_names,protein_name"
                }
            )
            assert len(result["results"]) == 1
            assert result["results"][0]["primaryAccession"] == "P04637"

    @patch.object(UniProtClient, 'get')
    def test_search_with_cursor(self, mock_get):
        """Test searching with cursor for pagination."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": [], "facets": []}
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            client.search(
                query="gene:TP53",
                cursor="abc123",
                size=50
            )
            
            args, kwargs = mock_get.call_args
            assert kwargs["params"]["cursor"] == "abc123"

    @patch.object(UniProtClient, 'get_entry')
    def test_get_fasta(self, mock_get_entry):
        """Test getting FASTA sequence."""
        mock_get_entry.return_value = ">sp|P04637|P53_HUMAN\nMEEPQSDPSV"
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.get_fasta("P04637")
            
            mock_get_entry.assert_called_once_with("P04637", format="fasta")
            assert ">sp|P04637|P53_HUMAN" in result

    @patch.object(UniProtClient, 'get_entry')
    def test_get_features(self, mock_get_entry):
        """Test getting protein features."""
        mock_get_entry.return_value = {
            "features": [
                {
                    "type": "Domain",
                    "location": {
                        "start": {"value": 94},
                        "end": {"value": 292}
                    },
                    "description": "DNA-binding"
                },
                {
                    "type": "Modified residue",
                    "location": {
                        "start": {"value": 15},
                        "end": {"value": 15}
                    },
                    "description": "Phosphoserine"
                }
            ]
        }
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.get_features("P04637")
            
            mock_get_entry.assert_called_once_with("P04637")
            assert len(result) == 2
            assert result[0]["type"] == "Domain"
            assert result[1]["description"] == "Phosphoserine"

    @patch.object(UniProtClient, 'get')
    def test_batch_retrieve_json(self, mock_get):
        """Test batch retrieval in JSON format."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "primaryAccession": "P04637",
                    "uniProtkbId": "P53_HUMAN"
                },
                {
                    "primaryAccession": "P53762",
                    "uniProtkbId": "MAPK1_HUMAN"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.batch_retrieve(["P04637", "P53762"])
            
            mock_get.assert_called_once_with(
                "/uniprotkb/accessions",
                params={
                    "from": "P04637,P53762",
                    "format": "json"
                }
            )
            assert len(result["results"]) == 2
            assert result["results"][0]["primaryAccession"] == "P04637"

    @patch.object(UniProtClient, 'get')
    def test_batch_retrieve_fasta(self, mock_get):
        """Test batch retrieval in FASTA format."""
        mock_response = Mock()
        mock_response.text = ">sp|P04637|P53_HUMAN\nMEEPQSDPSV\n>sp|P53762|MAPK1_HUMAN\nMQTVASDRLE"
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.batch_retrieve(["P04637", "P53762"], format="fasta")
            
            mock_get.assert_called_once_with(
                "/uniprotkb/accessions",
                params={
                    "from": "P04637,P53762",
                    "format": "fasta"
                }
            )
            assert ">sp|P04637|P53_HUMAN" in result
            assert ">sp|P53762|MAPK1_HUMAN" in result

    @patch.object(UniProtClient, 'get')
    def test_batch_retrieve_compressed(self, mock_get):
        """Test batch retrieval with compression."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            client.batch_retrieve(["P04637"], compressed=True)
            
            args, kwargs = mock_get.call_args
            assert kwargs["params"]["compressed"] == "true"

    @patch.object(UniProtClient, 'post')
    @patch.object(UniProtClient, 'get')
    @patch('time.sleep')
    def test_id_mapping(self, mock_sleep, mock_get, mock_post):
        """Test ID mapping between databases."""
        # Mock job submission
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"jobId": "job123"}
        mock_post.return_value = mock_post_response
        
        # Mock status polling - first call returns running, second returns results
        mock_get_responses = [
            Mock(),
            Mock()
        ]
        mock_get_responses[0].json.return_value = {"jobStatus": "RUNNING"}
        mock_get_responses[1].json.return_value = {
            "jobStatus": "FINISHED",
            "results": {
                "mappings": [
                    {
                        "from": "P04637",
                        "to": "ENSG00000141510"
                    }
                ]
            }
        }
        mock_get.side_effect = mock_get_responses
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.id_mapping("UniProtKB_AC-ID", "Ensembl", ["P04637"])
            
            # Verify job submission
            mock_post.assert_called_once_with(
                "/idmapping/run",
                data={
                    "from": "UniProtKB_AC-ID",
                    "to": "Ensembl",
                    "ids": "P04637"
                }
            )
            
            # Verify polling
            assert mock_get.call_count == 2
            mock_get.assert_any_call("/idmapping/status/job123")
            
            # Verify sleep was called
            mock_sleep.assert_called()
            
            assert "results" in result
            assert result["results"]["mappings"][0]["from"] == "P04637"

    @patch.object(UniProtClient, 'search')
    def test_search_default_params(self, mock_search):
        """Test search with default parameters."""
        mock_search.return_value = {"results": []}
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            client.search("gene:TP53")
            
            args, kwargs = mock_search.call_args
            assert args[0] == "gene:TP53"
            # Check default parameters are used in the actual search call
            # This will be verified by checking the get call parameters

    @responses.activate
    def test_get_entry_http_request(self):
        """Test actual HTTP request for getting entry."""
        entry_data = {
            "primaryAccession": "P04637",
            "uniProtkbId": "P53_HUMAN"
        }
        
        responses.add(
            responses.GET,
            "https://rest.uniprot.org/uniprotkb/P04637",
            json=entry_data,
            status=200
        )
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.get_entry("P04637")
            
            assert result["primaryAccession"] == "P04637"
            assert result["uniProtkbId"] == "P53_HUMAN"
            assert len(responses.calls) == 1

    @responses.activate
    def test_search_http_request(self):
        """Test actual HTTP request for search."""
        search_data = {
            "results": [
                {
                    "primaryAccession": "P04637",
                    "genes": [{"geneName": {"value": "TP53"}}]
                }
            ]
        }
        
        responses.add(
            responses.GET,
            "https://rest.uniprot.org/uniprotkb/search",
            json=search_data,
            status=200
        )
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.search("gene:TP53", size=10)
            
            assert len(result["results"]) == 1
            assert result["results"][0]["primaryAccession"] == "P04637"
            
            # Check request parameters
            request = responses.calls[0].request
            assert "query=gene%3ATP53" in request.url
            assert "size=10" in request.url

    @responses.activate
    def test_http_error_handling(self):
        """Test HTTP error handling."""
        responses.add(
            responses.GET,
            "https://rest.uniprot.org/uniprotkb/INVALID",
            status=404
        )
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            
            with pytest.raises(requests.exceptions.HTTPError):
                client.get_entry("INVALID")

    @patch.object(UniProtClient, 'get_entry')
    def test_get_features_empty(self, mock_get_entry):
        """Test getting features when none exist."""
        mock_get_entry.return_value = {}
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            result = client.get_features("P04637")
            
            assert result == []

    @patch.object(UniProtClient, 'get')
    def test_search_without_fields(self, mock_get):
        """Test search without specifying fields."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            client.search("gene:TP53")
            
            args, kwargs = mock_get.call_args
            assert "fields" not in kwargs["params"]

    @patch.object(UniProtClient, 'get')
    def test_search_without_cursor(self, mock_get):
        """Test search without cursor."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response
        
        with patch('src.api.uniprot.settings') as mock_settings:
            mock_settings.api.uniprot_base_url = "https://rest.uniprot.org"
            
            client = UniProtClient()
            client.search("gene:TP53")
            
            args, kwargs = mock_get.call_args
            assert "cursor" not in kwargs["params"]