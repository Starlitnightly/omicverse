"""Tests for PDB API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.pdb import PDBClient


class TestPDBClient:
    """Test PDB API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = PDBClient()
        assert client.base_url == "https://data.rcsb.org"
        assert client.rate_limit == 10
        assert client.search_url == "https://search.rcsb.org"
        
        # Test custom initialization
        client_custom = PDBClient(rate_limit=5)
        assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = PDBClient()
        headers = client.get_default_headers()
        
        assert headers["Accept"] == "application/json"
        assert "User-Agent" in headers

    @patch.object(PDBClient, 'get')
    def test_get_entry(self, mock_get):
        """Test getting PDB entry information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "rcsb_id": "1ABC",
            "struct": {
                "title": "Crystal structure of human hemoglobin",
                "pdbx_descriptor": "Hemoglobin subunit alpha, Hemoglobin subunit beta"
            },
            "pdbx_database_status": {
                "status_code": "REL",
                "deposit_date": "2020-01-15",
                "status_code_mr": "REPL"
            },
            "exptl": [
                {
                    "method": "X-RAY DIFFRACTION"
                }
            ],
            "refine": [
                {
                    "ls_d_res_high": 1.5,
                    "ls_R_factor_R_work": 0.183,
                    "ls_R_factor_R_free": 0.215
                }
            ],
            "cell": {
                "length_a": 63.15,
                "length_b": 83.59,
                "length_c": 53.80,
                "angle_alpha": 90.0,
                "angle_beta": 99.25,
                "angle_gamma": 90.0
            },
            "symmetry": {
                "space_group_name_H_M": "P 1 21 1"
            }
        }
        mock_get.return_value = mock_response
        
        client = PDBClient()
        result = client.get_entry("1abc")
        
        mock_get.assert_called_once_with("/rest/v1/core/entry/1ABC")
        assert result["rcsb_id"] == "1ABC"
        assert result["struct"]["title"] == "Crystal structure of human hemoglobin"
        assert result["exptl"][0]["method"] == "X-RAY DIFFRACTION"
        assert result["refine"][0]["ls_d_res_high"] == 1.5

    @patch.object(PDBClient, 'get')
    def test_get_structure(self, mock_get):
        """Test downloading structure file."""
        mock_response = Mock()
        mock_response.text = """HEADER    TRANSPORT PROTEIN                       15-JAN-20   1ABC              
TITLE     CRYSTAL STRUCTURE OF HUMAN HEMOGLOBIN                           
ATOM      1  N   VAL A   1      20.154  -7.859  15.681  1.00 25.00           N  
ATOM      2  CA  VAL A   1      19.939  -6.667  14.853  1.00 25.00           C  
END                                                                             """
        mock_get.return_value = mock_response
        
        client = PDBClient()
        result = client.get_structure("1abc", format="pdb")
        
        mock_get.assert_called_once_with("/download/1ABC.pdb")
        assert "HEADER    TRANSPORT PROTEIN" in result
        assert "ATOM      1  N   VAL A   1" in result

    @patch.object(PDBClient, 'get')
    def test_get_structure_cif_format(self, mock_get):
        """Test downloading structure file in CIF format."""
        mock_response = Mock()
        mock_response.text = """data_1ABC
#
_entry.id   1ABC
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
ATOM 1 N N"""
        mock_get.return_value = mock_response
        
        client = PDBClient()
        result = client.get_structure("1abc", format="cif")
        
        mock_get.assert_called_once_with("/download/1ABC.cif")
        assert "data_1ABC" in result

    @patch.object(PDBClient, 'search')
    def test_search(self, mock_search):
        """Test PDB search functionality."""
        mock_response = {
            "query_id": "12345",
            "result_type": "entry",
            "total_count": 2,
            "result_set": [
                {
                    "identifier": "1ABC",
                    "score": 1.0
                },
                {
                    "identifier": "2XYZ",
                    "score": 0.95
                }
            ]
        }
        mock_search.return_value = mock_response
        
        client = PDBClient()
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "value": "hemoglobin"
            }
        }
        result = client.search(query, return_type="entry", rows=25, start=0)
        
        expected_request = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "pager": {
                    "start": 0,
                    "rows": 25,
                }
            }
        }
        
        mock_search.assert_called_once()
        assert result["total_count"] == 2
        assert len(result["result_set"]) == 2
        assert result["result_set"][0]["identifier"] == "1ABC"

    @patch.object(PDBClient, 'search')
    def test_text_search(self, mock_search):
        """Test text search functionality."""
        mock_response = {
            "total_count": 1,
            "result_set": [
                {
                    "identifier": "1ABC",
                    "score": 0.98
                }
            ]
        }
        mock_search.return_value = mock_response
        
        client = PDBClient()
        result = client.text_search("insulin")
        
        expected_query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "value": "insulin",
            }
        }
        
        mock_search.assert_called_once_with(expected_query)
        assert result["total_count"] == 1
        assert result["result_set"][0]["identifier"] == "1ABC"

    @patch.object(PDBClient, 'post')
    def test_get_ligands(self, mock_post):
        """Test getting ligands for a PDB entry."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "nonpolymer_entities": [
                    {
                        "rcsb_id": "1ABC_ATP_1",
                        "rcsb_nonpolymer_entity": {
                            "comp_id": "ATP",
                            "name": "ADENOSINE-5'-TRIPHOSPHATE",
                            "formula": "C10 H16 N5 O13 P3",
                            "formula_weight": 507.181
                        }
                    },
                    {
                        "rcsb_id": "1ABC_MG_1",
                        "rcsb_nonpolymer_entity": {
                            "comp_id": "MG",
                            "name": "MAGNESIUM ION",
                            "formula": "Mg",
                            "formula_weight": 24.305
                        }
                    }
                ]
            }
        }
        mock_post.return_value = mock_response
        
        client = PDBClient()
        result = client.get_ligands("1abc")
        
        mock_post.assert_called_once_with("/graphql", json={"query": f"""
            {{
              nonpolymer_entities(entry_id: "1ABC") {{
                rcsb_id
                rcsb_nonpolymer_entity {{
                  comp_id
                  name
                  formula
                  formula_weight
                }}
              }}
            }}
            """})
        
        assert len(result) == 2
        assert result[0]["rcsb_nonpolymer_entity"]["comp_id"] == "ATP"
        assert result[1]["rcsb_nonpolymer_entity"]["comp_id"] == "MG"

    @patch.object(PDBClient, 'post')
    def test_get_ligands_no_data(self, mock_post):
        """Test getting ligands when no data is returned."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"nonpolymer_entities": None}}
        mock_post.return_value = mock_response
        
        client = PDBClient()
        result = client.get_ligands("9999")
        
        assert result == []

    @patch.object(PDBClient, 'get')
    def test_get_assemblies(self, mock_get):
        """Test getting biological assemblies."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "assembly_id": "1",
                "details": "author_and_software_defined_assembly",
                "method_details": "PISA",
                "oligomeric_details": "monomeric",
                "oligomeric_count": 1,
                "preferred": True
            }
        ]
        mock_get.return_value = mock_response
        
        client = PDBClient()
        result = client.get_assemblies("1abc")
        
        mock_get.assert_called_once_with("/rest/v1/core/assembly/1ABC")
        assert len(result) == 1
        assert result[0]["assembly_id"] == "1"
        assert result[0]["preferred"] is True

    @patch.object(PDBClient, 'post')
    def test_get_polymer_entities(self, mock_post):
        """Test getting polymer entities (chains)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "polymer_entities": [
                    {
                        "rcsb_id": "1ABC_1",
                        "entity_poly": {
                            "pdbx_seq_one_letter_code": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
                            "entity_id": "1"
                        },
                        "rcsb_polymer_entity_instance_container_identifiers": {
                            "auth_asym_ids": ["A", "C"]
                        },
                        "rcsb_polymer_entity_container_identifiers": {
                            "uniprot_ids": ["P69905"]
                        },
                        "rcsb_entity_source_organism": {
                            "ncbi_scientific_name": "Homo sapiens"
                        }
                    }
                ]
            }
        }
        mock_post.return_value = mock_response
        
        client = PDBClient()
        result = client.get_polymer_entities("1abc")
        
        mock_post.assert_called_once()
        assert len(result) == 1
        assert result[0]["entity_poly"]["entity_id"] == "1"
        assert "P69905" in result[0]["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"]

    @patch.object(PDBClient, 'search')
    def test_sequence_search(self, mock_search):
        """Test sequence similarity search."""
        mock_response = {
            "total_count": 3,
            "result_set": [
                {
                    "identifier": "1ABC",
                    "score": 0.95,
                    "services": [
                        {
                            "service_type": "sequence",
                            "nodes": [
                                {
                                    "node_id": 0,
                                    "original_score": 234.0,
                                    "norm_score": 0.95
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_search.return_value = mock_response
        
        client = PDBClient()
        sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRF"
        result = client.sequence_search(sequence, sequence_type="protein", e_value=0.01)
        
        expected_query = {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "value": sequence,
                "sequence_type": "protein",
                "evalue_cutoff": 0.01,
            }
        }
        
        mock_search.assert_called_once_with(expected_query)
        assert result["total_count"] == 3
        assert result["result_set"][0]["score"] == 0.95

    def test_search_error_handling(self):
        """Test search error handling."""
        client = PDBClient()
        
        # Test with missing search_client
        if not hasattr(client, 'search_client'):
            with pytest.raises(AttributeError):
                client.search({})

    @patch.object(PDBClient, 'get')
    def test_get_entry_error_handling(self, mock_get):
        """Test error handling for get_entry."""
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        client = PDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_entry("invalid")

    @patch.object(PDBClient, 'post')
    def test_get_ligands_error_handling(self, mock_post):
        """Test error handling for get_ligands."""
        mock_response = Mock()
        mock_response.json.return_value = {"errors": ["Entry not found"]}
        mock_post.return_value = mock_response
        
        client = PDBClient()
        result = client.get_ligands("invalid")
        
        assert result == []

    @patch.object(PDBClient, 'post')
    def test_get_polymer_entities_error_handling(self, mock_post):
        """Test error handling for get_polymer_entities."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"polymer_entities": None}}
        mock_post.return_value = mock_response
        
        client = PDBClient()
        result = client.get_polymer_entities("invalid")
        
        assert result == []

    @responses.activate
    def test_real_http_request_entry(self):
        """Test actual HTTP request for entry data."""
        # Mock the HTTP response
        responses.add(
            responses.GET,
            "https://data.rcsb.org/rest/v1/core/entry/1ABC",
            json={
                "rcsb_id": "1ABC",
                "struct": {
                    "title": "Test structure"
                }
            },
            status=200
        )
        
        client = PDBClient()
        result = client.get_entry("1abc")
        
        assert result["rcsb_id"] == "1ABC"
        assert len(responses.calls) == 1
        
        # Check the request
        request = responses.calls[0].request
        assert request.headers["Accept"] == "application/json"

    @responses.activate
    def test_real_http_request_structure(self):
        """Test actual HTTP request for structure download."""
        responses.add(
            responses.GET,
            "https://data.rcsb.org/download/1ABC.pdb",
            body="HEADER    TEST STRUCTURE\nATOM      1  N   VAL A   1\nEND",
            status=200
        )
        
        client = PDBClient()
        result = client.get_structure("1abc")
        
        assert "HEADER    TEST STRUCTURE" in result
        assert len(responses.calls) == 1

    @responses.activate
    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        responses.add(
            responses.GET,
            "https://data.rcsb.org/rest/v1/core/entry/INVALID",
            status=404
        )
        
        client = PDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_entry("invalid")

    def test_case_insensitive_pdb_id(self):
        """Test that PDB IDs are converted to uppercase."""
        with patch.object(PDBClient, 'get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"rcsb_id": "1ABC"}
            mock_get.return_value = mock_response
            
            client = PDBClient()
            client.get_entry("1abc")
            
            # Verify the PDB ID was converted to uppercase
            mock_get.assert_called_once_with("/rest/v1/core/entry/1ABC")