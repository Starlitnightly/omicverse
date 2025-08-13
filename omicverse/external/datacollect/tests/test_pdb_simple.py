"""Tests for simplified PDB API client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from omicverse.external.datacollect.api.pdb_simple import SimplePDBClient


class TestSimplePDBClient:
    """Test simplified PDB API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = SimplePDBClient()
        assert client.base_url == "https://files.rcsb.org"
        assert client.data_api_base == "https://data.rcsb.org"
        assert client.rate_limit == 10
        
        # Test custom initialization
        client_custom = SimplePDBClient(rate_limit=5)
        assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = SimplePDBClient()
        headers = client.get_default_headers()
        
        assert headers["User-Agent"] == "BioinformaticsDataCollector/0.1.0"
        assert headers["Accept"] == "application/json,text/plain"

    @patch('requests.Session.get')
    def test_get_entry(self, mock_get):
        """Test getting PDB entry data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "rcsb_id": "1ABC",
            "struct": {
                "title": "Crystal structure of human insulin",
                "pdbx_descriptor": "Insulin A chain, Insulin B chain"
            },
            "pdbx_database_status": {
                "status_code": "REL",
                "deposit_date": "1999-07-15",
                "revision_date": "2020-01-15"
            },
            "exptl": [
                {
                    "method": "X-RAY DIFFRACTION"
                }
            ],
            "refine": [
                {
                    "ls_d_res_high": 1.8,
                    "ls_R_factor_R_work": 0.195,
                    "ls_R_factor_R_free": 0.240
                }
            ],
            "rcsb_entry_info": {
                "molecular_weight": 5730.0,
                "polymer_entity_count": 2,
                "nonpolymer_entity_count": 1,
                "deposited_polymer_monomer_count": 51
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = SimplePDBClient()
        result = client.get_entry("1abc")
        
        # Verify the correct URL was called
        expected_url = "https://data.rcsb.org/rest/v1/core/entry/1ABC"
        mock_get.assert_called_once_with(
            expected_url,
            headers=client.get_default_headers(),
            timeout=client.timeout
        )
        
        assert result["rcsb_id"] == "1ABC"
        assert result["struct"]["title"] == "Crystal structure of human insulin"
        assert result["exptl"][0]["method"] == "X-RAY DIFFRACTION"
        assert result["refine"][0]["ls_d_res_high"] == 1.8

    @patch.object(SimplePDBClient, 'get')
    def test_get_structure(self, mock_get):
        """Test downloading structure file."""
        mock_response = Mock()
        mock_response.text = """HEADER    HORMONE                                 15-JUL-99   1ABC              
TITLE     CRYSTAL STRUCTURE OF HUMAN INSULIN                               
REMARK   2 RESOLUTION.    1.80 ANGSTROMS.                                   
ATOM      1  N   GLY A   1      -3.524   2.628  -1.686  1.00 18.00           N  
ATOM      2  CA  GLY A   1      -2.469   2.201  -2.600  1.00 18.00           C  
END                                                                             """
        mock_get.return_value = mock_response
        
        client = SimplePDBClient()
        result = client.get_structure("1abc", format="pdb")
        
        mock_get.assert_called_once_with("/download/1ABC.pdb")
        assert "HEADER    HORMONE" in result
        assert "CRYSTAL STRUCTURE OF HUMAN INSULIN" in result
        assert "ATOM      1  N   GLY A   1" in result

    @patch.object(SimplePDBClient, 'get')
    def test_get_structure_cif_format(self, mock_get):
        """Test downloading structure file in CIF format."""
        mock_response = Mock()
        mock_response.text = """data_1ABC
#
_entry.id   1ABC
#
_struct.title   'Crystal structure of human insulin'
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . GLY A 1 1 ? -3.524 2.628 -1.686 1.00 18.00 ? 1 GLY A N 1"""
        mock_get.return_value = mock_response
        
        client = SimplePDBClient()
        result = client.get_structure("1abc", format="cif")
        
        mock_get.assert_called_once_with("/download/1ABC.cif")
        assert "data_1ABC" in result
        assert "_struct.title   'Crystal structure of human insulin'" in result

    @patch('requests.Session.get')
    def test_get_polymer_info_success(self, mock_get):
        """Test getting polymer information successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rcsb_id": "1ABC_1",
            "entity_poly": {
                "entity_id": "1",
                "type": "polypeptide(L)",
                "pdbx_seq_one_letter_code": "GIVEQCCTSICSLYQLENYCN",
                "pdbx_strand_id": "A",
                "pdbx_seq_one_letter_code_can": "GIVEQCCTSICSLYQLENYCN"
            },
            "rcsb_polymer_entity": {
                "pdbx_description": "Insulin A chain",
                "formula_weight": 2382.32,
                "polymer_type": "Protein"
            },
            "rcsb_entity_source_organism": [
                {
                    "ncbi_scientific_name": "Homo sapiens",
                    "ncbi_taxonomy_id": 9606
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = SimplePDBClient()
        result = client.get_polymer_info("1abc")
        
        expected_url = "https://data.rcsb.org/rest/v1/core/polymer_entity/1ABC/1"
        mock_get.assert_called_once_with(
            expected_url,
            headers=client.get_default_headers(),
            timeout=client.timeout
        )
        
        assert result["rcsb_id"] == "1ABC_1"
        assert result["entity_poly"]["entity_id"] == "1"
        assert result["rcsb_polymer_entity"]["pdbx_description"] == "Insulin A chain"

    @patch('requests.Session.get')
    def test_get_polymer_info_not_found(self, mock_get):
        """Test getting polymer information when not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        client = SimplePDBClient()
        result = client.get_polymer_info("invalid")
        
        assert result == {}

    @patch('requests.Session.get')
    def test_get_polymer_info_exception(self, mock_get):
        """Test getting polymer information with exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        client = SimplePDBClient()
        result = client.get_polymer_info("1abc")
        
        assert result == {}

    @patch('requests.Session.get')
    def test_get_entry_error_handling(self, mock_get):
        """Test error handling for get_entry."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        client = SimplePDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_entry("invalid")

    @patch.object(SimplePDBClient, 'get')
    def test_get_structure_error_handling(self, mock_get):
        """Test error handling for get_structure."""
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        client = SimplePDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_structure("invalid")

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
                    "title": "Test structure",
                    "pdbx_descriptor": "Test protein"
                },
                "exptl": [
                    {
                        "method": "X-RAY DIFFRACTION"
                    }
                ]
            },
            status=200
        )
        
        client = SimplePDBClient()
        result = client.get_entry("1abc")
        
        assert result["rcsb_id"] == "1ABC"
        assert result["struct"]["title"] == "Test structure"
        assert len(responses.calls) == 1
        
        # Check the request
        request = responses.calls[0].request
        assert request.headers["Accept"] == "application/json,text/plain"

    @responses.activate
    def test_real_http_request_structure(self):
        """Test actual HTTP request for structure download."""
        responses.add(
            responses.GET,
            "https://files.rcsb.org/download/1ABC.pdb",
            body="HEADER    TEST STRUCTURE\nATOM      1  N   VAL A   1\nEND",
            status=200
        )
        
        client = SimplePDBClient()
        result = client.get_structure("1abc")
        
        assert "HEADER    TEST STRUCTURE" in result
        assert len(responses.calls) == 1

    @responses.activate
    def test_real_http_request_polymer_info(self):
        """Test actual HTTP request for polymer info."""
        responses.add(
            responses.GET,
            "https://data.rcsb.org/rest/v1/core/polymer_entity/1ABC/1",
            json={
                "rcsb_id": "1ABC_1",
                "entity_poly": {
                    "entity_id": "1",
                    "type": "polypeptide(L)",
                    "pdbx_seq_one_letter_code": "GIVEQCC"
                }
            },
            status=200
        )
        
        client = SimplePDBClient()
        result = client.get_polymer_info("1abc")
        
        assert result["rcsb_id"] == "1ABC_1"
        assert len(responses.calls) == 1

    @responses.activate
    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        responses.add(
            responses.GET,
            "https://data.rcsb.org/rest/v1/core/entry/INVALID",
            status=404
        )
        
        client = SimplePDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_entry("invalid")

    @responses.activate
    def test_http_structure_not_found(self):
        """Test handling of structure file not found."""
        responses.add(
            responses.GET,
            "https://files.rcsb.org/download/INVALID.pdb",
            status=404
        )
        
        client = SimplePDBClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_structure("invalid")

    def test_case_insensitive_pdb_id(self):
        """Test that PDB IDs are converted to uppercase."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"rcsb_id": "1ABC"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            client = SimplePDBClient()
            client.get_entry("1abc")
            
            # Verify the PDB ID was converted to uppercase in the URL
            expected_url = "https://data.rcsb.org/rest/v1/core/entry/1ABC"
            mock_get.assert_called_once_with(
                expected_url,
                headers=client.get_default_headers(),
                timeout=client.timeout
            )

    def test_mixed_case_pdb_id_structure(self):
        """Test that PDB IDs are converted to uppercase for structure requests."""
        with patch.object(SimplePDBClient, 'get') as mock_get:
            mock_response = Mock()
            mock_response.text = "HEADER    TEST"
            mock_get.return_value = mock_response
            
            client = SimplePDBClient()
            client.get_structure("1aBc")
            
            # Verify the PDB ID was converted to uppercase
            mock_get.assert_called_once_with("/download/1ABC.pdb")

    def test_polymer_info_url_construction(self):
        """Test URL construction for polymer info requests."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            mock_get.return_value = mock_response
            
            client = SimplePDBClient()
            client.get_polymer_info("2xyz")
            
            # Verify the correct URL was constructed
            expected_url = "https://data.rcsb.org/rest/v1/core/polymer_entity/2XYZ/1"
            mock_get.assert_called_once_with(
                expected_url,
                headers=client.get_default_headers(),
                timeout=client.timeout
            )

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        client = SimplePDBClient()
        assert hasattr(client, 'timeout')
        
        # Test that timeout is used in requests
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            client.get_entry("1abc")
            
            # Verify timeout was passed to the request
            call_kwargs = mock_get.call_args[1]
            assert 'timeout' in call_kwargs