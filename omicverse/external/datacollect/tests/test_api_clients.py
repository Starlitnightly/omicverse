"""Tests for API client classes."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict

from omicverse.external.datacollect.api.base import BaseAPIClient, RateLimiter
from omicverse.external.datacollect.api.uniprot import UniProtClient
from omicverse.external.datacollect.api.pdb import PDBClient


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_sync(self):
        """Test synchronous rate limiting."""
        rate_limiter = RateLimiter(calls_per_second=10)  # 10 calls/sec = 0.1 sec between calls
        
        start_time = time.time()
        rate_limiter.acquire_sync()
        rate_limiter.acquire_sync()
        elapsed = time.time() - start_time
        
        # Should take at least 0.1 seconds between calls
        assert elapsed >= 0.09  # Allow small margin for timing
    
    @pytest.mark.asyncio
    async def test_rate_limiter_async(self):
        """Test asynchronous rate limiting."""
        rate_limiter = RateLimiter(calls_per_second=10)
        
        start_time = time.time()
        await rate_limiter.acquire()
        await rate_limiter.acquire()
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.09


# Create a concrete test implementation of BaseAPIClient
class TestAPIClient(BaseAPIClient):
    """Concrete implementation for testing."""
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers."""
        return {
            "User-Agent": "TestClient/1.0",
            "Accept": "application/json",
        }


class TestBaseAPIClient:
    """Test base API client functionality."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = TestAPIClient(
            base_url="https://api.example.com",
            rate_limit=5,
            timeout=60,
            max_retries=5
        )
        
        assert client.base_url == "https://api.example.com"
        assert client.rate_limit == 5
        assert client.timeout == 60
        assert client.max_retries == 5
    
    def test_build_url(self):
        """Test URL building."""
        client = TestAPIClient(base_url="https://api.example.com")
        
        # Relative endpoint
        assert client.build_url("/endpoint") == "https://api.example.com/endpoint"
        assert client.build_url("endpoint") == "https://api.example.com/endpoint"
        
        # Absolute URL
        assert client.build_url("https://other.com/endpoint") == "https://other.com/endpoint"
    
    def test_default_headers(self):
        """Test default headers."""
        client = TestAPIClient(base_url="https://api.example.com")
        headers = client.get_default_headers()
        
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert headers["Accept"] == "application/json"
    
    @patch('requests.Session.request')
    def test_request_success(self, mock_request):
        """Test successful request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_request.return_value = mock_response
        
        client = TestAPIClient(base_url="https://api.example.com")
        response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"result": "success"}
    
    @patch('requests.Session.request')
    def test_request_with_params(self, mock_request):
        """Test request with parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        client = TestAPIClient(base_url="https://api.example.com")
        client.get("/test", params={"key": "value"})
        
        # Check that params were passed
        call_args = mock_request.call_args
        assert call_args[1]["params"] == {"key": "value"}


class TestUniProtClient:
    """Test UniProt API client."""
    
    def test_initialization(self):
        """Test UniProt client initialization."""
        client = UniProtClient()
        assert "uniprot.org" in client.base_url
        assert client.rate_limit == 10  # UniProt default
    
    @patch.object(UniProtClient, 'get')
    def test_get_entry(self, mock_get, mock_api_response):
        """Test getting a single entry."""
        mock_data = {
            "primaryAccession": "P12345",
            "uniProtkbId": "TEST_HUMAN",
            "organism": {"scientificName": "Homo sapiens", "taxonId": 9606}
        }
        mock_get.return_value = mock_api_response(mock_data)
        
        client = UniProtClient()
        result = client.get_entry("P12345")
        
        assert result["primaryAccession"] == "P12345"
        mock_get.assert_called_once_with("/uniprotkb/P12345")
    
    @patch.object(UniProtClient, 'get')
    def test_search(self, mock_get, mock_api_response):
        """Test search functionality."""
        mock_data = {
            "results": [
                {"primaryAccession": "P12345"},
                {"primaryAccession": "P67890"}
            ]
        }
        mock_get.return_value = mock_api_response(mock_data)
        
        client = UniProtClient()
        result = client.search("organism:human", size=10)
        
        assert len(result["results"]) == 2
        call_args = mock_get.call_args
        assert call_args[0][0] == "/uniprotkb/search"
        assert call_args[1]["params"]["query"] == "organism:human"
        assert call_args[1]["params"]["size"] == 10
    
    @patch.object(UniProtClient, 'get')
    def test_batch_retrieve(self, mock_get, mock_api_response):
        """Test batch retrieval."""
        mock_data = {"results": []}
        mock_get.return_value = mock_api_response(mock_data)
        
        client = UniProtClient()
        client.batch_retrieve(["P12345", "P67890"])
        
        call_args = mock_get.call_args
        assert call_args[0][0] == "/uniprotkb/accessions"
        assert call_args[1]["params"]["from"] == "P12345,P67890"


class TestPDBClient:
    """Test PDB API client."""
    
    def test_initialization(self):
        """Test PDB client initialization."""
        client = PDBClient()
        assert "rcsb.org" in client.base_url
        assert hasattr(client, 'search_url')
    
    @patch.object(PDBClient, 'get')
    def test_get_entry(self, mock_get, mock_api_response):
        """Test getting PDB entry."""
        mock_data = {
            "struct": {"title": "Test structure"},
            "rcsb_accession_info": {
                "deposit_date": "2023-01-01",
                "initial_release_date": "2023-01-15"
            }
        }
        mock_get.return_value = mock_api_response(mock_data)
        
        client = PDBClient()
        result = client.get_entry("1abc")
        
        assert result["struct"]["title"] == "Test structure"
        mock_get.assert_called_once_with("/rest/v1/core/entry/1ABC")
    
    @patch.object(PDBClient, 'get')
    def test_get_structure(self, mock_get, mock_api_response):
        """Test structure file download."""
        mock_pdb_content = "HEADER    TEST STRUCTURE"
        mock_get.return_value = mock_api_response(mock_pdb_content)
        
        client = PDBClient()
        result = client.get_structure("1abc", format="pdb")
        
        assert result == str(mock_pdb_content)
        mock_get.assert_called_once_with("/download/1ABC.pdb")
    
    def test_text_search(self):
        """Test text search query building."""
        client = PDBClient()
        
        with patch.object(client, 'search') as mock_search:
            client.text_search("hemoglobin")
            
            call_args = mock_search.call_args
            query = call_args[0][0]
            assert query["type"] == "terminal"
            assert query["service"] == "text"
            assert query["parameters"]["value"] == "hemoglobin"