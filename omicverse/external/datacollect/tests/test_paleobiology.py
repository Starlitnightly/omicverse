"""Tests for Paleobiology Database API client."""

import pytest
from unittest.mock import Mock, patch

from omicverse.external.datacollect.api.paleobiology import PaleobiologyClient


@pytest.fixture
def mock_occurrences_response():
    """Sample Paleobiology occurrences response."""
    return {
        "records": [
            {
                "oid": "occ:123456",
                "cid": "col:789",
                "tid": "txn:456",
                "tna": "Tyrannosaurus rex",
                "rnk": 3,
                "odl": "species",
                "idn": "Tyrannosaurus rex",
                "eag": 67.0,
                "lag": 66.0,
                "cc2": "US",
                "sta": "Montana",
                "lat": 45.123,
                "lng": -108.456,
                "prc": 1,
                "env": "terrestrial",
                "lth": "sandstone"
            }
        ],
        "records_found": 1,
        "records_returned": 1,
        "record_offset": 0
    }


@pytest.fixture
def mock_taxa_response():
    """Sample Paleobiology taxa response."""
    return {
        "records": [
            {
                "oid": "txn:456",
                "nam": "Tyrannosaurus",
                "rnk": 5,
                "noc": 10,
                "nco": 1500,
                "fea": 83.6,
                "fla": 66.0,
                "lea": 72.1,
                "lla": 66.0,
                "phl": "Chordata",
                "cll": "Reptilia",
                "odl": "Saurischia",
                "fml": "Tyrannosauridae",
                "gnl": "Tyrannosaurus",
                "ext": 1
            }
        ],
        "records_found": 1,
        "records_returned": 1,
        "record_offset": 0
    }


@pytest.fixture
def mock_collections_response():
    """Sample Paleobiology collections response."""
    return {
        "records": [
            {
                "oid": "col:789",
                "nam": "Hell Creek Formation",
                "cxi": 1,
                "n_occs": 150,
                "cc2": "US",
                "sta": "Montana",
                "lat": 45.123,
                "lng": -108.456,
                "gpl": 45.1,
                "gng": -108.5,
                "eag": 67.0,
                "lag": 66.0,
                "gsc": "Maastrichtian",
                "env": "terrestrial",
                "lth": "sandstone,mudstone"
            }
        ],
        "records_found": 1,
        "records_returned": 1,
        "record_offset": 0
    }


@pytest.fixture
def mock_intervals_response():
    """Sample Paleobiology intervals response."""
    return {
        "records": [
            {
                "oid": "int:123",
                "nam": "Cretaceous",
                "lvl": 2,
                "typ": "period",
                "eag": 145.0,
                "lag": 66.0,
                "col": "#7FC64E"
            },
            {
                "oid": "int:124",
                "nam": "Maastrichtian",
                "lvl": 5,
                "typ": "stage",
                "eag": 72.1,
                "lag": 66.0,
                "col": "#F0E68C"
            }
        ],
        "records_found": 2,
        "records_returned": 2
    }


@pytest.fixture
def mock_strata_response():
    """Sample Paleobiology strata response."""
    return {
        "records": [
            {
                "oid": "str:456",
                "nam": "Hell Creek Formation",
                "rnk": "formation",
                "cc2": "US",
                "sta": "Montana,North Dakota,South Dakota,Wyoming",
                "n_colls": 250
            }
        ],
        "records_found": 1,
        "records_returned": 1
    }


@pytest.fixture
def mock_diversity_response():
    """Sample Paleobiology diversity response."""
    return {
        "records": [
            {
                "interval_name": "Maastrichtian",
                "interval_mid_ma": 69.05,
                "n_occs": 1500,
                "n_taxa": 250
            },
            {
                "interval_name": "Campanian",
                "interval_mid_ma": 76.5,
                "n_occs": 1200,
                "n_taxa": 200
            }
        ],
        "records_found": 2,
        "records_returned": 2
    }


class TestPaleobiologyClient:
    """Test Paleobiology Database API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = PaleobiologyClient()
        assert "paleobiodb.org/data1.2" in client.base_url
        assert client.rate_limit == 5
    
    @patch.object(PaleobiologyClient, 'get')
    def test_search_occurrences(self, mock_get, mock_occurrences_response):
        """Test searching for occurrences."""
        mock_get.return_value = mock_occurrences_response
        
        client = PaleobiologyClient()
        result = client.search_occurrences(
            taxon_name="Tyrannosaurus",
            country="US",
            limit=100
        )
        
        mock_get.assert_called_once_with(
            "/occs/list.json",
            params={
                "show": "coords,age,strat,lith",
                "limit": 100,
                "offset": 0,
                "base_name": "Tyrannosaurus",
                "cc": "US"
            }
        )
        assert result == mock_occurrences_response
        assert result["records_found"] == 1
    
    @patch.object(PaleobiologyClient, 'get')
    def test_search_occurrences_with_age_range(self, mock_get, mock_occurrences_response):
        """Test searching occurrences with age range."""
        mock_get.return_value = mock_occurrences_response
        
        client = PaleobiologyClient()
        result = client.search_occurrences(
            min_age=66.0,
            max_age=72.1,
            country="US"
        )
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["min_ma"] == 66.0
        assert call_params["max_ma"] == 72.1
        assert call_params["cc"] == "US"
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_occurrence(self, mock_get):
        """Test getting single occurrence."""
        mock_get.return_value = {"records": [{"oid": "occ:123456"}]}
        
        client = PaleobiologyClient()
        result = client.get_occurrence("123456")
        
        mock_get.assert_called_once_with(
            "/occs/single.json",
            params={"id": "123456", "show": "full"}
        )
        assert result["oid"] == "occ:123456"
    
    @patch.object(PaleobiologyClient, 'get')
    def test_search_taxa(self, mock_get, mock_taxa_response):
        """Test searching for taxa."""
        mock_get.return_value = mock_taxa_response
        
        client = PaleobiologyClient()
        result = client.search_taxa(
            name="Tyrannosaurus",
            rank="genus",
            extinct=True
        )
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["name"] == "Tyrannosaurus"
        assert call_params["rank"] == "genus"
        assert call_params["extant"] == "no"
        assert result == mock_taxa_response
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_taxon(self, mock_get):
        """Test getting single taxon."""
        mock_get.return_value = {"records": [{"oid": "txn:456"}]}
        
        client = PaleobiologyClient()
        result = client.get_taxon("456")
        
        mock_get.assert_called_once_with(
            "/taxa/single.json",
            params={"id": "456", "show": "full"}
        )
        assert result["oid"] == "txn:456"
    
    @patch.object(PaleobiologyClient, 'get')
    def test_search_collections(self, mock_get, mock_collections_response):
        """Test searching for collections."""
        mock_get.return_value = mock_collections_response
        
        client = PaleobiologyClient()
        result = client.search_collections(
            taxon_name="Hell Creek",
            interval="Maastrichtian",
            country="US"
        )
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["base_name"] == "Hell Creek"
        assert call_params["interval"] == "Maastrichtian"
        assert call_params["cc"] == "US"
        assert result == mock_collections_response
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_collection(self, mock_get):
        """Test getting single collection."""
        mock_get.return_value = {"records": [{"oid": "col:789"}]}
        
        client = PaleobiologyClient()
        result = client.get_collection("789")
        
        mock_get.assert_called_once_with(
            "/colls/single.json",
            params={"id": "789", "show": "full"}
        )
        assert result["oid"] == "col:789"
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_intervals(self, mock_get, mock_intervals_response):
        """Test getting geological intervals."""
        mock_get.return_value = mock_intervals_response
        
        client = PaleobiologyClient()
        result = client.get_intervals(scale="international")
        
        mock_get.assert_called_once_with(
            "/intervals/list.json",
            params={"scale": "international"}
        )
        assert result == mock_intervals_response
        assert len(result["records"]) == 2
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_intervals_with_age_range(self, mock_get, mock_intervals_response):
        """Test getting intervals with age range."""
        mock_get.return_value = mock_intervals_response
        
        client = PaleobiologyClient()
        result = client.get_intervals(min_age=66.0, max_age=145.0)
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["min_ma"] == 66.0
        assert call_params["max_ma"] == 145.0
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_strata(self, mock_get, mock_strata_response):
        """Test getting stratigraphic units."""
        mock_get.return_value = {"records": [{"oid": "str:456"}]}
        
        client = PaleobiologyClient()
        result = client.get_strata(
            name="Hell Creek",
            rank="formation"
        )
        
        mock_get.assert_called_once_with(
            "/strata/list.json",
            params={
                "name": "Hell Creek",
                "rank": "formation"
            }
        )
        assert result == [{"oid": "str:456"}]
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_references(self, mock_get):
        """Test getting references."""
        mock_response_data = {
            "records": [
                {
                    "oid": "ref:123",
                    "author": "Smith, J.",
                    "year": 2020,
                    "title": "New dinosaur species"
                }
            ]
        }
        mock_get.return_value = mock_response_data
        
        client = PaleobiologyClient()
        result = client.get_references(author="Smith", year=2020)
        
        mock_get.assert_called_once_with(
            "/refs/list.json",
            params={
                "author": "Smith",
                "year": 2020
            }
        )
        assert len(result) == 1
    
    @patch.object(PaleobiologyClient, 'get')
    def test_get_diversity(self, mock_get, mock_diversity_response):
        """Test getting diversity data."""
        mock_get.return_value = mock_diversity_response
        
        client = PaleobiologyClient()
        result = client.get_diversity(
            taxon_name="Dinosauria",
            time_rule="stage",
            interval="Mesozoic"
        )
        
        mock_get.assert_called_once_with(
            "/occs/diversity.json",
            params={
                "base_name": "Dinosauria",
                "count": "species",
                "time_rule": "stage",
                "interval": "Mesozoic"
            }
        )
        assert result == mock_diversity_response["records"]
        assert len(result) == 2
    
    @patch.object(PaleobiologyClient, 'get')
    def test_search_with_custom_show_fields(self, mock_get):
        """Test search with default show fields."""
        mock_get.return_value = {"records": []}
        
        client = PaleobiologyClient()
        result = client.search_occurrences(
            taxon_name="Mammalia"
        )
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["show"] == "coords,age,strat,lith"
    
    @patch.object(PaleobiologyClient, 'get')
    def test_pagination(self, mock_get):
        """Test pagination parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"records": []}
        mock_get.return_value = mock_response
        
        client = PaleobiologyClient()
        result = client.search_taxa(limit=50, offset=100)
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["limit"] == 50
        assert call_params["offset"] == 100