"""Tests for IUCN Red List API client."""

import pytest
from unittest.mock import Mock, patch

from omicverse.external.datacollect.api.iucn import IUCNClient


@pytest.fixture
def mock_species_response():
    """Sample IUCN species response."""
    return {
        "name": "Panthera leo",
        "result": [
            {
                "taxonid": 15951,
                "scientific_name": "Panthera leo",
                "kingdom": "ANIMALIA",
                "phylum": "CHORDATA",
                "class": "MAMMALIA",
                "order": "CARNIVORA",
                "family": "FELIDAE",
                "genus": "Panthera",
                "main_common_name": "Lion",
                "category": "VU",
                "criteria": "A2abcd",
                "population_trend": "Decreasing",
                "marine_system": False,
                "freshwater_system": False,
                "terrestrial_system": True
            }
        ]
    }


@pytest.fixture
def mock_narrative_response():
    """Sample IUCN narrative response."""
    return {
        "name": "Panthera leo",
        "result": [
            {
                "species_id": 15951,
                "taxonomicnotes": "Taxonomy notes...",
                "rationale": "Conservation rationale...",
                "geographicrange": "Geographic range description...",
                "population": "Population information...",
                "populationtrend": "Decreasing",
                "habitat": "Habitat description...",
                "threats": "Major threats...",
                "conservationmeasures": "Conservation measures...",
                "usetrade": "Use and trade information..."
            }
        ]
    }


@pytest.fixture
def mock_threats_response():
    """Sample IUCN threats response."""
    return {
        "name": "Panthera leo",
        "result": [
            {
                "code": "2.1",
                "title": "Annual & perennial non-timber crops",
                "timing": "Ongoing",
                "scope": "Majority (50-90%)",
                "severity": "Slow, Significant Declines",
                "score": "Medium Impact",
                "invasive": None
            },
            {
                "code": "5.1",
                "title": "Hunting & trapping terrestrial animals",
                "timing": "Ongoing",
                "scope": "Majority (50-90%)",
                "severity": "Rapid Declines",
                "score": "High Impact",
                "invasive": None
            }
        ]
    }


@pytest.fixture
def mock_habitats_response():
    """Sample IUCN habitats response."""
    return {
        "name": "Panthera leo",
        "result": [
            {
                "code": "2",
                "habitat": "Savanna",
                "suitability": "Suitable",
                "season": "Resident",
                "majorimportance": "Yes"
            },
            {
                "code": "3",
                "habitat": "Shrubland",
                "suitability": "Suitable",
                "season": "Resident",
                "majorimportance": "No"
            }
        ]
    }


@pytest.fixture
def mock_countries_response():
    """Sample IUCN countries response."""
    return {
        "name": "Panthera leo",
        "result": [
            {
                "code": "KE",
                "country": "Kenya",
                "presence": "Extant",
                "origin": "Native",
                "distribution_code": "Native"
            },
            {
                "code": "TZ",
                "country": "Tanzania",
                "presence": "Extant",
                "origin": "Native",
                "distribution_code": "Native"
            }
        ]
    }


@pytest.fixture
def mock_regions_response():
    """Sample IUCN regions response."""
    return {
        "count": 3,
        "results": [
            {
                "name": "global",
                "identifier": "global"
            },
            {
                "name": "Mediterranean",
                "identifier": "mediterranean"
            },
            {
                "name": "Europe",
                "identifier": "europe"
            }
        ]
    }


class TestIUCNClient:
    """Test IUCN Red List API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = IUCNClient(api_token="test_token")
        assert "apiv3.iucnredlist.org" in client.base_url
        assert client.rate_limit == 2
        assert client.api_token == "test_token"
    
    def test_initialization_without_token(self):
        """Test client initialization without token."""
        with patch('src.api.iucn.logger') as mock_logger:
            client = IUCNClient()
            assert client.api_token is None
            mock_logger.warning.assert_called_once()
    
    @patch.object(IUCNClient, 'get')
    def test_get_species_by_name(self, mock_get, mock_species_response):
        """Test getting species by name."""
        mock_get.return_value = mock_species_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_species_by_name("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/species/Panthera leo",
            params={"token": "test_token"}
        )
        assert result == mock_species_response
    
    @patch.object(IUCNClient, 'get')
    def test_get_species_by_name_with_region(self, mock_get, mock_species_response):
        """Test getting species by name with region."""
        mock_get.return_value = mock_species_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_species_by_name("Panthera leo", region="mediterranean")
        
        mock_get.assert_called_once_with(
            "/species/region/mediterranean/name/Panthera leo",
            params={"token": "test_token"}
        )
        assert result == mock_species_response
    
    @patch.object(IUCNClient, 'get')
    def test_get_species_by_id(self, mock_get, mock_species_response):
        """Test getting species by ID."""
        mock_get.return_value = mock_species_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_species_by_id(15951)
        
        mock_get.assert_called_once_with(
            "/species/id/15951",
            params={"token": "test_token"}
        )
        assert result == mock_species_response
    
    @patch.object(IUCNClient, 'get')
    def test_search_species(self, mock_get, mock_species_response):
        """Test searching species with pagination."""
        mock_get.return_value = mock_species_response
        
        client = IUCNClient(api_token="test_token")
        result = client.search_species(page=0)
        
        mock_get.assert_called_once_with(
            "/species/page/0",
            params={"token": "test_token"}
        )
        assert result == mock_species_response
    
    @patch.object(IUCNClient, 'get')
    def test_get_species_narrative(self, mock_get, mock_narrative_response):
        """Test getting species narrative."""
        mock_get.return_value = mock_narrative_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_species_narrative("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/species/narrative/Panthera leo",
            params={"token": "test_token"}
        )
        assert result == mock_narrative_response
    
    @patch.object(IUCNClient, 'get')
    def test_get_threats_by_species(self, mock_get, mock_threats_response):
        """Test getting threats for species."""
        mock_get.return_value = mock_threats_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_threats_by_species("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/threats/species/name/Panthera%20leo",
            params={"token": "test_token"}
        )
        # get_threats_by_species returns the "result" field, not the full response
        assert result == mock_threats_response["result"]
        assert len(result) == 2
        assert result[1]["score"] == "High Impact"
    
    @patch.object(IUCNClient, 'get')
    def test_get_habitats_by_species(self, mock_get, mock_habitats_response):
        """Test getting habitats for species."""
        mock_get.return_value = mock_habitats_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_habitats_by_species("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/habitats/species/name/Panthera%20leo",
            params={"token": "test_token"}
        )
        # get_habitats_by_species returns the "result" field, not the full response
        assert result == mock_habitats_response["result"]
        assert len(result) == 2
        assert result[0]["habitat"] == "Savanna"
    
    @patch.object(IUCNClient, 'get')
    def test_get_conservation_measures(self, mock_get):
        """Test getting conservation measures."""
        mock_data = {
            "name": "Panthera leo",
            "result": [
                {
                    "code": "1.1",
                    "title": "Site/area protection"
                }
            ]
        }
        mock_get.return_value = mock_data
        
        client = IUCNClient(api_token="test_token")
        result = client.get_conservation_measures("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/measures/species/name/Panthera%20leo",
            params={"token": "test_token"}
        )
        # get_conservation_measures returns the "result" field, not the full response
        assert result == mock_data["result"]
        assert len(result) == 1
    
    @patch.object(IUCNClient, 'get')
    def test_get_countries_by_species(self, mock_get, mock_countries_response):
        """Test getting countries for species."""
        mock_get.return_value = mock_countries_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_countries_by_species("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/species/countries/name/Panthera%20leo",
            params={"token": "test_token"}
        )
        # get_countries_by_species returns the "result" field, not the full response
        assert result == mock_countries_response["result"]
        assert len(result) == 2
        assert result[0]["country"] == "Kenya"
    
    @patch.object(IUCNClient, 'get')
    def test_get_citation(self, mock_get):
        """Test getting citation."""
        mock_data = {
            "citation": "Bauer, H., Packer, C., Funston, P.F., Henschel, P. & Nowell, K. 2016..."
        }
        mock_get.return_value = mock_data
        
        client = IUCNClient(api_token="test_token")
        result = client.get_citation("Panthera leo")
        
        mock_get.assert_called_once_with(
            "/species/citation/Panthera leo",
            params={"token": "test_token"}
        )
        # get_citation returns the citation string, not the full response
        assert result == "Bauer, H., Packer, C., Funston, P.F., Henschel, P. & Nowell, K. 2016..."
        assert isinstance(result, str)
    
    @patch.object(IUCNClient, 'get')
    def test_get_regions(self, mock_get, mock_regions_response):
        """Test getting available regions."""
        mock_get.return_value = mock_regions_response
        
        client = IUCNClient(api_token="test_token")
        result = client.get_regions()
        
        mock_get.assert_called_once_with(
            "/region/list",
            params={"token": "test_token"}
        )
        # get_regions returns the "results" field, not the full response
        assert result == mock_regions_response["results"]
        assert len(result) == 3
        assert result[0]["name"] == "global"
    
    @patch.object(IUCNClient, 'get')
    def test_get_comprehensive_groups(self, mock_get):
        """Test getting comprehensive groups."""
        mock_data = {
            "result": [
                {"scientific_name": "Panthera leo", "category": "VU"},
                {"scientific_name": "Panthera tigris", "category": "EN"}
            ]
        }
        mock_get.return_value = mock_data
        
        client = IUCNClient(api_token="test_token")
        result = client.get_comprehensive_groups("mammals")
        
        mock_get.assert_called_once_with(
            "/comp-group/getspecies/mammals",
            params={"token": "test_token"}
        )
        # get_comprehensive_groups returns the "result" field, not the full response
        assert result == mock_data["result"]
        assert len(result) == 2
        assert result[0]["scientific_name"] == "Panthera leo"
    
    def test_add_token_with_token(self):
        """Test adding token to parameters."""
        client = IUCNClient(api_token="test_token")
        params = client._add_token()
        assert params == {"token": "test_token"}
        
        params = client._add_token({"key": "value"})
        assert params == {"key": "value", "token": "test_token"}
    
    def test_add_token_without_token(self):
        """Test adding token when no token provided."""
        client = IUCNClient()
        params = client._add_token()
        assert params == {}
        
        params = client._add_token({"key": "value"})
        assert params == {"key": "value"}