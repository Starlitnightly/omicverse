"""Tests for WoRMS API client."""

import pytest
from unittest.mock import Mock, patch
from datetime import date

from omicverse.external.datacollect.api.worms import WoRMSClient


@pytest.fixture
def mock_aphia_record():
    """Sample WoRMS AphiaRecord."""
    return {
        "AphiaID": 127160,
        "url": "https://www.marinespecies.org/aphia.php?p=taxdetails&id=127160",
        "scientificname": "Carcharodon carcharias",
        "authority": "(Linnaeus, 1758)",
        "status": "accepted",
        "unacceptreason": None,
        "taxonRankID": 220,
        "rank": "Species",
        "valid_AphiaID": 127160,
        "valid_name": "Carcharodon carcharias",
        "valid_authority": "(Linnaeus, 1758)",
        "parentNameUsageID": 105838,
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Elasmobranchii",
        "order": "Lamniformes",
        "family": "Lamnidae",
        "genus": "Carcharodon",
        "citation": "Froese, R. and D. Pauly. Editors. (2023). FishBase.",
        "lsid": "urn:lsid:marinespecies.org:taxname:127160",
        "isMarine": 1,
        "isBrackish": 0,
        "isFreshwater": 0,
        "isTerrestrial": 0,
        "isExtinct": 0,
        "match_type": "exact",
        "modified": "2008-01-15T18:27:08.177Z"
    }


@pytest.fixture
def mock_classification():
    """Sample WoRMS classification."""
    return {
        "AphiaID": 127160,
        "rank": "Species",
        "scientificname": "Carcharodon carcharias",
        "child": {
            "AphiaID": 105838,
            "rank": "Genus",
            "scientificname": "Carcharodon",
            "child": {
                "AphiaID": 105841,
                "rank": "Family",
                "scientificname": "Lamnidae",
                "child": {
                    "AphiaID": 10209,
                    "rank": "Order",
                    "scientificname": "Lamniformes"
                }
            }
        }
    }


@pytest.fixture
def mock_children_response():
    """Sample WoRMS children response."""
    return [
        {
            "AphiaID": 367477,
            "scientificname": "Carcharodon hubbelli",
            "authority": "Ehret, Macfadden, Jones, Devries, Foster & Salas-Gismondi, 2012",
            "status": "accepted",
            "rank": "Species"
        },
        {
            "AphiaID": 159902,
            "scientificname": "Carcharodon megalodon",
            "authority": "(Agassiz, 1843)",
            "status": "synonym",
            "rank": "Species"
        }
    ]


@pytest.fixture
def mock_synonyms_response():
    """Sample WoRMS synonyms response."""
    return [
        {
            "AphiaID": 217663,
            "scientificname": "Carcharias carcharias",
            "authority": "(Linnaeus, 1758)",
            "status": "synonym",
            "valid_AphiaID": 127160,
            "valid_name": "Carcharodon carcharias"
        },
        {
            "AphiaID": 246949,
            "scientificname": "Squalus carcharias",
            "authority": "Linnaeus, 1758",
            "status": "synonym",
            "valid_AphiaID": 127160,
            "valid_name": "Carcharodon carcharias"
        }
    ]


@pytest.fixture
def mock_vernaculars_response():
    """Sample WoRMS vernacular names response."""
    return [
        {
            "vernacular": "Great white shark",
            "language_code": "eng",
            "language": "English"
        },
        {
            "vernacular": "Grand requin blanc",
            "language_code": "fra",
            "language": "French"
        },
        {
            "vernacular": "Tiburón blanco",
            "language_code": "spa",
            "language": "Spanish"
        }
    ]


@pytest.fixture
def mock_distribution_response():
    """Sample WoRMS distribution response."""
    return [
        {
            "locationID": 26567,
            "locality": "Mediterranean Sea",
            "higherGeography": "Mediterranean Sea",
            "status": "present",
            "qualitystatus": "checked"
        },
        {
            "locationID": 5688,
            "locality": "South Africa",
            "higherGeography": "Indian Ocean",
            "status": "present",
            "qualitystatus": "checked"
        }
    ]


@pytest.fixture
def mock_attributes_response():
    """Sample WoRMS attributes response."""
    return [
        {
            "measurementTypeID": 3,
            "measurementType": "maximum length",
            "measurementValue": "6",
            "measurementUnit": "m",
            "source_id": 142063,
            "reference": "Froese, R. and D. Pauly"
        },
        {
            "measurementTypeID": 5,
            "measurementType": "habitat",
            "measurementValue": "pelagic-oceanic",
            "source_id": 142063
        }
    ]


class TestWoRMSClient:
    """Test WoRMS API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = WoRMSClient()
        assert "marinespecies.org/rest" in client.base_url
        assert client.rate_limit == 10
    
    @patch.object(WoRMSClient, 'get')
    def test_get_aphia_record_by_id(self, mock_get, mock_aphia_record):
        """Test getting AphiaRecord by ID."""
        mock_get.return_value = mock_aphia_record
        
        client = WoRMSClient()
        result = client.get_aphia_record_by_id(127160)
        
        mock_get.assert_called_once_with("/AphiaRecordByAphiaID/127160")
        assert result == mock_aphia_record
        assert result["scientificname"] == "Carcharodon carcharias"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_aphia_records_by_name(self, mock_get):
        """Test getting AphiaRecords by name."""
        mock_response = Mock()
        mock_response.json.return_value = [mock_aphia_record]
        mock_response.text = "[{}]"
        mock_get.return_value = mock_response
        
        client = WoRMSClient()
        result = client.get_aphia_records_by_name(
            "Carcharodon carcharias",
            fuzzy=True,
            marine_only=True
        )
        
        mock_get.assert_called_once_with(
            "/AphiaRecordsByName/Carcharodon carcharias",
            params={
                "like": "true",
                "fuzzy": "true",
                "marine_only": "true",
                "offset": 0
            }
        )
        assert len(result) == 1
    
    @patch.object(WoRMSClient, 'get')
    def test_get_aphia_records_by_name_empty(self, mock_get):
        """Test getting AphiaRecords with no results."""
        mock_get.return_value = []
        
        client = WoRMSClient()
        result = client.get_aphia_records_by_name("NonexistentSpecies")
        
        assert result == []
    
    @patch.object(WoRMSClient, 'get')
    def test_get_aphia_record_by_external_id(self, mock_get, mock_aphia_record):
        """Test getting AphiaRecord by external ID."""
        mock_get.return_value = mock_aphia_record
        
        client = WoRMSClient()
        result = client.get_aphia_record_by_external_id("123456", "tsn")
        
        mock_get.assert_called_once_with(
            "/AphiaRecordByExternalID/123456",
            params={"type": "tsn"}
        )
        assert result == mock_aphia_record
    
    @patch.object(WoRMSClient, 'get')
    def test_get_classification(self, mock_get, mock_classification):
        """Test getting classification."""
        mock_get.return_value = mock_classification
        
        client = WoRMSClient()
        result = client.get_classification(127160)
        
        mock_get.assert_called_once_with("/AphiaClassificationByAphiaID/127160")
        assert result == mock_classification
        assert result["scientificname"] == "Carcharodon carcharias"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_children(self, mock_get, mock_children_response):
        """Test getting children taxa."""
        mock_get.return_value = mock_children_response
        
        client = WoRMSClient()
        result = client.get_children(105838, marine_only=True)
        
        mock_get.assert_called_once_with(
            "/AphiaChildrenByAphiaID/105838",
            params={"marine_only": "true", "offset": 0}
        )
        assert len(result) == 2
        assert result[0]["scientificname"] == "Carcharodon hubbelli"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_synonyms(self, mock_get, mock_synonyms_response):
        """Test getting synonyms."""
        mock_get.return_value = mock_synonyms_response
        
        client = WoRMSClient()
        result = client.get_synonyms(127160)
        
        mock_get.assert_called_once_with("/AphiaSynonymsByAphiaID/127160")
        assert len(result) == 2
        assert result[0]["scientificname"] == "Carcharias carcharias"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_vernaculars(self, mock_get, mock_vernaculars_response):
        """Test getting vernacular names."""
        mock_get.return_value = mock_vernaculars_response
        
        client = WoRMSClient()
        result = client.get_vernaculars(127160)
        
        mock_get.assert_called_once_with("/AphiaVernacularsByAphiaID/127160")
        assert len(result) == 3
        assert result[0]["vernacular"] == "Great white shark"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_distribution(self, mock_get, mock_distribution_response):
        """Test getting distribution."""
        mock_get.return_value = mock_distribution_response
        
        client = WoRMSClient()
        result = client.get_distribution(127160)
        
        mock_get.assert_called_once_with("/AphiaDistributionsByAphiaID/127160")
        assert len(result) == 2
        assert result[0]["locality"] == "Mediterranean Sea"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_attributes(self, mock_get, mock_attributes_response):
        """Test getting attributes."""
        mock_get.return_value = mock_attributes_response
        
        client = WoRMSClient()
        result = client.get_attributes(127160, include_inherited=True)
        
        mock_get.assert_called_once_with(
            "/AphiaAttributesByAphiaID/127160",
            params={"include_inherited": "true"}
        )
        assert len(result) == 2
        assert result[0]["measurementType"] == "maximum length"
    
    @patch.object(WoRMSClient, 'get')
    def test_get_sources(self, mock_get):
        """Test getting sources."""
        mock_sources = [
            {
                "source_id": 142063,
                "reference": "Froese, R. and D. Pauly. Editors.",
                "year": 2023
            }
        ]
        mock_get.return_value = mock_sources
        
        client = WoRMSClient()
        result = client.get_sources(127160)
        
        mock_get.assert_called_once_with("/AphiaSourcesByAphiaID/127160")
        assert len(result) == 1
    
    @patch.object(WoRMSClient, 'post')
    def test_search_taxa(self, mock_post):
        """Test searching taxa."""
        mock_post.return_value = [[{"AphiaID": 127160}]]
        
        client = WoRMSClient()
        result = client.search_taxa("Carcharodon", marine_only=True)
        
        mock_post.assert_called_once_with(
            "/AphiaRecordsByMatchNames",
            json={
                "scientificnames[]": ["Carcharodon"],
                "marine_only": True,
                "fossil_only": False
            }
        )
        assert result == [{"AphiaID": 127160}]
    
    @patch.object(WoRMSClient, 'post')
    def test_batch_get_records_by_names(self, mock_post):
        """Test batch getting records by names."""
        mock_post.return_value = [
            [{"AphiaID": 127160, "scientificname": "Carcharodon carcharias"}],
            [{"AphiaID": 105696, "scientificname": "Isurus oxyrinchus"}]
        ]
        
        client = WoRMSClient()
        result = client.batch_get_records_by_names(
            ["Carcharodon carcharias", "Isurus oxyrinchus"]
        )
        
        mock_post.assert_called_once_with(
            "/AphiaRecordsByMatchNames",
            json={
                "scientificnames[]": ["Carcharodon carcharias", "Isurus oxyrinchus"],
                "marine_only": True
            }
        )
        assert len(result) == 2
    
    @patch.object(WoRMSClient, 'post')
    def test_batch_get_records_by_ids(self, mock_post):
        """Test batch getting records by IDs."""
        mock_post.return_value = [
            {"AphiaID": 127160, "scientificname": "Carcharodon carcharias"},
            {"AphiaID": 105696, "scientificname": "Isurus oxyrinchus"}
        ]
        
        client = WoRMSClient()
        result = client.batch_get_records_by_ids([127160, 105696])
        
        mock_post.assert_called_once_with(
            "/AphiaRecordsByAphiaIDs",
            json={"aphiaids[]": [127160, 105696]}
        )
        assert len(result) == 2
    
    @patch.object(WoRMSClient, 'get')
    def test_get_record_by_date(self, mock_get):
        """Test getting records by date."""
        mock_get.return_value = [{"AphiaID": 127160}]
        
        client = WoRMSClient()
        result = client.get_record_by_date(
            "2023-01-01",
            "2023-12-31",
            marine_only=True
        )
        
        mock_get.assert_called_once_with(
            "/AphiaRecordsByDate",
            params={
                "startdate": "2023-01-01",
                "enddate": "2023-12-31",
                "marine_only": "true",
                "offset": 0
            }
        )
        assert len(result) == 1
    
    @patch.object(WoRMSClient, 'get_aphia_record_by_id')
    @patch.object(WoRMSClient, 'get_children')
    def test_get_taxon_tree(self, mock_children, mock_record):
        """Test getting taxonomic tree."""
        mock_record.return_value = {"AphiaID": 105838, "scientificname": "Carcharodon"}
        mock_children.return_value = [
            {"AphiaID": 127160, "scientificname": "Carcharodon carcharias"}
        ]
        
        client = WoRMSClient()
        result = client.get_taxon_tree(105838, max_depth=2)
        
        assert result["record"]["AphiaID"] == 105838
        assert "children" in result
        mock_record.assert_called()
        mock_children.assert_called()
    
    @patch.object(WoRMSClient, 'get')
    def test_get_external_identifiers(self, mock_get):
        """Test getting external identifiers."""
        mock_get.return_value = [
            {"database": "fishbase", "identifier": "751"},
            {"database": "itis", "identifier": "159903"}
        ]
        
        client = WoRMSClient()
        result = client.get_external_identifiers(127160)
        
        mock_get.assert_called_once_with("/AphiaExternalIDByAphiaID/127160")
        assert len(result) == 2
    
    @patch.object(WoRMSClient, 'get')
    def test_get_images(self, mock_get):
        """Test getting images."""
        mock_get.return_value = [
            {
                "imageID": 12345,
                "imageURL": "https://example.com/image.jpg",
                "thumbnailURL": "https://example.com/thumb.jpg",
                "caption": "Great white shark"
            }
        ]
        
        client = WoRMSClient()
        result = client.get_images(127160)
        
        mock_get.assert_called_once_with("/AphiaImagesByAphiaID/127160")
        assert len(result) == 1
        assert result[0]["imageURL"] == "https://example.com/image.jpg"