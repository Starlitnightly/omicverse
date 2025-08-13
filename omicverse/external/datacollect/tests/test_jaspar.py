"""Tests for JASPAR API client."""

import pytest
from unittest.mock import Mock, patch

from omicverse.external.datacollect.api.jaspar import JASPARClient


@pytest.fixture
def mock_matrix_response():
    """Sample JASPAR matrix response."""
    return {
        "matrix_id": "MA0001.1",
        "name": "AGL3",
        "collection": "CORE",
        "tf_class": "MADS box factors",
        "tf_family": "MADS SRF-like",
        "species": [{"tax_id": "3702", "species": "Arabidopsis thaliana"}],
        "uniprot_ids": ["Q9C7C0"],
        "data_type": "SELEX",
        "medline": "21146531",
        "pfm": {
            "A": [0, 9, 1, 16, 0, 20, 0, 0],
            "C": [19, 2, 18, 0, 20, 0, 0, 0],
            "G": [1, 7, 0, 3, 0, 0, 20, 0],
            "T": [0, 2, 1, 1, 0, 0, 0, 20]
        },
        "url": "http://jaspar.genereg.net/matrix/MA0001.1"
    }


@pytest.fixture
def mock_search_response():
    """Sample JASPAR search response."""
    return {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [
            {
                "matrix_id": "MA0001.1",
                "name": "AGL3",
                "collection": "CORE",
                "tf_class": "MADS box factors",
                "tf_family": "MADS SRF-like"
            },
            {
                "matrix_id": "MA0002.2",
                "name": "RUNX1",
                "collection": "CORE",
                "tf_class": "Runt domain factors",
                "tf_family": "Runt-related"
            }
        ]
    }


@pytest.fixture
def mock_pfm_response():
    """Sample JASPAR PFM response."""
    return {
        "A": [0, 9, 1, 16, 0, 20, 0, 0],
        "C": [19, 2, 18, 0, 20, 0, 0, 0],
        "G": [1, 7, 0, 3, 0, 0, 20, 0],
        "T": [0, 2, 1, 1, 0, 0, 0, 20]
    }


@pytest.fixture
def mock_pwm_response():
    """Sample JASPAR PWM response."""
    return {
        "A": [-2.0, 0.5, -1.5, 1.2, -2.0, 1.5, -2.0, -2.0],
        "C": [1.3, -1.0, 1.2, -2.0, 1.5, -2.0, -2.0, -2.0],
        "G": [-1.5, 0.3, -2.0, -0.5, -2.0, -2.0, 1.5, -2.0],
        "T": [-2.0, -1.0, -1.5, -1.5, -2.0, -2.0, -2.0, 1.5]
    }


@pytest.fixture
def mock_collections_response():
    """Sample JASPAR collections response."""
    return [
        {
            "name": "CORE",
            "description": "Core vertebrate collection"
        },
        {
            "name": "CNE",
            "description": "Conserved non-coding elements"
        },
        {
            "name": "PHYLOFACTS",
            "description": "Phylogenetically derived matrices"
        }
    ]


@pytest.fixture
def mock_species_response():
    """Sample JASPAR species response."""
    return [
        {
            "tax_id": "9606",
            "species": "Homo sapiens"
        },
        {
            "tax_id": "10090",
            "species": "Mus musculus"
        },
        {
            "tax_id": "3702",
            "species": "Arabidopsis thaliana"
        }
    ]


@pytest.fixture
def mock_infer_response():
    """Sample JASPAR inference response."""
    return [
        {
            "matrix_id": "MA0001.1",
            "start": 10,
            "end": 18,
            "strand": "+",
            "score": 8.5,
            "relative_score": 0.85,
            "sequence": "CCATATATGG"
        },
        {
            "matrix_id": "MA0001.1",
            "start": 45,
            "end": 53,
            "strand": "-",
            "score": 7.2,
            "relative_score": 0.72,
            "sequence": "CCATATAAGG"
        }
    ]


class TestJASPARClient:
    """Test JASPAR API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = JASPARClient()
        assert "jaspar.elixir.no/api/v1" in client.base_url
        assert client.rate_limit == 10
    
    @patch.object(JASPARClient, 'get')
    def test_get_matrix(self, mock_get, mock_matrix_response):
        """Test getting a single matrix."""
        mock_response = Mock()
        mock_response.json.return_value = mock_matrix_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_matrix("MA0001.1")
        
        mock_get.assert_called_once_with("/matrix/MA0001.1/")
        assert result == mock_matrix_response
        assert result["matrix_id"] == "MA0001.1"
        assert result["name"] == "AGL3"
    
    @patch.object(JASPARClient, 'get')
    def test_search_matrices(self, mock_get, mock_search_response):
        """Test searching matrices."""
        mock_response = Mock()
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.search_matrices(
            name="AGL",
            collection="CORE",
            tf_class="MADS box factors",
            page=1,
            page_size=10
        )
        
        mock_get.assert_called_once_with(
            "/matrix/",
            params={
                "page": 1,
                "page_size": 10,
                "name": "AGL",
                "collection": "CORE",
                "tf_class": "MADS box factors"
            }
        )
        assert result == mock_search_response
        assert result["count"] == 2
    
    @patch.object(JASPARClient, 'get')
    def test_search_matrices_with_tax_id(self, mock_get, mock_search_response):
        """Test searching matrices with taxonomy ID."""
        mock_response = Mock()
        mock_response.json.return_value = mock_search_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.search_matrices(
            tax_id=9606,
            data_type="ChIP-seq"
        )
        
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["tax_id"] == 9606
        assert call_params["data_type"] == "ChIP-seq"
    
    @patch.object(JASPARClient, 'get')
    def test_get_matrix_pfm(self, mock_get, mock_pfm_response):
        """Test getting PFM for a matrix."""
        mock_response = Mock()
        mock_response.json.return_value = mock_pfm_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_matrix_pfm("MA0001.1")
        
        mock_get.assert_called_once_with("/matrix/MA0001.1/pfm/")
        assert result == mock_pfm_response
        assert len(result["A"]) == 8
    
    @patch.object(JASPARClient, 'get')
    def test_get_matrix_pwm(self, mock_get, mock_pwm_response):
        """Test getting PWM for a matrix."""
        mock_response = Mock()
        mock_response.json.return_value = mock_pwm_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_matrix_pwm("MA0001.1")
        
        mock_get.assert_called_once_with("/matrix/MA0001.1/pwm/")
        assert result == mock_pwm_response
        assert len(result["A"]) == 8
    
    @patch.object(JASPARClient, 'get')
    def test_get_matrix_jaspar_format(self, mock_get):
        """Test getting matrix in JASPAR format."""
        mock_response = Mock()
        mock_response.text = ">MA0001.1 AGL3\nA [ 0 9 1 16 ]\nC [ 19 2 18 0 ]\nG [ 1 7 0 3 ]\nT [ 0 2 1 1 ]"
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_matrix_jaspar_format("MA0001.1")
        
        mock_get.assert_called_once_with(
            "/matrix/MA0001.1/",
            headers={"Accept": "text/plain"}
        )
        assert ">MA0001.1" in result
    
    @patch.object(JASPARClient, 'get')
    def test_get_matrix_meme_format(self, mock_get):
        """Test getting matrix in MEME format."""
        mock_response = Mock()
        mock_response.text = "MEME version 4\n\nALPHABET= ACGT\n\nMOTIF MA0001.1 AGL3"
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_matrix_meme_format("MA0001.1")
        
        mock_get.assert_called_once_with("/matrix/MA0001.1/meme/")
        assert "MEME" in result
    
    @patch.object(JASPARClient, 'get')
    def test_get_collections(self, mock_get, mock_collections_response):
        """Test getting collections."""
        mock_response = Mock()
        mock_response.json.return_value = mock_collections_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_collections()
        
        mock_get.assert_called_once_with("/collections/")
        assert result == mock_collections_response
        assert len(result) == 3
    
    @patch.object(JASPARClient, 'get')
    def test_get_releases(self, mock_get):
        """Test getting releases."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"release": "2022", "date": "2022-01-01"},
            {"release": "2020", "date": "2020-01-01"}
        ]
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_releases()
        
        mock_get.assert_called_once_with("/releases/")
        assert len(result) == 2
    
    @patch.object(JASPARClient, 'get')
    def test_get_species(self, mock_get, mock_species_response):
        """Test getting species."""
        mock_response = Mock()
        mock_response.json.return_value = mock_species_response
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_species()
        
        mock_get.assert_called_once_with("/species/")
        assert result == mock_species_response
        assert len(result) == 3
    
    @patch.object(JASPARClient, 'get')
    def test_get_taxa(self, mock_get):
        """Test getting taxonomic groups."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"tax_group": "vertebrates"},
            {"tax_group": "plants"},
            {"tax_group": "insects"}
        ]
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_taxa()
        
        mock_get.assert_called_once_with("/taxa/")
        assert len(result) == 3
    
    @patch.object(JASPARClient, 'post')
    def test_infer_matrix(self, mock_post, mock_infer_response):
        """Test inferring binding sites."""
        mock_response = Mock()
        mock_response.json.return_value = mock_infer_response
        mock_post.return_value = mock_response
        
        client = JASPARClient()
        result = client.infer_matrix(
            sequence="ATGCCATATATGGCGATCGATATCG",
            matrix_id="MA0001.1",
            collection="CORE"
        )
        
        mock_post.assert_called_once_with(
            "/infer/",
            json={
                "sequence": "ATGCCATATATGGCGATCGATATCG",
                "collection": "CORE",
                "matrix_id": "MA0001.1"
            }
        )
        assert result == mock_infer_response
        assert len(result) == 2
        assert result[0]["score"] == 8.5
    
    @patch.object(JASPARClient, 'post')
    def test_batch_download(self, mock_post):
        """Test batch download of matrices."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"matrix_id": "MA0001.1", "name": "AGL3"},
            {"matrix_id": "MA0002.2", "name": "RUNX1"}
        ]
        mock_post.return_value = mock_response
        
        client = JASPARClient()
        result = client.batch_download(["MA0001.1", "MA0002.2"], format="json")
        
        mock_post.assert_called_once_with(
            "/batch/",
            json={
                "matrix_ids": ["MA0001.1", "MA0002.2"],
                "format": "json"
            }
        )
        assert len(result) == 2
    
    @patch.object(JASPARClient, 'get')
    def test_get_tf_flexible_models(self, mock_get):
        """Test getting TF flexible models."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"tffm_id": "TFFM0001", "name": "Model1"}
        ]
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.get_tf_flexible_models()
        
        mock_get.assert_called_once_with("/tffm/")
        assert len(result) == 1
    
    @patch.object(JASPARClient, 'post')
    def test_search_by_sequence_similarity(self, mock_post):
        """Test searching by sequence similarity."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"matrix_id": "MA0001.1", "similarity": 0.95},
            {"matrix_id": "MA0002.2", "similarity": 0.82}
        ]
        mock_post.return_value = mock_response
        
        client = JASPARClient()
        result = client.search_by_sequence_similarity(
            sequence="CCATATATGG",
            threshold=0.8
        )
        
        mock_post.assert_called_once_with(
            "/similarity/sequence/",
            json={
                "sequence": "CCATATATGG",
                "threshold": 0.8
            }
        )
        assert len(result) == 2
        assert result[0]["similarity"] == 0.95
    
    @patch.object(JASPARClient, 'get')
    def test_search_by_matrix_similarity(self, mock_get):
        """Test searching by matrix similarity."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"matrix_id": "MA0002.2", "similarity": 0.92},
            {"matrix_id": "MA0003.3", "similarity": 0.85}
        ]
        mock_get.return_value = mock_response
        
        client = JASPARClient()
        result = client.search_by_matrix_similarity("MA0001.1", threshold=0.8)
        
        mock_get.assert_called_once_with(
            "/matrix/MA0001.1/similar/",
            params={"threshold": 0.8}
        )
        assert len(result) == 2
        assert result[0]["similarity"] == 0.92