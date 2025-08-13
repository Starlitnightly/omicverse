"""Tests for ClinVar API client and collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.api.clinvar import ClinVarClient
from omicverse.external.datacollect.collectors.clinvar_collector import ClinVarCollector
from omicverse.external.datacollect.models.genomic import Variant, Gene


@pytest.fixture
def mock_clinvar_search_response():
    """Sample ClinVar search response."""
    return {
        "header": {
            "type": "esearch",
            "version": "0.3"
        },
        "esearchresult": {
            "count": "2",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["12345", "67890"],
            "translationset": [],
            "querytranslation": "BRCA1[gene]"
        }
    }


@pytest.fixture
def mock_clinvar_variant_response():
    """Sample ClinVar variant summary response."""
    return {
        "uid": "12345",
        "title": "NM_007294.4(BRCA1):c.5266dup (p.Gln1756fs)",
        "gene_symbol": "BRCA1",
        "gene_id": "672",
        "clinical_significance": {
            "description": "Pathogenic",
            "review_status": "reviewed by expert panel",
            "last_evaluated": "2022-01-15"
        },
        "variation_set": [
            {
                "variation_type": "Duplication",
                "variation_name": "NM_007294.4(BRCA1):c.5266dup",
                "variation_xrefs": [
                    {
                        "db_source": "dbSNP",
                        "db_id": "rs80357906"
                    }
                ],
                "variation_loc": [
                    {
                        "chr": "17",
                        "start": 43047642,
                        "stop": 43047642,
                        "assembly": "GRCh38"
                    }
                ]
            }
        ],
        "molecular_consequence": "frameshift_variant",
        "protein_change": "p.Gln1756fs",
        "trait_set": [
            {
                "trait_name": "Breast-ovarian cancer, familial 1"
            },
            {
                "trait_name": "Hereditary breast and ovarian cancer syndrome"
            }
        ]
    }


class TestClinVarClient:
    """Test ClinVar API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = ClinVarClient()
        assert "eutils.ncbi.nlm.nih.gov" in client.base_url
        assert client.rate_limit == 3
        assert client.database == "clinvar"
    
    def test_get_base_params(self):
        """Test base parameters."""
        client = ClinVarClient()
        params = client._get_base_params()
        assert params["db"] == "clinvar"
        assert params["retmode"] == "json"
    
    @patch.object(ClinVarClient, 'get')
    def test_search(self, mock_get, mock_clinvar_search_response):
        """Test search functionality."""
        mock_response = Mock()
        mock_response.json.return_value = mock_clinvar_search_response
        mock_get.return_value = mock_response
        
        client = ClinVarClient()
        result = client.search("BRCA1[gene]", max_results=20)
        
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "/esearch.fcgi"
        assert call_args[1]["params"]["term"] == "BRCA1[gene]"
        assert call_args[1]["params"]["retmax"] == 20
        
        assert result["esearchresult"]["count"] == "2"
        assert len(result["esearchresult"]["idlist"]) == 2
    
    @patch.object(ClinVarClient, 'get')
    def test_get_variant_by_id(self, mock_get, mock_clinvar_variant_response):
        """Test getting variant by ID."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "12345": mock_clinvar_variant_response
            }
        }
        mock_get.return_value = mock_response
        
        client = ClinVarClient()
        result = client.get_variant_by_id("12345")
        
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "/esummary.fcgi"
        assert call_args[1]["params"]["id"] == "12345"
        
        assert result["uid"] == "12345"
        assert result["gene_symbol"] == "BRCA1"
    
    @patch.object(ClinVarClient, 'search')
    @patch.object(ClinVarClient, 'get_variant_by_id')
    def test_get_variants_by_gene(self, mock_get_variant, mock_search, 
                                  mock_clinvar_search_response, mock_clinvar_variant_response):
        """Test getting variants by gene."""
        mock_search.return_value = mock_clinvar_search_response
        mock_get_variant.side_effect = [
            mock_clinvar_variant_response,
            {"uid": "67890", "gene_symbol": "BRCA1"}
        ]
        
        client = ClinVarClient()
        result = client.get_variants_by_gene("BRCA1", max_results=100)
        
        mock_search.assert_called_once_with("BRCA1[gene]", 100)
        assert mock_get_variant.call_count == 2
        assert len(result) == 2
        assert result[0]["gene_symbol"] == "BRCA1"
    
    @patch.object(ClinVarClient, 'search')
    def test_get_pathogenic_variants(self, mock_search):
        """Test getting pathogenic variants."""
        mock_search.return_value = {"esearchresult": {"idlist": []}}
        
        client = ClinVarClient()
        with patch.object(client, '_search_and_fetch') as mock_fetch:
            mock_fetch.return_value = []
            result = client.get_pathogenic_variants("BRCA1")
            
            mock_fetch.assert_called_once_with(
                "BRCA1[gene] AND pathogenic[clinsig]", 50
            )
    
    def test_parse_variant_summary(self, mock_clinvar_variant_response):
        """Test parsing variant summary."""
        client = ClinVarClient()
        result = client.parse_variant_summary(mock_clinvar_variant_response)
        
        assert result["clinvar_id"] == "12345"
        assert result["gene_symbol"] == "BRCA1"
        assert result["clinical_significance"] == "Pathogenic"
        assert result["review_status"] == "reviewed by expert panel"
        assert result["variation_type"] == "Duplication"
        assert result["protein_change"] == "p.Gln1756fs"
        assert result["dbsnp_id"] == "rs80357906"
        assert len(result["conditions"]) == 2
        assert result["genomic_location"]["chromosome"] == "17"
        assert result["genomic_location"]["start"] == 43047642
    
    @patch.object(ClinVarClient, 'get')
    def test_fetch_batch(self, mock_get):
        """Test batch fetching."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "12345": {"uid": "12345"},
                "67890": {"uid": "67890"}
            }
        }
        mock_get.return_value = mock_response
        
        client = ClinVarClient()
        result = client._fetch_batch(["12345", "67890"])
        
        assert len(result) == 2
        assert result[0]["uid"] == "12345"
        assert result[1]["uid"] == "67890"


class TestClinVarCollector:
    """Test ClinVar collector."""
    
    @patch('src.collectors.clinvar_collector.ClinVarClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = ClinVarCollector()
        mock_client_class.assert_called_once()
    
    def test_collect_single(self, mock_clinvar_variant_response):
        """Test collecting single variant."""
        collector = ClinVarCollector()
        
        with patch.object(collector.api_client, 'get_variant_by_id') as mock_get:
            with patch.object(collector.api_client, 'parse_variant_summary') as mock_parse:
                mock_get.return_value = mock_clinvar_variant_response
                mock_parse.return_value = {
                    "clinvar_id": "12345",
                    "gene_symbol": "BRCA1",
                    "clinical_significance": "Pathogenic"
                }
                
                result = collector.collect_single("12345")
                
                assert result["clinvar_id"] == "12345"
                assert result["gene_symbol"] == "BRCA1"
                assert result["raw_data"] == mock_clinvar_variant_response
    
    def test_collect_by_gene(self):
        """Test collecting variants by gene."""
        collector = ClinVarCollector()
        
        with patch.object(collector.api_client, 'get_pathogenic_variants') as mock_path:
            with patch.object(collector.api_client, 'get_variants_by_gene') as mock_gene:
                with patch.object(collector.api_client, 'parse_variant_summary') as mock_parse:
                    mock_variants = [
                        {"uid": "12345", "gene_symbol": "BRCA1"},
                        {"uid": "67890", "gene_symbol": "BRCA1"}
                    ]
                    mock_path.return_value = mock_variants
                    mock_gene.return_value = mock_variants
                    # Create a function that returns the parsed data
                    def parse_side_effect(variant):
                        return {"clinvar_id": variant["uid"], "gene_symbol": variant["gene_symbol"]}
                    
                    mock_parse.side_effect = parse_side_effect
                    
                    # Test pathogenic only
                    result = collector.collect_by_gene("BRCA1", pathogenic_only=True)
                    assert len(result) == 2
                    mock_path.assert_called_once_with("BRCA1", 100)
                    
                    # Test all variants
                    result = collector.collect_by_gene("BRCA1", pathogenic_only=False)
                    assert len(result) == 2
                    mock_gene.assert_called_once_with("BRCA1", 100)
    
    def test_save_to_database(self, test_db):
        """Test saving variant to database."""
        collector = ClinVarCollector(db_session=test_db)
        
        data = {
            "clinvar_id": "12345",
            "gene_symbol": "BRCA1",
            "variation_type": "Duplication",
            "protein_change": "p.Gln1756fs",
            "clinical_significance": "Pathogenic",
            "dbsnp_id": "rs80357906",
            "genomic_location": {
                "chromosome": "17",
                "start": 43047642
            },
            "conditions": ["Breast cancer", "Ovarian cancer"],
            "review_status": "reviewed by expert panel",
            "molecular_consequence": "frameshift_variant",
            "variant_name": "c.5266dup"
        }
        
        variant = collector.save_to_database(data)
        
        assert isinstance(variant, Variant)
        assert variant.variant_id == "clinvar:12345"
        assert variant.rsid == "rs80357906"
        assert variant.gene_symbol == "BRCA1"
        assert variant.variant_type == "Duplication"
        assert variant.protein_change == "p.Gln1756fs"
        assert variant.clinical_significance == "Pathogenic"
        assert variant.chromosome == "17"
        assert variant.position == 43047642
        assert "Breast cancer" in variant.disease_associations
        
        # Check annotations JSON
        annotations = json.loads(variant.annotations)
        assert annotations["clinvar_id"] == "12345"
        assert annotations["review_status"] == "reviewed by expert panel"
    
    @patch.object(ClinVarCollector, 'collect_by_gene')
    @patch.object(ClinVarCollector, 'save_variants_to_database')
    def test_update_gene_variants(self, mock_save_variants, mock_collect, test_db):
        """Test updating gene variants."""
        collector = ClinVarCollector(db_session=test_db)
        
        # Create test gene
        gene = Gene(
            id="test_gene",
            gene_id="BRCA1",
            symbol="BRCA1",
            source="test"
        )
        test_db.add(gene)
        test_db.commit()
        
        mock_collect.return_value = [
            {"clinvar_id": "12345"},
            {"clinvar_id": "67890"}
        ]
        mock_save_variants.return_value = [Mock(), Mock()]
        
        count = collector.update_gene_variants("BRCA1", pathogenic_only=True)
        
        assert count == 2
        mock_collect.assert_called_once_with("BRCA1", True)
        mock_save_variants.assert_called_once()
    
    @patch.object(ClinVarCollector, 'collect_single')
    @patch.object(ClinVarCollector, 'save_to_database')
    def test_search_variants(self, mock_save, mock_collect):
        """Test searching variants."""
        collector = ClinVarCollector()
        
        with patch.object(collector.api_client, 'search') as mock_search:
            mock_search.return_value = {
                "esearchresult": {
                    "idlist": ["12345", "67890"]
                }
            }
            
            mock_collect.side_effect = [
                {"clinvar_id": "12345"},
                {"clinvar_id": "67890"}
            ]
            
            mock_save.side_effect = [Mock(), Mock()]
            
            results = collector.search_variants("BRCA1", max_results=50)
            
            assert len(results) == 2
            mock_search.assert_called_once_with("BRCA1", 50)
            assert mock_collect.call_count == 2
            assert mock_save.call_count == 2