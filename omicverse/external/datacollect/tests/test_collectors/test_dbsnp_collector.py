"""Tests for dbSNP collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.collectors.dbsnp_collector import dbSNPCollector
from omicverse.external.datacollect.models.genomic import Variant, Gene


class TestdbSNPCollector:
    """Test cases for dbSNP collector."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.add = Mock()
        session.commit = Mock()
        return session
    
    @pytest.fixture
    def collector(self, mock_db_session):
        """Create a dbSNP collector instance."""
        collector = dbSNPCollector(db_session=mock_db_session)
        return collector
    
    @pytest.fixture
    def variant_data(self):
        """Sample variant data."""
        return {
            "rsid": "rs7412",
            "variant_type": "SNP",
            "chromosome": "19",
            "position": 44908684,
            "reference_allele": "C",
            "alternate_alleles": ["T"],
            "gene_symbols": ["APOE"],
            "consequences": [{
                "gene_name": "APOE",
                "consequence_type": "missense_variant",
                "hgvs": "NP_000032.1:p.Arg176Cys"
            }],
            "clinical_significance": [{
                "gene": "APOE",
                "clinical_significance": "risk factor",
                "review_status": "reviewed by expert panel"
            }],
            "global_frequency": 0.0816,
            "population_frequencies": {
                "European": {"1000Genomes": {"frequency": 0.1122}},
                "African": {"1000Genomes": {"frequency": 0.2105}}
            },
            "allele_annotations": [],
            "raw_data": {}
        }
    
    def test_collect_single(self, collector):
        """Test collecting single variant."""
        # Mock API responses
        variant_response = {
            "refsnp_id": "7412",
            "primary_snapshot_data": {}
        }
        
        with patch.object(collector.api_client, 'get_variant_by_rsid', return_value=variant_response):
            with patch.object(collector.api_client, 'get_variant_allele_annotations', return_value=[]):
                with patch.object(collector.api_client, 'get_population_frequency', return_value={"global_frequency": 0.08}):
                    with patch.object(collector.api_client, 'get_clinical_significance', return_value=[]):
                        with patch.object(collector.api_client, 'get_variant_consequences', return_value=[]):
                            with patch.object(collector, '_extract_variant_type', return_value="SNP"):
                                with patch.object(collector, '_extract_chromosome', return_value="19"):
                                    with patch.object(collector, '_extract_position', return_value=44908684):
                                        with patch.object(collector, '_extract_reference_allele', return_value="C"):
                                            with patch.object(collector, '_extract_alternate_alleles', return_value=["T"]):
                                                with patch.object(collector, '_extract_gene_symbols', return_value=["APOE"]):
                                                    result = collector.collect_single("rs7412")
                    
                                                    assert result["rsid"] == "rs7412"
                                                    assert result["variant_type"] == "SNP"
                                                    assert result["chromosome"] == "19"
                                                    assert result["position"] == 44908684
    
    def test_save_to_database(self, collector, variant_data, mock_db_session):
        """Test saving variant to database."""
        result = collector.save_to_database(variant_data)
        
        assert isinstance(result, Variant)
        assert result.rsid == "rs7412"
        assert result.variant_type == "SNP"
        assert result.chromosome == "19"
        assert result.position == 44908684
        assert result.reference_allele == "C"
        assert result.alternative_allele == "T"
        assert result.gene_symbol == "APOE"
        assert result.minor_allele_frequency == 0.0816
        assert mock_db_session.commit.called
    
    def test_save_to_database_update_existing(self, collector, variant_data, mock_db_session):
        """Test updating existing variant."""
        existing_variant = Variant(
            id="dbsnp_rs7412",
            rsid="rs7412",
            source="dbSNP"
        )
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = existing_variant
        
        result = collector.save_to_database(variant_data)
        
        assert result == existing_variant
        assert result.variant_type == "SNP"
        mock_db_session.add.assert_not_called()  # Should not add existing variant
        assert mock_db_session.commit.called
    
    def test_collect_by_gene(self, collector):
        """Test collecting variants by gene."""
        rs_ids = ["rs7412", "rs429358"]
        variant_data1 = {"rsid": "rs7412", "variant_type": "SNP"}
        variant_data2 = {"rsid": "rs429358", "variant_type": "SNP"}
        
        with patch.object(collector.api_client, 'search_by_gene', return_value=rs_ids):
            with patch.object(collector, 'collect_single', side_effect=[variant_data1, variant_data2]):
                results = collector.collect_by_gene("APOE", "human", 50)
                
                assert len(results) == 2
                assert results[0]["rsid"] == "rs7412"
                assert results[1]["rsid"] == "rs429358"
    
    def test_collect_by_position(self, collector):
        """Test collecting variants by position."""
        rs_ids = ["rs7412", "rs429358", "rs123"]  # More than limit
        variant_data = {"rsid": "rs7412", "variant_type": "SNP"}
        
        with patch.object(collector.api_client, 'search_by_position', return_value=rs_ids):
            with patch.object(collector, 'collect_single', return_value=variant_data):
                results = collector.collect_by_position("19", 44905791, 44909393, "human")
                
                # Should be limited to 50
                assert len(results) <= 50
    
    def test_save_gene_variants(self, collector, mock_db_session):
        """Test saving all variants for a gene."""
        # Mock gene in database
        gene = Gene(id="gene_apoe", symbol="APOE")
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = gene
        
        variants_data = [
            {"rsid": "rs7412", "variant_type": "SNP"},
            {"rsid": "rs429358", "variant_type": "SNP"}
        ]
        
        with patch.object(collector, 'collect_by_gene', return_value=variants_data):
            with patch.object(collector, 'save_to_database') as mock_save:
                mock_save.side_effect = [
                    Variant(rsid="rs7412"),
                    Variant(rsid="rs429358")
                ]
                
                results = collector.save_gene_variants("APOE", "human", 50)
                
                assert len(results) == 2
                assert mock_save.call_count == 2
    
    def test_extract_variant_type(self, collector):
        """Test extracting variant type."""
        variant_data = {
            "primary_snapshot_data": {
                "variant_type": "snv"
            }
        }
        
        result = collector._extract_variant_type(variant_data)
        assert result == "SNP"
        
        # Test type mapping
        variant_data["primary_snapshot_data"]["variant_type"] = "ins"
        result = collector._extract_variant_type(variant_data)
        assert result == "insertion"
    
    def test_extract_chromosome(self, collector):
        """Test extracting chromosome."""
        variant_data = {
            "primary_snapshot_data": {
                "placements_with_allele": [{
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [{
                            "is_chromosome": True,
                            "sequence_name": "NC_000019.10",
                            "traits": [{"trait_name": "Chr19"}]
                        }]
                    }
                }]
            }
        }
        
        result = collector._extract_chromosome(variant_data)
        assert result == "19"
    
    def test_extract_position(self, collector):
        """Test extracting position."""
        variant_data = {
            "primary_snapshot_data": {
                "placements_with_allele": [{
                    "alleles": [{
                        "location": {"position": 44908684}
                    }]
                }]
            }
        }
        
        result = collector._extract_position(variant_data)
        assert result == 44908684
    
    def test_extract_alleles(self, collector):
        """Test extracting reference and alternate alleles."""
        variant_data = {
            "primary_snapshot_data": {
                "placements_with_allele": [{
                    "alleles": [{
                        "allele": {
                            "spdi": {
                                "deleted_sequence": "C",
                                "inserted_sequence": "T"
                            }
                        }
                    }]
                }]
            }
        }
        
        ref = collector._extract_reference_allele(variant_data)
        assert ref == "C"
        
        alts = collector._extract_alternate_alleles(variant_data)
        assert alts == ["T"]
    
    def test_extract_gene_symbols(self, collector):
        """Test extracting gene symbols."""
        variant_data = {
            "primary_snapshot_data": {
                "placements_with_allele": [{
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [{
                            "genes": [
                                {"name": "APOE"},
                                {"name": "APOC1"}
                            ]
                        }]
                    }
                }]
            }
        }
        
        result = collector._extract_gene_symbols(variant_data)
        assert len(result) == 2
        assert "APOE" in result
        assert "APOC1" in result
    
    def test_collect_batch(self, collector):
        """Test batch collection."""
        with patch.object(collector, 'collect_single') as mock_collect:
            mock_collect.side_effect = [
                {"rsid": "rs7412"},
                {"rsid": "rs429358"}
            ]
            
            results = collector.collect_batch(["rs7412", "rs429358"])
            
            assert len(results) == 2
            assert results[0]["rsid"] == "rs7412"
            assert results[1]["rsid"] == "rs429358"