"""Tests for GEO collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.collectors.geo_collector import GEOCollector
from omicverse.external.datacollect.models.genomic import Expression


class TestGEOCollector:
    """Test cases for GEO collector."""
    
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
        """Create a GEO collector instance."""
        collector = GEOCollector(db_session=mock_db_session)
        return collector
    
    @pytest.fixture
    def series_data(self):
        """Sample series data."""
        return {
            "accession": "GSE12345",
            "type": "series",
            "title": "Test Series",
            "platform": "GPL570", 
            "summary": "Test summary",
            "overall_design": "Test design",
            "sample_count": 2,
            "samples": ["GSM1", "GSM2"],
            "supplementary_files": [],
            "raw_data": {}
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample sample data."""
        return {
            "accession": "GSM12345",
            "type": "sample",
            "title": "Test Sample",
            "source": "HeLa cells",
            "organism": "Homo sapiens",
            "platform": "GPL570",
            "series": "GSE12345",
            "characteristics": {
                "cell_type": "epithelial",
                "treatment": "control"
            },
            "raw_data": {}
        }
    
    def test_collect_single_series(self, collector):
        """Test collecting series data."""
        series_data = {
            "series": {"series_summary": "Test"},
            "samples": {}
        }
        expression_data = {
            "title": "Test Series",
            "platform": "GPL570",
            "samples": {"GSM1": {}, "GSM2": {}}
        }
        
        with patch.object(collector.api_client, 'get_series_matrix', return_value=series_data):
            with patch.object(collector.api_client, 'get_expression_data', return_value=expression_data):
                with patch.object(collector.api_client, 'download_supplementary_files', return_value=[]):
                    result = collector.collect_single("GSE12345")
                    
                    assert result["accession"] == "GSE12345"
                    assert result["type"] == "series"
                    assert result["title"] == "Test Series"
                    assert result["sample_count"] == 2
    
    def test_collect_single_sample(self, collector):
        """Test collecting sample data."""
        sample_data = {
            "sample": {
                "sample_title": "Test Sample",
                "sample_source_name_ch1": "HeLa cells",
                "sample_organism_ch1": "Homo sapiens",
                "sample_platform_id": "GPL570",
                "sample_series_id": "GSE12345",
                "sample_characteristics_ch1": ["cell line: HeLa", "treatment: control"]
            }
        }
        
        with patch.object(collector.api_client, 'get_sample_data', return_value=sample_data):
            result = collector.collect_single("GSM12345")
            
            assert result["accession"] == "GSM12345"
            assert result["type"] == "sample"
            assert result["title"] == "Test Sample"
            assert result["organism"] == "Homo sapiens"
            assert len(result["characteristics"]) > 0
    
    def test_collect_single_platform(self, collector):
        """Test collecting platform data."""
        platform_data = {
            "platform": {
                "platform_title": "Test Platform",
                "platform_technology": "microarray",
                "platform_organism": "Homo sapiens",
                "platform_manufacturer": "Affymetrix"
            }
        }
        
        with patch.object(collector.api_client, 'get_platform_data', return_value=platform_data):
            result = collector.collect_single("GPL570")
            
            assert result["accession"] == "GPL570"
            assert result["type"] == "platform"
            assert result["title"] == "Test Platform"
    
    def test_collect_single_dataset(self, collector):
        """Test collecting dataset."""
        dataset_data = {
            "accession": "GDS123",
            "title": "Test Dataset",
            "summary": "Test summary",
            "gpl": "GPL570",
            "gse": ["GSE123"],
            "n_samples": 10
        }
        
        with patch.object(collector.api_client, 'get_dataset_summary', return_value=dataset_data):
            result = collector.collect_single("GDS123")
            
            assert result["accession"] == "GDS123"
            assert result["type"] == "dataset"
            assert result["sample_count"] == 10
    
    def test_save_series_to_database(self, collector, series_data, mock_db_session):
        """Test saving series to database."""
        result = collector.save_series_to_database(series_data)
        
        # Should create 2 expression records
        assert len(result) == 2
        assert mock_db_session.add.call_count == 2
        assert mock_db_session.commit.called
        
        # Check expression records
        for expr in result:
            assert expr.dataset_id == "GSE12345"
            assert expr.platform == "GPL570"
            assert expr.source == "GEO"
    
    def test_save_sample_to_database(self, collector, sample_data, mock_db_session):
        """Test saving sample to database."""
        result = collector.save_sample_to_database(sample_data)
        
        assert isinstance(result, Expression)
        assert result.sample_id == "GSM12345"
        assert result.dataset_id == "GSE12345"
        assert result.tissue == "HeLa cells"
        assert result.cell_type == "epithelial"
        assert mock_db_session.commit.called
    
    def test_search_by_gene(self, collector):
        """Test searching by gene."""
        mock_datasets = [
            {"uid": "123", "accession": "GDS123"},
            {"uid": "456", "accession": "GDS456"}
        ]
        
        detailed_data = [
            {"accession": "GDS123", "type": "dataset", "title": "Dataset 1"},
            {"accession": "GDS456", "type": "dataset", "title": "Dataset 2"}
        ]
        
        with patch.object(collector.api_client, 'search_datasets_by_gene', return_value=mock_datasets):
            with patch.object(collector, 'collect_dataset', side_effect=detailed_data):
                results = collector.search_by_gene("TP53", "Homo sapiens", 10)
                
                assert len(results) == 2
                assert results[0]["title"] == "Dataset 1"
                assert results[1]["title"] == "Dataset 2"
    
    def test_collect_expression_values(self, collector):
        """Test collecting expression values."""
        expression_data = {
            "samples": {
                "GSM1": {"title": "Sample 1"},
                "GSM2": {"title": "Sample 2"}
            }
        }
        
        with patch.object(collector.api_client, 'get_expression_data', return_value=expression_data):
            result = collector.collect_expression_values("GSE12345", ["TP53", "BRCA1"])
            
            assert "GSM1" in result
            assert "GSM2" in result
            assert result["GSM1"]["TP53"] == 0.0
            assert result["GSM1"]["BRCA1"] == 0.0
    
    def test_extract_characteristics(self, collector):
        """Test extracting characteristics."""
        sample_info = {
            "sample_characteristics_ch1": [
                "cell line: HeLa",
                "treatment: control",
                "time point: 24h"
            ]
        }
        
        result = collector._extract_characteristics(sample_info)
        
        assert result["cell_line"] == "HeLa"
        assert result["treatment"] == "control"
        assert result["time_point"] == "24h"
    
    def test_collect_batch(self, collector):
        """Test batch collection."""
        with patch.object(collector, 'collect_single') as mock_collect:
            mock_collect.side_effect = [
                {"accession": "GSE1", "type": "series"},
                {"accession": "GSM1", "type": "sample"}
            ]
            
            results = collector.collect_batch(["GSE1", "GSM1"])
            
            assert len(results) == 2
            assert results[0]["accession"] == "GSE1"
            assert results[1]["accession"] == "GSM1"
    
    def test_error_handling(self, collector):
        """Test error handling in collect_single."""
        with pytest.raises(ValueError, match="Unknown GEO accession type"):
            collector.collect_single("INVALID123")