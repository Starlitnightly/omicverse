"""Tests for RegulomeDB API client."""

import pytest
from unittest.mock import patch, MagicMock

from omicverse.external.datacollect.api.regulomedb import RegulomeDBClient


class TestRegulomeDBClient:
    def setup_method(self):
        self.client = RegulomeDBClient()
        self.client.session = MagicMock()
    
    def test_query_variant(self):
        """Test querying variant by position."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "@graph": [
                {
                    "coordinates": {"chr": "chr1", "start": 12345, "end": 12345},
                    "regulome_score": {"ranking": "2a", "probability": 0.95},
                    "peaks": [{"dataset": "ENCODE", "biosample": "liver"}],
                    "motifs": [{"tf": "CTCF", "score": 0.9}]
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        result = self.client.query_variant("chr1", 12345)
        
        assert "@graph" in result
        assert len(result["@graph"]) == 1
        assert result["@graph"][0]["regulome_score"]["ranking"] == "2a"
    
    def test_query_rsid(self):
        """Test querying by rsID."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "@graph": [
                {
                    "rsid": "rs12345",
                    "regulome_score": {"ranking": "1f"},
                    "peaks": []
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        result = self.client.query_rsid("rs12345")
        
        assert "@graph" in result
        assert result["@graph"][0]["rsid"] == "rs12345"
    
    def test_query_region(self):
        """Test querying genomic region."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "@graph": [
                {"coordinates": {"start": 100}},
                {"coordinates": {"start": 200}},
                {"coordinates": {"start": 300}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        result = self.client.query_region("chr1", 100, 500)
        
        assert "@graph" in result
        assert len(result["@graph"]) == 3
    
    def test_get_regulatory_score(self):
        """Test getting regulatory score."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "@graph": [
                {
                    "regulome_score": {"ranking": "2b", "probability": 0.85},
                    "peaks": [{"test": "peak1"}, {"test": "peak2"}],
                    "motifs": [{"test": "motif1"}],
                    "qtls": [],
                    "chromatin_states": []
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        score = self.client.get_regulatory_score("chr1", 12345)
        
        assert score["score"] == "2b"
        assert score["probability"] == 0.85
        assert score["evidence_count"] == 2
        assert len(score["peaks"]) == 2
        assert len(score["motifs"]) == 1
    
    def test_batch_query(self):
        """Test batch querying variants."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "@graph": [
                {"coordinates": {"start": 100}},
                {"coordinates": {"start": 200}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        variants = [
            {"chr": "1", "pos": 100},
            {"chr": "2", "pos": 200}
        ]
        results = self.client.batch_query(variants)
        
        assert "@graph" in results
        assert len(results["@graph"]) == 2
    
    def test_get_peaks_at_position(self):
        """Test getting chromatin peaks."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "@graph": [
                {
                    "peaks": [
                        {"dataset": "ENCODE", "biosample": "K562"},
                        {"dataset": "Roadmap", "biosample": "liver"}
                    ]
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        peaks = self.client.get_peaks_at_position("chr1", 12345)
        
        assert len(peaks) == 2
        assert peaks[0]["dataset"] == "ENCODE"
        assert peaks[1]["biosample"] == "liver"