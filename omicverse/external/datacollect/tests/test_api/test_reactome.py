"""Tests for Reactome API client."""

import pytest
from unittest.mock import patch, MagicMock

from omicverse.external.datacollect.api.reactome import ReactomeClient


class TestReactomeClient:
    def setup_method(self):
        self.client = ReactomeClient()
        self.client.session = MagicMock()
    
    def test_search(self):
        """Test searching Reactome."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "stId": "R-HSA-109582",
                    "name": "Hemostasis",
                    "species": "Homo sapiens"
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        results = self.client.search("hemostasis", species="Homo sapiens")
        
        assert "results" in results
        assert len(results["results"]) == 1
        assert results["results"][0]["name"] == "Hemostasis"
    
    def test_get_pathways_by_gene(self):
        """Test getting pathways for a gene."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "stId": "R-HSA-1234",
                "displayName": "Test Pathway",
                "species": [{"displayName": "Homo sapiens"}]
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        pathways = self.client.get_pathways_by_gene("TP53")
        
        assert len(pathways) == 1
        assert pathways[0]["displayName"] == "Test Pathway"
    
    def test_get_pathway_details(self):
        """Test getting pathway details."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "stId": "R-HSA-109582",
            "displayName": "Hemostasis",
            "summation": [{"text": "Blood clotting"}],
            "species": [{"displayName": "Homo sapiens"}]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        details = self.client.get_pathway_details("R-HSA-109582")
        
        assert details["stId"] == "R-HSA-109582"
        assert details["displayName"] == "Hemostasis"
        assert len(details["summation"]) == 1
    
    def test_get_interactors(self):
        """Test getting protein interactors."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "acc": "P12345",
                "alias": "GENE1",
                "score": 0.95
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        interactors = self.client.get_interactors("P04637")
        
        assert len(interactors) == 1
        assert interactors[0]["acc"] == "P12345"
        assert interactors[0]["score"] == 0.95
    
    def test_analyze_expression_data(self):
        """Test pathway enrichment analysis."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "summary": {"token": "test_token"},
            "pathwaysFound": 10,
            "identifiersNotFound": ["UNKNOWN1"]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.post.return_value = mock_response
        
        results = self.client.analyze_expression_data(["TP53", "BRCA1", "EGFR"])
        
        assert results["pathwaysFound"] == 10
        assert "UNKNOWN1" in results["identifiersNotFound"]
    
    def test_get_top_level_pathways(self):
        """Test getting top-level pathways."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "stId": "R-HSA-1234",
                "displayName": "Metabolism"
            },
            {
                "stId": "R-HSA-5678",
                "displayName": "Signal Transduction"
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        pathways = self.client.get_top_level_pathways()
        
        assert len(pathways) == 2
        assert pathways[0]["displayName"] == "Metabolism"