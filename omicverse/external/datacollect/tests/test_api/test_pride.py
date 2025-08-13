"""Tests for PRIDE API client."""

import pytest
from unittest.mock import patch, MagicMock

from omicverse.external.datacollect.api.pride import PRIDEClient


class TestPRIDEClient:
    def setup_method(self):
        self.client = PRIDEClient()
        self.client.session = MagicMock()
    
    def test_search_projects(self):
        """Test searching for projects."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "_embedded": {
                "projects": [
                    {
                        "accession": "PXD000001",
                        "title": "Test Project",
                        "species": [{"name": "Homo sapiens"}]
                    }
                ]
            },
            "page": {"totalElements": 1}
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        results = self.client.search_projects(keyword="cancer", species="Homo sapiens")
        
        assert "_embedded" in results
        assert len(results["_embedded"]["projects"]) == 1
        assert results["_embedded"]["projects"][0]["accession"] == "PXD000001"
    
    def test_get_project(self):
        """Test getting project details."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "accession": "PXD000001",
            "title": "Test Project",
            "projectDescription": "Test description",
            "numAssays": 10,
            "species": [{"name": "Homo sapiens", "accession": "9606"}],
            "tissues": [{"name": "liver"}],
            "ptms": [{"name": "Phosphorylation"}]
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        project = self.client.get_project("PXD000001")
        
        assert project["accession"] == "PXD000001"
        assert project["numAssays"] == 10
        assert len(project["species"]) == 1
        assert project["species"][0]["name"] == "Homo sapiens"
    
    def test_search_peptides(self):
        """Test searching for peptides."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "_embedded": {
                "peptideevidences": [
                    {
                        "peptideSequence": "PEPTIDER",
                        "proteinAccession": "P12345",
                        "projectAccession": "PXD000001",
                        "charge": 2,
                        "mz": 500.25
                    }
                ]
            },
            "page": {"totalElements": 1}
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        results = self.client.search_peptides(sequence="PEPTIDER")
        
        assert "_embedded" in results
        assert len(results["_embedded"]["peptideevidences"]) == 1
        assert results["_embedded"]["peptideevidences"][0]["peptideSequence"] == "PEPTIDER"
    
    def test_search_proteins(self):
        """Test searching for proteins."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "_embedded": {
                "proteinevidences": [
                    {
                        "proteinAccession": "P12345",
                        "projectAccession": "PXD000001",
                        "numPSMs": 100,
                        "numPeptides": 20,
                        "sequenceCoverage": 45.5
                    }
                ]
            },
            "page": {"totalElements": 1}
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        results = self.client.search_proteins(protein_accession="P12345")
        
        assert "_embedded" in results
        assert results["_embedded"]["proteinevidences"][0]["proteinAccession"] == "P12345"
        assert results["_embedded"]["proteinevidences"][0]["sequenceCoverage"] == 45.5
    
    def test_get_modifications(self):
        """Test getting PTMs from a project."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "_embedded": {
                "peptideevidences": [
                    {
                        "peptideSequence": "PEPTIDER",
                        "ptmList": [
                            {"name": "Phosphorylation", "position": 3, "accession": "MOD:00046"},
                            {"name": "Phosphorylation", "position": 5, "accession": "MOD:00046"}
                        ]
                    },
                    {
                        "peptideSequence": "SEQUENCES",
                        "ptmList": [
                            {"name": "Acetylation", "position": 1, "accession": "MOD:00064"}
                        ]
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        modifications = self.client.get_modifications("PXD000001")
        
        assert len(modifications) == 2  # Phosphorylation and Acetylation
        phospho = next(m for m in modifications if m["name"] == "Phosphorylation")
        assert phospho["count"] == 2
    
    def test_get_project_statistics(self):
        """Test getting database statistics."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "numProjects": 15000,
            "numAssays": 100000,
            "numSpectra": 1000000000,
            "numProteins": 500000,
            "numPeptides": 5000000
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        stats = self.client.get_project_statistics()
        
        assert stats["numProjects"] == 15000
        assert stats["numSpectra"] == 1000000000