"""Tests for GtoPdb API client."""

import pytest
from unittest.mock import patch, MagicMock

from omicverse.external.datacollect.api.gtopdb import GtoPdbClient


class TestGtoPdbClient:
    def setup_method(self):
        self.client = GtoPdbClient()
        self.client.session = MagicMock()
    
    def test_search_targets(self):
        """Test searching for drug targets."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "targetId": 1,
                "name": "5-HT1A receptor",
                "abbreviation": "5-HT1A",
                "type": "gpcr",
                "familyName": "5-HT receptors"
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        targets = self.client.search_targets("serotonin", target_type="gpcr")
        
        assert len(targets) == 1
        assert targets[0]["name"] == "5-HT1A receptor"
        assert targets[0]["type"] == "gpcr"
    
    def test_get_target(self):
        """Test getting target details."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "targetId": 1,
            "name": "5-HT1A receptor",
            "geneSymbol": "HTR1A",
            "uniprotId": "P08908",
            "species": "Human",
            "tissueDistribution": "Brain, heart",
            "functionalCharacteristics": "Gi/o coupled receptor"
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        target = self.client.get_target(1)
        
        assert target["targetId"] == 1
        assert target["geneSymbol"] == "HTR1A"
        assert target["uniprotId"] == "P08908"
    
    def test_get_target_by_gene(self):
        """Test getting targets by gene symbol."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "targetId": 1,
                "name": "5-HT1A receptor",
                "geneSymbol": "HTR1A"
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        targets = self.client.get_target_by_gene("HTR1A")
        
        assert len(targets) == 1
        assert targets[0]["geneSymbol"] == "HTR1A"
    
    def test_search_ligands(self):
        """Test searching for ligands."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "ligandId": 1,
                "name": "serotonin",
                "abbreviation": "5-HT",
                "type": "Metabolite",
                "approved": False
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        ligands = self.client.search_ligands("serotonin")
        
        assert len(ligands) == 1
        assert ligands[0]["name"] == "serotonin"
        assert ligands[0]["type"] == "Metabolite"
    
    def test_get_ligand(self):
        """Test getting ligand details."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ligandId": 1,
            "name": "serotonin",
            "inn": None,
            "type": "Metabolite",
            "approved": False,
            "molecularFormula": "C10H12N2O",
            "molecularWeight": 176.22,
            "smiles": "NCCc1c[nH]c2ccc(O)cc12",
            "pubchemCid": 5202
        }
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        ligand = self.client.get_ligand(1)
        
        assert ligand["ligandId"] == 1
        assert ligand["molecularFormula"] == "C10H12N2O"
        assert ligand["pubchemCid"] == 5202
    
    def test_get_interactions(self):
        """Test getting target-ligand interactions."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "targetId": 1,
                "targetName": "5-HT1A receptor",
                "ligandId": 1,
                "ligandName": "serotonin",
                "type": "Agonist",
                "action": "Full agonist",
                "affinityType": "pKi",
                "affinityValue": 8.5,
                "approved": False
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        interactions = self.client.get_interactions(target_id=1)
        
        assert len(interactions) == 1
        assert interactions[0]["targetName"] == "5-HT1A receptor"
        assert interactions[0]["affinityValue"] == 8.5
    
    def test_get_diseases(self):
        """Test getting diseases."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "diseaseId": 1,
                "name": "Depression",
                "meshId": "D003866"
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        diseases = self.client.get_diseases()
        
        assert len(diseases) == 1
        assert diseases[0]["name"] == "Depression"
    
    def test_get_families(self):
        """Test getting target families."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "familyId": 1,
                "name": "5-HT receptors",
                "type": "gpcr"
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        families = self.client.get_families(family_type="gpcr")
        
        assert len(families) == 1
        assert families[0]["name"] == "5-HT receptors"
    
    def test_get_approved_drugs(self):
        """Test getting approved drugs."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "ligandId": 100,
                "name": "fluoxetine",
                "inn": "fluoxetine",
                "approved": True
            }
        ]
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        drugs = self.client.get_approved_drugs()
        
        assert len(drugs) == 1
        assert drugs[0]["name"] == "fluoxetine"
        assert drugs[0]["approved"] == True