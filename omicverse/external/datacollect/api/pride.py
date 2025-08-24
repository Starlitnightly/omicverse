"""PRIDE proteomics database API client."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIClient
from ..config import settings


logger = logging.getLogger(__name__)


class PRIDEClient(BaseAPIClient):
    """Client for PRIDE proteomics database API.
    
    PRIDE (PRoteomics IDEntifications) is a centralized, standards compliant,
    public data repository for proteomics data, including protein and peptide
    identifications, post-translational modifications and supporting spectral evidence.
    
    API Documentation: https://www.ebi.ac.uk/pride/ws/archive/v2/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://www.ebi.ac.uk/pride/ws/archive/v2")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get PRIDE-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def search_projects(self, keyword: str = None, species: str = None,
                       instrument: str = None, **kwargs) -> Dict[str, Any]:
        """Search for proteomics projects.
        
        Args:
            keyword: Search keyword
            species: Species filter
            instrument: Instrument filter
            **kwargs: Additional filters
        
        Returns:
            Search results with projects
        """
        endpoint = "/projects"
        params = {
            "pageSize": kwargs.get("page_size", 100),
            "page": kwargs.get("page", 0)
        }
        
        if keyword:
            params["keyword"] = keyword
        if species:
            params["speciesFilter"] = species
        if instrument:
            params["instrumentFilter"] = instrument
        
        # Add additional filters
        for key in ["diseaseFilter", "tissueFilter", "ptmFilter", "quantificationFilter"]:
            if key in kwargs:
                params[key] = kwargs[key]
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_project(self, project_accession: str) -> Dict[str, Any]:
        """Get detailed project information.
        
        Args:
            project_accession: PRIDE project accession (e.g., PXD000001)
        
        Returns:
            Project details
        """
        endpoint = f"/projects/{project_accession}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_project_files(self, project_accession: str) -> List[Dict[str, Any]]:
        """Get files associated with a project.
        
        Args:
            project_accession: PRIDE project accession
        
        Returns:
            List of project files
        """
        endpoint = f"/files/byProject"
        params = {
            "accession": project_accession,
            "pageSize": 1000,
            "page": 0
        }
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def search_peptides(self, sequence: str = None, protein_accession: str = None,
                       project_accession: str = None, **kwargs) -> Dict[str, Any]:
        """Search for peptides.
        
        Args:
            sequence: Peptide sequence
            protein_accession: Protein accession
            project_accession: Project accession
            **kwargs: Additional parameters
        
        Returns:
            Peptide search results
        """
        endpoint = "/peptideevidences"
        params = {
            "pageSize": kwargs.get("page_size", 100),
            "page": kwargs.get("page", 0)
        }
        
        if sequence:
            params["peptideSequence"] = sequence
        if protein_accession:
            params["proteinAccession"] = protein_accession
        if project_accession:
            params["projectAccession"] = project_accession
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def search_proteins(self, protein_accession: str = None, project_accession: str = None,
                       **kwargs) -> Dict[str, Any]:
        """Search for protein identifications.
        
        Args:
            protein_accession: Protein accession
            project_accession: Project accession
            **kwargs: Additional parameters
        
        Returns:
            Protein search results
        """
        endpoint = "/proteinevidences"
        params = {
            "pageSize": kwargs.get("page_size", 100),
            "page": kwargs.get("page", 0)
        }
        
        if protein_accession:
            params["proteinAccession"] = protein_accession
        if project_accession:
            params["projectAccession"] = project_accession
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_spectra(self, project_accession: str, assay_accession: str = None,
                   **kwargs) -> Dict[str, Any]:
        """Get spectra data.
        
        Args:
            project_accession: Project accession
            assay_accession: Assay accession
            **kwargs: Additional parameters
        
        Returns:
            Spectra data
        """
        endpoint = "/spectra"
        params = {
            "projectAccession": project_accession,
            "pageSize": kwargs.get("page_size", 100),
            "page": kwargs.get("page", 0)
        }
        
        if assay_accession:
            params["assayAccession"] = assay_accession
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_psms(self, project_accession: str, peptide_sequence: str = None,
                **kwargs) -> Dict[str, Any]:
        """Get peptide-spectrum matches (PSMs).
        
        Args:
            project_accession: Project accession
            peptide_sequence: Peptide sequence filter
            **kwargs: Additional parameters
        
        Returns:
            PSM data
        """
        endpoint = "/psms"
        params = {
            "projectAccession": project_accession,
            "pageSize": kwargs.get("page_size", 100),
            "page": kwargs.get("page", 0)
        }
        
        if peptide_sequence:
            params["peptideSequence"] = peptide_sequence
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_project_statistics(self) -> Dict[str, Any]:
        """Get PRIDE database statistics.
        
        Returns:
            Database statistics
        """
        endpoint = "/stats"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def search_by_organism(self, organism: str, **kwargs) -> Dict[str, Any]:
        """Search projects by organism.
        
        Args:
            organism: Organism name or taxonomy ID
            **kwargs: Additional parameters
        
        Returns:
            Projects for the organism
        """
        return self.search_projects(species=organism, **kwargs)
    
    def search_by_disease(self, disease: str, **kwargs) -> Dict[str, Any]:
        """Search projects by disease.
        
        Args:
            disease: Disease name
            **kwargs: Additional parameters
        
        Returns:
            Disease-related projects
        """
        return self.search_projects(diseaseFilter=disease, **kwargs)
    
    def search_by_tissue(self, tissue: str, **kwargs) -> Dict[str, Any]:
        """Search projects by tissue.
        
        Args:
            tissue: Tissue name
            **kwargs: Additional parameters
        
        Returns:
            Tissue-specific projects
        """
        return self.search_projects(tissueFilter=tissue, **kwargs)
    
    def get_modifications(self, project_accession: str) -> List[Dict[str, Any]]:
        """Get post-translational modifications in a project.
        
        Args:
            project_accession: Project accession
        
        Returns:
            List of PTMs
        """
        # Get peptides with modifications
        peptides = self.search_peptides(project_accession=project_accession)
        
        modifications = {}
        for peptide in peptides.get("_embedded", {}).get("peptideevidences", []):
            if peptide.get("ptmList"):
                for ptm in peptide["ptmList"]:
                    mod_name = ptm.get("name", "Unknown")
                    if mod_name not in modifications:
                        modifications[mod_name] = {
                            "name": mod_name,
                            "accession": ptm.get("accession"),
                            "positions": [],
                            "count": 0
                        }
                    modifications[mod_name]["count"] += 1
                    modifications[mod_name]["positions"].append(ptm.get("position"))
        
        return list(modifications.values())
