"""Reactome API client."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIClient
from omicverse.external.datacollect.config.config import settings


logger = logging.getLogger(__name__)


class ReactomeClient(BaseAPIClient):
    """Client for Reactome pathway database API.
    
    Reactome is a free, open-source, curated and peer-reviewed pathway database
    that provides intuitive bioinformatics tools for the visualization,
    interpretation and analysis of pathway knowledge.
    
    API Documentation: https://reactome.org/ContentService/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://reactome.org/ContentService")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get Reactome-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def search(self, query: str, species: Optional[str] = None,
              types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search Reactome for entities.
        
        Args:
            query: Search query
            species: Species name or taxon ID
            types: Entity types to search (Pathway, Reaction, PhysicalEntity, etc.)
        
        Returns:
            Search results
        """
        endpoint = "/search/query"
        params = {
            "query": query,
            "cluster": True
        }
        
        if species:
            params["species"] = species
        if types:
            params["types"] = ",".join(types)
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_pathways_by_gene(self, gene_name: str, species: str = "Homo sapiens") -> List[Dict[str, Any]]:
        """Get pathways containing a gene.
        
        Args:
            gene_name: Gene name or identifier
            species: Species name
        
        Returns:
            List of pathways
        """
        endpoint = f"/data/pathways/low/diagram/entity/{gene_name}/allForms"
        params = {"species": species}
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        
        if response.status_code == 404:
            # Try alternative endpoint
            endpoint = f"/data/query/{gene_name}/pathways"
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                headers=self.get_default_headers()
            )
        
        response.raise_for_status()
        return response.json()
    
    def get_pathway_details(self, pathway_id: str) -> Dict[str, Any]:
        """Get detailed information about a pathway.
        
        Args:
            pathway_id: Reactome pathway stable ID (e.g., R-HSA-109582)
        
        Returns:
            Pathway details
        """
        endpoint = f"/data/query/{pathway_id}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_pathway_participants(self, pathway_id: str) -> Dict[str, Any]:
        """Get participants (molecules) in a pathway.
        
        Args:
            pathway_id: Reactome pathway ID
        
        Returns:
            Pathway participants
        """
        endpoint = f"/data/pathway/{pathway_id}/containedEntities"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_pathway_events(self, pathway_id: str) -> List[Dict[str, Any]]:
        """Get events (reactions) in a pathway.
        
        Args:
            pathway_id: Reactome pathway ID
        
        Returns:
            List of events
        """
        endpoint = f"/data/pathway/{pathway_id}/containedEvents"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_reaction_details(self, reaction_id: str) -> Dict[str, Any]:
        """Get detailed information about a reaction.
        
        Args:
            reaction_id: Reactome reaction ID
        
        Returns:
            Reaction details
        """
        endpoint = f"/data/query/{reaction_id}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_entity_ancestors(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get ancestor pathways of an entity.
        
        Args:
            entity_id: Entity stable ID
        
        Returns:
            List of ancestor pathways
        """
        endpoint = f"/data/event/{entity_id}/ancestors"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_interactors(self, protein_accession: str) -> List[Dict[str, Any]]:
        """Get protein-protein interactions.
        
        Args:
            protein_accession: Protein accession (e.g., UniProt ID)
        
        Returns:
            List of interactors
        """
        endpoint = f"/interactors/static/molecule/{protein_accession}/details"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        
        if response.status_code == 404:
            return []
        
        response.raise_for_status()
        return response.json()
    
    def get_disease_pathways(self, disease_name: str) -> List[Dict[str, Any]]:
        """Get pathways associated with a disease.
        
        Args:
            disease_name: Disease name
        
        Returns:
            List of disease pathways
        """
        # Search for disease
        search_results = self.search(disease_name, types=["Pathway"])
        
        disease_pathways = []
        for result in search_results.get("results", []):
            if "disease" in result.get("name", "").lower():
                disease_pathways.append(result)
        
        return disease_pathways
    
    def export_pathway(self, pathway_id: str, format: str = "json") -> Any:
        """Export pathway in various formats.
        
        Args:
            pathway_id: Reactome pathway ID
            format: Export format (json, sbml, biopax, pdf, png, svg)
        
        Returns:
            Pathway data in requested format
        """
        format_endpoints = {
            "json": f"/data/query/{pathway_id}",
            "sbml": f"/exporter/sbml/{pathway_id}.xml",
            "biopax": f"/exporter/biopax/{pathway_id}.owl",
            "pdf": f"/exporter/document/pdf/{pathway_id}",
            "png": f"/exporter/diagram/{pathway_id}.png",
            "svg": f"/exporter/diagram/{pathway_id}.svg"
        }
        
        endpoint = format_endpoints.get(format)
        if not endpoint:
            raise ValueError(f"Unsupported format: {format}")
        
        headers = self.get_default_headers()
        if format in ["pdf", "png", "svg"]:
            headers["Accept"] = f"application/{format}" if format == "pdf" else f"image/{format}"
        elif format in ["sbml", "biopax"]:
            headers["Accept"] = "application/xml"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=headers
        )
        response.raise_for_status()
        
        if format == "json":
            return response.json()
        else:
            return response.content
    
    def get_orthology(self, entity_id: str, species: str) -> List[Dict[str, Any]]:
        """Get orthologous entities in another species.
        
        Args:
            entity_id: Source entity ID
            species: Target species
        
        Returns:
            List of orthologous entities
        """
        endpoint = f"/data/orthology/{entity_id}/species/{species}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        
        if response.status_code == 404:
            return []
        
        response.raise_for_status()
        return response.json()
    
    def get_species_list(self) -> List[Dict[str, Any]]:
        """Get list of species in Reactome.
        
        Returns:
            List of species
        """
        endpoint = "/data/species/all"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def analyze_expression_data(self, gene_list: List[str],
                               species: str = "Homo sapiens") -> Dict[str, Any]:
        """Analyze gene expression data for pathway enrichment.
        
        Args:
            gene_list: List of gene identifiers
            species: Species name
        
        Returns:
            Pathway enrichment analysis results
        """
        endpoint = "/analysis/identifier"
        
        data = {
            "identifiers": "\n".join(gene_list),
            "species": species,
            "includeDisease": True
        }
        
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_top_level_pathways(self, species: str = "Homo sapiens") -> List[Dict[str, Any]]:
        """Get top-level pathways for a species.
        
        Args:
            species: Species name
        
        Returns:
            List of top-level pathways
        """
        endpoint = f"/data/pathways/top/{species}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()