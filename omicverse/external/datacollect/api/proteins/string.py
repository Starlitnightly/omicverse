"""STRING API client."""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class STRINGClient(BaseAPIClient):
    """Client for STRING database API.
    
    STRING (Search Tool for the Retrieval of Interacting Genes/Proteins)
    is a database of known and predicted protein-protein interactions.
    
    API Documentation: https://string-db.org/help/api/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://string-db.org/api")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
        self.api_version = "v11"
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get STRING-specific headers."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json",
        }
    
    def get_string_ids(self, identifiers: List[str], species: int = 9606) -> List[Dict[str, Any]]:
        """Map external identifiers to STRING IDs.
        
        Args:
            identifiers: List of protein identifiers (UniProt, Ensembl, etc.)
            species: NCBI taxonomy ID (default: 9606 for human)
        
        Returns:
            List of STRING ID mappings
        """
        endpoint = f"/{self.api_version}/json/get_string_ids"
        params = {
            "identifiers": "\r".join(identifiers),  # STRING expects \r separated
            "species": species,
            "echo_query": 1,
            "format": "json"
        }
        
        response = self.post(endpoint, data=params)
        return response.json()
    
    def get_network(self, identifiers: List[str], species: int = 9606,
                   required_score: int = 400, network_type: str = "functional",
                   add_nodes: int = 0) -> List[Dict[str, Any]]:
        """Get protein-protein interaction network.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            required_score: Minimum required interaction score (0-1000)
            network_type: Type of network edges (functional or physical)
            add_nodes: Number of additional interactors to add
        
        Returns:
            List of interactions
        """
        endpoint = f"/{self.api_version}/json/network"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
            "required_score": required_score,
            "network_type": network_type,
            "add_nodes": add_nodes,
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_interaction_partners(self, identifiers: List[str], species: int = 9606,
                               required_score: int = 400, limit: int = 10) -> List[Dict[str, Any]]:
        """Get interaction partners for proteins.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            required_score: Minimum required interaction score
            limit: Maximum number of interactors per protein
        
        Returns:
            List of interaction partners
        """
        endpoint = f"/{self.api_version}/json/interaction_partners"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
            "required_score": required_score,
            "limit": limit,
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_enrichment(self, identifiers: List[str], species: int = 9606,
                      background_string_identifiers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get functional enrichment for a set of proteins.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            background_string_identifiers: Optional background set
        
        Returns:
            List of enriched terms
        """
        endpoint = f"/{self.api_version}/json/enrichment"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
        }
        
        if background_string_identifiers:
            params["background_string_identifiers"] = "\r".join(background_string_identifiers)
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_ppi_enrichment(self, identifiers: List[str], species: int = 9606,
                          required_score: int = 400) -> Dict[str, Any]:
        """Test if proteins are enriched for interactions.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            required_score: Minimum required interaction score
        
        Returns:
            PPI enrichment statistics
        """
        endpoint = f"/{self.api_version}/json/ppi_enrichment"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
            "required_score": required_score,
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_homology(self, identifiers: List[str], species: int = 9606) -> List[Dict[str, Any]]:
        """Get homologous proteins in other species.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID of source species
        
        Returns:
            List of homologs
        """
        endpoint = f"/{self.api_version}/json/homology"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_actions(self, identifiers: List[str], species: int = 9606,
                   required_score: int = 400, limit: int = 10) -> List[Dict[str, Any]]:
        """Get detailed interaction types (actions).
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            required_score: Minimum required interaction score
            limit: Maximum number of interactions
        
        Returns:
            List of interaction actions
        """
        endpoint = f"/{self.api_version}/json/actions"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
            "required_score": required_score,
            "limit": limit,
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def resolve(self, identifiers: List[str], species: int = 9606) -> List[Dict[str, Any]]:
        """Resolve ambiguous protein names.
        
        Args:
            identifiers: List of protein names/identifiers
            species: NCBI taxonomy ID
        
        Returns:
            List of resolved protein matches
        """
        endpoint = f"/{self.api_version}/json/resolve"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_image(self, identifiers: List[str], species: int = 9606,
                 required_score: int = 400, network_flavor: str = "confidence",
                 add_color_nodes: int = 10, add_white_nodes: int = 0) -> bytes:
        """Get network image.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            required_score: Minimum required interaction score
            network_flavor: Visual style of network
            add_color_nodes: Number of colored interaction partners to add
            add_white_nodes: Number of white interaction partners to add
        
        Returns:
            PNG image data
        """
        endpoint = f"/{self.api_version}/image/network"
        params = {
            "identifiers": "\r".join(identifiers),
            "species": species,
            "required_score": required_score,
            "network_flavor": network_flavor,
            "add_color_nodes": add_color_nodes,
            "add_white_nodes": add_white_nodes,
        }
        
        response = self.get(endpoint, params=params)
        return response.content