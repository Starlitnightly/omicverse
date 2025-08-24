"""Simplified PDB API client using direct REST endpoints."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIClient
from ..config.config import settings


logger = logging.getLogger(__name__)


class SimplePDBClient(BaseAPIClient):
    """Simplified client for RCSB PDB REST API."""
    
    def __init__(self, **kwargs):
        # Use the download base URL which has simpler endpoints
        rate_limit = kwargs.pop("rate_limit", 10)
        super().__init__(
            base_url="https://files.rcsb.org",
            rate_limit=rate_limit,
            **kwargs
        )
        self.data_api_base = "https://data.rcsb.org"
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get PDB-specific headers."""
        return {
            "User-Agent": "BioinformaticsDataCollector/0.1.0",
            "Accept": "application/json,text/plain",
        }
    
    def get_entry(self, pdb_id: str) -> Dict[str, Any]:
        """Get basic PDB entry data.
        
        Args:
            pdb_id: 4-character PDB ID
        
        Returns:
            PDB entry data
        """
        # Use the summary endpoint which gives basic info
        url = f"{self.data_api_base}/rest/v1/core/entry/{pdb_id.upper()}"
        
        response = self.session.get(url, headers=self.get_default_headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_structure(self, pdb_id: str, format: str = "pdb") -> str:
        """Download structure file.
        
        Args:
            pdb_id: 4-character PDB ID  
            format: File format (pdb, cif)
        
        Returns:
            Structure file content
        """
        endpoint = f"/download/{pdb_id.upper()}.{format}"
        response = self.get(endpoint)
        return response.text
    
    def get_polymer_info(self, pdb_id: str) -> Dict[str, Any]:
        """Get simplified polymer/chain information.
        
        Args:
            pdb_id: 4-character PDB ID
        
        Returns:
            Polymer information
        """
        # Get entity information
        url = f"{self.data_api_base}/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
        
        # This might fail for some structures, so we'll return empty data
        try:
            response = self.session.get(url, headers=self.get_default_headers(), timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {}
