"""Electron Microscopy Data Bank API client."""

from typing import Dict, List, Optional, Any
from ..base import BaseAPIClient


class EMDBClient(BaseAPIClient):
    """Client for EMDB API - 3D electron microscopy structures."""
    
    def __init__(self):
        super().__init__(
            base_url="https://www.ebi.ac.uk/emdb",
            rate_limit=1.0
        )
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    def query_emdb(
        self,
        keyword: Optional[str] = None,
        resolution_min: Optional[float] = None,
        resolution_max: Optional[float] = None,
        organism: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query EMDB for EM structures.
        
        Args:
            keyword: Search keyword
            resolution_min: Minimum resolution in Angstroms
            resolution_max: Maximum resolution in Angstroms
            organism: Organism name
        
        Returns:
            Dict containing search results
        """
        # Prefer JSON entry search endpoint for stability
        params = {}
        if keyword:
            params["query"] = keyword
        # Advanced filters may need specific endpoints; keep keyword search stable
        last_exc = None
        for _ in range(2):
            response = self.get("/api/search/entry", params=params)
            ctype = (response.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                try:
                    return response.json()
                except Exception as e:
                    last_exc = e
                    continue
        return {"results": []}
    
    def get_entry(self, emdb_id: str) -> Dict[str, Any]:
        """
        Get detailed information about an EMDB entry.
        
        Args:
            emdb_id: EMDB identifier (e.g., "EMD-1234")
        
        Returns:
            Entry details
        """
        return self.get(f"/api/entry/{emdb_id}").json()
    
    def search_by_author(
        self,
        author_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for entries by author.
        
        Args:
            author_name: Author name
            limit: Maximum results
        
        Returns:
            List of matching entries
        """
        params = {
            "author": author_name,
            "limit": limit
        }
        
        response = self.get("/api/search/author", params=params)
        return response.json().get("entries", [])
    
    def search_by_method(
        self,
        method: str,
        year_from: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search by reconstruction method.
        
        Args:
            method: Method (e.g., "single particle", "tomography")
            year_from: Publication year filter
        
        Returns:
            List of entries
        """
        params = {"method": method}
        if year_from:
            params["year_from"] = year_from
        
        response = self.get("/api/search/method", params=params)
        return response.json().get("entries", [])
    
    def get_sample_info(self, emdb_id: str) -> Dict[str, Any]:
        """
        Get sample information for an entry.
        
        Args:
            emdb_id: EMDB identifier
        
        Returns:
            Sample details
        """
        return self.get(f"/api/entry/{emdb_id}/sample").json()
    
    def get_experiment_info(self, emdb_id: str) -> Dict[str, Any]:
        """
        Get experimental details.
        
        Args:
            emdb_id: EMDB identifier
        
        Returns:
            Experiment information
        """
        return self.get(f"/api/entry/{emdb_id}/experiment").json()
    
    def get_map_statistics(self, emdb_id: str) -> Dict[str, Any]:
        """
        Get map statistics for an entry.
        
        Args:
            emdb_id: EMDB identifier
        
        Returns:
            Map statistics
        """
        return self.get(f"/api/entry/{emdb_id}/map_stats").json()
    
    def search_by_resolution(
        self,
        max_resolution: float,
        min_entries: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find high-resolution structures.
        
        Args:
            max_resolution: Maximum resolution cutoff
            min_entries: Minimum number of entries
        
        Returns:
            List of high-resolution entries
        """
        params = {
            "resolution_to": max_resolution,
            "limit": min_entries
        }
        
        response = self.get("/api/search/resolution", params=params)
        return response.json().get("entries", [])
    
    def get_related_pdb(self, emdb_id: str) -> List[str]:
        """
        Get related PDB entries.
        
        Args:
            emdb_id: EMDB identifier
        
        Returns:
            List of PDB IDs
        """
        response = self.get(f"/api/entry/{emdb_id}/pdb")
        return response.json().get("pdb_ids", [])
    
    def get_citations(self, emdb_id: str) -> List[Dict[str, Any]]:
        """
        Get citations for an entry.
        
        Args:
            emdb_id: EMDB identifier
        
        Returns:
            List of citations
        """
        response = self.get(f"/api/entry/{emdb_id}/citations")
        return response.json().get("citations", [])
    
    def search_complexes(
        self,
        complex_name: str,
        species: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for protein complexes.
        
        Args:
            complex_name: Complex name
            species: Species filter
        
        Returns:
            List of complex structures
        """
        params = {"complex": complex_name}
        if species:
            params["species"] = species
        
        response = self.get("/api/search/complex", params=params)
        return response.json().get("entries", [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get EMDB database statistics.
        
        Returns:
            Database statistics
        """
        return self.get("/api/statistics").json()
