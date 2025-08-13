"""Paleobiology Database API client for fossil data."""

from typing import Dict, List, Optional, Any
from omicverse.external.datacollect.api.base import BaseAPIClient


class PaleobiologyClient(BaseAPIClient):
    """Client for Paleobiology Database API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://paleobiodb.org/data1.2",
            rate_limit=5
        )
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    def query_paleobiology(
        self,
        taxon_name: Optional[str] = None,
        time_interval: Optional[str] = None,
        continent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query Paleobiology Database for fossil records.
        
        Args:
            taxon_name: Taxonomic name
            time_interval: Geological time interval
            continent: Continent name
        
        Returns:
            Dict containing fossil data
        """
        params = {"show": "full"}
        
        if taxon_name:
            params["taxon_name"] = taxon_name
        if time_interval:
            params["interval"] = time_interval
        if continent:
            params["continent"] = continent
        
        response = self.get("/occs/list.json", params=params)
        return response
    
    def get_occurrences(
        self,
        taxon_name: str,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get fossil occurrences.
        
        Args:
            taxon_name: Taxonomic name
            min_age: Minimum age in Ma
            max_age: Maximum age in Ma
            country: Country code
        
        Returns:
            List of occurrences
        """
        params = {
            "base_name": taxon_name,
            "show": "coords,age,strat,lith"
        }
        
        if min_age:
            params["min_ma"] = min_age
        if max_age:
            params["max_ma"] = max_age
        if country:
            params["cc"] = country
        
        response = self.get("/occs/list.json", params=params)
        return response.get("records", [])
    
    def search_occurrences(
        self,
        taxon_name: Optional[str] = None,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        country: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search for fossil occurrences (alias for get_occurrences but returns full response)."""
        params = {
            "show": "coords,age,strat,lith",
            "limit": limit,
            "offset": offset
        }
        
        if taxon_name:
            params["base_name"] = taxon_name
        if min_age:
            params["min_ma"] = min_age
        if max_age:
            params["max_ma"] = max_age
        if country:
            params["cc"] = country
        
        return self.get("/occs/list.json", params=params)
    
    def get_occurrence(self, occurrence_id: str) -> Dict[str, Any]:
        """Get single occurrence by ID."""
        params = {"id": occurrence_id, "show": "full"}
        response = self.get("/occs/single.json", params=params)
        return response.get("records", [{}])[0] if response.get("records") else {}
    
    def get_taxa(
        self,
        name: str,
        rank: Optional[str] = None,
        extinct: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get taxonomic information.
        
        Args:
            name: Taxonomic name
            rank: Taxonomic rank
            extinct: Filter extinct taxa
        
        Returns:
            List of taxa
        """
        params = {
            "name": name,
            "show": "attr,app,size,ecospace"
        }
        
        if rank:
            params["rank"] = rank
        if extinct is not None:
            params["extant"] = "no" if extinct else "yes"
        
        response = self.get("/taxa/list.json", params=params)
        return response.get("records", [])
    
    def search_taxa(
        self,
        name: Optional[str] = None,
        rank: Optional[str] = None,
        extinct: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search for taxa (returns full response)."""
        params = {
            "show": "attr,app,size,ecospace",
            "limit": limit,
            "offset": offset
        }
        
        if name:
            params["name"] = name
        if rank:
            params["rank"] = rank
        if extinct is not None:
            params["extant"] = "no" if extinct else "yes"
        
        return self.get("/taxa/list.json", params=params)
    
    def get_taxon(self, taxon_id: str) -> Dict[str, Any]:
        """Get single taxon by ID."""
        params = {"id": taxon_id, "show": "full"}
        response = self.get("/taxa/single.json", params=params)
        return response.get("records", [{}])[0] if response.get("records") else {}
    
    def get_collections(
        self,
        taxon_name: Optional[str] = None,
        interval: Optional[str] = None,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get fossil collections.
        
        Args:
            taxon_name: Taxonomic name
            interval: Time interval
            country: Country code
        
        Returns:
            List of collections
        """
        params = {"show": "loc,stratext"}
        
        if taxon_name:
            params["base_name"] = taxon_name
        if interval:
            params["interval"] = interval
        if country:
            params["cc"] = country
        
        response = self.get("/colls/list.json", params=params)
        return response.get("records", [])
    
    def search_collections(
        self,
        taxon_name: Optional[str] = None,
        interval: Optional[str] = None,
        country: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search for fossil collections."""
        params = {
            "show": "loc,stratext",
            "limit": limit,
            "offset": offset
        }
        
        if taxon_name:
            params["base_name"] = taxon_name
        if interval:
            params["interval"] = interval
        if country:
            params["cc"] = country
        
        return self.get("/colls/list.json", params=params)
    
    def get_collection(self, collection_id: str) -> Dict[str, Any]:
        """Get single collection by ID."""
        params = {"id": collection_id, "show": "full"}
        response = self.get("/colls/single.json", params=params)
        return response.get("records", [{}])[0] if response.get("records") else {}
    
    def get_time_intervals(self) -> List[Dict[str, Any]]:
        """
        Get geological time intervals.
        
        Returns:
            List of time intervals
        """
        params = {"scale": "all"}
        response = self.get("/intervals/list.json", params=params)
        return response.get("records", [])
    
    def get_intervals(
        self,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        scale: str = "all"
    ) -> Dict[str, Any]:
        """Get geological time intervals (alias returning full response)."""
        params = {"scale": scale}
        
        if min_age:
            params["min_ma"] = min_age
        if max_age:
            params["max_ma"] = max_age
        
        return self.get("/intervals/list.json", params=params)
    
    def get_diversity(
        self,
        taxon_name: str,
        time_rule: str = "major",
        interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get diversity through time.
        
        Args:
            taxon_name: Taxonomic name
            time_rule: Time resolution
            interval: Optional time interval
        
        Returns:
            Diversity data
        """
        params = {
            "base_name": taxon_name,
            "count": "species",
            "time_rule": time_rule
        }
        
        if interval:
            params["interval"] = interval
        
        response = self.get("/occs/diversity.json", params=params)
        return response.get("records", [])
    
    def get_references(
        self,
        author: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get bibliographic references.
        
        Args:
            author: Author name
            year: Publication year
        
        Returns:
            List of references
        """
        params = {}
        
        if author:
            params["author"] = author
        if year:
            params["year"] = year
        
        response = self.get("/refs/list.json", params=params)
        return response.get("records", [])
    
    def get_measurements(
        self,
        taxon_name: str,
        measurement_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get specimen measurements.
        
        Args:
            taxon_name: Taxonomic name
            measurement_type: Type of measurement
        
        Returns:
            List of measurements
        """
        params = {"taxon_name": taxon_name}
        
        if measurement_type:
            params["meas_type"] = measurement_type
        
        response = self.get("/specs/measurements.json", params=params)
        return response.get("records", [])
    
    def get_strata(
        self,
        name: Optional[str] = None,
        rank: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get stratigraphic units.
        
        Args:
            name: Stratigraphic name
            rank: Stratigraphic rank
        
        Returns:
            List of strata
        """
        params = {}
        
        if name:
            params["name"] = name
        if rank:
            params["rank"] = rank
        
        response = self.get("/strata/list.json", params=params)
        return response.get("records", [])