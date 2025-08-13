"""IUCN Red List API client for conservation status."""

import logging
from typing import Dict, List, Optional, Any
from omicverse.external.datacollect.api.base import BaseAPIClient

logger = logging.getLogger(__name__)


class IUCNClient(BaseAPIClient):
    """Client for IUCN Red List API."""
    
    def __init__(self, api_token: Optional[str] = None):
        super().__init__(
            base_url="https://apiv3.iucnredlist.org/api/v3",
            rate_limit=2
        )
        self.api_token = api_token
        if not api_token:
            logger.warning("No API token provided for IUCN Red List API")
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    
    def query_iucn(
        self,
        species_name: Optional[str] = None,
        iucn_id: Optional[int] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query IUCN for species conservation status.
        
        Args:
            species_name: Scientific species name
            iucn_id: IUCN species ID
            region: Geographic region code
        
        Returns:
            Dict containing conservation data
        """
        if iucn_id:
            return self.get_by_id(iucn_id, region)
        elif species_name:
            return self.get_by_name(species_name, region)
        else:
            raise ValueError("Either species_name or iucn_id required")
    
    def get_species_by_name(
        self,
        name: str,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get species by scientific name.
        
        Args:
            name: Scientific name
            region: Region code
        
        Returns:
            Species assessment data
        """
        if region:
            endpoint = f"/species/region/{region}/name/{name}"
        else:
            endpoint = f"/species/{name}"
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response
    
    def get_by_name(
        self,
        name: str,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Alias for get_species_by_name for backward compatibility."""
        return self.get_species_by_name(name, region)
    
    def get_species_by_id(
        self,
        species_id: int,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get species by IUCN ID.
        
        Args:
            species_id: IUCN species ID
            region: Region code
        
        Returns:
            Species data
        """
        if region:
            endpoint = f"/species/region/{region}/id/{species_id}"
        else:
            endpoint = f"/species/id/{species_id}"
        
        params = {"token": self.api_token} if self.api_token else {}
        return self.get(endpoint, params=params)
    
    def get_by_id(
        self,
        species_id: int,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Alias for get_species_by_id for backward compatibility."""
        return self.get_species_by_id(species_id, region)
    
    def get_threats(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get threats for a species.
        
        Args:
            species_name: Scientific name
            species_id: IUCN ID
        
        Returns:
            List of threats
        """
        if species_id:
            endpoint = f"/threats/species/id/{species_id}"
        elif species_name:
            endpoint = f"/threats/species/name/{species_name.replace(' ', '%20')}"
        else:
            raise ValueError("Species name or ID required")
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("result", [])
    
    def get_threats_by_species(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Alias for get_threats."""
        return self.get_threats(species_name, species_id)
    
    def get_habitats(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get habitat information.
        
        Args:
            species_name: Scientific name
            species_id: IUCN ID
        
        Returns:
            List of habitats
        """
        if species_id:
            endpoint = f"/habitats/species/id/{species_id}"
        elif species_name:
            endpoint = f"/habitats/species/name/{species_name.replace(' ', '%20')}"
        else:
            raise ValueError("Species name or ID required")
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("result", [])
    
    def get_habitats_by_species(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Alias for get_habitats."""
        return self.get_habitats(species_name, species_id)
    
    def get_conservation_measures(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conservation measures.
        
        Args:
            species_name: Scientific name
            species_id: IUCN ID
        
        Returns:
            List of conservation measures
        """
        if species_id:
            endpoint = f"/measures/species/id/{species_id}"
        elif species_name:
            endpoint = f"/measures/species/name/{species_name.replace(' ', '%20')}"
        else:
            raise ValueError("Species name or ID required")
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("result", [])
    
    def get_countries(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get countries of occurrence.
        
        Args:
            species_name: Scientific name
            species_id: IUCN ID
        
        Returns:
            List of countries
        """
        if species_id:
            endpoint = f"/species/countries/id/{species_id}"
        elif species_name:
            endpoint = f"/species/countries/name/{species_name.replace(' ', '%20')}"
        else:
            raise ValueError("Species name or ID required")
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("result", [])
    
    def get_countries_by_species(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Alias for get_countries."""
        return self.get_countries(species_name, species_id)
    
    def search_by_category(
        self,
        category: str,
        region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search species by threat category.
        
        Args:
            category: IUCN category (e.g., "CR", "EN", "VU")
            region: Region code
        
        Returns:
            List of species
        """
        endpoint = f"/species/category/{category}"
        if region:
            endpoint = f"/species/region/{region}/category/{category}"
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("result", [])
    
    def get_regions(self) -> List[Dict[str, Any]]:
        """
        Get list of regions.
        
        Returns:
            List of available regions
        """
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get("/region/list", params=params)
        return response.get("results", [])
    
    def get_comprehensive_assessment(
        self,
        species_name: str,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive species assessment.
        
        Args:
            species_name: Scientific name
            region: Region code
        
        Returns:
            Comprehensive assessment data
        """
        assessment = self.get_by_name(species_name, region)
        
        if assessment.get("result"):
            species_data = assessment["result"][0] if isinstance(assessment["result"], list) else assessment["result"]
            
            # Enrich with additional data
            species_data["threats"] = self.get_threats(species_name=species_name)
            species_data["habitats"] = self.get_habitats(species_name=species_name)
            species_data["measures"] = self.get_conservation_measures(species_name=species_name)
            species_data["countries"] = self.get_countries(species_name=species_name)
        
        return species_data if 'species_data' in locals() else assessment
    
    def search_species(
        self,
        query: Optional[str] = None,
        page: Optional[int] = None,
        region: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for species.
        
        Args:
            query: Search query
            page: Page number for pagination
            region: Region filter
            category: Category filter
        
        Returns:
            Dict containing search results
        """
        if page is not None:
            endpoint = f"/species/page/{page}"
            params = {"token": self.api_token} if self.api_token else {}
            return self.get(endpoint, params=params)
        
        if category:
            results = self.search_by_category(category, region)
            return {"result": results}
        
        if query:
            # Use name search as default
            return self.get_species_by_name(query, region)
        
        return {"result": []}
    
    def get_species_narrative(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get species narrative information.
        
        Args:
            species_name: Scientific name
            species_id: IUCN ID
        
        Returns:
            Narrative data
        """
        if species_id:
            endpoint = f"/species/narrative/id/{species_id}"
        elif species_name:
            endpoint = f"/species/narrative/{species_name}"
        else:
            raise ValueError("Species name or ID required")
        
        params = {"token": self.api_token} if self.api_token else {}
        return self.get(endpoint, params=params)
    
    def get_citation(
        self,
        species_name: Optional[str] = None,
        species_id: Optional[int] = None
    ) -> str:
        """
        Get citation for a species.
        
        Args:
            species_name: Scientific name
            species_id: IUCN ID
        
        Returns:
            Citation string
        """
        if species_id:
            endpoint = f"/species/citation/id/{species_id}"
        elif species_name:
            endpoint = f"/species/citation/{species_name}"
        else:
            raise ValueError("Species name or ID required")
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("citation", "")
    
    def get_comprehensive_groups(self, group_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get comprehensive assessment groups.
        
        Args:
            group_name: Optional group name filter
        
        Returns:
            List of groups
        """
        if group_name:
            endpoint = f"/comp-group/getspecies/{group_name}"
        else:
            endpoint = "/comp-group/list"
        
        params = {"token": self.api_token} if self.api_token else {}
        response = self.get(endpoint, params=params)
        return response.get("result", [])
    
    def _add_token(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add API token to parameters.
        
        Args:
            params: Optional parameters dict
        
        Returns:
            Updated parameters
        """
        if params is None:
            params = {}
        if self.api_token:
            params["token"] = self.api_token
        return params