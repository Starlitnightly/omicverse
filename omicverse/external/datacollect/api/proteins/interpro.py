"""InterPro Protein Families and Domains API client."""

from typing import Dict, List, Optional, Any
from omicverse.external.datacollect.api.base import BaseAPIClient


class InterProClient(BaseAPIClient):
    """Client for InterPro API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://www.ebi.ac.uk/interpro/api",
            rate_limit=20
        )
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    def query_interpro(
        self,
        protein_id: Optional[str] = None,
        interpro_id: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query InterPro for protein families and domains.
        
        Args:
            protein_id: UniProt accession
            interpro_id: InterPro entry ID
            search_term: Search keyword
        
        Returns:
            Dict containing domain/family data
        """
        if interpro_id:
            return self.get_entry(interpro_id)
        elif protein_id:
            return self.get_protein_entries(protein_id)
        elif search_term:
            return self.search(search_term)
        else:
            raise ValueError("One of protein_id, interpro_id, or search_term required")
    
    def get_entry(self, interpro_id: str) -> Dict[str, Any]:
        """
        Get InterPro entry details.
        
        Args:
            interpro_id: InterPro entry ID
        
        Returns:
            Entry details
        """
        response = self.get(f"/entry/interpro/{interpro_id}")
        # Handle mock response
        if hasattr(response, 'json'):
            return response.json()
        # Wrap in metadata structure if not already
        if response and isinstance(response, dict) and "metadata" not in response:
            response = {"metadata": response}
        return response
    
    def get_protein_entries(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get InterPro entries for a protein.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            Protein's InterPro entries
        """
        response = self.get(f"/protein/UniProt/{uniprot_id}/entry/interpro")
        # Handle mock response
        if hasattr(response, 'json'):
            return response.json()
        return response
    
    def search(
        self,
        query: str,
        entry_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search InterPro database.
        
        Args:
            query: Search term
            entry_type: Entry type filter (replaces type_filter)
            limit: Maximum results
        
        Returns:
            Search results
        """
        params = {
            "search": query,
            "page_size": limit
        }
        
        if entry_type:
            params["type"] = entry_type
        
        response = self.get("/search/entry", params=params)
        return response.get("results", [])
    
    def get_member_databases(self) -> List[Dict[str, Any]]:
        """
        Get list of member databases.
        
        Returns:
            List of databases
        """
        response = self.get("/database")
        return response.get("results", [])
    
    def get_go_terms(self, interpro_id: str) -> List[Dict[str, Any]]:
        """
        Get GO terms for an entry.
        
        Args:
            interpro_id: InterPro entry ID
        
        Returns:
            List of GO terms
        """
        response = self.get(f"/entry/interpro/{interpro_id}/go_terms")
        return response.get("results", [])
    
    def get_pathways(self, interpro_id: str) -> List[Dict[str, Any]]:
        """
        Get pathway associations.
        
        Args:
            interpro_id: InterPro entry ID
        
        Returns:
            List of pathways
        """
        response = self.get(f"/entry/interpro/{interpro_id}/pathways")
        return response.get("results", [])
    
    def get_structural_models(self, interpro_id: str) -> List[Dict[str, Any]]:
        """
        Get structural models.
        
        Args:
            interpro_id: InterPro entry ID
        
        Returns:
            List of structures
        """
        response = self.get(f"/entry/interpro/{interpro_id}/structures")
        return response.get("results", [])
    
    def get_taxonomy_distribution(self, interpro_id: str) -> Dict[str, Any]:
        """
        Get taxonomic distribution.
        
        Args:
            interpro_id: InterPro entry ID
        
        Returns:
            Taxonomy data
        """
        return self.get(f"/entry/interpro/{interpro_id}/taxonomy")
    
    def get_overlapping_entries(
        self,
        interpro_id: str,
        db_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get overlapping entries from member databases.
        
        Args:
            interpro_id: InterPro entry ID
            db_filter: Filter by database
        
        Returns:
            List of overlapping entries
        """
        endpoint = f"/entry/interpro/{interpro_id}/member_databases"
        if db_filter:
            endpoint += f"/{db_filter}"
        
        response = self.get(endpoint)
        return response.get("results", [])
    
    def get_domains_for_protein(self, uniprot_id: str) -> List[Dict[str, Any]]:
        """
        Get domains for a specific protein.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            List of domain information
        """
        # Get all entries for the protein
        entries = self.get_protein_entries(uniprot_id)
        
        # Handle if entries is not a dict
        if not isinstance(entries, dict):
            return []
        
        domains = []
        for result in entries.get("results", []):
            metadata = result.get("metadata", {})
            if metadata.get("type") == "domain" or "domain" in metadata.get("name", "").lower():
                domain_info = {
                    "interpro_id": metadata.get("accession"),
                    "name": metadata.get("name"),
                    "entry_type": metadata.get("type", "domain"),
                    "locations": []
                }
                
                # Extract location information if available
                for protein in result.get("proteins", []):
                    if protein.get("accession") == uniprot_id:
                        for loc in protein.get("entry_protein_locations", []):
                            domain_info["locations"].append(loc)
                
                domains.append(domain_info)
        
        return domains