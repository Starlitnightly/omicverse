"""AlphaFold Protein Structure Database API client."""

from typing import Dict, List, Optional, Any
from omicverse.external.datacollect.api.base import BaseAPIClient


class AlphaFoldClient(BaseAPIClient):
    """Client for AlphaFold API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://alphafold.ebi.ac.uk/api",
            rate_limit=10
        )
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    def query_alphafold(
        self,
        uniprot_id: Optional[str] = None,
        gene_name: Optional[str] = None,
        organism: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query AlphaFold for protein structures.
        
        Args:
            uniprot_id: UniProt accession
            gene_name: Gene name
            organism: Organism name
        
        Returns:
            Dict containing structure data
        """
        if uniprot_id:
            return self.get_prediction_by_uniprot(uniprot_id)
        elif gene_name:
            return self.search_by_gene(gene_name, organism)
        else:
            raise ValueError("Either uniprot_id or gene_name required")
    
    def get_prediction_by_uniprot(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get structure by UniProt ID.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            Structure data
        """
        response = self.get(f"/prediction/{uniprot_id}")
        # Handle mock objects for testing
        if hasattr(response, 'json'):
            return response.json()
        return response
    
    def search_by_gene(
        self,
        gene: str,
        organism: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for structures by gene name.
        
        Args:
            gene: Gene name
            organism: Organism filter
        
        Returns:
            List of matching structures
        """
        params = {"gene": gene}
        if organism:
            params["organism"] = organism
        
        response = self.get("/search", params=params)
        return response if isinstance(response, list) else [response]
    
    def get_plddt_scores(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get pLDDT confidence scores.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            Confidence scores
        """
        return self.get(f"/prediction/{uniprot_id}/confidence")
    
    def get_pae_matrix(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get predicted aligned error matrix.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            PAE matrix data
        """
        return self.get(f"/prediction/{uniprot_id}/pae")
    
    def search_by_organism(
        self,
        organism: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all structures for an organism.
        
        Args:
            organism: Organism name or tax ID
            limit: Maximum results
        
        Returns:
            List of structures
        """
        params = {
            "organism": organism,
            "limit": limit
        }
        
        response = self.get("/search", params=params)
        return response if isinstance(response, list) else []
    
    def get_structure_summary(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get structure summary information.
        
        Args:
            uniprot_id: UniProt accession
        
        Returns:
            Summary data
        """
        return self.get(f"/summary/{uniprot_id}")
    
    def batch_query(self, uniprot_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Query multiple structures.
        
        Args:
            uniprot_ids: List of UniProt IDs
        
        Returns:
            List of structure data
        """
        results = []
        for uid in uniprot_ids:
            try:
                results.append(self.get_prediction_by_uniprot(uid))
            except Exception as e:
                results.append({"uniprot_id": uid, "error": str(e)})
        
        return results
    
    def get_structure_file(self, uniprot_id: str, format: str = "pdb") -> str:
        """
        Download structure file.
        
        Args:
            uniprot_id: UniProt accession
            format: File format (pdb, cif, bcif)
        
        Returns:
            Structure file content
        """
        # Build URL directly based on pattern
        base_url = "https://alphafold.ebi.ac.uk/files"
        file_extensions = {
            "pdb": "model_v4.pdb",
            "cif": "model_v4.cif",
            "bcif": "model_v4.bcif"
        }
        
        if format not in file_extensions:
            raise ValueError(f"Invalid format: {format}")
        
        # Construct file URL
        file_url = f"{base_url}/AF-{uniprot_id}-F1-{file_extensions[format]}"
        
        # Download file directly
        response = self.session.get(file_url)
        response.raise_for_status()
        return response.text