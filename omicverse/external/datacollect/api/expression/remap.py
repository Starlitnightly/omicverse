"""ReMap Transcription Factor Binding Database API client."""

from typing import Dict, List, Optional, Any
from ..base import BaseAPIClient


class ReMapClient(BaseAPIClient):
    """Client for ReMap API - regulatory regions from ChIP-seq."""
    
    def __init__(self):
        super().__init__(
            base_url="https://remap.univ-amu.fr",
            rate_limit=1.0
        )
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    def query_remap(
        self,
        chromosome: str,
        start: int,
        end: int,
        species: str = "human",
        assembly: str = "hg38"
    ) -> Dict[str, Any]:
        """
        Query ReMap for TF binding sites in a region.
        
        Args:
            chromosome: Chromosome (e.g., "chr1")
            start: Start position
            end: End position
            species: Species (human, mouse, fly, arabidopsis)
            assembly: Genome assembly
        
        Returns:
            Dict containing TF binding data
        """
        params = {
            "chr": chromosome,
            "start": start,
            "end": end,
            "species": species,
            "assembly": assembly
        }
        
        response = self.get("/api/v1/peaks", params=params)
        return response.json()
    
    def search_by_tf(
        self,
        tf_name: str,
        species: str = "human",
        cell_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search for binding sites of a specific TF.
        
        Args:
            tf_name: Transcription factor name
            species: Species
            cell_type: Specific cell type
            limit: Maximum results
        
        Returns:
            List of TF binding sites
        """
        params = {
            "tf": tf_name,
            "species": species,
            "limit": limit
        }
        
        if cell_type:
            params["cell_type"] = cell_type
        
        response = self.get("/api/v1/search/tf", params=params)
        return response.json().get("peaks", [])
    
    def search_by_gene(
        self,
        gene_name: str,
        window: int = 10000,
        species: str = "human"
    ) -> List[Dict[str, Any]]:
        """
        Find TF binding sites near a gene.
        
        Args:
            gene_name: Gene symbol
            window: Distance from TSS in bp
            species: Species
        
        Returns:
            List of TF binding sites
        """
        params = {
            "gene": gene_name,
            "window": window,
            "species": species
        }
        
        response = self.get("/api/v1/search/gene", params=params)
        return response.json().get("peaks", [])
    
    def get_tf_list(
        self,
        species: str = "human",
        cell_type: Optional[str] = None
    ) -> List[str]:
        """
        Get list of available TFs.
        
        Args:
            species: Species
            cell_type: Filter by cell type
        
        Returns:
            List of TF names
        """
        params = {"species": species}
        if cell_type:
            params["cell_type"] = cell_type
        
        response = self.get("/api/v1/tfs", params=params)
        return response.json().get("tfs", [])
    
    def get_cell_types(self, species: str = "human") -> List[str]:
        """
        Get list of available cell types.
        
        Args:
            species: Species
        
        Returns:
            List of cell types
        """
        params = {"species": species}
        response = self.get("/api/v1/cell_types", params=params)
        return response.json().get("cell_types", [])
    
    def get_peak_details(self, peak_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a peak.
        
        Args:
            peak_id: Peak identifier
        
        Returns:
            Peak details
        """
        return self.get(f"/api/v1/peak/{peak_id}").json()
    
    def get_enrichment(
        self,
        chromosome: str,
        start: int,
        end: int,
        species: str = "human"
    ) -> Dict[str, Any]:
        """
        Get TF enrichment in a region.
        
        Args:
            chromosome: Chromosome
            start: Start position
            end: End position
            species: Species
        
        Returns:
            Enrichment statistics
        """
        params = {
            "chr": chromosome,
            "start": start,
            "end": end,
            "species": species
        }
        
        return self.get("/api/v1/enrichment", params=params).json()
    
    def batch_query_regions(
        self,
        regions: List[Dict[str, Any]],
        species: str = "human"
    ) -> List[Dict[str, Any]]:
        """
        Query multiple regions for TF binding.
        
        Args:
            regions: List of regions with chr, start, end
            species: Species
        
        Returns:
            Results for each region
        """
        results = []
        for region in regions:
            try:
                peaks = self.query_remap(
                    region["chr"],
                    region["start"],
                    region["end"],
                    species
                )
                results.append({
                    "region": region,
                    "peaks": peaks.get("peaks", [])
                })
            except Exception as e:
                results.append({
                    "region": region,
                    "error": str(e)
                })
        
        return results
    
    def get_colocalization(
        self,
        tf1: str,
        tf2: str,
        species: str = "human"
    ) -> Dict[str, Any]:
        """
        Find co-localization between two TFs.
        
        Args:
            tf1: First TF name
            tf2: Second TF name
            species: Species
        
        Returns:
            Co-localization statistics
        """
        params = {
            "tf1": tf1,
            "tf2": tf2,
            "species": species
        }
        
        return self.get("/api/v1/colocalization", params=params).json()