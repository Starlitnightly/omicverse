"""ENCODE cCRE (candidate cis-Regulatory Elements) API client.

Updated to use the ENCODE Portal search API which returns JSON when
`Accept: application/json` and `format=json` are provided.
"""

from typing import Dict, List, Optional, Any
from ..base import BaseAPIClient


class CCREClient(BaseAPIClient):
    """Client for ENCODE Portal cCRE search API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://www.encodeproject.org",
            rate_limit=2.0
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }

    def region_to_ccre_screen(
        self,
        chromosome: str,
        start: int,
        end: int,
        genome: str = "GRCh38"
    ) -> Dict[str, Any]:
        """Find cCREs in a genomic region via ENCODE Portal search.

        Args:
            chromosome: Chromosome (e.g., "chr1")
            start: Start position
            end: End position
            genome: Genome assembly (GRCh38 or mm10)

        Returns:
            Dict containing cCRE search results
        """
        params = {
            "type": "Annotation",
            "annotation_type": "candidate Cis-Regulatory Elements",
            "assembly": genome,
            "region": f"{chromosome}:{start}-{end}",
            "format": "json",
            "limit": 100
        }
        response = self.get("/search/", params=params)
        return response.json()
    
    def get_ccre_details(self, accession: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific cCRE.
        
        Args:
            accession: cCRE accession ID
        
        Returns:
            Dict containing cCRE details
        """
        return self.get(f"/api/v1/ccre/{accession}").json()
    
    def get_genes_near_ccre(
        self,
        accession: str,
        distance: int = 100000,
        genome: str = "GRCh38"
    ) -> List[Dict[str, Any]]:
        """
        Find genes near a cCRE.
        
        Args:
            accession: cCRE accession ID
            distance: Distance threshold in bp
            genome: Genome assembly
        
        Returns:
            List of nearby genes
        """
        params = {
            "distance": distance,
            "genome": genome
        }
        
        response = self.get(f"/api/v1/ccre/{accession}/genes", params=params)
        return response.json().get("genes", [])
    
    def search_ccres(
        self,
        cell_type: Optional[str] = None,
        ccre_type: Optional[str] = None,
        genome: str = "GRCh38",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for cCREs by various criteria.
        
        Args:
            cell_type: Cell type or tissue
            ccre_type: Type of cCRE (e.g., "promoter", "enhancer")
            genome: Genome assembly
            limit: Maximum results
        
        Returns:
            List of matching cCREs
        """
        params = {
            "genome": genome,
            "limit": limit
        }
        
        if cell_type:
            params["cell_type"] = cell_type
        if ccre_type:
            params["type"] = ccre_type
        
        response = self.get("/api/v1/ccres/search", params=params)
        return response.json().get("ccres", [])
    
    def get_ccre_expression(
        self,
        accession: str,
        cell_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get expression data for a cCRE.
        
        Args:
            accession: cCRE accession ID
            cell_type: Specific cell type
        
        Returns:
            Expression data
        """
        params = {}
        if cell_type:
            params["cell_type"] = cell_type
        
        return self.get(f"/api/v1/ccre/{accession}/expression", params=params).json()
    
    def get_ccre_chromatin_state(
        self,
        accession: str,
        assay_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get chromatin state data for a cCRE.
        
        Args:
            accession: cCRE accession ID
            assay_type: Type of chromatin assay
        
        Returns:
            Chromatin state data
        """
        params = {}
        if assay_type:
            params["assay"] = assay_type
        
        return self.get(f"/api/v1/ccre/{accession}/chromatin", params=params).json()
    
    def batch_query_ccres(
        self,
        regions: List[Dict[str, Any]],
        genome: str = "GRCh38"
    ) -> List[Dict[str, Any]]:
        """
        Query multiple genomic regions for cCREs.
        
        Args:
            regions: List of regions with chr, start, end
            genome: Genome assembly
        
        Returns:
            List of cCRE results for each region
        """
        results = []
        for region in regions:
            try:
                ccres = self.region_to_ccre_screen(
                    region["chr"],
                    region["start"],
                    region["end"],
                    genome
                )
                results.append({
                    "region": region,
                    "ccres": ccres.get("ccres", [])
                })
            except Exception as e:
                results.append({
                    "region": region,
                    "error": str(e)
                })
        
        return results
