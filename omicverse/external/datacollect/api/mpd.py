"""Mouse Phenome Database API client."""

from typing import Dict, List, Optional, Any
import requests
from .base import BaseAPIClient


class MPDClient(BaseAPIClient):
    """Client for Mouse Phenome Database API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://phenome.jax.org",
            rate_limit=1.0
        )
    
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json"
        }
    def query_mpd(
        self,
        measure_id: Optional[int] = None,
        strain: Optional[str] = None,
        dataset_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query MPD for phenotype data.
        
        Args:
            measure_id: Specific measure ID
            strain: Mouse strain name
            dataset_id: Dataset identifier
        
        Returns:
            Dict containing phenotype data
        """
        params = {}
        if measure_id:
            params["measnum"] = measure_id
        if strain:
            params["strain"] = strain
        if dataset_id:
            params["dataset"] = dataset_id
        
        response = self.get("/api/pheno/retrieve", params=params)
        return response.json()
    
    def search_strains(
        self,
        keyword: Optional[str] = None,
        panel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for mouse strains.
        
        Args:
            keyword: Search keyword
            panel: Strain panel (e.g., "inbred", "BXD")
        
        Returns:
            List of matching strains
        """
        params = {}
        if keyword:
            params["q"] = keyword
        if panel:
            params["panel"] = panel
        
        try:
            response = self.get("/api/strains/search", params=params)
            data = response.json()
            # Newer API may return list directly
            return data.get("strains", data if isinstance(data, list) else [])
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # Try alternative endpoints if route changed
            for alt in ("/api/strains", "/api/strain/search"):
                try:
                    resp = self.get(alt, params=params)
                    try:
                        data = resp.json()
                    except Exception:
                        data = []
                    return data.get("strains", data if isinstance(data, list) else [])
                except requests.exceptions.HTTPError:
                    continue
            # If all fail, bubble original error
            raise
    
    def get_strain_info(self, strain_name: str) -> Dict[str, Any]:
        """
        Get detailed strain information.
        
        Args:
            strain_name: Strain name
        
        Returns:
            Strain details
        """
        params = {"strain": strain_name}
        return self.get("/api/strains/info", params=params).json()
    
    def search_phenotypes(
        self,
        keyword: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for phenotype measures.
        
        Args:
            keyword: Search term
            category: Phenotype category
        
        Returns:
            List of matching phenotypes
        """
        params = {"q": keyword}
        if category:
            params["category"] = category
        
        response = self.get("/api/pheno/search", params=params)
        return response.json().get("measures", [])
    
    def get_measure_data(
        self,
        measure_id: int,
        sex: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get data for a specific measure.
        
        Args:
            measure_id: Measure identifier
            sex: Filter by sex (M/F)
        
        Returns:
            Measure data
        """
        params = {"measnum": measure_id}
        if sex:
            params["sex"] = sex
        
        return self.get("/api/pheno/measure", params=params).json()
    
    def get_dataset_info(self, dataset_id: int) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Dataset details
        """
        params = {"dataset": dataset_id}
        return self.get("/api/datasets/info", params=params).json()
    
    def get_qtl_data(
        self,
        chromosome: Optional[str] = None,
        start_mb: Optional[float] = None,
        end_mb: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get QTL mapping data.
        
        Args:
            chromosome: Chromosome number
            start_mb: Start position in Mb
            end_mb: End position in Mb
        
        Returns:
            List of QTL data
        """
        params = {}
        if chromosome:
            params["chr"] = chromosome
        if start_mb:
            params["start"] = start_mb
        if end_mb:
            params["end"] = end_mb
        
        response = self.get("/api/qtl/retrieve", params=params)
        return response.json().get("qtls", [])
    
    def get_gene_expression(
        self,
        gene_symbol: str,
        tissue: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get gene expression data.
        
        Args:
            gene_symbol: Gene symbol
            tissue: Tissue type
        
        Returns:
            Expression data
        """
        params = {"gene": gene_symbol}
        if tissue:
            params["tissue"] = tissue
        
        return self.get("/api/expression/gene", params=params).json()
    
    def compare_strains(
        self,
        strain1: str,
        strain2: str,
        measure_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare phenotypes between strains.
        
        Args:
            strain1: First strain
            strain2: Second strain
            measure_id: Specific measure to compare
        
        Returns:
            Comparison results
        """
        params = {
            "strain1": strain1,
            "strain2": strain2
        }
        if measure_id:
            params["measnum"] = measure_id
        
        return self.get("/api/strains/compare", params=params).json()
    
    def get_correlations(
        self,
        measure_id: int,
        min_correlation: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find correlated phenotypes.
        
        Args:
            measure_id: Reference measure
            min_correlation: Minimum correlation coefficient
        
        Returns:
            List of correlated measures
        """
        params = {
            "measnum": measure_id,
            "min_r": min_correlation
        }
        
        response = self.get("/api/pheno/correlations", params=params)
        return response.json().get("correlations", [])
