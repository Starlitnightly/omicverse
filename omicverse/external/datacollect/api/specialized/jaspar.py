"""JASPAR API client for transcription factor binding profiles."""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseAPIClient
from ...config import settings

logger = logging.getLogger(__name__)


class JASPARClient(BaseAPIClient):
    """Client for JASPAR REST API.
    
    JASPAR is an open-access database of curated, non-redundant transcription
    factor binding profiles stored as position frequency matrices (PFMs).
    
    API Documentation: https://jaspar.elixir.no/api/v1/docs/
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://jaspar.elixir.no/api/v1",
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get JASPAR-specific headers."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    
    def get_matrix(self, matrix_id: str) -> Dict[str, Any]:
        """Get a single matrix by ID.
        
        Args:
            matrix_id: Matrix ID (e.g., 'MA0001.1')
        
        Returns:
            Matrix data including PFM, PWM, and metadata
        """
        endpoint = f"/matrix/{matrix_id}/"
        response = self.get(endpoint)
        return response.json()
    
    def search_matrices(
        self,
        name: Optional[str] = None,
        collection: Optional[str] = None,
        tax_group: Optional[str] = None,
        tax_id: Optional[int] = None,
        tf_class: Optional[str] = None,
        tf_family: Optional[str] = None,
        data_type: Optional[str] = None,
        version: Optional[str] = None,
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """Search for matrices with various filters.
        
        Args:
            name: TF name or partial name
            collection: Collection name (e.g., 'CORE', 'CNE', 'PHYLOFACTS')
            tax_group: Taxonomic group (e.g., 'vertebrates', 'plants')
            tax_id: NCBI taxonomy ID
            tf_class: TF class
            tf_family: TF family
            data_type: Data type (e.g., 'ChIP-seq', 'SELEX')
            version: JASPAR version
            page: Page number for pagination
            page_size: Number of results per page
        
        Returns:
            Search results with pagination info
        """
        params = {
            "page": page,
            "page_size": page_size
        }
        
        if name:
            params["name"] = name
        if collection:
            params["collection"] = collection
        if tax_group:
            params["tax_group"] = tax_group
        if tax_id:
            params["tax_id"] = tax_id
        if tf_class:
            params["tf_class"] = tf_class
        if tf_family:
            params["tf_family"] = tf_family
        if data_type:
            params["data_type"] = data_type
        if version:
            params["version"] = version
        
        response = self.get("/matrix/", params=params)
        return response.json()
    
    def get_matrix_pfm(self, matrix_id: str) -> Dict[str, List[float]]:
        """Get Position Frequency Matrix (PFM) for a matrix.
        
        Args:
            matrix_id: Matrix ID
        
        Returns:
            PFM data with nucleotide frequencies
        """
        endpoint = f"/matrix/{matrix_id}/pfm/"
        response = self.get(endpoint)
        return response.json()
    
    def get_matrix_pwm(self, matrix_id: str) -> Dict[str, List[float]]:
        """Get Position Weight Matrix (PWM) for a matrix.
        
        Args:
            matrix_id: Matrix ID
        
        Returns:
            PWM data with position weights
        """
        endpoint = f"/matrix/{matrix_id}/pwm/"
        response = self.get(endpoint)
        return response.json()
    
    def get_matrix_jaspar_format(self, matrix_id: str) -> str:
        """Get matrix in JASPAR format.
        
        Args:
            matrix_id: Matrix ID
        
        Returns:
            Matrix in JASPAR text format
        """
        endpoint = f"/matrix/{matrix_id}/"
        headers = {"Accept": "text/plain"}
        response = self.get(endpoint, headers=headers)
        return response.text
    
    def get_matrix_meme_format(self, matrix_id: str) -> str:
        """Get matrix in MEME format.
        
        Args:
            matrix_id: Matrix ID
        
        Returns:
            Matrix in MEME format
        """
        endpoint = f"/matrix/{matrix_id}/meme/"
        response = self.get(endpoint)
        return response.text
    
    def get_matrix_transfac_format(self, matrix_id: str) -> str:
        """Get matrix in TRANSFAC format.
        
        Args:
            matrix_id: Matrix ID
        
        Returns:
            Matrix in TRANSFAC format
        """
        endpoint = f"/matrix/{matrix_id}/transfac/"
        response = self.get(endpoint)
        return response.text
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all available collections.
        
        Returns:
            List of collections with metadata
        """
        response = self.get("/collections/")
        return response.json()
    
    def get_releases(self) -> List[Dict[str, Any]]:
        """Get all JASPAR releases.
        
        Returns:
            List of releases with version info
        """
        response = self.get("/releases/")
        return response.json()
    
    def get_species(self) -> List[Dict[str, Any]]:
        """Get all species in JASPAR.
        
        Returns:
            List of species with taxonomy info
        """
        response = self.get("/species/")
        return response.json()
    
    def get_taxa(self) -> List[Dict[str, Any]]:
        """Get all taxonomic groups.
        
        Returns:
            List of taxonomic groups
        """
        response = self.get("/taxa/")
        return response.json()
    
    def infer_matrix(
        self,
        sequence: str,
        matrix_id: Optional[str] = None,
        collection: str = "CORE",
        tax_group: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Infer TF binding sites in a sequence.
        
        Args:
            sequence: DNA sequence to scan
            matrix_id: Specific matrix to use (if None, uses all matching criteria)
            collection: Collection to search in
            tax_group: Taxonomic group filter
        
        Returns:
            List of predicted binding sites with scores
        """
        data = {
            "sequence": sequence,
            "collection": collection
        }
        
        if matrix_id:
            data["matrix_id"] = matrix_id
        if tax_group:
            data["tax_group"] = tax_group
        
        response = self.post("/infer/", json=data)
        return response.json()
    
    def batch_download(
        self,
        matrix_ids: List[str],
        format: str = "json"
    ) -> Any:
        """Download multiple matrices in batch.
        
        Args:
            matrix_ids: List of matrix IDs
            format: Output format ('json', 'jaspar', 'meme', 'transfac')
        
        Returns:
            Matrices in requested format
        """
        data = {
            "matrix_ids": matrix_ids,
            "format": format
        }
        
        response = self.post("/batch/", json=data)
        
        if format == "json":
            return response.json()
        return response.text
    
    def get_tf_flexible_models(self) -> List[Dict[str, Any]]:
        """Get TF flexible models information.
        
        Returns:
            List of TF flexible models
        """
        response = self.get("/tffm/")
        return response.json()
    
    def search_by_sequence_similarity(
        self,
        sequence: str,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Search for similar binding profiles based on sequence.
        
        Args:
            sequence: Query sequence
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of similar matrices with similarity scores
        """
        data = {
            "sequence": sequence,
            "threshold": threshold
        }
        
        response = self.post("/similarity/sequence/", json=data)
        return response.json()
    
    def search_by_matrix_similarity(
        self,
        matrix_id: str,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find matrices similar to a given matrix.
        
        Args:
            matrix_id: Reference matrix ID
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of similar matrices with similarity scores
        """
        params = {
            "threshold": threshold
        }
        
        response = self.get(f"/matrix/{matrix_id}/similar/", params=params)
        return response.json()
