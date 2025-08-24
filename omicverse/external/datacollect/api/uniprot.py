"""UniProt API client."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIClient
from ..config import settings


logger = logging.getLogger(__name__)


class UniProtClient(BaseAPIClient):
    """Client for UniProt REST API."""
    
    def __init__(self, **kwargs):
        rate_limit = kwargs.pop("rate_limit", 10)  # UniProt allows 10 req/sec
        super().__init__(
            base_url=settings.api.uniprot_base_url,
            rate_limit=rate_limit,
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get UniProt-specific headers."""
        headers = super().get_default_headers()
        headers.update({
            "Accept": "application/json",
        })
        return headers
    
    def get_entry(self, accession: str, format: str = "json") -> Dict[str, Any]:
        """Get a single UniProt entry.
        
        Args:
            accession: UniProt accession (e.g., P12345)
            format: Response format (json, xml, fasta, etc.)
        
        Returns:
            Entry data in requested format
        """
        endpoint = f"/uniprotkb/{accession}"
        if format != "json":
            endpoint += f".{format}"
        
        response = self.get(endpoint)
        
        if format == "json":
            return response.json()
        return response.text
    
    def search(
        self,
        query: str,
        format: str = "json",
        fields: Optional[List[str]] = None,
        size: int = 25,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search UniProt database.
        
        Args:
            query: Search query in UniProt query language
            format: Response format
            fields: List of fields to return
            size: Number of results per page
            cursor: Cursor for pagination
        
        Returns:
            Search results
        """
        params = {
            "query": query,
            "format": format,
            "size": size,
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        if cursor:
            params["cursor"] = cursor
        
        response = self.get("/uniprotkb/search", params=params)
        return response.json()
    
    def get_fasta(self, accession: str) -> str:
        """Get FASTA sequence for an entry."""
        return self.get_entry(accession, format="fasta")
    
    def get_features(self, accession: str) -> List[Dict[str, Any]]:
        """Get features for a protein."""
        entry = self.get_entry(accession)
        return entry.get("features", [])
    
    def batch_retrieve(
        self,
        accessions: List[str],
        format: str = "json",
        compressed: bool = False,
    ) -> Any:
        """Retrieve multiple entries in batch.
        
        Args:
            accessions: List of UniProt accessions
            format: Response format
            compressed: Whether to compress response
        
        Returns:
            Batch results
        """
        from_param = ",".join(accessions)
        params = {
            "from": from_param,
            "format": format,
        }
        
        if compressed:
            params["compressed"] = "true"
        
        response = self.get("/uniprotkb/accessions", params=params)
        
        if format == "json":
            return response.json()
        return response.text
    
    def id_mapping(
        self,
        from_db: str,
        to_db: str,
        ids: List[str],
    ) -> Dict[str, Any]:
        """Map IDs between databases.
        
        Args:
            from_db: Source database
            to_db: Target database
            ids: List of IDs to map
        
        Returns:
            Mapping results
        """
        data = {
            "from": from_db,
            "to": to_db,
            "ids": ",".join(ids),
        }
        
        # Submit job
        response = self.post("/idmapping/run", data=data)
        job_id = response.json()["jobId"]
        
        # Poll for results (simplified - in production, implement proper polling)
        import time
        while True:
            status_response = self.get(f"/idmapping/status/{job_id}")
            status_data = status_response.json()
            
            if "results" in status_data:
                return status_data
            
            time.sleep(2)  # Wait before next poll
