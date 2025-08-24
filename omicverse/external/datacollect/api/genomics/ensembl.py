"""Ensembl API client."""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class EnsemblClient(BaseAPIClient):
    """Client for Ensembl REST API.
    
    Ensembl is a genome browser for vertebrate genomes that supports
    research in comparative genomics, evolution, sequence variation
    and transcriptional regulation.
    
    API Documentation: https://rest.ensembl.org/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://rest.ensembl.org")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 15),  # Ensembl allows 15 req/sec
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get Ensembl-specific headers."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    def lookup_id(self, identifier: str, species: Optional[str] = None,
                 db_type: Optional[str] = None, expand: bool = False) -> Dict[str, Any]:
        """Look up Ensembl ID.
        
        Args:
            identifier: Ensembl stable ID
            species: Species name (optional)
            db_type: Database type (optional)
            expand: Expand the search to include transcripts, translations
        
        Returns:
            Gene/transcript/protein information
        """
        endpoint = f"/lookup/id/{identifier}"
        params = {}
        
        if species:
            params["species"] = species
        if db_type:
            params["db_type"] = db_type
        if expand:
            params["expand"] = 1
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def lookup_symbol(self, symbol: str, species: str, expand: bool = False) -> Dict[str, Any]:
        """Look up by gene symbol.
        
        Args:
            symbol: Gene symbol (e.g., "BRCA2")
            species: Species name (e.g., "human")
            expand: Include transcripts and translations
        
        Returns:
            Gene information
        """
        endpoint = f"/lookup/symbol/{species}/{symbol}"
        params = {"expand": 1} if expand else {}
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_sequence(self, identifier: str, object_type: str = "genomic",
                    format: str = "fasta", multiple_sequences: bool = False) -> str:
        """Get sequence for an Ensembl ID.
        
        Args:
            identifier: Ensembl ID
            object_type: Type of sequence (genomic, cds, cdna, protein)
            format: Output format (fasta or json)
            multiple_sequences: For genes, get all transcripts
        
        Returns:
            Sequence data
        """
        endpoint = f"/sequence/id/{identifier}"
        params = {
            "type": object_type,
            "format": format,
        }
        
        if multiple_sequences:
            params["multiple_sequences"] = 1
        
        headers = self.get_default_headers()
        if format == "fasta":
            headers["Accept"] = "text/x-fasta"
        
        response = self.get(endpoint, params=params, headers=headers)
        
        if format == "fasta":
            return response.text
        return response.json()
    
    def get_variant(self, species: str, variant_id: str) -> Dict[str, Any]:
        """Get variant information.
        
        Args:
            species: Species name
            variant_id: Variant ID (e.g., rs56116432)
        
        Returns:
            Variant information
        """
        endpoint = f"/variation/{species}/{variant_id}"
        response = self.get(endpoint)
        return response.json()
    
    def get_variant_consequences(self, species: str, region: str, allele: str) -> List[Dict[str, Any]]:
        """Get variant consequences by region.
        
        Args:
            species: Species name
            region: Genomic region (e.g., "9:22125503-22125502:1")
            allele: Alternative allele
        
        Returns:
            List of consequences
        """
        endpoint = f"/vep/{species}/region/{region}/{allele}"
        response = self.get(endpoint)
        return response.json()
    
    def get_homology(self, identifier: str, target_species: Optional[str] = None,
                    type: str = "all") -> Dict[str, Any]:
        """Get homology information.
        
        Args:
            identifier: Ensembl gene ID
            target_species: Filter by target species
            type: Type of homology (orthologues, paralogues, all)
        
        Returns:
            Homology data
        """
        endpoint = f"/homology/id/{identifier}"
        params = {"type": type}
        
        if target_species:
            params["target_species"] = target_species
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_xrefs(self, identifier: str, external_db: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cross-references.
        
        Args:
            identifier: Ensembl ID
            external_db: Filter by external database
        
        Returns:
            List of cross-references
        """
        endpoint = f"/xrefs/id/{identifier}"
        params = {}
        
        if external_db:
            params["external_db"] = external_db
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_overlap_region(self, species: str, region: str, 
                          feature: List[str] = None) -> List[Dict[str, Any]]:
        """Get features overlapping a region.
        
        Args:
            species: Species name
            region: Genomic region (e.g., "7:140424943-140624564")
            feature: Feature types to include (gene, transcript, cds, exon)
        
        Returns:
            List of overlapping features
        """
        endpoint = f"/overlap/region/{species}/{region}"
        params = {}
        
        if feature:
            params["feature"] = feature
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_regulatory(self, species: str, identifier: str) -> Dict[str, Any]:
        """Get regulatory feature information.
        
        Args:
            species: Species name
            identifier: Regulatory feature ID
        
        Returns:
            Regulatory feature data
        """
        endpoint = f"/regulatory/{species}/{identifier}"
        response = self.get(endpoint)
        return response.json()
    
    def post_lookup_ids(self, identifiers: List[str], species: Optional[str] = None,
                       db_type: Optional[str] = None, expand: bool = False) -> List[Dict[str, Any]]:
        """Batch lookup of multiple IDs.
        
        Args:
            identifiers: List of Ensembl IDs
            species: Species name
            db_type: Database type
            expand: Include additional information
        
        Returns:
            List of lookup results
        """
        endpoint = "/lookup/id"
        data = {"ids": identifiers}
        params = {}
        
        if species:
            params["species"] = species
        if db_type:
            params["db_type"] = db_type
        if expand:
            params["expand"] = 1
        
        response = self.post(endpoint, json=data, params=params)
        return response.json()
    
    def get_species(self) -> List[Dict[str, Any]]:
        """Get all available species.
        
        Returns:
            List of species information
        """
        endpoint = "/info/species"
        response = self.get(endpoint)
        return response.json()["species"]
    
    def get_info_ping(self) -> Dict[str, Any]:
        """Ping the Ensembl REST API.
        
        Returns:
            Service status
        """
        endpoint = "/info/ping"
        response = self.get(endpoint)
        return response.json()
