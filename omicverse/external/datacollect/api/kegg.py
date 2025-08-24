"""KEGG API client."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseAPIClient
from ..config.config import settings


logger = logging.getLogger(__name__)


class KEGGClient(BaseAPIClient):
    """Client for KEGG REST API.
    
    KEGG (Kyoto Encyclopedia of Genes and Genomes) is a database resource
    for understanding high-level functions and utilities of biological systems.
    
    API Documentation: https://www.kegg.jp/kegg/rest/keggapi.html
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://rest.kegg.jp")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get KEGG-specific headers."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "text/plain",  # KEGG returns plain text
        }
    
    def get(self, endpoint: str, **kwargs) -> Any:
        """Override get to handle text responses."""
        response = super().get(endpoint, **kwargs)
        # KEGG returns text, not JSON
        return response
    
    def info(self, database: str) -> Dict[str, str]:
        """Get information about a KEGG database.
        
        Args:
            database: Database name (e.g., "pathway", "compound")
        
        Returns:
            Database information
        """
        endpoint = f"/info/{database}"
        response = self.get(endpoint)
        
        # Parse text response
        info = {}
        for line in response.text.strip().split('\n'):
            if '\t' in line:
                key, value = line.split('\t', 1)
                info[key] = value
        
        return info
    
    def get_entry(self, kegg_id: str) -> Dict[str, Any]:
        """Get a KEGG entry.
        
        Args:
            kegg_id: KEGG ID (e.g., "hsa:7157" for human TP53)
        
        Returns:
            Entry data
        """
        endpoint = f"/get/{kegg_id}"
        response = self.get(endpoint)
        return self._parse_entry(response.text)
    
    def find(self, database: str, query: str) -> List[Dict[str, str]]:
        """Find entries in a KEGG database.
        
        Args:
            database: Database to search (e.g., "genes", "pathway")
            query: Search query
        
        Returns:
            List of matching entries
        """
        endpoint = f"/find/{database}/{query}"
        response = self.get(endpoint)
        
        results = []
        for line in response.text.strip().split('\n'):
            if '\t' in line:
                entry_id, description = line.split('\t', 1)
                results.append({
                    "entry_id": entry_id,
                    "description": description
                })
        
        return results
    
    def link(self, target_db: str, source_db: str, source_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Get linked entries between databases.
        
        Args:
            target_db: Target database
            source_db: Source database
            source_id: Optional specific source ID
        
        Returns:
            List of links
        """
        if source_id:
            endpoint = f"/link/{target_db}/{source_id}"
        else:
            endpoint = f"/link/{target_db}/{source_db}"
            
        response = self.get(endpoint)
        
        links = []
        for line in response.text.strip().split('\n'):
            if '\t' in line:
                source, target = line.split('\t')
                links.append({
                    "source": source,
                    "target": target
                })
        
        return links
    
    def get_pathway(self, pathway_id: str) -> Dict[str, Any]:
        """Get pathway information.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa04110")
        
        Returns:
            Pathway data
        """
        return self.get_entry(pathway_id)
    
    def get_gene(self, gene_id: str) -> Dict[str, Any]:
        """Get gene information.
        
        Args:
            gene_id: KEGG gene ID (e.g., "hsa:7157")
        
        Returns:
            Gene data
        """
        return self.get_entry(gene_id)
    
    def conv(self, target_db: str, source_db: str, ids: Optional[List[str]] = None) -> Dict[str, str]:
        """Convert IDs between databases.
        
        Args:
            target_db: Target database (e.g., "uniprot")
            source_db: Source database (e.g., "hsa")
            ids: Optional list of IDs to convert
        
        Returns:
            ID mapping
        """
        if ids:
            # Convert specific IDs
            endpoint = f"/conv/{target_db}/{'+'.join(ids)}"
        else:
            # Convert all entries
            endpoint = f"/conv/{target_db}/{source_db}"
            
        response = self.get(endpoint)
        
        mapping = {}
        for line in response.text.strip().split('\n'):
            if '\t' in line:
                source, target = line.split('\t')
                mapping[source] = target
        
        return mapping
    
    def get_pathways_by_gene(self, gene_id: str) -> List[Dict[str, str]]:
        """Get pathways containing a gene.
        
        Args:
            gene_id: KEGG gene ID
        
        Returns:
            List of pathways
        """
        return self.link("pathway", "genes", gene_id)
    
    def get_genes_by_pathway(self, pathway_id: str) -> List[Dict[str, str]]:
        """Get genes in a pathway.
        
        Args:
            pathway_id: KEGG pathway ID
        
        Returns:
            List of genes
        """
        return self.link("genes", "pathway", pathway_id)
    
    def get_compounds_by_pathway(self, pathway_id: str) -> List[Dict[str, str]]:
        """Get compounds in a pathway.
        
        Args:
            pathway_id: KEGG pathway ID
        
        Returns:
            List of compounds
        """
        return self.link("compound", "pathway", pathway_id)
    
    def _parse_entry(self, text: str) -> Dict[str, Any]:
        """Parse KEGG entry text format."""
        entry = {}
        current_key = None
        current_value = []
        
        for line in text.strip().split('\n'):
            if line.startswith('///'):
                break
                
            if line.startswith(' '):
                # Continuation of previous field
                if current_key:
                    current_value.append(line.strip())
            else:
                # New field
                if current_key:
                    entry[current_key] = '\n'.join(current_value) if len(current_value) > 1 else current_value[0] if current_value else ''
                
                if ' ' in line:
                    # Split only on first space to preserve spacing in value
                    first_space = line.index(' ')
                    current_key = line[:first_space]
                    # Trim leading spaces but preserve internal spacing
                    current_value = [line[first_space+1:].lstrip()]
                else:
                    current_key = line
                    current_value = []
        
        # Don't forget the last field
        if current_key:
            entry[current_key] = '\n'.join(current_value) if len(current_value) > 1 else current_value[0] if current_value else ''
        
        return entry
