"""RegulomeDB API client."""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class RegulomeDBClient(BaseAPIClient):
    """Client for RegulomeDB regulatory variants API.
    
    RegulomeDB is a database that annotates SNPs with known and predicted
    regulatory elements in the intergenic regions of the human genome.
    
    API Documentation: https://regulomedb.org/regulome-help/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://regulomedb.org/regulome-search")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 5),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get RegulomeDB-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def query_variant(self, chromosome: str, position: int,
                     genome_assembly: str = "GRCh38") -> Dict[str, Any]:
        """Query regulatory information for a variant.
        
        Args:
            chromosome: Chromosome (e.g., 'chr1' or '1')
            position: Genomic position
            genome_assembly: Genome assembly (GRCh37 or GRCh38)
        
        Returns:
            Regulatory annotation data
        """
        # Ensure chromosome format
        if not chromosome.startswith("chr"):
            chromosome = f"chr{chromosome}"
        
        params = {
            "regions": f"{chromosome}:{position}-{position}",
            "genome": genome_assembly,
            "format": "json"
        }
        
        response = self.session.get(
            self.base_url,
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def query_rsid(self, rsid: str, genome_assembly: str = "GRCh38") -> Dict[str, Any]:
        """Query regulatory information by rsID.
        
        Args:
            rsid: dbSNP rsID (e.g., 'rs12345')
            genome_assembly: Genome assembly
        
        Returns:
            Regulatory annotation data
        """
        # Ensure rsid format
        if not rsid.startswith("rs"):
            rsid = f"rs{rsid}"
        
        params = {
            "regions": rsid,
            "genome": genome_assembly,
            "format": "json"
        }
        
        response = self.session.get(
            self.base_url,
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def query_region(self, chromosome: str, start: int, end: int,
                    genome_assembly: str = "GRCh38") -> Dict[str, Any]:
        """Query regulatory variants in a genomic region.
        
        Args:
            chromosome: Chromosome
            start: Start position
            end: End position
            genome_assembly: Genome assembly
        
        Returns:
            List of regulatory variants in region
        """
        # Ensure chromosome format
        if not chromosome.startswith("chr"):
            chromosome = f"chr{chromosome}"
        
        params = {
            "regions": f"{chromosome}:{start}-{end}",
            "genome": genome_assembly,
            "format": "json",
            "limit": 1000  # Maximum variants to return
        }
        
        response = self.session.get(
            self.base_url,
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def batch_query(self, variants: List[Dict[str, Any]],
                   genome_assembly: str = "GRCh38") -> List[Dict[str, Any]]:
        """Query multiple variants.
        
        Args:
            variants: List of variant dictionaries with 'chr' and 'pos' keys
            genome_assembly: Genome assembly
        
        Returns:
            List of regulatory annotations
        """
        regions = []
        for variant in variants:
            chr = variant.get("chr", variant.get("chromosome"))
            pos = variant.get("pos", variant.get("position"))
            if not chr.startswith("chr"):
                chr = f"chr{chr}"
            regions.append(f"{chr}:{pos}-{pos}")
        
        params = {
            "regions": " ".join(regions),
            "genome": genome_assembly,
            "format": "json"
        }
        
        response = self.session.get(
            self.base_url,
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_regulatory_score(self, chromosome: str, position: int,
                           genome_assembly: str = "GRCh38") -> Dict[str, Any]:
        """Get RegulomeDB score for a variant.
        
        Args:
            chromosome: Chromosome
            position: Position
            genome_assembly: Genome assembly
        
        Returns:
            RegulomeDB score and evidence
        """
        result = self.query_variant(chromosome, position, genome_assembly)
        
        if not result or "@graph" not in result:
            return {
                "score": None,
                "evidence": [],
                "message": "No regulatory data found"
            }
        
        variants = result.get("@graph", [])
        if not variants:
            return {
                "score": None,
                "evidence": [],
                "message": "No regulatory data found"
            }
        
        variant_data = variants[0]
        
        return {
            "score": variant_data.get("regulome_score", {}).get("ranking"),
            "probability": variant_data.get("regulome_score", {}).get("probability"),
            "evidence_count": len(variant_data.get("peaks", [])),
            "peaks": variant_data.get("peaks", []),
            "motifs": variant_data.get("motifs", []),
            "qtls": variant_data.get("qtls", []),
            "chromatin_states": variant_data.get("chromatin_states", [])
        }
    
    def search_motifs(self, motif_name: str) -> List[Dict[str, Any]]:
        """Search for regulatory motifs.
        
        Args:
            motif_name: Motif name or pattern
        
        Returns:
            List of matching motifs
        """
        params = {
            "type": "motif",
            "search": motif_name,
            "format": "json"
        }
        
        response = self.session.get(
            f"{self.base_url}/search",
            params=params,
            headers=self.get_default_headers()
        )
        
        if response.status_code == 404:
            return []
        
        response.raise_for_status()
        return response.json()
    
    def get_peaks_at_position(self, chromosome: str, position: int,
                             genome_assembly: str = "GRCh38") -> List[Dict[str, Any]]:
        """Get chromatin accessibility peaks at a position.
        
        Args:
            chromosome: Chromosome
            position: Position
            genome_assembly: Genome assembly
        
        Returns:
            List of peaks
        """
        result = self.query_variant(chromosome, position, genome_assembly)
        
        if not result or "@graph" not in result:
            return []
        
        variants = result.get("@graph", [])
        if not variants:
            return []
        
        return variants[0].get("peaks", [])
    
    def get_qtls_at_position(self, chromosome: str, position: int,
                            genome_assembly: str = "GRCh38") -> List[Dict[str, Any]]:
        """Get QTLs at a position.
        
        Args:
            chromosome: Chromosome
            position: Position
            genome_assembly: Genome assembly
        
        Returns:
            List of QTLs
        """
        result = self.query_variant(chromosome, position, genome_assembly)
        
        if not result or "@graph" not in result:
            return []
        
        variants = result.get("@graph", [])
        if not variants:
            return []
        
        return variants[0].get("qtls", [])
    
    def get_chromatin_states(self, chromosome: str, position: int,
                            genome_assembly: str = "GRCh38") -> List[Dict[str, Any]]:
        """Get chromatin states at a position.
        
        Args:
            chromosome: Chromosome
            position: Position
            genome_assembly: Genome assembly
        
        Returns:
            List of chromatin states
        """
        result = self.query_variant(chromosome, position, genome_assembly)
        
        if not result or "@graph" not in result:
            return []
        
        variants = result.get("@graph", [])
        if not variants:
            return []
        
        return variants[0].get("chromatin_states", [])