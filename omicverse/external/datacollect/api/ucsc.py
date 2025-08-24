"""UCSC Genome Browser API client."""

import logging
from typing import Any, Dict, List, Optional
import json

from .base import BaseAPIClient
from ..config.config import settings


logger = logging.getLogger(__name__)


class UCSCClient(BaseAPIClient):
    """Client for UCSC Genome Browser REST API.
    
    The UCSC Genome Browser provides access to genome assemblies and 
    annotations for vertebrate and selected model organisms.
    
    API Documentation: https://api.genome.ucsc.edu/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://api.genome.ucsc.edu")
        rate_limit = kwargs.pop("rate_limit", 10)
        super().__init__(
            base_url=base_url,
            rate_limit=rate_limit,
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get UCSC-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def list_genomes(self) -> Dict[str, Any]:
        """List available genome assemblies.
        
        Returns:
            Dictionary of available genomes
        """
        endpoint = "/list/ucscGenomes"
        response = self.get(endpoint)
        return response.json()
    
    def get_chromosomes(self, genome: str) -> Dict[str, Any]:
        """Get chromosome information for a genome.
        
        Args:
            genome: Genome assembly (e.g., "hg38", "mm10")
        
        Returns:
            Chromosome information
        """
        endpoint = f"/list/chromosomes"
        params = {"genome": genome}
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_tracks(self, genome: str, track: Optional[str] = None) -> Dict[str, Any]:
        """Get track information for a genome.
        
        Args:
            genome: Genome assembly
            track: Optional specific track name
        
        Returns:
            Track information
        """
        endpoint = "/list/tracks"
        params = {"genome": genome}
        if track:
            params["track"] = track
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_data(self, genome: str, track: str, chrom: str, 
                 start: int, end: int) -> Dict[str, Any]:
        """Get track data for a genomic region.
        
        Args:
            genome: Genome assembly
            track: Track name
            chrom: Chromosome
            start: Start position
            end: End position
        
        Returns:
            Track data for the region
        """
        endpoint = "/getData/track"
        params = {
            "genome": genome,
            "track": track,
            "chrom": chrom,
            "start": start,
            "end": end
        }
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_sequence(self, genome: str, chrom: str, start: int, end: int) -> Dict[str, Any]:
        """Get DNA sequence for a genomic region.
        
        Args:
            genome: Genome assembly
            chrom: Chromosome
            start: Start position
            end: End position
        
        Returns:
            DNA sequence
        """
        endpoint = "/getData/sequence"
        params = {
            "genome": genome,
            "chrom": chrom,
            "start": start,
            "end": end
        }
        response = self.get(endpoint, params=params)
        return response.json()
    
    def search(self, genome: str, query: str, type: Optional[str] = None) -> Dict[str, Any]:
        """Search for genomic features.
        
        Args:
            genome: Genome assembly
            query: Search query (gene name, position, etc.)
            type: Optional search type filter
        
        Returns:
            Search results
        """
        endpoint = "/search"
        params = {
            "genome": genome,
            "query": query
        }
        if type:
            params["type"] = type
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_gene_predictions(self, genome: str, gene_name: str) -> Dict[str, Any]:
        """Get gene predictions/annotations.
        
        Args:
            genome: Genome assembly
            gene_name: Gene name
        
        Returns:
            Gene predictions
        """
        search_result = self.search(genome, gene_name, type="gene")
        
        if not search_result.get("matches"):
            return {"error": f"Gene {gene_name} not found"}
        
        # Get detailed gene information
        gene_info = []
        for match in search_result["matches"]:
            if "position" in match:
                try:
                    chrom, positions = match["position"].split(":")
                    start, end = positions.split("-")
                    
                    # Get gene track data
                    gene_data = self.get_data(
                        genome, "ncbiRefSeqCurated", 
                        chrom, int(start), int(end)
                    )
                    gene_info.append(gene_data)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Malformed position data: {match['position']}")
                    continue
        
        return {
            "gene": gene_name,
            "genome": genome,
            "predictions": gene_info
        }
    
    def get_snp_data(self, genome: str, chrom: str, position: int,
                     window: int = 100) -> Dict[str, Any]:
        """Get SNP data around a position.
        
        Args:
            genome: Genome assembly
            chrom: Chromosome
            position: Position
            window: Window size around position
        
        Returns:
            SNP data
        """
        start = max(0, position - window)
        end = position + window
        
        # Try common SNP tracks
        snp_tracks = ["snp151", "snp150", "snp144", "snp142"]
        
        for track in snp_tracks:
            try:
                data = self.get_data(genome, track, chrom, start, end)
                if data and not data.get("error"):
                    return data
            except Exception:
                continue
        
        return {"error": "No SNP data available for this region"}
    
    def get_conservation_scores(self, genome: str, chrom: str,
                               start: int, end: int) -> Dict[str, Any]:
        """Get conservation scores for a region.
        
        Args:
            genome: Genome assembly
            chrom: Chromosome  
            start: Start position
            end: End position
        
        Returns:
            Conservation scores
        """
        # Try phyloP and phastCons tracks
        conservation_data = {}
        
        try:
            phylop = self.get_data(genome, "phyloP100way", chrom, start, end)
            conservation_data["phyloP"] = phylop
        except Exception:
            pass
        
        try:
            phastcons = self.get_data(genome, "phastCons100way", chrom, start, end)
            conservation_data["phastCons"] = phastcons
        except Exception:
            pass
        
        return conservation_data
    
    def get_regulatory_elements(self, genome: str, chrom: str,
                               start: int, end: int) -> Dict[str, Any]:
        """Get regulatory elements in a region.
        
        Args:
            genome: Genome assembly
            chrom: Chromosome
            start: Start position
            end: End position
        
        Returns:
            Regulatory elements
        """
        regulatory_data = {}
        
        # ENCODE regulatory tracks
        tracks = [
            "encRegTfbsClustered",
            "encRegDnaseClustered",
            "encRegH3K27Ac",
            "encRegH3K4Me1",
            "encRegH3K4Me3"
        ]
        
        for track in tracks:
            try:
                data = self.get_data(genome, track, chrom, start, end)
                if data and not data.get("error"):
                    regulatory_data[track] = data
            except Exception:
                continue
        
        return regulatory_data
