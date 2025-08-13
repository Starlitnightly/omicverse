"""dbSNP API client."""

import logging
from typing import Any, Dict, List, Optional, Union
import json

from .base import BaseAPIClient
from config.config import settings


logger = logging.getLogger(__name__)


class dbSNPClient(BaseAPIClient):
    """Client for NCBI dbSNP API.
    
    dbSNP is a public-domain archive for a broad collection of simple genetic
    polymorphisms including single-nucleotide polymorphisms (SNPs), insertions/deletions,
    microsatellites, and small-scale variations.
    
    API Documentation: https://api.ncbi.nlm.nih.gov/variation/v0/
    """
    
    def __init__(self, **kwargs):
        # Two different endpoints - E-utilities and Variation Services API
        base_url = kwargs.pop("base_url", "https://api.ncbi.nlm.nih.gov/variation/v0")
        self.api_key = settings.api.ncbi_api_key
        self.eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.database = "snp"
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 10),  # NCBI allows 10 req/sec with API key
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get dbSNP-specific headers."""
        headers = {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers
    
    def get_variant_by_rsid(self, rsid: str) -> Dict[str, Any]:
        """Get variant information by RS ID.
        
        Args:
            rsid: dbSNP RS ID (e.g., "rs7412" or just "7412")
        
        Returns:
            Variant information
        """
        # Clean RS ID
        if rsid.lower().startswith("rs"):
            rsid = rsid[2:]
        
        endpoint = f"/beta/refsnp/{rsid}"
        
        response = self.get(endpoint)
        return response.json()
    
    def get_variant_allele_annotations(self, rsid: str) -> List[Dict[str, Any]]:
        """Get allele-specific annotations for a variant.
        
        Args:
            rsid: dbSNP RS ID
        
        Returns:
            List of allele annotations
        """
        if rsid.lower().startswith("rs"):
            rsid = rsid[2:]
        
        endpoint = f"/beta/refsnp/{rsid}/alleles"
        
        response = self.get(endpoint)
        data = response.json()
        
        return data.get("alleles", [])
    
    def search_by_gene(self, gene_symbol: str, organism: str = "human",
                      max_results: int = 100) -> List[str]:
        """Search for variants in a gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "APOE")
            organism: Organism (default: human)
            max_results: Maximum number of results
        
        Returns:
            List of RS IDs
        """
        # Use E-utilities for search
        search_url = f"{self.eutils_base}/esearch.fcgi"
        params = {
            "db": self.database,
            "term": f"{gene_symbol}[Gene Name] AND {organism}[Organism]",
            "retmax": max_results,
            "retmode": "json"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self.session.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        rs_ids = []
        
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            # IDs returned are internal IDs, need to convert to RS IDs
            id_list = data["esearchresult"]["idlist"]
            
            if id_list:
                # Fetch summaries to get RS IDs
                summary_url = f"{self.eutils_base}/esummary.fcgi"
                summary_params = {
                    "db": self.database,
                    "id": ",".join(id_list[:20]),  # Limit to 20 at a time
                    "retmode": "json"
                }
                
                if self.api_key:
                    summary_params["api_key"] = self.api_key
                
                summary_response = self.session.get(summary_url, params=summary_params)
                summary_data = summary_response.json()
                
                if "result" in summary_data:
                    for uid in summary_data["result"].get("uids", []):
                        if uid in summary_data["result"]:
                            snp_data = summary_data["result"][uid]
                            if "snp_id" in snp_data:
                                rs_ids.append(f"rs{snp_data['snp_id']}")
        
        return rs_ids
    
    def search_by_position(self, chromosome: str, start: int, end: int,
                          organism: str = "human") -> List[str]:
        """Search for variants by genomic position.
        
        Args:
            chromosome: Chromosome (e.g., "7", "X")
            start: Start position
            end: End position
            organism: Organism
        
        Returns:
            List of RS IDs
        """
        # Use E-utilities search with position
        if organism.lower() == "human":
            assembly = "GRCh38"
        else:
            assembly = ""
        
        search_term = f"{chromosome}[CHR] AND {start}:{end}[CHRPOS]"
        if assembly:
            search_term += f" AND {assembly}[Assembly]"
        
        search_url = f"{self.eutils_base}/esearch.fcgi"
        params = {
            "db": self.database,
            "term": search_term,
            "retmax": 1000,
            "retmode": "json"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self.session.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        rs_ids = []
        
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            # Convert internal IDs to RS IDs as above
            id_list = data["esearchresult"]["idlist"]
            
            # Process in batches
            for i in range(0, len(id_list), 20):
                batch_ids = id_list[i:i+20]
                rs_batch = self._convert_ids_to_rsids(batch_ids)
                rs_ids.extend(rs_batch)
        
        return rs_ids
    
    def get_clinical_significance(self, rsid: str) -> List[Dict[str, Any]]:
        """Get clinical significance annotations.
        
        Args:
            rsid: dbSNP RS ID
        
        Returns:
            List of clinical annotations
        """
        variant_data = self.get_variant_by_rsid(rsid)
        
        clinical_data = []
        
        # Extract ClinVar data if present
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            # Look for allele annotations
            for allele_ann in snapshot.get("allele_annotations", []):
                for assembly_ann in allele_ann.get("assembly_annotation", []):
                    for gene in assembly_ann.get("genes", []):
                        # Clinical annotations may be in gene annotations
                        if "clinical_significance" in gene:
                            clinical_data.append({
                                "gene": gene.get("name", ""),
                                "clinical_significance": gene["clinical_significance"],
                                "review_status": gene.get("review_status", "")
                            })
        
        return clinical_data
    
    def get_population_frequency(self, rsid: str) -> Dict[str, Any]:
        """Get population frequency data.
        
        Args:
            rsid: dbSNP RS ID
        
        Returns:
            Population frequency information
        """
        variant_data = self.get_variant_by_rsid(rsid)
        
        frequency_data = {
            "rsid": f"rs{rsid}" if not rsid.startswith("rs") else rsid,
            "global_frequency": None,
            "population_frequencies": {}
        }
        
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            # Look for frequency data
            for allele_ann in snapshot.get("allele_annotations", []):
                for freq in allele_ann.get("frequency", []):
                    study = freq.get("study_name", "Unknown")
                    
                    # Store frequency by population
                    for pop_freq in freq.get("populations", []):
                        pop_name = pop_freq.get("population", "")
                        allele_count = pop_freq.get("allele_count", 0)
                        total_count = pop_freq.get("total_count", 0)
                        
                        if total_count > 0:
                            freq_value = allele_count / total_count
                            
                            if pop_name not in frequency_data["population_frequencies"]:
                                frequency_data["population_frequencies"][pop_name] = {}
                            
                            frequency_data["population_frequencies"][pop_name][study] = {
                                "frequency": freq_value,
                                "allele_count": allele_count,
                                "total_count": total_count
                            }
                            
                            # Set global frequency from largest study
                            if study == "1000Genomes" and pop_name == "Total":
                                frequency_data["global_frequency"] = freq_value
        
        return frequency_data
    
    def get_variant_consequences(self, rsid: str) -> List[Dict[str, Any]]:
        """Get predicted consequences of a variant.
        
        Args:
            rsid: dbSNP RS ID
        
        Returns:
            List of predicted consequences
        """
        variant_data = self.get_variant_by_rsid(rsid)
        consequences = []
        
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            for placement in snapshot.get("placements_with_allele", []):
                for allele in placement.get("alleles", []):
                    # Extract consequence information
                    hgvs = allele.get("hgvs", "")
                    
                    for assembly_ann in placement.get("placement_annot", {}).get("seq_id_traits_by_assembly", []):
                        for gene_info in assembly_ann.get("genes", []):
                            gene_name = gene_info.get("name", "")
                            gene_id = gene_info.get("id", "")
                            
                            # Get consequence type
                            consequence_type = gene_info.get("consequence_type", "")
                            if not consequence_type and "rnas" in gene_info:
                                # Check RNA consequences
                                for rna in gene_info["rnas"]:
                                    if "consequence_type" in rna:
                                        consequence_type = rna["consequence_type"]
                            
                            if gene_name or consequence_type:
                                consequences.append({
                                    "gene_name": gene_name,
                                    "gene_id": gene_id,
                                    "consequence_type": consequence_type,
                                    "hgvs": hgvs,
                                    "assembly": assembly_ann.get("assembly_name", "")
                                })
        
        return consequences
    
    def _convert_ids_to_rsids(self, internal_ids: List[str]) -> List[str]:
        """Convert internal dbSNP IDs to RS IDs."""
        rs_ids = []
        
        summary_url = f"{self.eutils_base}/esummary.fcgi"
        summary_params = {
            "db": self.database,
            "id": ",".join(internal_ids),
            "retmode": "json"
        }
        
        if self.api_key:
            summary_params["api_key"] = self.api_key
        
        try:
            summary_response = self.session.get(summary_url, params=summary_params)
            summary_data = summary_response.json()
            
            if "result" in summary_data:
                for uid in summary_data["result"].get("uids", []):
                    if uid in summary_data["result"]:
                        snp_data = summary_data["result"][uid]
                        if "snp_id" in snp_data:
                            rs_ids.append(f"rs{snp_data['snp_id']}")
        except Exception as e:
            logger.error(f"Failed to convert IDs to RS IDs: {e}")
        
        return rs_ids