"""ClinVar API client."""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .base import BaseAPIClient
from ..config.config import settings


logger = logging.getLogger(__name__)


class ClinVarClient(BaseAPIClient):
    """Client for NCBI ClinVar API.
    
    ClinVar is a freely accessible, public archive of reports of the
    relationships among human variations and phenotypes, with supporting evidence.
    
    API Documentation: https://www.ncbi.nlm.nih.gov/clinvar/docs/api_http/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 3),  # NCBI limit: 3 req/sec
            **kwargs
        )
        self.api_key = settings.api.ncbi_api_key
        self.database = "clinvar"
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get ClinVar-specific headers."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json",
        }
    
    def _get_base_params(self) -> Dict[str, str]:
        """Get base parameters for all requests."""
        params = {
            "db": self.database,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    def search(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """Search ClinVar for variants.
        
        Args:
            query: Search query (e.g., "BRCA1[gene]", "pathogenic[clinsig]")
            max_results: Maximum number of results
        
        Returns:
            Search results with variant IDs
        """
        endpoint = "/esearch.fcgi"
        params = self._get_base_params()
        params.update({
            "term": query,
            "retmax": max_results,
            "usehistory": "y"
        })
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_variant_by_id(self, variant_id: Union[str, int]) -> Dict[str, Any]:
        """Get variant details by ClinVar ID.
        
        Args:
            variant_id: ClinVar variant ID
        
        Returns:
            Variant details
        """
        endpoint = "/esummary.fcgi"
        params = self._get_base_params()
        params["id"] = str(variant_id)
        
        response = self.get(endpoint, params=params)
        data = response.json()
        
        # Extract variant from result
        if "result" in data and str(variant_id) in data["result"]:
            return data["result"][str(variant_id)]
        return {}
    
    def get_variants_by_gene(self, gene_symbol: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get all variants for a gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "BRCA1")
            max_results: Maximum number of variants
        
        Returns:
            List of variants
        """
        # Search for variants in gene
        search_results = self.search(f"{gene_symbol}[gene]", max_results)
        
        variants = []
        if "esearchresult" in search_results:
            variant_ids = search_results["esearchresult"].get("idlist", [])
            
            # Get details for each variant
            for variant_id in variant_ids:
                try:
                    variant = self.get_variant_by_id(variant_id)
                    if variant:
                        variants.append(variant)
                except Exception as e:
                    logger.error(f"Failed to get variant {variant_id}: {e}")
        
        return variants
    
    def get_variants_by_rsid(self, rsid: str) -> List[Dict[str, Any]]:
        """Get variants by dbSNP RS ID.
        
        Args:
            rsid: dbSNP RS ID (e.g., "rs80357906")
        
        Returns:
            List of ClinVar variants with this RS ID
        """
        # Search for variants with this rsid
        search_results = self.search(f"{rsid}[snp]")
        
        variants = []
        if "esearchresult" in search_results:
            variant_ids = search_results["esearchresult"].get("idlist", [])
            
            for variant_id in variant_ids:
                try:
                    variant = self.get_variant_by_id(variant_id)
                    if variant:
                        variants.append(variant)
                except Exception as e:
                    logger.error(f"Failed to get variant {variant_id}: {e}")
        
        return variants
    
    def get_pathogenic_variants(self, gene_symbol: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get pathogenic variants for a gene.
        
        Args:
            gene_symbol: Gene symbol
            max_results: Maximum number of variants
        
        Returns:
            List of pathogenic variants
        """
        query = f"{gene_symbol}[gene] AND pathogenic[clinsig]"
        return self._search_and_fetch(query, max_results)
    
    def get_variants_by_disease(self, disease: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get variants associated with a disease.
        
        Args:
            disease: Disease name or OMIM ID
            max_results: Maximum number of variants
        
        Returns:
            List of disease-associated variants
        """
        query = f'"{disease}"[disease]'
        return self._search_and_fetch(query, max_results)
    
    def _search_and_fetch(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Helper to search and fetch full variant details."""
        search_results = self.search(query, max_results)
        
        variants = []
        if "esearchresult" in search_results:
            variant_ids = search_results["esearchresult"].get("idlist", [])
            
            # Batch fetch if many IDs
            if len(variant_ids) > 20:
                # Process in batches
                for i in range(0, len(variant_ids), 20):
                    batch_ids = variant_ids[i:i+20]
                    batch_variants = self._fetch_batch(batch_ids)
                    variants.extend(batch_variants)
            else:
                # Fetch individually
                for variant_id in variant_ids:
                    try:
                        variant = self.get_variant_by_id(variant_id)
                        if variant:
                            variants.append(variant)
                    except Exception as e:
                        logger.error(f"Failed to get variant {variant_id}: {e}")
        
        return variants
    
    def _fetch_batch(self, variant_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple variants in one request."""
        endpoint = "/esummary.fcgi"
        params = self._get_base_params()
        params["id"] = ",".join(str(vid) for vid in variant_ids)
        
        response = self.get(endpoint, params=params)
        data = response.json()
        
        variants = []
        if "result" in data:
            for vid in variant_ids:
                if str(vid) in data["result"]:
                    variants.append(data["result"][str(vid)])
        
        return variants
    
    def parse_variant_summary(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ClinVar variant summary into structured format.
        
        Args:
            variant: Raw variant data from API
        
        Returns:
            Parsed variant information
        """
        parsed = {
            "clinvar_id": variant.get("uid"),
            "title": variant.get("title", ""),
            "gene_symbol": variant.get("gene_symbol", ""),
            "gene_id": variant.get("gene_id"),
            "clinical_significance": variant.get("clinical_significance", {}).get("description", ""),
            "review_status": variant.get("clinical_significance", {}).get("review_status", ""),
            "last_evaluated": variant.get("clinical_significance", {}).get("last_evaluated"),
            "variation_type": variant.get("variation_set", [{}])[0].get("variation_type", ""),
            "molecular_consequence": variant.get("molecular_consequence", ""),
            "protein_change": variant.get("protein_change", ""),
            "variant_name": variant.get("variation_set", [{}])[0].get("variation_name", ""),
            "conditions": [],
            "dbsnp_id": None,
            "genomic_location": None
        }
        
        # Extract conditions
        trait_set = variant.get("trait_set", [])
        for trait in trait_set:
            if "trait_name" in trait:
                parsed["conditions"].append(trait["trait_name"])
        
        # Extract dbSNP ID
        variation_xrefs = variant.get("variation_set", [{}])[0].get("variation_xrefs", [])
        for xref in variation_xrefs:
            if xref.get("db_source") == "dbSNP":
                parsed["dbsnp_id"] = xref.get("db_id")
        
        # Extract genomic location
        variation_loc = variant.get("variation_set", [{}])[0].get("variation_loc", [])
        if variation_loc:
            loc = variation_loc[0]
            parsed["genomic_location"] = {
                "chromosome": loc.get("chr"),
                "start": loc.get("start"),
                "stop": loc.get("stop"),
                "assembly": loc.get("assembly")
            }
        
        return parsed
