"""GWAS Catalog API client."""

import logging
from typing import Any, Dict, List, Optional
import json

from .base import BaseAPIClient
from ..config import settings


logger = logging.getLogger(__name__)


class GWASCatalogClient(BaseAPIClient):
    """Client for GWAS Catalog REST API.
    
    The GWAS Catalog provides a curated collection of all published 
    genome-wide association studies, enabling researchers to identify 
    causal variants, understand disease mechanisms, and establish targets for treatment.
    
    API Documentation: https://www.ebi.ac.uk/gwas/docs/api
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://www.ebi.ac.uk/gwas/rest/api")
        rate_limit = kwargs.pop("rate_limit", 10)
        super().__init__(
            base_url=base_url,
            rate_limit=rate_limit,
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get GWAS Catalog-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def get_study(self, study_accession: str) -> Dict[str, Any]:
        """Get study information.
        
        Args:
            study_accession: Study accession (e.g., GCST000001)
        
        Returns:
            Study information
        """
        endpoint = f"/studies/{study_accession}"
        response = self.get(endpoint)
        return response.json()
    
    def search_studies(self, query: str, page: int = 0, size: int = 20) -> Dict[str, Any]:
        """Search for studies.
        
        Args:
            query: Search query
            page: Page number
            size: Page size
        
        Returns:
            Search results
        """
        endpoint = "/studies/search"
        params = {
            "q": query,
            "page": page,
            "size": size
        }
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_associations(self, study_id: Optional[str] = None,
                        trait: Optional[str] = None,
                        gene: Optional[str] = None,
                        variant: Optional[str] = None,
                        page: int = 0, size: int = 20) -> Dict[str, Any]:
        """Get associations with various filters.
        
        Args:
            study_id: Study accession
            trait: Trait name
            gene: Gene name
            variant: Variant rsID
            page: Page number
            size: Page size
        
        Returns:
            Association data
        """
        endpoint = "/associations"
        params = {
            "page": page,
            "size": size
        }
        
        if study_id:
            params["studyAccessionId"] = study_id
        if trait:
            params["efoTrait"] = trait
        if gene:
            params["geneName"] = gene
        if variant:
            params["rsId"] = variant
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_association(self, association_id: str) -> Dict[str, Any]:
        """Get specific association.
        
        Args:
            association_id: Association ID
        
        Returns:
            Association details
        """
        endpoint = f"/associations/{association_id}"
        response = self.get(endpoint)
        return response.json()
    
    def get_single_nucleotide_polymorphisms(self, rsid: str) -> Dict[str, Any]:
        """Get SNP information.
        
        Args:
            rsid: SNP rsID
        
        Returns:
            SNP information
        """
        endpoint = f"/singleNucleotidePolymorphisms/{rsid}"
        response = self.get(endpoint)
        return response.json()
    
    def search_snps(self, query: str, page: int = 0, size: int = 20) -> Dict[str, Any]:
        """Search for SNPs.
        
        Args:
            query: Search query (rsID or genomic location)
            page: Page number
            size: Page size
        
        Returns:
            SNP search results
        """
        endpoint = "/singleNucleotidePolymorphisms/search"
        params = {
            "q": query,
            "page": page,
            "size": size
        }
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_trait(self, trait_uri: str) -> Dict[str, Any]:
        """Get trait/disease information.
        
        Args:
            trait_uri: EFO trait URI
        
        Returns:
            Trait information
        """
        # Convert URI to ID if needed
        if trait_uri.startswith("http"):
            trait_id = trait_uri.split("/")[-1]
        else:
            trait_id = trait_uri
        
        endpoint = f"/efoTraits/{trait_id}"
        response = self.get(endpoint)
        return response.json()
    
    def search_traits(self, query: str, page: int = 0, size: int = 20) -> Dict[str, Any]:
        """Search for traits/diseases.
        
        Args:
            query: Search query
            page: Page number
            size: Page size
        
        Returns:
            Trait search results
        """
        endpoint = "/efoTraits/search"
        params = {
            "q": query,
            "page": page,
            "size": size
        }
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_genes_by_symbol(self, gene_symbol: str) -> Dict[str, Any]:
        """Get gene information by symbol.
        
        Args:
            gene_symbol: Gene symbol
        
        Returns:
            Gene information
        """
        endpoint = "/genes/search/findByGeneName"
        params = {"geneName": gene_symbol}
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_ancestry(self, ancestry_id: str) -> Dict[str, Any]:
        """Get ancestry information.
        
        Args:
            ancestry_id: Ancestry ID
        
        Returns:
            Ancestry information
        """
        endpoint = f"/ancestries/{ancestry_id}"
        response = self.get(endpoint)
        return response.json()
    
    def get_publications(self, pmid: str) -> Dict[str, Any]:
        """Get publication information.
        
        Args:
            pmid: PubMed ID
        
        Returns:
            Publication information
        """
        endpoint = f"/publications/{pmid}"
        response = self.get(endpoint)
        return response.json()
    
    def get_genomic_contexts(self, chromosome: str, start: int, end: int,
                           page: int = 0, size: int = 20) -> Dict[str, Any]:
        """Get associations in a genomic region.
        
        Args:
            chromosome: Chromosome number
            start: Start position
            end: End position
            page: Page number
            size: Page size
        
        Returns:
            Associations in the region
        """
        endpoint = "/genomicContexts"
        params = {
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "page": page,
            "size": size
        }
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_risk_alleles(self, association_id: str) -> Dict[str, Any]:
        """Get risk alleles for an association.
        
        Args:
            association_id: Association ID
        
        Returns:
            Risk allele information
        """
        endpoint = f"/associations/{association_id}/riskAlleles"
        response = self.get(endpoint)
        return response.json()
    
    def get_loci(self, association_id: str) -> Dict[str, Any]:
        """Get loci for an association.
        
        Args:
            association_id: Association ID
        
        Returns:
            Loci information
        """
        endpoint = f"/associations/{association_id}/loci"
        response = self.get(endpoint)
        return response.json()
    
    def get_summary_statistics(self, study_accession: str) -> Dict[str, Any]:
        """Get summary statistics download links for a study.
        
        Args:
            study_accession: Study accession
        
        Returns:
            Summary statistics information
        """
        endpoint = f"/studies/{study_accession}/associations"
        params = {"projection": "summaryStatistics"}
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_top_associations_by_trait(self, trait: str, p_value_threshold: float = 5e-8,
                                     size: int = 100) -> Dict[str, Any]:
        """Get top associations for a trait.
        
        Args:
            trait: EFO trait
            p_value_threshold: P-value threshold
            size: Maximum number of results
        
        Returns:
            Top associations
        """
        endpoint = "/associations/search/findByEfoTrait"
        params = {
            "efoTrait": trait,
            "pvalueMantissaLessThan": int(p_value_threshold * 1e9) / 1e9,
            "size": size,
            "sort": "pvalueMantissa,asc"
        }
        response = self.get(endpoint, params=params)
        return response.json()
