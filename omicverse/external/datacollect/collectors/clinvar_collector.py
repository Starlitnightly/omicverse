"""ClinVar data collector."""

import logging
import json
from typing import Any, Dict, List, Optional, Union

from omicverse.external.datacollect.api.clinvar import ClinVarClient
from omicverse.external.datacollect.models.genomic import Variant, Gene
from .base import BaseCollector
from ..config import settings


logger = logging.getLogger(__name__)


class ClinVarCollector(BaseCollector):
    """Collector for ClinVar clinical variant data."""
    
    def __init__(self, db_session=None):
        api_client = ClinVarClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, variant_id: Union[str, int]) -> Dict[str, Any]:
        """Collect data for a single ClinVar variant.
        
        Args:
            variant_id: ClinVar variant ID
        
        Returns:
            Collected variant data
        """
        logger.info(f"Collecting ClinVar variant {variant_id}")
        
        # Get variant details
        raw_variant = self.api_client.get_variant_by_id(variant_id)
        
        if not raw_variant:
            raise ValueError(f"Variant {variant_id} not found in ClinVar")
        
        # Parse variant data
        variant_data = self.api_client.parse_variant_summary(raw_variant)
        
        # Add raw data
        variant_data["raw_data"] = raw_variant
        
        return variant_data
    
    def collect_by_gene(self, gene_symbol: str, pathogenic_only: bool = False,
                       max_results: int = 100) -> List[Dict[str, Any]]:
        """Collect variants for a gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "BRCA1")
            pathogenic_only: Only collect pathogenic variants
            max_results: Maximum number of variants
        
        Returns:
            List of variant data
        """
        logger.info(f"Collecting ClinVar variants for gene {gene_symbol}")
        
        if pathogenic_only:
            raw_variants = self.api_client.get_pathogenic_variants(gene_symbol, max_results)
        else:
            raw_variants = self.api_client.get_variants_by_gene(gene_symbol, max_results)
        
        variants = []
        for raw_variant in raw_variants:
            try:
                variant_data = self.api_client.parse_variant_summary(raw_variant)
                variant_data["raw_data"] = raw_variant
                variants.append(variant_data)
            except Exception as e:
                logger.error(f"Failed to parse variant: {e}")
        
        logger.info(f"Collected {len(variants)} variants for {gene_symbol}")
        return variants
    
    def collect_by_disease(self, disease: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Collect variants associated with a disease.
        
        Args:
            disease: Disease name or OMIM ID
            max_results: Maximum number of variants
        
        Returns:
            List of variant data
        """
        logger.info(f"Collecting ClinVar variants for disease '{disease}'")
        
        raw_variants = self.api_client.get_variants_by_disease(disease, max_results)
        
        variants = []
        for raw_variant in raw_variants:
            try:
                variant_data = self.api_client.parse_variant_summary(raw_variant)
                variant_data["raw_data"] = raw_variant
                variants.append(variant_data)
            except Exception as e:
                logger.error(f"Failed to parse variant: {e}")
        
        logger.info(f"Collected {len(variants)} variants for disease '{disease}'")
        return variants
    
    def save_to_database(self, data: Dict[str, Any]) -> Variant:
        """Save ClinVar variant to database.
        
        Args:
            data: Parsed variant data
        
        Returns:
            Saved Variant instance
        """
        clinvar_id = str(data["clinvar_id"])
        
        # Check if variant exists
        existing = self.db_session.query(Variant).filter_by(
            variant_id=f"clinvar:{clinvar_id}"
        ).first()
        
        if existing:
            logger.info(f"Updating existing variant clinvar:{clinvar_id}")
            variant = existing
        else:
            variant = Variant(
                id=self.generate_id("clinvar", clinvar_id),
                source="ClinVar",
                variant_id=f"clinvar:{clinvar_id}"
            )
        
        # Update basic fields
        variant.rsid = data.get("dbsnp_id")
        variant.gene_symbol = data.get("gene_symbol")
        variant.variant_type = data.get("variation_type", "")
        variant.protein_change = data.get("protein_change")
        variant.clinical_significance = data.get("clinical_significance", "")
        
        # Set reference and alternate alleles (required fields)
        # For ClinVar, we may not always have explicit alleles, so use defaults
        variant.reference_allele = data.get("reference_allele", "N")
        variant.alternate_allele = data.get("alternate_allele", "N")
        
        # Update genomic location
        if data.get("genomic_location"):
            loc = data["genomic_location"]
            variant.chromosome = loc.get("chromosome", "Unknown")
            variant.position = loc.get("start", 0)
        else:
            # Set defaults if no location provided
            variant.chromosome = data.get("chromosome", "Unknown")
            variant.position = data.get("position", 0)
        
        # Store conditions as disease associations
        if data.get("conditions"):
            variant.disease_associations = "; ".join(data["conditions"])
        
        # Store all data as annotations
        variant.annotations = json.dumps({
            "clinvar_id": clinvar_id,
            "title": data.get("title", ""),
            "review_status": data.get("review_status", ""),
            "last_evaluated": data.get("last_evaluated"),
            "molecular_consequence": data.get("molecular_consequence", ""),
            "variant_name": data.get("variant_name", ""),
            "conditions": data.get("conditions", [])
        })
        
        # Save to database
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved variant clinvar:{clinvar_id}")
        
        return variant
    
    def save_variants_to_database(self, variants: List[Dict[str, Any]]) -> List[Variant]:
        """Save multiple variants to database.
        
        Args:
            variants: List of parsed variant data
        
        Returns:
            List of saved Variant instances
        """
        saved_variants = []
        
        for variant_data in variants:
            try:
                variant = self.save_to_database(variant_data)
                saved_variants.append(variant)
            except Exception as e:
                logger.error(f"Failed to save variant {variant_data.get('clinvar_id')}: {e}")
        
        return saved_variants
    
    def update_gene_variants(self, gene_symbol: str, pathogenic_only: bool = True) -> int:
        """Update all variants for a gene.
        
        Args:
            gene_symbol: Gene symbol
            pathogenic_only: Only update pathogenic variants
        
        Returns:
            Number of variants updated
        """
        # Check if gene exists in database
        gene = self.db_session.query(Gene).filter_by(
            symbol=gene_symbol
        ).first()
        
        if not gene:
            logger.warning(f"Gene {gene_symbol} not found in database")
        
        # Collect variants
        variants = self.collect_by_gene(gene_symbol, pathogenic_only)
        
        # Save to database
        saved_variants = self.save_variants_to_database(variants)
        
        logger.info(f"Updated {len(saved_variants)} variants for {gene_symbol}")
        return len(saved_variants)
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect ClinVar data for {identifier}: {e}")
        return results
    
    def search_variants(self, query: str, max_results: int = 50) -> List[Variant]:
        """Search for variants using ClinVar search.
        
        Args:
            query: Search query
            max_results: Maximum results
        
        Returns:
            List of Variant instances
        """
        logger.info(f"Searching ClinVar for '{query}'")
        
        # Search ClinVar
        search_results = self.api_client.search(query, max_results)
        
        saved_variants = []
        if "esearchresult" in search_results:
            variant_ids = search_results["esearchresult"].get("idlist", [])
            
            for variant_id in variant_ids:
                try:
                    # Collect and save variant
                    data = self.collect_single(variant_id)
                    variant = self.save_to_database(data)
                    saved_variants.append(variant)
                except Exception as e:
                    logger.error(f"Failed to process variant {variant_id}: {e}")
        
        return saved_variants