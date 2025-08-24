"""OpenTargets Genetics data collector."""

import logging
from typing import Any, Dict, List, Optional

from src.api.opentargets_genetics import OpenTargetsGeneticsClient
from src.models.genomic import Gene, Variant
from .base import BaseCollector
from ..config.config import settings


logger = logging.getLogger(__name__)


class OpenTargetsGeneticsCollector(BaseCollector):
    """Collector for OpenTargets Genetics GWAS data."""
    
    def __init__(self, db_session=None):
        api_client = OpenTargetsGeneticsClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Gene ID (Ensembl), variant ID, or study ID
            **kwargs: Additional parameters (type)
        
        Returns:
            Collected data
        """
        id_type = kwargs.get('type', 'gene')
        
        if id_type == 'gene':
            return self.collect_gene_associations(identifier, **kwargs)
        elif id_type == 'variant':
            return self.collect_variant_associations(identifier, **kwargs)
        elif id_type == 'study':
            return self.collect_study_data(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_gene_associations(self, gene_id: str, **kwargs) -> Dict[str, Any]:
        """Collect GWAS associations for a gene.
        
        Args:
            gene_id: Ensembl gene ID
            **kwargs: Additional parameters
        
        Returns:
            Gene association data
        """
        logger.info(f"Collecting OpenTargets Genetics data for gene {gene_id}")
        
        # Get gene info
        gene_info = self.api_client.get_gene_info(gene_id)
        
        if not gene_info:
            raise ValueError(f"Gene {gene_id} not found")
        
        # Get associations
        associations_data = self.api_client.get_associations_for_gene(
            gene_id, page_size=kwargs.get('page_size', 100)
        )
        
        data = {
            "gene_id": gene_id,
            "gene_info": gene_info,
            "associations": associations_data.get("associatedStudiesForGene", []),
            "association_count": len(associations_data.get("associatedStudiesForGene", []))
        }
        
        # Get colocalization data if requested
        if kwargs.get('include_colocalization'):
            coloc_data = []
            for assoc in data["associations"][:10]:  # Limit to top 10
                study_id = assoc["study"]["studyId"]
                coloc = self.api_client.get_colocalization(gene_id, study_id)
                if coloc:
                    coloc_data.extend(coloc)
            data["colocalization"] = coloc_data
        
        return data
    
    def collect_variant_associations(self, variant_id: str, **kwargs) -> Dict[str, Any]:
        """Collect GWAS associations for a variant.
        
        Args:
            variant_id: Variant ID (chr_pos_ref_alt format)
            **kwargs: Additional parameters
        
        Returns:
            Variant association data
        """
        logger.info(f"Collecting OpenTargets Genetics data for variant {variant_id}")
        
        # Get variant info
        variant_info = self.api_client.get_variant_info(variant_id)
        
        if not variant_info:
            raise ValueError(f"Variant {variant_id} not found")
        
        # Get associations
        associations_data = self.api_client.get_associations_for_variant(
            variant_id, page_size=kwargs.get('page_size', 100)
        )
        
        data = {
            "variant_id": variant_id,
            "variant_info": variant_info,
            "associations": associations_data.get("associatedStudiesForVariant", []),
            "association_count": len(associations_data.get("associatedStudiesForVariant", []))
        }
        
        # Get PheWAS data if requested
        if kwargs.get('include_phewas'):
            phewas = self.api_client.get_pheWAS(variant_id)
            data["phewas"] = phewas
            data["phewas_count"] = len(phewas)
        
        return data
    
    def collect_study_data(self, study_id: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a GWAS study.
        
        Args:
            study_id: Study ID
            **kwargs: Additional parameters
        
        Returns:
            Study data
        """
        logger.info(f"Collecting OpenTargets Genetics study data for {study_id}")
        
        # Get study info
        study_info = self.api_client.get_study_info(study_id)
        
        if not study_info:
            raise ValueError(f"Study {study_id} not found")
        
        data = {
            "study_id": study_id,
            "study_info": study_info
        }
        
        # Get Manhattan plot data if requested
        if kwargs.get('include_manhattan'):
            manhattan = self.api_client.get_manhattan_plot_data(study_id)
            data["manhattan_data"] = manhattan
            if manhattan.get("associations"):
                data["total_variants"] = len(manhattan["associations"])
        
        return data
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect OpenTargets Genetics data for {identifier}: {e}")
        return results
    
    def save_gene_associations(self, data: Dict[str, Any]) -> Gene:
        """Save gene association data to database.
        
        Args:
            data: Collected gene association data
        
        Returns:
            Saved Gene instance
        """
        gene_info = data["gene_info"]
        gene_id = gene_info["id"]
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            ensembl_id=gene_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene {gene_id}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("otg_gene", gene_id),
                gene_id=gene_id,
                source="OpenTargets_Genetics"
            )
        
        # Update fields
        gene.ensembl_id = gene_id
        gene.symbol = gene_info.get("symbol")
        gene.description = gene_info.get("description")
        gene.chromosome = gene_info.get("chromosome")
        gene.start_position = gene_info.get("start")
        gene.end_position = gene_info.get("end")
        gene.biotype = gene_info.get("bioType")
        
        # Store GWAS associations as JSON
        if data.get("associations"):
            gene.gwas_associations = {
                "association_count": data["association_count"],
                "top_associations": [
                    {
                        "study": assoc["study"]["traitReported"],
                        "pmid": assoc["study"].get("pmid"),
                        "pval": assoc.get("pval"),
                        "variant": assoc["variant"].get("rsId")
                    }
                    for assoc in data["associations"][:10]  # Store top 10
                ]
            }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved gene {gene_id} with {data['association_count']} associations")
        
        return gene
    
    def save_variant_associations(self, data: Dict[str, Any]) -> Variant:
        """Save variant association data to database.
        
        Args:
            data: Collected variant association data
        
        Returns:
            Saved Variant instance
        """
        variant_info = data["variant_info"]
        variant_id = variant_info["id"]
        
        # Parse variant ID (chr_pos_ref_alt)
        parts = variant_id.split("_")
        if len(parts) >= 4:
            chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
        else:
            raise ValueError(f"Invalid variant ID format: {variant_id}")
        
        # Check if variant exists
        existing = self.db_session.query(Variant).filter_by(
            variant_id=variant_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing variant {variant_id}")
            variant = existing
        else:
            variant = Variant(
                id=self.generate_id("otg_variant", variant_id),
                variant_id=variant_id,
                source="OpenTargets_Genetics"
            )
        
        # Update fields
        variant.rsid = variant_info.get("rsId")
        variant.chromosome = chrom
        variant.position = int(pos)
        variant.reference_allele = ref
        variant.alternate_allele = alt
        variant.consequence = variant_info.get("mostSevereConsequence")
        
        # Store additional info
        if variant_info.get("nearestGene"):
            variant.nearest_gene = variant_info["nearestGene"]["symbol"]
            variant.nearest_gene_distance = variant_info.get("nearestGeneDistance")
        
        # Store GWAS associations as JSON
        if data.get("associations"):
            variant.gwas_associations = {
                "association_count": data["association_count"],
                "top_associations": [
                    {
                        "study": assoc["study"]["traitReported"],
                        "pmid": assoc["study"].get("pmid"),
                        "pval": assoc.get("pval")
                    }
                    for assoc in data["associations"][:10]
                ]
            }
        
        # Store PheWAS data if available
        if data.get("phewas"):
            variant.phewas_data = {
                "phewas_count": data.get("phewas_count", 0),
                "top_associations": [
                    {
                        "trait": p["study"]["traitReported"],
                        "pval": p.get("pval")
                    }
                    for p in data["phewas"][:10]
                ]
            }
        
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved variant {variant_id} with {data['association_count']} associations")
        
        return variant
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save data to database based on data type."""
        if "gene_info" in data:
            return self.save_gene_associations(data)
        elif "variant_info" in data:
            return self.save_variant_associations(data)
        else:
            logger.warning("Unknown data type, cannot save to database")
            return None
    
    def search_studies(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for GWAS studies.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching studies
        """
        logger.info(f"Searching OpenTargets Genetics for studies: '{query}'")
        return self.api_client.search_studies(query, page_size=limit)
    
    def get_credible_sets(self, study_id: str, variant_id: str) -> List[Dict[str, Any]]:
        """Get credible set variants.
        
        Args:
            study_id: Study ID
            variant_id: Lead variant ID
        
        Returns:
            Credible set variants
        """
        logger.info(f"Getting credible sets for {study_id} / {variant_id}")
        return self.api_client.get_credible_sets(study_id, variant_id)
