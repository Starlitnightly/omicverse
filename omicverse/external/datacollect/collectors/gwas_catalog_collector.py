"""GWAS Catalog data collector."""

import logging
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.gwas_catalog import GWASCatalogClient
from omicverse.external.datacollect.models.genomic import Gene, Variant
from omicverse.external.datacollect.models.disease import Disease
from .base import BaseCollector
from omicverse.external.datacollect.config.config import settings


logger = logging.getLogger(__name__)


class GWASCatalogCollector(BaseCollector):
    """Collector for GWAS Catalog association data."""
    
    def __init__(self, db_session=None):
        api_client = GWASCatalogClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Study accession, gene symbol, variant rsID, or trait
            **kwargs: Additional parameters (type)
        
        Returns:
            Collected data
        """
        id_type = kwargs.get('type', 'study')
        
        if id_type == 'study':
            return self.collect_study(identifier, **kwargs)
        elif id_type == 'gene':
            return self.collect_gene_associations(identifier, **kwargs)
        elif id_type == 'variant':
            return self.collect_variant_associations(identifier, **kwargs)
        elif id_type == 'trait':
            return self.collect_trait_associations(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_study(self, study_accession: str, **kwargs) -> Dict[str, Any]:
        """Collect GWAS study data.
        
        Args:
            study_accession: Study accession (e.g., GCST000001)
            **kwargs: Additional parameters
        
        Returns:
            Study data
        """
        logger.info(f"Collecting GWAS Catalog study {study_accession}")
        
        # Get study info
        study = self.api_client.get_study(study_accession)
        
        data = {
            "study_accession": study_accession,
            "study_info": study,
            "title": study.get("initialSampleSize"),
            "author": study.get("author"),
            "publication_date": study.get("publicationDate"),
            "pmid": study.get("pubmedId")
        }
        
        # Get associations for this study
        associations = self.api_client.get_associations(
            study_id=study_accession,
            size=kwargs.get('association_limit', 100)
        )
        
        if associations and "_embedded" in associations:
            assoc_list = associations["_embedded"].get("associations", [])
            data["associations"] = assoc_list
            data["association_count"] = len(assoc_list)
            
            # Extract unique traits
            traits = set()
            for assoc in assoc_list:
                if "efoTraits" in assoc:
                    for trait in assoc["efoTraits"]:
                        traits.add(trait.get("trait"))
            data["traits"] = list(traits)
            
            # Extract unique genes
            genes = set()
            for assoc in assoc_list:
                if "loci" in assoc:
                    for locus in assoc["loci"]:
                        if "authorReportedGenes" in locus:
                            for gene in locus["authorReportedGenes"]:
                                genes.add(gene.get("geneName"))
            data["genes"] = list(genes)
        
        # Get summary statistics info if available
        if kwargs.get('include_sumstats'):
            sumstats = self.api_client.get_summary_statistics(study_accession)
            data["summary_statistics"] = sumstats
        
        return data
    
    def collect_gene_associations(self, gene_symbol: str, **kwargs) -> Dict[str, Any]:
        """Collect associations for a gene.
        
        Args:
            gene_symbol: Gene symbol
            **kwargs: Additional parameters
        
        Returns:
            Gene association data
        """
        logger.info(f"Collecting GWAS Catalog associations for gene {gene_symbol}")
        
        # Get gene info
        gene_info = self.api_client.get_genes_by_symbol(gene_symbol)
        
        data = {
            "gene_symbol": gene_symbol,
            "gene_info": gene_info
        }
        
        # Get associations for this gene
        associations = self.api_client.get_associations(
            gene=gene_symbol,
            size=kwargs.get('association_limit', 100)
        )
        
        if associations and "_embedded" in associations:
            assoc_list = associations["_embedded"].get("associations", [])
            data["associations"] = assoc_list
            data["association_count"] = len(assoc_list)
            
            # Group by trait
            traits = {}
            for assoc in assoc_list:
                if "efoTraits" in assoc:
                    for trait in assoc["efoTraits"]:
                        trait_name = trait.get("trait")
                        if trait_name not in traits:
                            traits[trait_name] = []
                        traits[trait_name].append({
                            "p_value": assoc.get("pvalueMantissa", 0) * 10 ** assoc.get("pvalueExponent", 0),
                            "variant": assoc.get("strongestRiskAlleles", [{}])[0].get("riskAlleleName"),
                            "study": assoc.get("study", {}).get("accessionId")
                        })
            data["associations_by_trait"] = traits
            data["trait_count"] = len(traits)
        
        return data
    
    def collect_variant_associations(self, rsid: str, **kwargs) -> Dict[str, Any]:
        """Collect associations for a variant.
        
        Args:
            rsid: Variant rsID
            **kwargs: Additional parameters
        
        Returns:
            Variant association data
        """
        logger.info(f"Collecting GWAS Catalog associations for variant {rsid}")
        
        # Get SNP info
        try:
            snp_info = self.api_client.get_single_nucleotide_polymorphisms(rsid)
            data = {
                "rsid": rsid,
                "snp_info": snp_info
            }
        except Exception as e:
            logger.warning(f"Could not get SNP info for {rsid}: {e}")
            data = {"rsid": rsid}
        
        # Get associations for this variant
        associations = self.api_client.get_associations(
            variant=rsid,
            size=kwargs.get('association_limit', 100)
        )
        
        if associations and "_embedded" in associations:
            assoc_list = associations["_embedded"].get("associations", [])
            data["associations"] = assoc_list
            data["association_count"] = len(assoc_list)
            
            # Extract top associations
            top_assocs = []
            for assoc in assoc_list:
                p_value = assoc.get("pvalueMantissa", 0) * 10 ** assoc.get("pvalueExponent", 0)
                for trait in assoc.get("efoTraits", []):
                    top_assocs.append({
                        "trait": trait.get("trait"),
                        "p_value": p_value,
                        "or_beta": assoc.get("orPerCopyNum") or assoc.get("betaNum"),
                        "study": assoc.get("study", {}).get("accessionId")
                    })
            
            # Sort by p-value
            top_assocs.sort(key=lambda x: x["p_value"])
            data["top_associations"] = top_assocs[:10]
        
        return data
    
    def collect_trait_associations(self, trait: str, **kwargs) -> Dict[str, Any]:
        """Collect associations for a trait.
        
        Args:
            trait: Trait/disease name
            **kwargs: Additional parameters
        
        Returns:
            Trait association data
        """
        logger.info(f"Collecting GWAS Catalog associations for trait {trait}")
        
        # Search for trait
        trait_search = self.api_client.search_traits(trait, size=1)
        
        data = {
            "trait": trait,
            "trait_search": trait_search
        }
        
        # Get top associations for this trait
        associations = self.api_client.get_top_associations_by_trait(
            trait,
            p_value_threshold=kwargs.get('p_value_threshold', 5e-8),
            size=kwargs.get('association_limit', 100)
        )
        
        if associations and "_embedded" in associations:
            assoc_list = associations["_embedded"].get("associations", [])
            data["associations"] = assoc_list
            data["association_count"] = len(assoc_list)
            
            # Extract lead variants
            lead_variants = []
            genes_involved = set()
            
            for assoc in assoc_list:
                p_value = assoc.get("pvalueMantissa", 0) * 10 ** assoc.get("pvalueExponent", 0)
                
                # Get risk alleles
                for risk_allele in assoc.get("strongestRiskAlleles", []):
                    lead_variants.append({
                        "variant": risk_allele.get("riskAlleleName"),
                        "p_value": p_value,
                        "risk_frequency": risk_allele.get("riskFrequency"),
                        "study": assoc.get("study", {}).get("accessionId")
                    })
                
                # Get genes
                for locus in assoc.get("loci", []):
                    for gene in locus.get("authorReportedGenes", []):
                        genes_involved.add(gene.get("geneName"))
            
            data["lead_variants"] = lead_variants[:20]
            data["genes_involved"] = list(genes_involved)
            data["gene_count"] = len(genes_involved)
        
        return data
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect GWAS Catalog data for {identifier}: {e}")
        return results
    
    def save_variant_associations(self, data: Dict[str, Any]) -> Optional[Variant]:
        """Save variant association data to database.
        
        Args:
            data: Collected variant data
        
        Returns:
            Saved Variant instance
        """
        rsid = data["rsid"]
        
        # Check if variant exists
        existing = self.db_session.query(Variant).filter_by(
            rsid=rsid
        ).first()
        
        if existing:
            logger.info(f"Updating existing variant {rsid}")
            variant = existing
        else:
            variant = Variant(
                id=self.generate_id("gwas_variant", rsid),
                variant_id=f"gwas:{rsid}",
                rsid=rsid,
                source="GWAS_Catalog"
            )
        
        # Update SNP info if available
        if "snp_info" in data and data["snp_info"]:
            snp = data["snp_info"]
            if "locations" in snp:
                for loc in snp["locations"]:
                    variant.chromosome = str(loc.get("chromosomeName"))
                    variant.position = loc.get("chromosomePosition")
                    break
        
        # Store GWAS associations
        if data.get("associations"):
            variant.gwas_catalog_data = {
                "association_count": data["association_count"],
                "top_associations": data.get("top_associations", [])
            }
        
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved variant {rsid} with {data.get('association_count', 0)} associations")
        
        return variant
    
    def save_gene_associations(self, data: Dict[str, Any]) -> Optional[Gene]:
        """Save gene association data to database.
        
        Args:
            data: Collected gene data
        
        Returns:
            Saved Gene instance
        """
        gene_symbol = data["gene_symbol"]
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            symbol=gene_symbol
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene {gene_symbol}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("gwas_gene", gene_symbol),
                gene_id=f"gwas:{gene_symbol}",
                symbol=gene_symbol,
                source="GWAS_Catalog"
            )
        
        # Store GWAS associations
        if data.get("associations"):
            gene.gwas_catalog_data = {
                "association_count": data["association_count"],
                "trait_count": data.get("trait_count", 0),
                "associations_by_trait": data.get("associations_by_trait", {})
            }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved gene {gene_symbol} with {data.get('association_count', 0)} associations")
        
        return gene
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save data to database based on data type."""
        if "rsid" in data:
            return self.save_variant_associations(data)
        elif "gene_symbol" in data:
            return self.save_gene_associations(data)
        else:
            logger.warning("Cannot determine data type to save")
            return None
    
    def search_studies(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for GWAS studies.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching studies
        """
        logger.info(f"Searching GWAS Catalog for studies: '{query}'")
        result = self.api_client.search_studies(query, size=limit)
        
        if result and "_embedded" in result:
            return result["_embedded"].get("studies", [])
        return []
    
    def get_region_associations(self, chromosome: str, start: int, end: int,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get associations in a genomic region.
        
        Args:
            chromosome: Chromosome
            start: Start position
            end: End position
            limit: Maximum results
        
        Returns:
            Associations in region
        """
        logger.info(f"Getting GWAS associations for {chromosome}:{start}-{end}")
        result = self.api_client.get_genomic_contexts(
            chromosome, start, end, size=limit
        )
        
        if result and "_embedded" in result:
            return result["_embedded"].get("associations", [])
        return []