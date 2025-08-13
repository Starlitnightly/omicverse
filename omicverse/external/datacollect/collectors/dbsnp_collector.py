"""dbSNP data collector."""

import logging
import json
from typing import Any, Dict, List, Optional, Union

from omicverse.external.datacollect.api.dbsnp import dbSNPClient
from omicverse.external.datacollect.models.genomic import Variant, Gene
from .base import BaseCollector
from ..config import settings


logger = logging.getLogger(__name__)


class dbSNPCollector(BaseCollector):
    """Collector for dbSNP variant data."""
    
    def __init__(self, db_session=None):
        api_client = dbSNPClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single variant.
        
        Args:
            identifier: RS ID (e.g., "rs7412" or "7412")
            **kwargs: Additional parameters
        
        Returns:
            Collected variant data
        """
        logger.info(f"Collecting dbSNP variant {identifier}")
        
        # Get basic variant information
        variant_data = self.api_client.get_variant_by_rsid(identifier)
        
        # Get allele annotations
        allele_annotations = self.api_client.get_variant_allele_annotations(identifier)
        
        # Get population frequencies
        frequency_data = self.api_client.get_population_frequency(identifier)
        
        # Get clinical significance
        clinical_data = self.api_client.get_clinical_significance(identifier)
        
        # Get consequences
        consequences = self.api_client.get_variant_consequences(identifier)
        
        # Extract key information
        rsid = f"rs{identifier}" if not identifier.startswith("rs") else identifier
        
        data = {
            "rsid": rsid,
            "variant_type": self._extract_variant_type(variant_data),
            "chromosome": self._extract_chromosome(variant_data),
            "position": self._extract_position(variant_data),
            "reference_allele": self._extract_reference_allele(variant_data),
            "alternate_alleles": self._extract_alternate_alleles(variant_data),
            "gene_symbols": self._extract_gene_symbols(variant_data),
            "consequences": consequences,
            "clinical_significance": clinical_data,
            "global_frequency": frequency_data.get("global_frequency"),
            "population_frequencies": frequency_data.get("population_frequencies", {}),
            "allele_annotations": allele_annotations,
            "raw_data": variant_data
        }
        
        return data
    
    def save_to_database(self, data: Dict[str, Any]) -> Variant:
        """Save variant data to database.
        
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
                id=self.generate_id("dbsnp", rsid),
                source="dbSNP",
                variant_id=rsid,
                rsid=rsid
            )
        
        # Update fields
        variant.variant_type = data.get("variant_type", "")
        variant.chromosome = str(data.get("chromosome", ""))
        variant.position = data.get("position")
        variant.reference_allele = data.get("reference_allele", "")
        
        # Handle multiple alternate alleles
        alt_alleles = data.get("alternate_alleles", [])
        if alt_alleles:
            variant.alternative_allele = ",".join(alt_alleles)
        
        # Set gene symbol (first one if multiple)
        gene_symbols = data.get("gene_symbols", [])
        if gene_symbols:
            variant.gene_symbol = gene_symbols[0]
        
        # Set global MAF
        variant.minor_allele_frequency = data.get("global_frequency")
        
        # Extract clinical significance
        if data.get("clinical_significance"):
            clin_sigs = [cs.get("clinical_significance", "") 
                        for cs in data["clinical_significance"]]
            variant.clinical_significance = "; ".join(filter(None, clin_sigs))
        
        # Extract consequence
        if data.get("consequences"):
            consequences = [c.get("consequence_type", "") 
                           for c in data["consequences"]]
            # Get unique consequences
            unique_consequences = list(set(filter(None, consequences)))
            if unique_consequences:
                variant.consequence = "; ".join(unique_consequences)
        
        # Store all data as annotations
        variant.annotations = json.dumps({
            "gene_symbols": gene_symbols,
            "consequences": data.get("consequences", []),
            "clinical_data": data.get("clinical_significance", []),
            "population_frequencies": data.get("population_frequencies", {})
        })
        
        # Save to database
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved variant {rsid}")
        
        return variant
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple variants."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect dbSNP data for {identifier}: {e}")
        return results
    
    def collect_by_gene(self, gene_symbol: str, organism: str = "human",
                       max_results: int = 100) -> List[Dict[str, Any]]:
        """Collect variants for a gene.
        
        Args:
            gene_symbol: Gene symbol
            organism: Organism name
            max_results: Maximum number of variants
        
        Returns:
            List of variant data
        """
        logger.info(f"Collecting dbSNP variants for gene {gene_symbol}")
        
        # Search for RS IDs
        rs_ids = self.api_client.search_by_gene(gene_symbol, organism, max_results)
        
        logger.info(f"Found {len(rs_ids)} variants for {gene_symbol}")
        
        # Collect data for each variant
        variants = []
        for rsid in rs_ids:
            try:
                data = self.collect_single(rsid)
                variants.append(data)
            except Exception as e:
                logger.error(f"Failed to collect variant {rsid}: {e}")
        
        return variants
    
    def collect_by_position(self, chromosome: str, start: int, end: int,
                          organism: str = "human") -> List[Dict[str, Any]]:
        """Collect variants in a genomic region.
        
        Args:
            chromosome: Chromosome
            start: Start position
            end: End position
            organism: Organism name
        
        Returns:
            List of variant data
        """
        logger.info(f"Collecting dbSNP variants for chr{chromosome}:{start}-{end}")
        
        # Search for RS IDs
        rs_ids = self.api_client.search_by_position(chromosome, start, end, organism)
        
        logger.info(f"Found {len(rs_ids)} variants in region")
        
        # Collect data for each variant
        variants = []
        for rsid in rs_ids[:50]:  # Limit to 50 variants
            try:
                data = self.collect_single(rsid)
                variants.append(data)
            except Exception as e:
                logger.error(f"Failed to collect variant {rsid}: {e}")
        
        return variants
    
    def save_gene_variants(self, gene_symbol: str, organism: str = "human",
                          max_results: int = 50) -> List[Variant]:
        """Collect and save all variants for a gene.
        
        Args:
            gene_symbol: Gene symbol
            organism: Organism
            max_results: Maximum variants to collect
        
        Returns:
            List of saved Variant instances
        """
        # Check if gene exists in database
        gene = self.db_session.query(Gene).filter_by(
            symbol=gene_symbol
        ).first()
        
        if not gene:
            logger.warning(f"Gene {gene_symbol} not found in database")
        
        # Collect variants
        variants_data = self.collect_by_gene(gene_symbol, organism, max_results)
        
        # Save to database
        saved_variants = []
        for var_data in variants_data:
            try:
                variant = self.save_to_database(var_data)
                saved_variants.append(variant)
            except Exception as e:
                logger.error(f"Failed to save variant {var_data.get('rsid')}: {e}")
        
        logger.info(f"Saved {len(saved_variants)} variants for {gene_symbol}")
        return saved_variants
    
    def _extract_variant_type(self, variant_data: Dict[str, Any]) -> str:
        """Extract variant type from raw data."""
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            variant_type = snapshot.get("variant_type", "")
            
            # Map to standard types
            type_mapping = {
                "snv": "SNP",
                "mnv": "MNP",
                "ins": "insertion",
                "del": "deletion",
                "delins": "indel"
            }
            
            return type_mapping.get(variant_type.lower(), variant_type)
        
        return ""
    
    def _extract_chromosome(self, variant_data: Dict[str, Any]) -> str:
        """Extract chromosome from raw data."""
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            for placement in snapshot.get("placements_with_allele", []):
                for seq_id in placement.get("placement_annot", {}).get("seq_id_traits_by_assembly", []):
                    if seq_id.get("is_chromosome", False):
                        # Extract chromosome from sequence name
                        seq_name = seq_id.get("sequence_name", "")
                        if seq_name.startswith("NC_"):
                            # Map RefSeq to chromosome
                            chr_mapping = {
                                "NC_000001": "1", "NC_000002": "2", "NC_000003": "3",
                                "NC_000004": "4", "NC_000005": "5", "NC_000006": "6",
                                "NC_000007": "7", "NC_000008": "8", "NC_000009": "9",
                                "NC_000010": "10", "NC_000011": "11", "NC_000012": "12",
                                "NC_000013": "13", "NC_000014": "14", "NC_000015": "15",
                                "NC_000016": "16", "NC_000017": "17", "NC_000018": "18",
                                "NC_000019": "19", "NC_000020": "20", "NC_000021": "21",
                                "NC_000022": "22", "NC_000023": "X", "NC_000024": "Y"
                            }
                            for ref, chr_num in chr_mapping.items():
                                if seq_name.startswith(ref):
                                    return chr_num
                        
                        # Try to extract from traits
                        traits = seq_id.get("traits", [])
                        for trait in traits:
                            if trait.get("trait_name", "").startswith("Chr"):
                                return trait["trait_name"][3:]
        
        return ""
    
    def _extract_position(self, variant_data: Dict[str, Any]) -> Optional[int]:
        """Extract genomic position from raw data."""
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            for placement in snapshot.get("placements_with_allele", []):
                for allele in placement.get("alleles", []):
                    if "location" in allele:
                        return allele["location"].get("position", None)
        
        return None
    
    def _extract_reference_allele(self, variant_data: Dict[str, Any]) -> str:
        """Extract reference allele from raw data."""
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            for placement in snapshot.get("placements_with_allele", []):
                for allele in placement.get("alleles", []):
                    spdi = allele.get("allele", {}).get("spdi", {})
                    if spdi.get("deleted_sequence"):
                        return spdi["deleted_sequence"]
        
        return ""
    
    def _extract_alternate_alleles(self, variant_data: Dict[str, Any]) -> List[str]:
        """Extract alternate alleles from raw data."""
        alt_alleles = []
        
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            for placement in snapshot.get("placements_with_allele", []):
                for allele in placement.get("alleles", []):
                    spdi = allele.get("allele", {}).get("spdi", {})
                    if spdi.get("inserted_sequence"):
                        alt = spdi["inserted_sequence"]
                        if alt and alt not in alt_alleles:
                            alt_alleles.append(alt)
        
        return alt_alleles
    
    def _extract_gene_symbols(self, variant_data: Dict[str, Any]) -> List[str]:
        """Extract associated gene symbols from raw data."""
        gene_symbols = []
        
        if "primary_snapshot_data" in variant_data:
            snapshot = variant_data["primary_snapshot_data"]
            
            for placement in snapshot.get("placements_with_allele", []):
                for seq_trait in placement.get("placement_annot", {}).get("seq_id_traits_by_assembly", []):
                    for gene in seq_trait.get("genes", []):
                        gene_name = gene.get("name", "")
                        if gene_name and gene_name not in gene_symbols:
                            gene_symbols.append(gene_name)
        
        return gene_symbols