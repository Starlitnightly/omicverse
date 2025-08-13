"""Ensembl data collector."""

import logging
import json
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.ensembl import EnsemblClient
from omicverse.external.datacollect.models.genomic import Gene, Variant
from omicverse.external.datacollect.models.protein import Protein
from .base import BaseCollector
from ..config import settings


logger = logging.getLogger(__name__)


class EnsemblCollector(BaseCollector):
    """Collector for Ensembl genomic data."""
    
    def __init__(self, db_session=None):
        api_client = EnsemblClient()
        super().__init__(api_client, db_session)
        self.default_species = "human"
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Ensembl ID or gene symbol
            **kwargs: Additional parameters (expand, species)
        
        Returns:
            Collected data
        """
        expand = kwargs.get('expand', True)
        species = kwargs.get('species', self.default_species)
        
        # Try to detect if it's an Ensembl ID or symbol
        if identifier.startswith('ENS'):
            return self.collect_gene(identifier, expand)
        else:
            return self.collect_gene_by_symbol(identifier, species)
    
    def save_to_database(self, data: Dict[str, Any]) -> Gene:
        """Save collected data to database."""
        return self.save_gene_to_database(data)
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect Ensembl data for {identifier}: {e}")
        return results
    
    def collect_gene(self, gene_id: str, expand: bool = True) -> Dict[str, Any]:
        """Collect data for an Ensembl gene.
        
        Args:
            gene_id: Ensembl gene ID (e.g., ENSG00000141510)
            expand: Include transcripts and translations
        
        Returns:
            Collected gene data
        """
        logger.info(f"Collecting Ensembl gene {gene_id}")
        
        # Get gene information
        gene_info = self.api_client.lookup_id(gene_id, expand=expand)
        
        # Get cross-references
        xrefs = self.api_client.get_xrefs(gene_id)
        
        # Get homology
        homology = self.api_client.get_homology(gene_id)
        
        # Get sequence
        sequence = self.api_client.get_sequence(gene_id, "genomic", "json")
        
        data = {
            "gene_id": gene_id,
            "gene_info": gene_info,
            "xrefs": xrefs,
            "homology": homology,
            "sequence_info": sequence,
            "transcripts": [],
            "proteins": []
        }
        
        # Process transcripts if expanded
        if expand and "Transcript" in gene_info:
            for transcript_info in gene_info["Transcript"]:
                transcript_id = transcript_info["id"]
                
                # Get transcript sequence
                transcript_seq = self.api_client.get_sequence(
                    transcript_id, "cdna", "json"
                )
                
                transcript_data = {
                    "id": transcript_id,
                    "info": transcript_info,
                    "sequence": transcript_seq
                }
                
                data["transcripts"].append(transcript_data)
                
                # Process translation if available
                if "Translation" in transcript_info:
                    protein_id = transcript_info["Translation"]["id"]
                    protein_seq = self.api_client.get_sequence(
                        protein_id, "protein", "json"
                    )
                    
                    protein_data = {
                        "id": protein_id,
                        "transcript_id": transcript_id,
                        "sequence": protein_seq
                    }
                    
                    data["proteins"].append(protein_data)
        
        return data
    
    def collect_gene_by_symbol(self, symbol: str, species: str = None) -> Dict[str, Any]:
        """Collect gene data by symbol.
        
        Args:
            symbol: Gene symbol (e.g., "TP53")
            species: Species name (default: human)
        
        Returns:
            Gene data
        """
        if species is None:
            species = self.default_species
            
        logger.info(f"Collecting Ensembl data for {symbol} ({species})")
        
        # Look up by symbol
        gene_info = self.api_client.lookup_symbol(symbol, species, expand=True)
        
        # Use the gene ID to collect full data
        return self.collect_gene(gene_info["id"])
    
    def save_gene_to_database(self, data: Dict[str, Any]) -> Gene:
        """Save Ensembl gene data to database.
        
        Args:
            data: Collected gene data
        
        Returns:
            Saved Gene instance
        """
        gene_info = data["gene_info"]
        gene_id = data["gene_id"]
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            ensembl_id=gene_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene {gene_id}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("ensembl_gene", gene_id),
                gene_id=gene_id,  # Set the required gene_id field
                source="Ensembl",
            )
        
        # Update fields
        gene.ensembl_id = gene_id
        gene.symbol = gene_info.get("display_name", "")
        gene.description = gene_info.get("description", "")
        gene.chromosome = gene_info.get("seq_region_name", "")
        gene.start_position = gene_info.get("start")
        gene.end_position = gene_info.get("end")
        gene.strand = gene_info.get("strand")
        gene.biotype = gene_info.get("biotype", "")
        gene.species = gene_info.get("species", "")
        
        # Store cross-references
        xref_dict = {}
        for xref in data.get("xrefs", []):
            db_name = xref.get("dbname", "")
            if db_name:
                if db_name not in xref_dict:
                    xref_dict[db_name] = []
                xref_dict[db_name].append(xref.get("primary_id", ""))
        
        gene.external_ids = json.dumps(xref_dict)
        
        # Extract specific IDs
        if "UniProtKB/Swiss-Prot" in xref_dict:
            gene.uniprot_ids = ",".join(xref_dict["UniProtKB/Swiss-Prot"])
        
        if "HGNC" in xref_dict and xref_dict["HGNC"]:
            gene.hgnc_id = xref_dict["HGNC"][0]
        
        # Count transcripts and proteins
        gene.transcript_count = len(data.get("transcripts", []))
        gene.protein_count = len(data.get("proteins", []))
        
        # Save to database
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved gene {gene.symbol} ({gene.ensembl_id})")
        
        # Link to proteins if they exist
        self._link_gene_to_proteins(gene, data)
        
        return gene
    
    def _link_gene_to_proteins(self, gene: Gene, data: Dict[str, Any]) -> None:
        """Link gene to existing proteins in database."""
        # Get UniProt IDs from cross-references
        uniprot_ids = []
        for xref in data.get("xrefs", []):
            if xref.get("dbname") in ["UniProtKB/Swiss-Prot", "UniProtKB/TrEMBL"]:
                uniprot_ids.append(xref.get("primary_id"))
        
        for uniprot_id in uniprot_ids:
            protein = self.db_session.query(Protein).filter_by(
                accession=uniprot_id
            ).first()
            
            if protein:
                # Update protein's gene information
                if not protein.gene_name:
                    protein.gene_name = gene.symbol
                if not protein.ensembl_gene_id:
                    protein.ensembl_gene_id = gene.ensembl_id
        
        self.db_session.commit()
    
    def collect_variant(self, species: str, variant_id: str) -> Dict[str, Any]:
        """Collect variant data.
        
        Args:
            species: Species name
            variant_id: Variant ID (e.g., rs56116432)
        
        Returns:
            Variant data
        """
        logger.info(f"Collecting variant {variant_id} for {species}")
        
        # Get variant information
        variant_info = self.api_client.get_variant(species, variant_id)
        
        data = {
            "variant_id": variant_id,
            "species": species,
            "info": variant_info,
            "mappings": variant_info.get("mappings", []),
            "synonyms": variant_info.get("synonyms", []),
            "most_severe_consequence": variant_info.get("most_severe_consequence"),
            "minor_allele": variant_info.get("minor_allele"),
            "minor_allele_freq": variant_info.get("MAF")
        }
        
        return data
    
    def save_variant_to_database(self, data: Dict[str, Any]) -> Variant:
        """Save variant data to database.
        
        Args:
            data: Variant data
        
        Returns:
            Saved Variant instance
        """
        variant_id = data["variant_id"]
        info = data["info"]
        
        # Check if variant exists
        existing = self.db_session.query(Variant).filter_by(
            rsid=variant_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing variant {variant_id}")
            variant = existing
        else:
            variant = Variant(
                id=self.generate_id("ensembl_variant", variant_id),
                variant_id=f"ensembl:{variant_id}",  # Set the required variant_id field
                source="Ensembl",
            )
        
        # Update fields
        variant.rsid = variant_id
        variant.variant_type = info.get("var_class", "")
        
        # Get first mapping for position info
        if data["mappings"]:
            mapping = data["mappings"][0]
            variant.chromosome = mapping.get("seq_region_name", "Unknown")
            variant.position = mapping.get("start", 0)
            variant.reference_allele = mapping.get("ancestral_allele", "N")
            variant.alternate_allele = data.get("minor_allele", "N")
            variant.alternative_allele = data.get("minor_allele", "N")  # Also set alternative_allele
        else:
            # Set defaults if no mapping
            variant.chromosome = "Unknown"
            variant.position = 0
            variant.reference_allele = "N"
            variant.alternate_allele = "N"
            variant.alternative_allele = "N"
        
        variant.minor_allele_frequency = data.get("minor_allele_freq")
        variant.clinical_significance = data.get("most_severe_consequence", "")
        
        # Store all info as JSON
        variant.annotations = json.dumps(info)
        
        # Save to database
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved variant {variant.rsid}")
        
        return variant
    
    def get_overlap_features(self, species: str, region: str,
                           feature_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get features overlapping a genomic region.
        
        Args:
            species: Species name
            region: Genomic region (e.g., "7:140424943-140624564")
            feature_types: Types of features to retrieve
        
        Returns:
            List of overlapping features
        """
        if feature_types is None:
            feature_types = ["gene", "transcript", "exon", "cds"]
        
        logger.info(f"Getting features in region {region} for {species}")
        
        features = self.api_client.get_overlap_region(
            species, region, feature_types
        )
        
        # Organize by type
        organized = {}
        for feature in features:
            ftype = feature.get("feature_type", "unknown")
            if ftype not in organized:
                organized[ftype] = []
            organized[ftype].append(feature)
        
        return organized
    
    def collect_regulatory(self, species: str, regulatory_id: str) -> Dict[str, Any]:
        """Collect regulatory feature data.
        
        Args:
            species: Species name
            regulatory_id: Regulatory feature ID
        
        Returns:
            Regulatory feature data
        """
        logger.info(f"Collecting regulatory feature {regulatory_id}")
        
        reg_info = self.api_client.get_regulatory(species, regulatory_id)
        
        return {
            "regulatory_id": regulatory_id,
            "species": species,
            "info": reg_info,
            "feature_type": reg_info.get("feature_type"),
            "description": reg_info.get("description"),
            "bound_start": reg_info.get("bound_start"),
            "bound_end": reg_info.get("bound_end"),
            "activity": reg_info.get("activity", {})
        }