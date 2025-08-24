"""RegulomeDB regulatory variant data collector."""

import logging
from typing import Any, Dict, List, Optional

from src.api.regulomedb import RegulomeDBClient
from src.models.genomic import Variant
from .base import BaseCollector
from ..config.config import settings


logger = logging.getLogger(__name__)


class RegulomeDBCollector(BaseCollector):
    """Collector for RegulomeDB regulatory variant annotations."""
    
    def __init__(self, db_session=None):
        api_client = RegulomeDBClient()
        super().__init__(api_client, db_session)
        self.default_assembly = "GRCh38"
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect regulatory data for a variant.
        
        Args:
            identifier: rsID or chr:pos format
            **kwargs: Additional parameters (assembly, type)
        
        Returns:
            Regulatory annotation data
        """
        assembly = kwargs.get('assembly', self.default_assembly)
        
        # Determine identifier type
        if identifier.startswith("rs"):
            return self.collect_rsid_data(identifier, assembly, **kwargs)
        elif ":" in identifier:
            # chr:pos format
            parts = identifier.split(":")
            chr = parts[0]
            pos = int(parts[1])
            return self.collect_position_data(chr, pos, assembly, **kwargs)
        else:
            raise ValueError(f"Invalid identifier format: {identifier}")
    
    def collect_rsid_data(self, rsid: str, assembly: str = None, **kwargs) -> Dict[str, Any]:
        """Collect regulatory data for an rsID.
        
        Args:
            rsid: dbSNP rsID
            assembly: Genome assembly
            **kwargs: Additional parameters
        
        Returns:
            Regulatory data
        """
        if assembly is None:
            assembly = self.default_assembly
        
        logger.info(f"Collecting RegulomeDB data for {rsid} on {assembly}")
        
        # Query RegulomeDB
        result = self.api_client.query_rsid(rsid, assembly)
        
        if not result or "@graph" not in result:
            raise ValueError(f"No regulatory data found for {rsid}")
        
        variants = result.get("@graph", [])
        if not variants:
            raise ValueError(f"No regulatory data found for {rsid}")
        
        variant_data = variants[0]
        
        data = {
            "rsid": rsid,
            "assembly": assembly,
            "coordinates": variant_data.get("coordinates", {}),
            "variant_type": variant_data.get("variant_type"),
            "reference_allele": variant_data.get("ref"),
            "alternate_alleles": variant_data.get("alt", [])
        }
        
        # RegulomeDB score
        if variant_data.get("regulome_score"):
            score_data = variant_data["regulome_score"]
            data["regulome_score"] = {
                "ranking": score_data.get("ranking"),
                "probability": score_data.get("probability"),
                "evidence_count": len(variant_data.get("peaks", []))
            }
        
        # Process regulatory evidence
        data["regulatory_evidence"] = self._process_regulatory_evidence(variant_data)
        
        # Process chromatin accessibility peaks
        if variant_data.get("peaks"):
            data["chromatin_peaks"] = self._process_peaks(variant_data["peaks"])
        
        # Process transcription factor motifs
        if variant_data.get("motifs"):
            data["tf_motifs"] = self._process_motifs(variant_data["motifs"])
        
        # Process QTLs
        if variant_data.get("qtls"):
            data["qtls"] = self._process_qtls(variant_data["qtls"])
        
        # Process chromatin states
        if variant_data.get("chromatin_states"):
            data["chromatin_states"] = self._process_chromatin_states(variant_data["chromatin_states"])
        
        # Get additional annotations if requested
        if kwargs.get('include_targets'):
            data["target_genes"] = self._extract_target_genes(variant_data)
        
        return data
    
    def collect_position_data(self, chromosome: str, position: int,
                            assembly: str = None, **kwargs) -> Dict[str, Any]:
        """Collect regulatory data for a genomic position.
        
        Args:
            chromosome: Chromosome
            position: Genomic position
            assembly: Genome assembly
            **kwargs: Additional parameters
        
        Returns:
            Regulatory data
        """
        if assembly is None:
            assembly = self.default_assembly
        
        logger.info(f"Collecting RegulomeDB data for {chromosome}:{position} on {assembly}")
        
        # Get regulatory score
        score_data = self.api_client.get_regulatory_score(chromosome, position, assembly)
        
        data = {
            "chromosome": chromosome,
            "position": position,
            "assembly": assembly,
            "regulome_score": score_data.get("score"),
            "probability": score_data.get("probability"),
            "evidence_count": score_data.get("evidence_count", 0)
        }
        
        # Get peaks at position
        peaks = self.api_client.get_peaks_at_position(chromosome, position, assembly)
        if peaks:
            data["chromatin_peaks"] = self._process_peaks(peaks)
        
        # Get QTLs at position
        qtls = self.api_client.get_qtls_at_position(chromosome, position, assembly)
        if qtls:
            data["qtls"] = self._process_qtls(qtls)
        
        # Get chromatin states
        states = self.api_client.get_chromatin_states(chromosome, position, assembly)
        if states:
            data["chromatin_states"] = self._process_chromatin_states(states)
        
        # Process regulatory evidence
        data["regulatory_evidence"] = {
            "peaks": score_data.get("peaks", []),
            "motifs": score_data.get("motifs", []),
            "qtls": score_data.get("qtls", []),
            "chromatin_states": score_data.get("chromatin_states", [])
        }
        
        return data
    
    def collect_region_data(self, chromosome: str, start: int, end: int,
                          assembly: str = None, **kwargs) -> Dict[str, Any]:
        """Collect regulatory variants in a region.
        
        Args:
            chromosome: Chromosome
            start: Start position
            end: End position
            assembly: Genome assembly
            **kwargs: Additional parameters
        
        Returns:
            Region regulatory data
        """
        if assembly is None:
            assembly = self.default_assembly
        
        logger.info(f"Collecting RegulomeDB data for region {chromosome}:{start}-{end} on {assembly}")
        
        # Query region
        result = self.api_client.query_region(chromosome, start, end, assembly)
        
        data = {
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "assembly": assembly,
            "variant_count": 0,
            "variants": []
        }
        
        if result and "@graph" in result:
            variants = result["@graph"]
            data["variant_count"] = len(variants)
            
            # Process variants
            score_distribution = {
                "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": []
            }
            
            for variant in variants[:100]:  # Process top 100 variants
                variant_info = {
                    "position": variant.get("coordinates", {}).get("start"),
                    "rsid": variant.get("rsid"),
                    "score": variant.get("regulome_score", {}).get("ranking"),
                    "probability": variant.get("regulome_score", {}).get("probability"),
                    "peak_count": len(variant.get("peaks", [])),
                    "motif_count": len(variant.get("motifs", [])),
                    "qtl_count": len(variant.get("qtls", []))
                }
                
                data["variants"].append(variant_info)
                
                # Categorize by score
                score = variant_info.get("score")
                if score and score[0] in score_distribution:
                    score_distribution[score[0]].append(variant_info)
            
            data["score_distribution"] = {
                score: {
                    "count": len(variants),
                    "examples": variants[:3]  # Top 3 examples per score
                }
                for score, variants in score_distribution.items()
            }
        
        return data
    
    def _process_regulatory_evidence(self, variant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process regulatory evidence from variant data."""
        evidence = {
            "summary": [],
            "peak_count": len(variant_data.get("peaks", [])),
            "motif_count": len(variant_data.get("motifs", [])),
            "qtl_count": len(variant_data.get("qtls", [])),
            "chromatin_state_count": len(variant_data.get("chromatin_states", []))
        }
        
        # Build evidence summary
        if evidence["peak_count"] > 0:
            evidence["summary"].append(f"{evidence['peak_count']} chromatin accessibility peaks")
        if evidence["motif_count"] > 0:
            evidence["summary"].append(f"{evidence['motif_count']} TF binding motifs")
        if evidence["qtl_count"] > 0:
            evidence["summary"].append(f"{evidence['qtl_count']} QTLs")
        if evidence["chromatin_state_count"] > 0:
            evidence["summary"].append(f"{evidence['chromatin_state_count']} chromatin states")
        
        return evidence
    
    def _process_peaks(self, peaks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chromatin accessibility peaks."""
        processed = []
        
        for peak in peaks[:20]:  # Process top 20 peaks
            peak_info = {
                "dataset": peak.get("dataset"),
                "file": peak.get("file"),
                "biosample": peak.get("biosample"),
                "organ": peak.get("organ"),
                "cell_type": peak.get("cell_type"),
                "treatment": peak.get("treatment"),
                "strand": peak.get("strand"),
                "value": peak.get("value")
            }
            processed.append(peak_info)
        
        return processed
    
    def _process_motifs(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process TF binding motifs."""
        processed = []
        
        for motif in motifs[:20]:  # Process top 20 motifs
            motif_info = {
                "tf": motif.get("tf"),
                "motif_id": motif.get("motif_id"),
                "strand": motif.get("strand"),
                "score": motif.get("score"),
                "matched_sequence": motif.get("matched_sequence")
            }
            processed.append(motif_info)
        
        return processed
    
    def _process_qtls(self, qtls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process QTLs."""
        processed = []
        
        for qtl in qtls[:20]:  # Process top 20 QTLs
            qtl_info = {
                "type": qtl.get("qtl_type"),
                "gene": qtl.get("gene"),
                "tissue": qtl.get("tissue"),
                "pvalue": qtl.get("pvalue"),
                "beta": qtl.get("beta"),
                "study": qtl.get("study")
            }
            processed.append(qtl_info)
        
        return processed
    
    def _process_chromatin_states(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chromatin states."""
        processed = []
        
        for state in states[:20]:  # Process top 20 states
            state_info = {
                "state": state.get("state"),
                "biosample": state.get("biosample"),
                "organ": state.get("organ"),
                "cell_type": state.get("cell_type")
            }
            processed.append(state_info)
        
        return processed
    
    def _extract_target_genes(self, variant_data: Dict[str, Any]) -> List[str]:
        """Extract potential target genes from regulatory data."""
        target_genes = set()
        
        # Extract from QTLs
        for qtl in variant_data.get("qtls", []):
            if qtl.get("gene"):
                target_genes.add(qtl["gene"])
        
        # Extract from nearby features
        for peak in variant_data.get("peaks", []):
            if peak.get("target_gene"):
                target_genes.add(peak["target_gene"])
        
        return list(target_genes)
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple variants."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect RegulomeDB data for {identifier}: {e}")
                results.append({
                    "identifier": identifier,
                    "error": str(e)
                })
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Variant:
        """Save regulatory variant data to database.
        
        Args:
            data: RegulomeDB data
        
        Returns:
            Saved Variant instance
        """
        # Determine variant ID
        if data.get("rsid"):
            variant_id = data["rsid"]
        else:
            variant_id = f"{data['chromosome']}:{data['position']}"
        
        # Check if variant exists
        existing = self.db_session.query(Variant).filter_by(
            rsid=data.get("rsid") if data.get("rsid") else None
        ).first()
        
        if not existing and not data.get("rsid"):
            existing = self.db_session.query(Variant).filter_by(
                chromosome=str(data["chromosome"]),
                position=data["position"]
            ).first()
        
        if existing:
            logger.info(f"Updating existing variant {variant_id}")
            variant = existing
        else:
            variant = Variant(
                id=self.generate_id("regulomedb_variant", variant_id),
                variant_id=variant_id,
                chromosome=str(data.get("chromosome", "Unknown")),
                position=data.get("position", 0),
                reference_allele=data.get("reference_allele", "N"),
                alternate_allele=data.get("alternate_alleles", ["N"])[0] if data.get("alternate_alleles") else "N",
                source="RegulomeDB"
            )
        
        # Update fields
        if data.get("rsid"):
            variant.rsid = data["rsid"]
        
        # Store RegulomeDB-specific data
        if not variant.annotations:
            variant.annotations = {}
        
        variant.annotations["regulomedb"] = {
            "assembly": data.get("assembly"),
            "regulome_score": data.get("regulome_score"),
            "regulatory_evidence": data.get("regulatory_evidence", {}),
            "chromatin_peaks": data.get("chromatin_peaks", [])[:10],  # Store top 10
            "tf_motifs": data.get("tf_motifs", [])[:10],
            "qtls": data.get("qtls", [])[:10],
            "chromatin_states": data.get("chromatin_states", [])[:10],
            "target_genes": data.get("target_genes", [])
        }
        
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved RegulomeDB data for variant {variant_id}")
        
        return variant
