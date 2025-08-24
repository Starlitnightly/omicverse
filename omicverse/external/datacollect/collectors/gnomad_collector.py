"""gnomAD data collector."""

import logging
from typing import Any, Dict, List, Optional

from src.api.gnomad import GnomADClient
from src.models.genomic import Gene, Variant
from .base import BaseCollector
from ..config.config import settings


logger = logging.getLogger(__name__)


class GnomADCollector(BaseCollector):
    """Collector for gnomAD population frequency data."""
    
    def __init__(self, db_session=None):
        api_client = GnomADClient()
        super().__init__(api_client, db_session)
        self.default_dataset = "gnomad_r3"
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Gene symbol, variant ID, or transcript ID
            **kwargs: Additional parameters (type, dataset)
        
        Returns:
            Collected data
        """
        id_type = kwargs.get('type', 'gene')
        dataset = kwargs.get('dataset', self.default_dataset)
        
        if id_type == 'gene':
            return self.collect_gene_data(identifier, dataset, **kwargs)
        elif id_type == 'variant':
            return self.collect_variant_data(identifier, dataset, **kwargs)
        elif id_type == 'transcript':
            return self.collect_transcript_data(identifier, dataset, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_gene_data(self, gene_symbol: str, dataset: str = None, **kwargs) -> Dict[str, Any]:
        """Collect gnomAD data for a gene.
        
        Args:
            gene_symbol: Gene symbol
            dataset: Dataset version
            **kwargs: Additional parameters
        
        Returns:
            Gene data with variants
        """
        if dataset is None:
            dataset = self.default_dataset
        
        logger.info(f"Collecting gnomAD data for gene {gene_symbol} from {dataset}")
        
        # Get gene info with variants
        gene_data = self.api_client.get_gene(gene_symbol, dataset)
        
        if not gene_data:
            raise ValueError(f"Gene {gene_symbol} not found in gnomAD")
        
        data = {
            "gene_symbol": gene_symbol,
            "gene_id": gene_data.get("gene_id"),
            "gene_name": gene_data.get("name"),
            "chromosome": gene_data.get("chrom"),
            "start": gene_data.get("start"),
            "stop": gene_data.get("stop"),
            "strand": gene_data.get("strand"),
            "canonical_transcript": gene_data.get("canonical_transcript_id"),
            "dataset": dataset
        }
        
        # Process variants
        variants = gene_data.get("variants", [])
        data["variant_count"] = len(variants)
        
        # Categorize variants by consequence
        consequences = {}
        lof_variants = []
        rare_variants = []
        
        for variant in variants:
            cons = variant.get("consequence", "unknown")
            if cons not in consequences:
                consequences[cons] = 0
            consequences[cons] += 1
            
            # Track LoF variants
            if variant.get("lof"):
                lof_variants.append({
                    "variant_id": variant["variant_id"],
                    "rsid": variant.get("rsid"),
                    "consequence": cons,
                    "af": variant.get("genome", {}).get("af") or variant.get("exome", {}).get("af", 0)
                })
            
            # Track rare variants (AF < 0.001)
            af = variant.get("genome", {}).get("af") or variant.get("exome", {}).get("af", 0)
            if af and af < 0.001:
                rare_variants.append({
                    "variant_id": variant["variant_id"],
                    "rsid": variant.get("rsid"),
                    "consequence": cons,
                    "af": af
                })
        
        data["consequences"] = consequences
        data["lof_variants"] = lof_variants[:20]  # Top 20 LoF variants
        data["lof_count"] = len(lof_variants)
        data["rare_variants"] = rare_variants[:20]  # Top 20 rare variants
        data["rare_count"] = len(rare_variants)
        
        # Get constraint scores if requested
        if kwargs.get('include_constraint'):
            try:
                constraint = self.api_client.get_constraint_scores(gene_symbol)
                data["constraint"] = constraint
            except Exception as e:
                logger.warning(f"Could not get constraint scores: {e}")
        
        # Store sample variants for detailed analysis
        data["sample_variants"] = variants[:100] if kwargs.get('include_variants') else []
        
        return data
    
    def collect_variant_data(self, variant_id: str, dataset: str = None, **kwargs) -> Dict[str, Any]:
        """Collect gnomAD data for a variant.
        
        Args:
            variant_id: Variant ID (chr-pos-ref-alt format)
            dataset: Dataset version
            **kwargs: Additional parameters
        
        Returns:
            Variant data with population frequencies
        """
        if dataset is None:
            dataset = self.default_dataset
        
        logger.info(f"Collecting gnomAD data for variant {variant_id} from {dataset}")
        
        # Get variant info
        variant_data = self.api_client.get_variant(variant_id, dataset)
        
        if not variant_data:
            raise ValueError(f"Variant {variant_id} not found in gnomAD")
        
        data = {
            "variant_id": variant_id,
            "chromosome": variant_data.get("chrom"),
            "position": variant_data.get("pos"),
            "reference": variant_data.get("ref"),
            "alternate": variant_data.get("alt"),
            "rsid": variant_data.get("rsid"),
            "dataset": dataset
        }
        
        # Process genome frequencies
        if variant_data.get("genome"):
            genome = variant_data["genome"]
            data["genome_frequencies"] = {
                "ac": genome.get("ac"),
                "an": genome.get("an"),
                "af": genome.get("af"),
                "homozygote_count": genome.get("homozygote_count"),
                "hemizygote_count": genome.get("hemizygote_count"),
                "filters": genome.get("filters", [])
            }
            
            # Population frequencies
            if genome.get("populations"):
                pop_freqs = {}
                for pop in genome["populations"]:
                    pop_id = pop["id"]
                    pop_freqs[pop_id] = {
                        "ac": pop.get("ac"),
                        "an": pop.get("an"),
                        "af": pop.get("ac", 0) / pop.get("an", 1) if pop.get("an") else 0,
                        "homozygotes": pop.get("homozygote_count", 0)
                    }
                data["genome_population_frequencies"] = pop_freqs
        
        # Process exome frequencies
        if variant_data.get("exome"):
            exome = variant_data["exome"]
            data["exome_frequencies"] = {
                "ac": exome.get("ac"),
                "an": exome.get("an"),
                "af": exome.get("af"),
                "homozygote_count": exome.get("homozygote_count"),
                "hemizygote_count": exome.get("hemizygote_count"),
                "filters": exome.get("filters", [])
            }
            
            # Population frequencies
            if exome.get("populations"):
                pop_freqs = {}
                for pop in exome["populations"]:
                    pop_id = pop["id"]
                    pop_freqs[pop_id] = {
                        "ac": pop.get("ac"),
                        "an": pop.get("an"),
                        "af": pop.get("ac", 0) / pop.get("an", 1) if pop.get("an") else 0,
                        "homozygotes": pop.get("homozygote_count", 0)
                    }
                data["exome_population_frequencies"] = pop_freqs
        
        # Transcript consequences
        if variant_data.get("transcript_consequences"):
            consequences = []
            for cons in variant_data["transcript_consequences"]:
                consequences.append({
                    "gene_symbol": cons.get("gene_symbol"),
                    "transcript_id": cons.get("transcript_id"),
                    "consequence": cons.get("consequence"),
                    "hgvsc": cons.get("hgvsc"),
                    "hgvsp": cons.get("hgvsp"),
                    "lof": cons.get("lof"),
                    "polyphen": cons.get("polyphen_prediction"),
                    "sift": cons.get("sift_prediction")
                })
            data["transcript_consequences"] = consequences
        
        # In silico predictors
        if variant_data.get("in_silico_predictors"):
            predictors = variant_data["in_silico_predictors"]
            data["predictions"] = {
                "cadd_phred": predictors.get("cadd", {}).get("phred"),
                "revel_score": predictors.get("revel", {}).get("score"),
                "splice_ai": predictors.get("splice_ai", {})
            }
        
        # Quality metrics if requested
        if kwargs.get('include_quality') and variant_data.get("quality_metrics"):
            data["quality_metrics"] = variant_data["quality_metrics"]
        
        return data
    
    def collect_transcript_data(self, transcript_id: str, dataset: str = None, **kwargs) -> Dict[str, Any]:
        """Collect gnomAD data for a transcript.
        
        Args:
            transcript_id: Transcript ID
            dataset: Dataset version
            **kwargs: Additional parameters
        
        Returns:
            Transcript data with variants
        """
        if dataset is None:
            dataset = self.default_dataset
        
        logger.info(f"Collecting gnomAD data for transcript {transcript_id} from {dataset}")
        
        # Get transcript info
        transcript_data = self.api_client.get_transcript(transcript_id, dataset)
        
        if not transcript_data:
            raise ValueError(f"Transcript {transcript_id} not found in gnomAD")
        
        data = {
            "transcript_id": transcript_id,
            "gene_id": transcript_data.get("gene_id"),
            "gene_symbol": transcript_data.get("gene_symbol"),
            "chromosome": transcript_data.get("chrom"),
            "start": transcript_data.get("start"),
            "stop": transcript_data.get("stop"),
            "strand": transcript_data.get("strand"),
            "dataset": dataset
        }
        
        # Process exons
        if transcript_data.get("exons"):
            data["exons"] = transcript_data["exons"]
            data["exon_count"] = len(transcript_data["exons"])
        
        # Process variants
        variants = transcript_data.get("variants", [])
        data["variant_count"] = len(variants)
        
        # Categorize variants
        coding_variants = []
        lof_variants = []
        
        for variant in variants:
            if variant.get("hgvsc"):
                coding_variants.append({
                    "variant_id": variant["variant_id"],
                    "rsid": variant.get("rsid"),
                    "consequence": variant.get("consequence"),
                    "hgvsc": variant.get("hgvsc"),
                    "hgvsp": variant.get("hgvsp"),
                    "af": variant.get("genome", {}).get("af") or variant.get("exome", {}).get("af", 0)
                })
            
            if variant.get("lof"):
                lof_variants.append({
                    "variant_id": variant["variant_id"],
                    "rsid": variant.get("rsid"),
                    "consequence": variant.get("consequence"),
                    "af": variant.get("genome", {}).get("af") or variant.get("exome", {}).get("af", 0)
                })
        
        data["coding_variants"] = coding_variants[:50]
        data["coding_count"] = len(coding_variants)
        data["lof_variants"] = lof_variants
        data["lof_count"] = len(lof_variants)
        
        return data
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect gnomAD data for {identifier}: {e}")
        return results
    
    def save_gene_data(self, data: Dict[str, Any]) -> Gene:
        """Save gene data to database.
        
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
                id=self.generate_id("gnomad_gene", gene_symbol),
                gene_id=data.get("gene_id", f"gnomad:{gene_symbol}"),
                source="gnomAD"
            )
        
        # Update fields
        gene.symbol = gene_symbol
        gene.name = data.get("gene_name")
        gene.chromosome = data.get("chromosome")
        gene.start_position = data.get("start")
        gene.end_position = data.get("stop")
        gene.strand = data.get("strand")
        
        # Store gnomAD-specific data
        gene.gnomad_data = {
            "dataset": data.get("dataset"),
            "variant_count": data.get("variant_count", 0),
            "lof_count": data.get("lof_count", 0),
            "rare_count": data.get("rare_count", 0),
            "consequences": data.get("consequences", {}),
            "constraint": data.get("constraint", {}),
            "canonical_transcript": data.get("canonical_transcript")
        }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved gene {gene_symbol} with {data.get('variant_count', 0)} variants")
        
        return gene
    
    def save_variant_data(self, data: Dict[str, Any]) -> Variant:
        """Save variant data to database.
        
        Args:
            data: Collected variant data
        
        Returns:
            Saved Variant instance
        """
        variant_id = data["variant_id"]
        
        # Check if variant exists
        existing = self.db_session.query(Variant).filter_by(
            variant_id=variant_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing variant {variant_id}")
            variant = existing
        else:
            variant = Variant(
                id=self.generate_id("gnomad_variant", variant_id),
                variant_id=variant_id,
                source="gnomAD"
            )
        
        # Update fields
        variant.chromosome = str(data.get("chromosome"))
        variant.position = data.get("position")
        variant.reference_allele = data.get("reference")
        variant.alternate_allele = data.get("alternate")
        variant.rsid = data.get("rsid")
        
        # Store population frequencies
        variant.gnomad_data = {
            "dataset": data.get("dataset"),
            "genome_frequencies": data.get("genome_frequencies", {}),
            "exome_frequencies": data.get("exome_frequencies", {}),
            "genome_populations": data.get("genome_population_frequencies", {}),
            "exome_populations": data.get("exome_population_frequencies", {}),
            "predictions": data.get("predictions", {}),
            "transcript_consequences": data.get("transcript_consequences", [])[:5]  # Store top 5
        }
        
        # Set overall allele frequency
        if data.get("genome_frequencies"):
            variant.allele_frequency = data["genome_frequencies"].get("af")
        elif data.get("exome_frequencies"):
            variant.allele_frequency = data["exome_frequencies"].get("af")
        
        if not existing:
            self.db_session.add(variant)
        
        self.db_session.commit()
        logger.info(f"Saved variant {variant_id}")
        
        return variant
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save data to database based on data type."""
        if "gene_symbol" in data and "variants" not in data:
            return self.save_gene_data(data)
        elif "variant_id" in data and "position" in data:
            return self.save_variant_data(data)
        else:
            logger.warning("Cannot determine data type to save")
            return None
    
    def get_region_variants(self, chrom: str, start: int, stop: int,
                           dataset: str = None) -> List[Dict[str, Any]]:
        """Get variants in a genomic region.
        
        Args:
            chrom: Chromosome
            start: Start position
            stop: Stop position
            dataset: Dataset version
        
        Returns:
            Variants in region
        """
        if dataset is None:
            dataset = self.default_dataset
        
        logger.info(f"Getting gnomAD variants for {chrom}:{start}-{stop}")
        return self.api_client.get_region_variants(chrom, start, stop, dataset)
    
    def search_by_rsid(self, rsid: str, dataset: str = None) -> List[Dict[str, Any]]:
        """Search for variants by rsID.
        
        Args:
            rsid: rsID
            dataset: Dataset version
        
        Returns:
            Matching variants
        """
        if dataset is None:
            dataset = self.default_dataset
        
        logger.info(f"Searching gnomAD for rsID {rsid}")
        return self.api_client.search_variants_by_rsid(rsid, dataset)
