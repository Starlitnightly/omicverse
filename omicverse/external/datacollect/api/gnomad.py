"""gnomAD API client."""

import logging
from typing import Any, Dict, List, Optional
import time
import requests
import json

from .base import BaseAPIClient
from ..config import settings


logger = logging.getLogger(__name__)


class GnomADClient(BaseAPIClient):
    """Client for gnomAD GraphQL API.
    
    gnomAD (Genome Aggregation Database) is a resource that aggregates
    and harmonizes exome and genome sequencing data from large-scale
    sequencing projects to provide allele frequencies across populations.
    
    API Documentation: https://gnomad.broadinstitute.org/api
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://gnomad.broadinstitute.org/api")
        rate_limit = kwargs.pop("rate_limit", 10)
        super().__init__(
            base_url=base_url,
            rate_limit=rate_limit,
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get gnomAD-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def query(self, graphql_query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query.
        
        Args:
            graphql_query: GraphQL query string
            variables: Optional query variables
        
        Returns:
            Query results
        """
        payload = {"query": graphql_query}
        if variables:
            payload["variables"] = variables
        
        last_err = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.base_url,
                    json=payload,
                    headers=self.get_default_headers()
                )
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    # Attach response body for easier debugging
                    body = ""
                    try:
                        body = response.text[:1000]
                    except Exception:
                        pass
                    raise requests.exceptions.HTTPError(f"{e}; body: {body}") from e
                return response.json()
            except requests.exceptions.RequestException as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        # Exhausted retries
        raise last_err
    
    def get_gene(self, gene_symbol: str, dataset: str = "gnomad_r3") -> Dict[str, Any]:
        """Get gene information and variants.
        
        Args:
            gene_symbol: Gene symbol
            dataset: Dataset version (gnomad_r3, gnomad_r2_1, exac)
        
        Returns:
            Gene information with variants
        """
        query = """
        query GeneInfo($geneSymbol: String!, $dataset: DatasetId!) {
            gene(gene_symbol: $geneSymbol) {
                gene_id
                gene_symbol
                name
                canonical_transcript_id
                chrom
                start
                stop
                strand
                variants(dataset: $dataset) {
                    variant_id
                    pos
                    ref
                    alt
                    rsid
                    consequence
                    hgvs
                    lof
                    genome {
                        ac
                        an
                        af
                        homozygote_count
                    }
                    exome {
                        ac
                        an
                        af
                        homozygote_count
                    }
                    populations {
                        id
                        ac
                        an
                        homozygote_count
                    }
                }
            }
        }
        """
        
        variables = {
            "geneSymbol": gene_symbol,
            "dataset": dataset
        }
        
        result = self.query(query, variables)
        gene_data = result.get("data", {}).get("gene")
        return gene_data if gene_data is not None else {}
    
    def get_variant(self, variant_id: str, dataset: str = "gnomad_r3") -> Dict[str, Any]:
        """Get variant information.
        
        Args:
            variant_id: Variant ID (format: chr-pos-ref-alt)
            dataset: Dataset version
        
        Returns:
            Variant information
        """
        query = """
        query VariantInfo($variantId: String!, $dataset: DatasetId!) {
            variant(variant_id: $variantId, dataset: $dataset) {
                variant_id
                chrom
                pos
                ref
                alt
                rsid
                reference_genome
                quality_metrics {
                    allele_balance {
                        alt {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                    }
                    genotype_depth {
                        all {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                        alt {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                    }
                    genotype_quality {
                        all {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                        alt {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                    }
                    site_quality_metrics {
                        BaseQRankSum
                        ClippingRankSum
                        DP
                        FS
                        InbreedingCoeff
                        MQ
                        MQRankSum
                        QD
                        ReadPosRankSum
                        SOR
                        VQSLOD
                        VQSR_NEGATIVE_TRAIN_SITE
                        VQSR_POSITIVE_TRAIN_SITE
                    }
                }
                genome {
                    ac
                    an
                    af
                    homozygote_count
                    hemizygote_count
                    filters
                    populations {
                        id
                        ac
                        an
                        homozygote_count
                        hemizygote_count
                    }
                    age_distribution {
                        het {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                        hom {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                    }
                }
                exome {
                    ac
                    an
                    af
                    homozygote_count
                    hemizygote_count
                    filters
                    populations {
                        id
                        ac
                        an
                        homozygote_count
                        hemizygote_count
                    }
                    age_distribution {
                        het {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                        hom {
                            bin_edges
                            bin_freq
                            n_smaller
                            n_larger
                        }
                    }
                }
                transcript_consequences {
                    gene_id
                    gene_symbol
                    transcript_id
                    consequence
                    hgvsc
                    hgvsp
                    lof
                    lof_filter
                    lof_flags
                    polyphen_prediction
                    sift_prediction
                }
                in_silico_predictors {
                    cadd {
                        phred
                        raw
                    }
                    revel {
                        score
                    }
                    splice_ai {
                        delta_score
                        splice_consequence
                    }
                }
            }
        }
        """
        
        variables = {
            "variantId": variant_id,
            "dataset": dataset
        }
        
        result = self.query(query, variables)
        return result.get("data", {}).get("variant", {})
    
    def search_variants_by_rsid(self, rsid: str, dataset: str = "gnomad_r3") -> List[Dict[str, Any]]:
        """Search for variants by rsID.
        
        Args:
            rsid: rsID
            dataset: Dataset version
        
        Returns:
            List of variants with this rsID
        """
        query = """
        query SearchByRsid($rsid: String!, $dataset: DatasetId!) {
            searchVariants(query: $rsid, dataset: $dataset) {
                variant_id
                chrom
                pos
                ref
                alt
                rsid
                genome {
                    af
                }
                exome {
                    af
                }
            }
        }
        """
        
        variables = {
            "rsid": rsid,
            "dataset": dataset
        }
        
        result = self.query(query, variables)
        return result.get("data", {}).get("searchVariants", [])
    
    def get_region_variants(self, chrom: str, start: int, stop: int,
                           dataset: str = "gnomad_r3") -> List[Dict[str, Any]]:
        """Get variants in a genomic region.
        
        Args:
            chrom: Chromosome
            start: Start position
            stop: Stop position
            dataset: Dataset version
        
        Returns:
            Variants in the region
        """
        query = """
        query RegionVariants($chrom: String!, $start: Int!, $stop: Int!, $dataset: DatasetId!) {
            region(chrom: $chrom, start: $start, stop: $stop) {
                variants(dataset: $dataset) {
                    variant_id
                    pos
                    ref
                    alt
                    rsid
                    consequence
                    genome {
                        ac
                        an
                        af
                    }
                    exome {
                        ac
                        an
                        af
                    }
                }
            }
        }
        """
        
        variables = {
            "chrom": chrom,
            "start": start,
            "stop": stop,
            "dataset": dataset
        }
        
        result = self.query(query, variables)
        region_data = result.get("data", {}).get("region", {})
        return region_data.get("variants", [])
    
    def get_transcript(self, transcript_id: str, dataset: str = "gnomad_r3") -> Dict[str, Any]:
        """Get transcript information with variants.
        
        Args:
            transcript_id: Transcript ID
            dataset: Dataset version
        
        Returns:
            Transcript information
        """
        query = """
        query TranscriptInfo($transcriptId: String!, $dataset: DatasetId!) {
            transcript(transcript_id: $transcriptId) {
                transcript_id
                gene_id
                gene_symbol
                chrom
                start
                stop
                strand
                exons {
                    feature_type
                    start
                    stop
                }
                variants(dataset: $dataset) {
                    variant_id
                    pos
                    ref
                    alt
                    rsid
                    consequence
                    hgvsc
                    hgvsp
                    lof
                    genome {
                        af
                    }
                    exome {
                        af
                    }
                }
            }
        }
        """
        
        variables = {
            "transcriptId": transcript_id,
            "dataset": dataset
        }
        
        result = self.query(query, variables)
        return result.get("data", {}).get("transcript", {})
    
    def get_constraint_scores(self, gene_symbol: str) -> Dict[str, Any]:
        """Get gene constraint scores.
        
        Args:
            gene_symbol: Gene symbol
        
        Returns:
            Constraint scores
        """
        query = """
        query GeneConstraint($geneSymbol: String!) {
            gene(gene_symbol: $geneSymbol) {
                gene_id
                gene_symbol
                constraint {
                    exp_lof
                    exp_mis
                    exp_syn
                    obs_lof
                    obs_mis
                    obs_syn
                    oe_lof
                    oe_lof_lower
                    oe_lof_upper
                    oe_mis
                    oe_mis_lower
                    oe_mis_upper
                    oe_syn
                    oe_syn_lower
                    oe_syn_upper
                    lof_z
                    mis_z
                    syn_z
                    pLI
                    pRec
                    pNull
                }
            }
        }
        """
        
        variables = {"geneSymbol": gene_symbol}
        
        result = self.query(query, variables)
        gene_data = result.get("data", {}).get("gene", {})
        return gene_data.get("constraint", {})
    
    def get_coverage(self, chrom: str, start: int, stop: int,
                    dataset: str = "gnomad_r3", data_type: str = "exome") -> Dict[str, Any]:
        """Get coverage statistics for a region.
        
        Args:
            chrom: Chromosome
            start: Start position
            stop: Stop position
            dataset: Dataset version
            data_type: exome or genome
        
        Returns:
            Coverage statistics
        """
        query = """
        query Coverage($chrom: String!, $start: Int!, $stop: Int!, $dataset: DatasetId!, $dataType: DataType!) {
            region(chrom: $chrom, start: $start, stop: $stop) {
                coverage(dataset: $dataset, data_type: $dataType) {
                    pos
                    mean
                    median
                    over_1
                    over_5
                    over_10
                    over_15
                    over_20
                    over_25
                    over_30
                    over_50
                    over_100
                }
            }
        }
        """
        
        variables = {
            "chrom": chrom,
            "start": start,
            "stop": stop,
            "dataset": dataset,
            "dataType": data_type
        }
        
        result = self.query(query, variables)
        region_data = result.get("data", {}).get("region", {})
        return region_data.get("coverage", [])
