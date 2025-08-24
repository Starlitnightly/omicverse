"""OpenTargets Genetics API client."""

import logging
from typing import Any, Dict, List, Optional
import json

from .base import BaseAPIClient
from ..config import settings


logger = logging.getLogger(__name__)


class OpenTargetsGeneticsClient(BaseAPIClient):
    """Client for OpenTargets Genetics GraphQL API.
    
    OpenTargets Genetics provides genetic association data from GWAS studies
    and functional genomics data to identify and prioritize drug targets.
    
    API Documentation: https://genetics-docs.opentargets.org/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://api.genetics.opentargets.org/graphql")
        rate_limit = kwargs.pop("rate_limit", 10)
        super().__init__(
            base_url=base_url,
            rate_limit=rate_limit,
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get OpenTargets-specific headers."""
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
        
        response = self.session.post(
            self.base_url,
            json=payload,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_gene_info(self, gene_id: str) -> Dict[str, Any]:
        """Get gene information.
        
        Args:
            gene_id: Ensembl gene ID
        
        Returns:
            Gene information
        """
        query = """
        query GeneInfo($geneId: String!) {
            geneInfo(geneId: $geneId) {
                id
                symbol
                description
                chromosome
                start
                end
                bioType
                tss
                strand
            }
        }
        """
        
        variables = {"geneId": gene_id}
        result = self.query(query, variables)
        gene_data = result.get("data", {}).get("geneInfo")
        return gene_data if gene_data is not None else {}
    
    def get_variant_info(self, variant_id: str) -> Dict[str, Any]:
        """Get variant information.
        
        Args:
            variant_id: Variant ID (format: chr_pos_ref_alt)
        
        Returns:
            Variant information
        """
        query = """
        query VariantInfo($variantId: String!) {
            variantInfo(variantId: $variantId) {
                id
                rsId
                chromosome
                position
                refAllele
                altAllele
                nearestGene {
                    id
                    symbol
                    distance
                }
                nearestGeneDistance
                mostSevereConsequence
                caddPhred
                gnomadAFR
                gnomadAMR
                gnomadEAS
                gnomadEUR
                gnomadNFE
                gnomadSAS
            }
        }
        """
        
        variables = {"variantId": variant_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("variantInfo", {})
    
    def get_study_info(self, study_id: str) -> Dict[str, Any]:
        """Get GWAS study information.
        
        Args:
            study_id: Study ID
        
        Returns:
            Study information
        """
        query = """
        query StudyInfo($studyId: String!) {
            studyInfo(studyId: $studyId) {
                studyId
                traitReported
                traitCategory
                pubAuthor
                pubDate
                pubJournal
                pmid
                nCases
                nTotal
                nInitial
                nReplication
                hasSumstats
            }
        }
        """
        
        variables = {"studyId": study_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("studyInfo", {})
    
    def get_associations_for_gene(self, gene_id: str, page_size: int = 50) -> Dict[str, Any]:
        """Get GWAS associations for a gene.
        
        Args:
            gene_id: Ensembl gene ID
            page_size: Number of results per page
        
        Returns:
            Gene associations
        """
        query = """
        query GeneAssociations($geneId: String!, $pageSize: Int) {
            geneInfo(geneId: $geneId) {
                id
                symbol
            }
            associatedStudiesForGene(geneId: $geneId, pageSize: $pageSize) {
                study {
                    studyId
                    traitReported
                    pmid
                    pubAuthor
                    pubDate
                }
                variant {
                    id
                    rsId
                }
                pval
                beta
                oddsRatio
                ci95Lower
                ci95Upper
            }
        }
        """
        
        variables = {"geneId": gene_id, "pageSize": page_size}
        result = self.query(query, variables)
        return result.get("data", {})
    
    def get_associations_for_variant(self, variant_id: str, page_size: int = 50) -> Dict[str, Any]:
        """Get GWAS associations for a variant.
        
        Args:
            variant_id: Variant ID
            page_size: Number of results per page
        
        Returns:
            Variant associations
        """
        query = """
        query VariantAssociations($variantId: String!, $pageSize: Int) {
            variantInfo(variantId: $variantId) {
                id
                rsId
            }
            associatedStudiesForVariant(variantId: $variantId, pageSize: $pageSize) {
                study {
                    studyId
                    traitReported
                    pmid
                    pubAuthor
                }
                pval
                beta
                oddsRatio
                ci95Lower
                ci95Upper
            }
        }
        """
        
        variables = {"variantId": variant_id, "pageSize": page_size}
        result = self.query(query, variables)
        return result.get("data", {})
    
    def get_colocalization(self, gene_id: str, study_id: str) -> Dict[str, Any]:
        """Get colocalization analysis between gene and study.
        
        Args:
            gene_id: Ensembl gene ID
            study_id: Study ID
        
        Returns:
            Colocalization data
        """
        query = """
        query Colocalization($geneId: String!, $studyId: String!) {
            colocalisationForGene(geneId: $geneId) {
                leftStudy {
                    studyId
                    traitReported
                }
                rightStudy {
                    studyId
                    traitReported
                }
                h0
                h1
                h2
                h3
                h4
                log2h4h3
            }
        }
        """
        
        variables = {"geneId": gene_id, "studyId": study_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("colocalisationForGene", [])
    
    def get_pheWAS(self, variant_id: str, page_size: int = 100) -> Dict[str, Any]:
        """Get PheWAS (phenome-wide association) data for a variant.
        
        Args:
            variant_id: Variant ID
            page_size: Number of results
        
        Returns:
            PheWAS associations
        """
        query = """
        query PheWAS($variantId: String!, $pageSize: Int) {
            pheWAS(variantId: $variantId, pageSize: $pageSize) {
                study {
                    studyId
                    traitReported
                    traitCategory
                }
                pval
                beta
                oddsRatio
            }
        }
        """
        
        variables = {"variantId": variant_id, "pageSize": page_size}
        result = self.query(query, variables)
        return result.get("data", {}).get("pheWAS", [])
    
    def get_credible_sets(self, study_id: str, variant_id: str) -> Dict[str, Any]:
        """Get credible set variants for a lead variant in a study.
        
        Args:
            study_id: Study ID
            variant_id: Lead variant ID
        
        Returns:
            Credible set variants
        """
        query = """
        query CredibleSets($studyId: String!, $variantId: String!) {
            credibleSets(studyId: $studyId, variantId: $variantId) {
                variant {
                    id
                    rsId
                }
                posteriorProbability
                pval
                beta
                standardError
            }
        }
        """
        
        variables = {"studyId": study_id, "variantId": variant_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("credibleSets", [])
    
    def search_studies(self, query_string: str, page_size: int = 50) -> List[Dict[str, Any]]:
        """Search for GWAS studies.
        
        Args:
            query_string: Search query
            page_size: Number of results
        
        Returns:
            Matching studies
        """
        query = """
        query SearchStudies($queryString: String!, $pageSize: Int) {
            search(queryString: $queryString, pageSize: $pageSize) {
                studies {
                    studyId
                    traitReported
                    pubAuthor
                    pmid
                }
            }
        }
        """
        
        variables = {"queryString": query_string, "pageSize": page_size}
        result = self.query(query, variables)
        return result.get("data", {}).get("search", {}).get("studies", [])
    
    def get_manhattan_plot_data(self, study_id: str) -> Dict[str, Any]:
        """Get data for Manhattan plot visualization.
        
        Args:
            study_id: Study ID
        
        Returns:
            Manhattan plot data
        """
        query = """
        query ManhattanPlot($studyId: String!) {
            manhattan(studyId: $studyId) {
                associations {
                    variant {
                        id
                        chromosome
                        position
                    }
                    pval
                }
            }
        }
        """
        
        variables = {"studyId": study_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("manhattan", {})
