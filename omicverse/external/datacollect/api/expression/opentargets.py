"""OpenTargets Platform API client."""

import logging
from typing import Any, Dict, List, Optional
import json

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class OpenTargetsClient(BaseAPIClient):
    """Client for OpenTargets Platform GraphQL API.
    
    OpenTargets Platform integrates genetics, genomics, transcriptomics, drugs,
    animal models and scientific literature to score and rank target-disease associations.
    
    API Documentation: https://platform-docs.opentargets.org/
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://api.platform.opentargets.org/api/v4/graphql")
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
    
    def get_target(self, ensembl_id: str) -> Dict[str, Any]:
        """Get minimal target information (stable GraphQL fields)."""
        query = """
        query($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
          }
        }
        """
        variables = {"ensemblId": ensembl_id}
        result = self.query(query, variables)
        target_data = result.get("data", {}).get("target")
        return target_data if target_data is not None else {}
    
    def get_disease(self, disease_id: str) -> Dict[str, Any]:
        """Get disease information.
        
        Args:
            disease_id: Disease ID (EFO, Orphanet, etc.)
        
        Returns:
            Disease information
        """
        query = """
        query DiseaseInfo($efoId: String!) {
            disease(efoId: $efoId) {
                id
                name
                description
                synonyms
                therapeuticAreas {
                    id
                    name
                }
                parents {
                    id
                    name
                }
                children {
                    id
                    name
                }
            }
        }
        """
        
        variables = {"efoId": disease_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("disease", {})
    
    def get_associations(self, target_id: str, disease_id: Optional[str] = None,
                        size: int = 50) -> Dict[str, Any]:
        """Get target-disease associations.
        
        Args:
            target_id: Ensembl gene ID
            disease_id: Optional disease ID to filter
            size: Number of results
        
        Returns:
            Association data
        """
        if disease_id:
            query = """
            query AssociationScore($targetId: String!, $diseaseId: String!) {
                target(ensemblId: $targetId) {
                    id
                    approvedSymbol
                    associatedDiseases(efoId: $diseaseId) {
                        score
                        datatypeScores {
                            id
                            score
                        }
                        disease {
                            id
                            name
                        }
                    }
                }
            }
            """
            variables = {"targetId": target_id, "diseaseId": disease_id}
        else:
            query = """
            query TargetAssociations($targetId: String!, $size: Int) {
                target(ensemblId: $targetId) {
                    id
                    approvedSymbol
                    associatedDiseases(page: {size: $size}) {
                        rows {
                            score
                            datatypeScores {
                                id
                                score
                            }
                            disease {
                                id
                                name
                            }
                        }
                        count
                    }
                }
            }
            """
            variables = {"targetId": target_id, "size": size}
        
        result = self.query(query, variables)
        return result.get("data", {}).get("target", {})
    
    def get_evidence(self, target_id: str, disease_id: str,
                    datasource: Optional[str] = None, size: int = 50) -> Dict[str, Any]:
        """Get evidence for target-disease association.
        
        Args:
            target_id: Ensembl gene ID
            disease_id: Disease ID
            datasource: Optional datasource filter
            size: Number of results
        
        Returns:
            Evidence data
        """
        query = """
        query Evidence($targetId: String!, $diseaseId: String!, $datasource: String, $size: Int) {
            evidences(
                ensemblIds: [$targetId]
                efoIds: [$diseaseId]
                datasourceIds: $datasource
                size: $size
            ) {
                rows {
                    id
                    score
                    datasourceId
                    datatypeId
                    targetFromSourceId
                    diseaseFromSourceMappedId
                    publicationYear
                    publicationFirstAuthor
                    literature
                }
                count
            }
        }
        """
        
        variables = {
            "targetId": target_id,
            "diseaseId": disease_id,
            "size": size
        }
        if datasource:
            variables["datasource"] = f"[{datasource}]"
        
        result = self.query(query, variables)
        return result.get("data", {}).get("evidences", {})
    
    def get_drug_info(self, drug_id: str) -> Dict[str, Any]:
        """Get drug information.
        
        Args:
            drug_id: ChEMBL drug ID
        
        Returns:
            Drug information
        """
        query = """
        query DrugInfo($chemblId: String!) {
            drug(chemblId: $chemblId) {
                id
                name
                description
                drugType
                maximumClinicalTrialPhase
                synonyms
                tradeNames
                yearOfFirstApproval
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        targetName
                        targets {
                            id
                            approvedSymbol
                        }
                    }
                }
                indications {
                    rows {
                        disease {
                            id
                            name
                        }
                        maxPhaseForIndication
                    }
                }
                adverseEvents {
                    count
                    criticalValue
                }
            }
        }
        """
        
        variables = {"chemblId": drug_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("drug", {})
    
    def search_targets(self, query_string: str, size: int = 50) -> List[Dict[str, Any]]:
        """Search for targets (genes).
        
        Args:
            query_string: Search query
            size: Number of results
        
        Returns:
            Matching targets
        """
        query = """
        query SearchTargets($queryString: String!, $size: Int) {
            search(queryString: $queryString, entityNames: ["target"], page: {size: $size}) {
                hits {
                    id
                    entity
                    name
                    description
                    score
                }
                total
            }
        }
        """
        
        variables = {"queryString": query_string, "size": size}
        result = self.query(query, variables)
        search_result = result.get("data", {}).get("search", {})
        return search_result.get("hits", [])
    
    def search_diseases(self, query_string: str, size: int = 50) -> List[Dict[str, Any]]:
        """Search for diseases.
        
        Args:
            query_string: Search query
            size: Number of results
        
        Returns:
            Matching diseases
        """
        query = """
        query SearchDiseases($queryString: String!, $size: Int) {
            search(queryString: $queryString, entityNames: ["disease"], page: {size: $size}) {
                hits {
                    id
                    entity
                    name
                    description
                    score
                }
                total
            }
        }
        """
        
        variables = {"queryString": query_string, "size": size}
        result = self.query(query, variables)
        search_result = result.get("data", {}).get("search", {})
        return search_result.get("hits", [])
    
    def search_drugs(self, query_string: str, size: int = 50) -> List[Dict[str, Any]]:
        """Search for drugs.
        
        Args:
            query_string: Search query
            size: Number of results
        
        Returns:
            Matching drugs
        """
        query = """
        query SearchDrugs($queryString: String!, $size: Int) {
            search(queryString: $queryString, entityNames: ["drug"], page: {size: $size}) {
                hits {
                    id
                    entity
                    name
                    description
                    score
                }
                total
            }
        }
        """
        
        variables = {"queryString": query_string, "size": size}
        result = self.query(query, variables)
        search_result = result.get("data", {}).get("search", {})
        return search_result.get("hits", [])
    
    def get_known_drugs_for_target(self, target_id: str) -> Dict[str, Any]:
        """Get known drugs for a target.
        
        Args:
            target_id: Ensembl gene ID
        
        Returns:
            Known drugs data
        """
        query = """
        query KnownDrugs($targetId: String!) {
            target(ensemblId: $targetId) {
                id
                approvedSymbol
                knownDrugs {
                    uniqueDrugs
                    uniqueDiseases
                    uniqueTargets
                    count
                    rows {
                        drug {
                            id
                            name
                            drugType
                            maximumClinicalTrialPhase
                        }
                        disease {
                            id
                            name
                        }
                        phase
                        status
                        mechanismOfAction
                        references {
                            source
                            ids
                        }
                    }
                }
            }
        }
        """
        
        variables = {"targetId": target_id}
        result = self.query(query, variables)
        return result.get("data", {}).get("target", {}).get("knownDrugs", {})
    
    def get_similar_targets(self, target_id: str, threshold: float = 0.5,
                           size: int = 20) -> List[Dict[str, Any]]:
        """Get similar targets based on various criteria.
        
        Args:
            target_id: Ensembl gene ID
            threshold: Similarity threshold
            size: Number of results
        
        Returns:
            Similar targets
        """
        query = """
        query SimilarTargets($targetId: String!, $threshold: Float, $size: Int) {
            target(ensemblId: $targetId) {
                id
                similarEntities(threshold: $threshold, size: $size) {
                    score
                    target {
                        id
                        approvedSymbol
                        approvedName
                    }
                }
            }
        }
        """
        
        variables = {"targetId": target_id, "threshold": threshold, "size": size}
        result = self.query(query, variables)
        target_data = result.get("data", {}).get("target", {})
        return target_data.get("similarEntities", [])
