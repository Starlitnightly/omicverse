"""Guide to PHARMACOLOGY database API client."""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class GtoPdbClient(BaseAPIClient):
    """Client for Guide to PHARMACOLOGY (GtoPdb) API.
    
    The IUPHAR/BPS Guide to PHARMACOLOGY is an expert-curated database of
    drug targets and the substances that act on them.
    
    API Documentation: https://www.guidetopharmacology.org/webServices.jsp
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://www.guidetopharmacology.org/services")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get GtoPdb-specific headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def search_targets(self, query: str, target_type: str = None) -> List[Dict[str, Any]]:
        """Search for drug targets.
        
        Args:
            query: Search query
            target_type: Target type filter (gpcr, lgic, vgic, nuclear, enzyme, transporter, other)
        
        Returns:
            List of matching targets
        """
        endpoint = "/targets"
        params = {"searchTerm": query}
        
        if target_type:
            params["type"] = target_type
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_target(self, target_id: int) -> Dict[str, Any]:
        """Get detailed target information.
        
        Args:
            target_id: GtoPdb target ID
        
        Returns:
            Target details
        """
        endpoint = f"/targets/{target_id}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_target_by_gene(self, gene_symbol: str) -> List[Dict[str, Any]]:
        """Get targets associated with a gene.
        
        Args:
            gene_symbol: Gene symbol
        
        Returns:
            List of targets for the gene
        """
        endpoint = "/targets"
        params = {"geneSymbol": gene_symbol}
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def search_ligands(self, query: str, ligand_type: str = None) -> List[Dict[str, Any]]:
        """Search for ligands.
        
        Args:
            query: Search query
            ligand_type: Ligand type (Synthetic organic, Metabolite, Natural product, Peptide, Antibody, etc.)
        
        Returns:
            List of matching ligands
        """
        endpoint = "/ligands"
        params = {"searchTerm": query}
        
        if ligand_type:
            params["type"] = ligand_type
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_ligand(self, ligand_id: int) -> Dict[str, Any]:
        """Get detailed ligand information.
        
        Args:
            ligand_id: GtoPdb ligand ID
        
        Returns:
            Ligand details
        """
        endpoint = f"/ligands/{ligand_id}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_interactions(self, target_id: int = None, ligand_id: int = None) -> List[Dict[str, Any]]:
        """Get target-ligand interactions.
        
        Args:
            target_id: Target ID filter
            ligand_id: Ligand ID filter
        
        Returns:
            List of interactions
        """
        endpoint = "/interactions"
        params = {}
        
        if target_id:
            params["targetId"] = target_id
        if ligand_id:
            params["ligandId"] = ligand_id
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_diseases(self) -> List[Dict[str, Any]]:
        """Get all diseases in the database.
        
        Returns:
            List of diseases
        """
        endpoint = "/diseases"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_disease(self, disease_id: int) -> Dict[str, Any]:
        """Get disease information.
        
        Args:
            disease_id: Disease ID
        
        Returns:
            Disease details
        """
        endpoint = f"/diseases/{disease_id}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_disease_targets(self, disease_id: int) -> List[Dict[str, Any]]:
        """Get targets associated with a disease.
        
        Args:
            disease_id: Disease ID
        
        Returns:
            List of disease targets
        """
        endpoint = f"/diseases/{disease_id}/targets"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_families(self, family_type: str = None) -> List[Dict[str, Any]]:
        """Get target families.
        
        Args:
            family_type: Family type filter
        
        Returns:
            List of families
        """
        endpoint = "/families"
        params = {}
        
        if family_type:
            params["type"] = family_type
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_family(self, family_id: int) -> Dict[str, Any]:
        """Get family information.
        
        Args:
            family_id: Family ID
        
        Returns:
            Family details
        """
        endpoint = f"/families/{family_id}"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_family_targets(self, family_id: int) -> List[Dict[str, Any]]:
        """Get targets in a family.
        
        Args:
            family_id: Family ID
        
        Returns:
            List of family targets
        """
        endpoint = f"/families/{family_id}/targets"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def search_by_structure(self, smiles: str, similarity: float = 0.8) -> List[Dict[str, Any]]:
        """Search ligands by chemical structure.
        
        Args:
            smiles: SMILES string
            similarity: Similarity threshold (0-1)
        
        Returns:
            List of similar ligands
        """
        endpoint = "/ligands/structure"
        params = {
            "smiles": smiles,
            "similarity": similarity
        }
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_approved_drugs(self) -> List[Dict[str, Any]]:
        """Get all approved drugs.
        
        Returns:
            List of approved drugs
        """
        endpoint = "/ligands"
        params = {"approved": True}
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_clinical_trials(self, target_id: int = None, ligand_id: int = None) -> List[Dict[str, Any]]:
        """Get clinical trial information.
        
        Args:
            target_id: Target ID filter
            ligand_id: Ligand ID filter
        
        Returns:
            List of clinical trials
        """
        endpoint = "/clinicalTrials"
        params = {}
        
        if target_id:
            params["targetId"] = target_id
        if ligand_id:
            params["ligandId"] = ligand_id
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Database statistics
        """
        endpoint = "/stats"
        
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_default_headers()
        )
        response.raise_for_status()
        return response.json()
