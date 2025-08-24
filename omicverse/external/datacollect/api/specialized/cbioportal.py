"""cBioPortal API client for cancer genomics data."""

import logging
from typing import Any, Dict, List, Optional, Union

from ..base import BaseAPIClient
from ...config import settings

logger = logging.getLogger(__name__)


class cBioPortalClient(BaseAPIClient):
    """Client for cBioPortal REST API.
    
    cBioPortal provides visualization, analysis and download of large-scale
    cancer genomics data sets.
    
    API Documentation: https://www.cbioportal.org/api/swagger-ui/index.html
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://www.cbioportal.org/api",
            rate_limit=kwargs.get("rate_limit", 10),
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get cBioPortal-specific headers."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    
    # Cancer Studies
    def get_all_studies(
        self,
        keyword: Optional[str] = None,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get all cancer studies.
        
        Args:
            keyword: Search keyword
            projection: Level of detail (SUMMARY, DETAILED, META)
        
        Returns:
            List of cancer studies
        """
        params = {"projection": projection}
        if keyword:
            params["keyword"] = keyword
        
        response = self.get("/studies", params=params)
        return response.json()
    
    def get_study(self, study_id: str) -> Dict[str, Any]:
        """Get a specific cancer study.
        
        Args:
            study_id: Study identifier (e.g., 'tcga_pan_can_atlas_2018')
        
        Returns:
            Study details
        """
        endpoint = f"/studies/{study_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Samples
    def get_samples_in_study(
        self,
        study_id: str,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get all samples in a study.
        
        Args:
            study_id: Study identifier
            projection: Level of detail
        
        Returns:
            List of samples
        """
        endpoint = f"/studies/{study_id}/samples"
        params = {"projection": projection}
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_sample(
        self,
        study_id: str,
        sample_id: str
    ) -> Dict[str, Any]:
        """Get a specific sample.
        
        Args:
            study_id: Study identifier
            sample_id: Sample identifier
        
        Returns:
            Sample details
        """
        endpoint = f"/studies/{study_id}/samples/{sample_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Patients
    def get_patients_in_study(
        self,
        study_id: str,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get all patients in a study.
        
        Args:
            study_id: Study identifier
            projection: Level of detail
        
        Returns:
            List of patients
        """
        endpoint = f"/studies/{study_id}/patients"
        params = {"projection": projection}
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_patient(
        self,
        study_id: str,
        patient_id: str
    ) -> Dict[str, Any]:
        """Get a specific patient.
        
        Args:
            study_id: Study identifier
            patient_id: Patient identifier
        
        Returns:
            Patient details
        """
        endpoint = f"/studies/{study_id}/patients/{patient_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Molecular Profiles
    def get_molecular_profiles(
        self,
        study_id: str,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get molecular profiles in a study.
        
        Args:
            study_id: Study identifier
            projection: Level of detail
        
        Returns:
            List of molecular profiles
        """
        endpoint = f"/studies/{study_id}/molecular-profiles"
        params = {"projection": projection}
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_molecular_profile(self, molecular_profile_id: str) -> Dict[str, Any]:
        """Get a specific molecular profile.
        
        Args:
            molecular_profile_id: Molecular profile identifier
        
        Returns:
            Molecular profile details
        """
        endpoint = f"/molecular-profiles/{molecular_profile_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Mutations
    def get_mutations_in_molecular_profile(
        self,
        molecular_profile_id: str,
        sample_list_id: Optional[str] = None,
        entrez_gene_ids: Optional[List[int]] = None,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get mutations in a molecular profile.
        
        Args:
            molecular_profile_id: Molecular profile identifier
            sample_list_id: Sample list identifier
            entrez_gene_ids: List of Entrez gene IDs
            projection: Level of detail
        
        Returns:
            List of mutations
        """
        endpoint = f"/molecular-profiles/{molecular_profile_id}/mutations"
        params = {"projection": projection}
        
        if sample_list_id:
            params["sampleListId"] = sample_list_id
        if entrez_gene_ids:
            params["entrezGeneId"] = ",".join(map(str, entrez_gene_ids))
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def fetch_mutations(
        self,
        molecular_profile_id: str,
        sample_ids: List[str],
        entrez_gene_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch mutations for specific samples and genes.
        
        Args:
            molecular_profile_id: Molecular profile identifier
            sample_ids: List of sample IDs
            entrez_gene_ids: List of Entrez gene IDs
        
        Returns:
            List of mutations
        """
        endpoint = f"/molecular-profiles/{molecular_profile_id}/mutations/fetch"
        
        data = {
            "sampleIds": sample_ids
        }
        
        if entrez_gene_ids:
            data["entrezGeneIds"] = entrez_gene_ids
        
        response = self.post(endpoint, json=data)
        return response.json()
    
    # Gene Expression
    def get_gene_expression_data(
        self,
        molecular_profile_id: str,
        sample_list_id: str,
        entrez_gene_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Get gene expression data.
        
        Args:
            molecular_profile_id: Molecular profile identifier
            sample_list_id: Sample list identifier
            entrez_gene_ids: List of Entrez gene IDs
        
        Returns:
            Gene expression data
        """
        endpoint = f"/molecular-profiles/{molecular_profile_id}/molecular-data"
        params = {"sampleListId": sample_list_id}
        
        if entrez_gene_ids:
            params["entrezGeneId"] = ",".join(map(str, entrez_gene_ids))
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def fetch_molecular_data(
        self,
        molecular_profile_id: str,
        sample_ids: List[str],
        entrez_gene_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch molecular data for specific samples.
        
        Args:
            molecular_profile_id: Molecular profile identifier
            sample_ids: List of sample IDs
            entrez_gene_ids: List of Entrez gene IDs
        
        Returns:
            Molecular data
        """
        endpoint = f"/molecular-profiles/{molecular_profile_id}/molecular-data/fetch"
        
        data = {
            "sampleIds": sample_ids
        }
        
        if entrez_gene_ids:
            data["entrezGeneIds"] = entrez_gene_ids
        
        response = self.post(endpoint, json=data)
        return response.json()
    
    # Copy Number Alterations
    def get_discrete_cna_data(
        self,
        molecular_profile_id: str,
        sample_list_id: str,
        entrez_gene_ids: Optional[List[int]] = None,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get discrete copy number alteration data.
        
        Args:
            molecular_profile_id: Molecular profile identifier
            sample_list_id: Sample list identifier
            entrez_gene_ids: List of Entrez gene IDs
            projection: Level of detail
        
        Returns:
            CNA data
        """
        endpoint = f"/molecular-profiles/{molecular_profile_id}/discrete-copy-number"
        params = {
            "sampleListId": sample_list_id,
            "projection": projection
        }
        
        if entrez_gene_ids:
            params["entrezGeneId"] = ",".join(map(str, entrez_gene_ids))
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    # Clinical Data
    def get_clinical_data_in_study(
        self,
        study_id: str,
        clinical_data_type: str = "SAMPLE",
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get clinical data for a study.
        
        Args:
            study_id: Study identifier
            clinical_data_type: Type of clinical data (SAMPLE or PATIENT)
            projection: Level of detail
        
        Returns:
            Clinical data
        """
        endpoint = f"/studies/{study_id}/clinical-data"
        params = {
            "clinicalDataType": clinical_data_type,
            "projection": projection
        }
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def fetch_clinical_data(
        self,
        study_id: str,
        sample_ids: Optional[List[str]] = None,
        patient_ids: Optional[List[str]] = None,
        attribute_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch clinical data for specific samples or patients.
        
        Args:
            study_id: Study identifier
            sample_ids: List of sample IDs
            patient_ids: List of patient IDs
            attribute_ids: List of clinical attribute IDs
        
        Returns:
            Clinical data
        """
        endpoint = f"/studies/{study_id}/clinical-data/fetch"
        
        data = {}
        if sample_ids:
            data["sampleIds"] = sample_ids
        if patient_ids:
            data["patientIds"] = patient_ids
        if attribute_ids:
            data["attributeIds"] = attribute_ids
        
        response = self.post(endpoint, json=data)
        return response.json()
    
    # Genes
    def get_gene(self, gene_id: Union[int, str]) -> Dict[str, Any]:
        """Get gene information.
        
        Args:
            gene_id: Entrez gene ID or Hugo gene symbol
        
        Returns:
            Gene information
        """
        endpoint = f"/genes/{gene_id}"
        response = self.get(endpoint)
        return response.json()
    
    def fetch_genes(
        self,
        gene_ids: Optional[List[Union[int, str]]] = None,
        hugo_symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch multiple genes.
        
        Args:
            gene_ids: List of Entrez gene IDs
            hugo_symbols: List of Hugo gene symbols
        
        Returns:
            List of genes
        """
        endpoint = "/genes/fetch"
        
        data = {}
        if gene_ids:
            data["geneIds"] = gene_ids
        if hugo_symbols:
            data["hugoGeneSymbols"] = hugo_symbols
        
        if not data:
            data["geneIds"] = []  # Fetch all genes if no filters
        
        response = self.post(endpoint, json=data)
        return response.json()
    
    # Gene Panels
    def get_gene_panels(self) -> List[Dict[str, Any]]:
        """Get all gene panels.
        
        Returns:
            List of gene panels
        """
        response = self.get("/gene-panels")
        return response.json()
    
    def get_gene_panel(self, panel_id: str) -> Dict[str, Any]:
        """Get a specific gene panel.
        
        Args:
            panel_id: Gene panel identifier
        
        Returns:
            Gene panel details
        """
        endpoint = f"/gene-panels/{panel_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Sample Lists
    def get_sample_lists(self, study_id: str) -> List[Dict[str, Any]]:
        """Get sample lists in a study.
        
        Args:
            study_id: Study identifier
        
        Returns:
            List of sample lists
        """
        endpoint = f"/studies/{study_id}/sample-lists"
        response = self.get(endpoint)
        return response.json()
    
    def get_sample_list(self, sample_list_id: str) -> Dict[str, Any]:
        """Get a specific sample list.
        
        Args:
            sample_list_id: Sample list identifier
        
        Returns:
            Sample list details
        """
        endpoint = f"/sample-lists/{sample_list_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Cancer Types
    def get_cancer_types(self) -> List[Dict[str, Any]]:
        """Get all cancer types.
        
        Returns:
            List of cancer types
        """
        response = self.get("/cancer-types")
        return response.json()
    
    def get_cancer_type(self, cancer_type_id: str) -> Dict[str, Any]:
        """Get a specific cancer type.
        
        Args:
            cancer_type_id: Cancer type identifier
        
        Returns:
            Cancer type details
        """
        endpoint = f"/cancer-types/{cancer_type_id}"
        response = self.get(endpoint)
        return response.json()
    
    # Clinical Attributes
    def get_clinical_attributes(
        self,
        study_id: str,
        projection: str = "SUMMARY"
    ) -> List[Dict[str, Any]]:
        """Get clinical attributes in a study.
        
        Args:
            study_id: Study identifier
            projection: Level of detail
        
        Returns:
            List of clinical attributes
        """
        endpoint = f"/studies/{study_id}/clinical-attributes"
        params = {"projection": projection}
        response = self.get(endpoint, params=params)
        return response.json()
    
    # Server Status
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and version information.
        
        Returns:
            Server status information
        """
        response = self.get("/info")
        return response.json()
