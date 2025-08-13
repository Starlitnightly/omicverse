"""Tests for cBioPortal API client."""

import pytest
from unittest.mock import Mock, patch

from omicverse.external.datacollect.api.cbioportal import cBioPortalClient


@pytest.fixture
def mock_studies_response():
    """Sample cBioPortal studies response."""
    return [
        {
            "studyId": "tcga_pan_can_atlas_2018",
            "name": "Pan-Cancer Atlas",
            "description": "TCGA PanCancer Atlas studies",
            "publicStudy": True,
            "groups": ["PUBLIC"],
            "status": 1,
            "cancerTypeId": "mixed",
            "cancerType": {
                "cancerTypeId": "mixed",
                "name": "Mixed",
                "dedicatedColor": "Dark Grey",
                "shortName": "MIXED",
                "parent": "tissue"
            },
            "referenceGenome": "hg19"
        },
        {
            "studyId": "msk_impact_2017",
            "name": "MSK-IMPACT Clinical Sequencing Cohort",
            "description": "Targeted sequencing of tumor samples",
            "publicStudy": True,
            "cancerTypeId": "mixed"
        }
    ]


@pytest.fixture
def mock_samples_response():
    """Sample cBioPortal samples response."""
    return [
        {
            "sampleId": "TCGA-OR-A5J1-01",
            "patientId": "TCGA-OR-A5J1",
            "studyId": "tcga_pan_can_atlas_2018",
            "sampleType": "Primary Solid Tumor"
        },
        {
            "sampleId": "TCGA-OR-A5J2-01",
            "patientId": "TCGA-OR-A5J2",
            "studyId": "tcga_pan_can_atlas_2018",
            "sampleType": "Primary Solid Tumor"
        }
    ]


@pytest.fixture
def mock_patients_response():
    """Sample cBioPortal patients response."""
    return [
        {
            "patientId": "TCGA-OR-A5J1",
            "studyId": "tcga_pan_can_atlas_2018",
            "sex": "FEMALE",
            "age": 65
        },
        {
            "patientId": "TCGA-OR-A5J2",
            "studyId": "tcga_pan_can_atlas_2018",
            "sex": "MALE",
            "age": 58
        }
    ]


@pytest.fixture
def mock_molecular_profiles_response():
    """Sample cBioPortal molecular profiles response."""
    return [
        {
            "molecularProfileId": "tcga_pan_can_atlas_2018_mutations",
            "studyId": "tcga_pan_can_atlas_2018",
            "molecularAlterationType": "MUTATION_EXTENDED",
            "datatype": "MAF",
            "name": "Mutations",
            "description": "Mutation data from whole exome sequencing",
            "showProfileInAnalysisTab": True
        },
        {
            "molecularProfileId": "tcga_pan_can_atlas_2018_gistic",
            "studyId": "tcga_pan_can_atlas_2018",
            "molecularAlterationType": "COPY_NUMBER_ALTERATION",
            "datatype": "DISCRETE",
            "name": "Putative copy-number alterations from GISTIC"
        }
    ]


@pytest.fixture
def mock_mutations_response():
    """Sample cBioPortal mutations response."""
    return [
        {
            "sampleId": "TCGA-OR-A5J1-01",
            "patientId": "TCGA-OR-A5J1",
            "studyId": "tcga_pan_can_atlas_2018",
            "entrezGeneId": 7157,
            "gene": {
                "entrezGeneId": 7157,
                "hugoGeneSymbol": "TP53",
                "type": "protein-coding"
            },
            "mutationType": "Missense_Mutation",
            "proteinChange": "p.R273H",
            "variantType": "SNP",
            "chr": "17",
            "startPosition": 7577539,
            "endPosition": 7577539,
            "referenceAllele": "G",
            "variantAllele": "A",
            "mutationStatus": "Somatic"
        }
    ]


@pytest.fixture
def mock_gene_expression_response():
    """Sample cBioPortal gene expression response."""
    return [
        {
            "sampleId": "TCGA-OR-A5J1-01",
            "patientId": "TCGA-OR-A5J1",
            "studyId": "tcga_pan_can_atlas_2018",
            "entrezGeneId": 7157,
            "molecularProfileId": "tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
            "value": 2.456
        },
        {
            "sampleId": "TCGA-OR-A5J2-01",
            "patientId": "TCGA-OR-A5J2",
            "studyId": "tcga_pan_can_atlas_2018",
            "entrezGeneId": 7157,
            "molecularProfileId": "tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
            "value": 1.234
        }
    ]


@pytest.fixture
def mock_clinical_data_response():
    """Sample cBioPortal clinical data response."""
    return [
        {
            "clinicalAttributeId": "CANCER_TYPE",
            "value": "Breast Invasive Carcinoma",
            "sampleId": "TCGA-OR-A5J1-01",
            "patientId": "TCGA-OR-A5J1",
            "studyId": "tcga_pan_can_atlas_2018"
        },
        {
            "clinicalAttributeId": "OVERALL_SURVIVAL_MONTHS",
            "value": "24.5",
            "patientId": "TCGA-OR-A5J1",
            "studyId": "tcga_pan_can_atlas_2018"
        }
    ]


@pytest.fixture
def mock_gene_response():
    """Sample cBioPortal gene response."""
    return {
        "entrezGeneId": 7157,
        "hugoGeneSymbol": "TP53",
        "type": "protein-coding",
        "cytoband": "17p13.1",
        "length": 19149
    }


@pytest.fixture
def mock_cancer_types_response():
    """Sample cBioPortal cancer types response."""
    return [
        {
            "cancerTypeId": "brca",
            "name": "Invasive Breast Carcinoma",
            "dedicatedColor": "HotPink",
            "shortName": "BRCA",
            "parent": "breast"
        },
        {
            "cancerTypeId": "luad",
            "name": "Lung Adenocarcinoma",
            "dedicatedColor": "Gainsboro",
            "shortName": "LUAD",
            "parent": "lung"
        }
    ]


class TestcBioPortalClient:
    """Test cBioPortal API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = cBioPortalClient()
        assert "cbioportal.org/api" in client.base_url
        assert client.rate_limit == 10
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_all_studies(self, mock_get, mock_studies_response):
        """Test getting all studies."""
        mock_response = Mock()
        mock_response.json.return_value = mock_studies_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_all_studies()
        
        mock_get.assert_called_once_with(
            "/studies",
            params={"projection": "SUMMARY"}
        )
        assert result == mock_studies_response
        assert len(result) == 2
        assert result[0]["studyId"] == "tcga_pan_can_atlas_2018"
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_all_studies_with_keyword(self, mock_get, mock_studies_response):
        """Test getting studies with keyword search."""
        mock_response = Mock()
        mock_response.json.return_value = mock_studies_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_all_studies(keyword="TCGA", projection="DETAILED")
        
        mock_get.assert_called_once_with(
            "/studies",
            params={"projection": "DETAILED", "keyword": "TCGA"}
        )
        assert result == mock_studies_response
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_study(self, mock_get, mock_studies_response):
        """Test getting a specific study."""
        mock_response = Mock()
        mock_response.json.return_value = mock_studies_response[0]
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_study("tcga_pan_can_atlas_2018")
        
        mock_get.assert_called_once_with("/studies/tcga_pan_can_atlas_2018")
        assert result["studyId"] == "tcga_pan_can_atlas_2018"
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_samples_in_study(self, mock_get, mock_samples_response):
        """Test getting samples in a study."""
        mock_response = Mock()
        mock_response.json.return_value = mock_samples_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_samples_in_study("tcga_pan_can_atlas_2018")
        
        mock_get.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/samples",
            params={"projection": "SUMMARY"}
        )
        assert result == mock_samples_response
        assert len(result) == 2
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_sample(self, mock_get, mock_samples_response):
        """Test getting a specific sample."""
        mock_response = Mock()
        mock_response.json.return_value = mock_samples_response[0]
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_sample("tcga_pan_can_atlas_2018", "TCGA-OR-A5J1-01")
        
        mock_get.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/samples/TCGA-OR-A5J1-01"
        )
        assert result["sampleId"] == "TCGA-OR-A5J1-01"
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_patients_in_study(self, mock_get, mock_patients_response):
        """Test getting patients in a study."""
        mock_response = Mock()
        mock_response.json.return_value = mock_patients_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_patients_in_study("tcga_pan_can_atlas_2018")
        
        mock_get.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/patients",
            params={"projection": "SUMMARY"}
        )
        assert result == mock_patients_response
        assert len(result) == 2
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_patient(self, mock_get, mock_patients_response):
        """Test getting a specific patient."""
        mock_response = Mock()
        mock_response.json.return_value = mock_patients_response[0]
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_patient("tcga_pan_can_atlas_2018", "TCGA-OR-A5J1")
        
        mock_get.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/patients/TCGA-OR-A5J1"
        )
        assert result["patientId"] == "TCGA-OR-A5J1"
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_molecular_profiles(self, mock_get, mock_molecular_profiles_response):
        """Test getting molecular profiles."""
        mock_response = Mock()
        mock_response.json.return_value = mock_molecular_profiles_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_molecular_profiles("tcga_pan_can_atlas_2018")
        
        mock_get.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/molecular-profiles",
            params={"projection": "SUMMARY"}
        )
        assert result == mock_molecular_profiles_response
        assert len(result) == 2
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_mutations_in_molecular_profile(self, mock_get, mock_mutations_response):
        """Test getting mutations."""
        mock_response = Mock()
        mock_response.json.return_value = mock_mutations_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_mutations_in_molecular_profile(
            "tcga_pan_can_atlas_2018_mutations",
            sample_list_id="sample_list_1",
            entrez_gene_ids=[7157]
        )
        
        mock_get.assert_called_once_with(
            "/molecular-profiles/tcga_pan_can_atlas_2018_mutations/mutations",
            params={
                "projection": "SUMMARY",
                "sampleListId": "sample_list_1",
                "entrezGeneId": "7157"
            }
        )
        assert result == mock_mutations_response
    
    @patch.object(cBioPortalClient, 'post')
    def test_fetch_mutations(self, mock_post, mock_mutations_response):
        """Test fetching mutations for specific samples."""
        mock_response = Mock()
        mock_response.json.return_value = mock_mutations_response
        mock_post.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.fetch_mutations(
            "tcga_pan_can_atlas_2018_mutations",
            ["TCGA-OR-A5J1-01"],
            entrez_gene_ids=[7157]
        )
        
        mock_post.assert_called_once_with(
            "/molecular-profiles/tcga_pan_can_atlas_2018_mutations/mutations/fetch",
            json={
                "sampleIds": ["TCGA-OR-A5J1-01"],
                "entrezGeneIds": [7157]
            }
        )
        assert result == mock_mutations_response
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_gene_expression_data(self, mock_get, mock_gene_expression_response):
        """Test getting gene expression data."""
        mock_response = Mock()
        mock_response.json.return_value = mock_gene_expression_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_gene_expression_data(
            "tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
            "sample_list_1",
            entrez_gene_ids=[7157]
        )
        
        mock_get.assert_called_once_with(
            "/molecular-profiles/tcga_pan_can_atlas_2018_rna_seq_v2_mrna/molecular-data",
            params={
                "sampleListId": "sample_list_1",
                "entrezGeneId": "7157"
            }
        )
        assert result == mock_gene_expression_response
    
    @patch.object(cBioPortalClient, 'post')
    def test_fetch_molecular_data(self, mock_post, mock_gene_expression_response):
        """Test fetching molecular data."""
        mock_response = Mock()
        mock_response.json.return_value = mock_gene_expression_response
        mock_post.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.fetch_molecular_data(
            "tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
            ["TCGA-OR-A5J1-01", "TCGA-OR-A5J2-01"]
        )
        
        mock_post.assert_called_once_with(
            "/molecular-profiles/tcga_pan_can_atlas_2018_rna_seq_v2_mrna/molecular-data/fetch",
            json={"sampleIds": ["TCGA-OR-A5J1-01", "TCGA-OR-A5J2-01"]}
        )
        assert result == mock_gene_expression_response
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_clinical_data_in_study(self, mock_get, mock_clinical_data_response):
        """Test getting clinical data."""
        mock_response = Mock()
        mock_response.json.return_value = mock_clinical_data_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_clinical_data_in_study(
            "tcga_pan_can_atlas_2018",
            clinical_data_type="SAMPLE"
        )
        
        mock_get.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/clinical-data",
            params={
                "clinicalDataType": "SAMPLE",
                "projection": "SUMMARY"
            }
        )
        assert result == mock_clinical_data_response
    
    @patch.object(cBioPortalClient, 'post')
    def test_fetch_clinical_data(self, mock_post, mock_clinical_data_response):
        """Test fetching clinical data."""
        mock_response = Mock()
        mock_response.json.return_value = mock_clinical_data_response
        mock_post.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.fetch_clinical_data(
            "tcga_pan_can_atlas_2018",
            sample_ids=["TCGA-OR-A5J1-01"]
        )
        
        mock_post.assert_called_once_with(
            "/studies/tcga_pan_can_atlas_2018/clinical-data/fetch",
            json={"sampleIds": ["TCGA-OR-A5J1-01"]}
        )
        assert result == mock_clinical_data_response
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_gene(self, mock_get, mock_gene_response):
        """Test getting gene information."""
        mock_response = Mock()
        mock_response.json.return_value = mock_gene_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_gene(7157)
        
        mock_get.assert_called_once_with("/genes/7157")
        assert result == mock_gene_response
        assert result["hugoGeneSymbol"] == "TP53"
    
    @patch.object(cBioPortalClient, 'post')
    def test_fetch_genes(self, mock_post):
        """Test fetching multiple genes."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"entrezGeneId": 7157, "hugoGeneSymbol": "TP53"},
            {"entrezGeneId": 675, "hugoGeneSymbol": "BRCA2"}
        ]
        mock_post.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.fetch_genes(hugo_symbols=["TP53", "BRCA2"])
        
        mock_post.assert_called_once_with(
            "/genes/fetch",
            json={"hugoGeneSymbols": ["TP53", "BRCA2"]}
        )
        assert len(result) == 2
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_cancer_types(self, mock_get, mock_cancer_types_response):
        """Test getting cancer types."""
        mock_response = Mock()
        mock_response.json.return_value = mock_cancer_types_response
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_cancer_types()
        
        mock_get.assert_called_once_with("/cancer-types")
        assert result == mock_cancer_types_response
        assert len(result) == 2
    
    @patch.object(cBioPortalClient, 'get')
    def test_get_server_status(self, mock_get):
        """Test getting server status."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "version": "5.0.0",
            "db_version": "2.13.1",
            "git_branch": "master",
            "git_commit": "abc123"
        }
        mock_get.return_value = mock_response
        
        client = cBioPortalClient()
        result = client.get_server_status()
        
        mock_get.assert_called_once_with("/info")
        assert "version" in result