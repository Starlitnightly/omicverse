"""Tests for GWAS Catalog API client."""

import pytest
from unittest.mock import Mock, patch
import responses
import requests

from omicverse.external.datacollect.api.gwas_catalog import GWASCatalogClient


class TestGWASCatalogClient:
    """Test GWAS Catalog API client."""

    def test_initialization(self):
        """Test client initialization."""
        client = GWASCatalogClient()
        assert client.base_url == "https://www.ebi.ac.uk/gwas/rest/api"
        assert client.rate_limit == 10
        
        # Test custom initialization
        client_custom = GWASCatalogClient(
            base_url="https://custom.gwas.org/api",
            rate_limit=5
        )
        assert client_custom.base_url == "https://custom.gwas.org/api"
        assert client_custom.rate_limit == 5

    def test_get_default_headers(self):
        """Test default headers."""
        client = GWASCatalogClient()
        headers = client.get_default_headers()
        
        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"

    @patch.object(GWASCatalogClient, 'get')
    def test_get_study(self, mock_get):
        """Test getting study information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "accessionId": "GCST000001",
            "fullPvalueset": True,
            "gxe": False,
            "gxg": False,
            "snpCount": 2173762,
            "qualifier": "",
            "imputed": True,
            "pooled": False,
            "studyDesignComment": "",
            "publicationInfo": {
                "pubmedId": "17463246",
                "publication": {
                    "title": "Genome-wide association study of 14,000 cases of seven common diseases",
                    "author": {
                        "fullname": "Burton PR",
                        "orcid": ""
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_study("GCST000001")
        
        mock_get.assert_called_once_with("/studies/GCST000001")
        assert result["accessionId"] == "GCST000001"
        assert result["snpCount"] == 2173762

    @patch.object(GWASCatalogClient, 'get')
    def test_search_studies(self, mock_get):
        """Test searching studies."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "studies": [
                    {
                        "accessionId": "GCST000001",
                        "fullPvalueset": True,
                        "publicationInfo": {
                            "pubmedId": "17463246"
                        }
                    }
                ]
            },
            "page": {
                "size": 20,
                "totalElements": 1,
                "totalPages": 1,
                "number": 0
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.search_studies("diabetes", page=0, size=20)
        
        mock_get.assert_called_once_with(
            "/studies/search",
            params={"q": "diabetes", "page": 0, "size": 20}
        )
        assert len(result["_embedded"]["studies"]) == 1
        assert result["page"]["totalElements"] == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_associations_all_filters(self, mock_get):
        """Test getting associations with all filters."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "associations": [
                    {
                        "id": "15608",
                        "riskFrequency": "0.14",
                        "pvalueMantissa": 2,
                        "pvalueExponent": -11,
                        "pvalueDescription": "",
                        "multiSnpHaplotype": False,
                        "snpInteraction": False,
                        "snpType": "known",
                        "standardError": 0.0305,
                        "range": "",
                        "orPerCopyNum": 1.18,
                        "orPerCopyRecip": 0.85,
                        "orPerCopyRecipRange": "",
                        "betaNum": 0.166,
                        "betaUnit": "",
                        "betaDirection": "increase"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_associations(
            study_id="GCST000001",
            trait="diabetes",
            gene="TCF7L2",
            variant="rs7903146"
        )
        
        mock_get.assert_called_once_with(
            "/associations",
            params={
                "page": 0,
                "size": 20,
                "studyAccessionId": "GCST000001",
                "efoTrait": "diabetes",
                "geneName": "TCF7L2",
                "rsId": "rs7903146"
            }
        )
        assert len(result["_embedded"]["associations"]) == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_associations_no_filters(self, mock_get):
        """Test getting associations with no filters."""
        mock_response = Mock()
        mock_response.json.return_value = {"_embedded": {"associations": []}}
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_associations(page=1, size=50)
        
        mock_get.assert_called_once_with(
            "/associations",
            params={"page": 1, "size": 50}
        )

    @patch.object(GWASCatalogClient, 'get')
    def test_get_association(self, mock_get):
        """Test getting specific association."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "15608",
            "riskFrequency": "0.14",
            "pvalueMantissa": 2,
            "pvalueExponent": -11,
            "orPerCopyNum": 1.18
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_association("15608")
        
        mock_get.assert_called_once_with("/associations/15608")
        assert result["id"] == "15608"
        assert result["orPerCopyNum"] == 1.18

    @patch.object(GWASCatalogClient, 'get')
    def test_get_single_nucleotide_polymorphisms(self, mock_get):
        """Test getting SNP information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "rsId": "rs7903146",
            "merged": 0,
            "functionalClass": "intron_variant",
            "lastUpdateDate": "2023-10-23T00:00:00.000+0000",
            "locations": [
                {
                    "chromosomeName": "10",
                    "chromosomePosition": 112998590,
                    "region": {
                        "name": "10q25.2"
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_single_nucleotide_polymorphisms("rs7903146")
        
        mock_get.assert_called_once_with("/singleNucleotidePolymorphisms/rs7903146")
        assert result["rsId"] == "rs7903146"
        assert result["functionalClass"] == "intron_variant"

    @patch.object(GWASCatalogClient, 'get')
    def test_search_snps(self, mock_get):
        """Test searching SNPs."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "singleNucleotidePolymorphisms": [
                    {
                        "rsId": "rs7903146",
                        "functionalClass": "intron_variant"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.search_snps("10:112998590")
        
        mock_get.assert_called_once_with(
            "/singleNucleotidePolymorphisms/search",
            params={"q": "10:112998590", "page": 0, "size": 20}
        )
        assert len(result["_embedded"]["singleNucleotidePolymorphisms"]) == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_trait_with_uri(self, mock_get):
        """Test getting trait information with URI."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "trait": "type II diabetes mellitus",
            "uri": "http://www.ebi.ac.uk/efo/EFO_0001360",
            "shortForm": "EFO_0001360"
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_trait("http://www.ebi.ac.uk/efo/EFO_0001360")
        
        mock_get.assert_called_once_with("/efoTraits/EFO_0001360")
        assert result["trait"] == "type II diabetes mellitus"

    @patch.object(GWASCatalogClient, 'get')
    def test_get_trait_with_id(self, mock_get):
        """Test getting trait information with ID."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "trait": "type II diabetes mellitus",
            "shortForm": "EFO_0001360"
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_trait("EFO_0001360")
        
        mock_get.assert_called_once_with("/efoTraits/EFO_0001360")

    @patch.object(GWASCatalogClient, 'get')
    def test_search_traits(self, mock_get):
        """Test searching traits."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "efoTraits": [
                    {
                        "trait": "type II diabetes mellitus",
                        "shortForm": "EFO_0001360"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.search_traits("diabetes")
        
        mock_get.assert_called_once_with(
            "/efoTraits/search",
            params={"q": "diabetes", "page": 0, "size": 20}
        )
        assert len(result["_embedded"]["efoTraits"]) == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_genes_by_symbol(self, mock_get):
        """Test getting gene information by symbol."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "genes": [
                    {
                        "geneName": "TCF7L2",
                        "entrezGeneId": "6934",
                        "ensemblGeneId": "ENSG00000148737",
                        "genomicContexts": []
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_genes_by_symbol("TCF7L2")
        
        mock_get.assert_called_once_with(
            "/genes/search/findByGeneName",
            params={"geneName": "TCF7L2"}
        )
        assert len(result["_embedded"]["genes"]) == 1
        assert result["_embedded"]["genes"][0]["geneName"] == "TCF7L2"

    @patch.object(GWASCatalogClient, 'get')
    def test_get_ancestry(self, mock_get):
        """Test getting ancestry information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "ancestralGroup": "European",
            "countryOfOrigin": "United Kingdom",
            "countryOfRecruitment": "United Kingdom",
            "sampleDescription": "European ancestry",
            "numberOfIndividuals": 1924
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_ancestry("1234")
        
        mock_get.assert_called_once_with("/ancestries/1234")
        assert result["ancestralGroup"] == "European"
        assert result["numberOfIndividuals"] == 1924

    @patch.object(GWASCatalogClient, 'get')
    def test_get_publications(self, mock_get):
        """Test getting publication information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "pubmedId": "17463246",
            "title": "Genome-wide association study of 14,000 cases",
            "author": {
                "fullname": "Burton PR"
            },
            "publicationDate": "2007-06-07T00:00:00.000+0000"
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_publications("17463246")
        
        mock_get.assert_called_once_with("/publications/17463246")
        assert result["pubmedId"] == "17463246"
        assert result["author"]["fullname"] == "Burton PR"

    @patch.object(GWASCatalogClient, 'get')
    def test_get_genomic_contexts(self, mock_get):
        """Test getting genomic contexts."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "genomicContexts": [
                    {
                        "chromosome": {
                            "chromosomeName": "10"
                        },
                        "distance": 0,
                        "isIntergenic": False,
                        "isUpstream": False,
                        "isDownstream": False
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_genomic_contexts("10", 112990000, 113000000)
        
        mock_get.assert_called_once_with(
            "/genomicContexts",
            params={
                "chromosome": "10",
                "start": 112990000,
                "end": 113000000,
                "page": 0,
                "size": 20
            }
        )
        assert len(result["_embedded"]["genomicContexts"]) == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_risk_alleles(self, mock_get):
        """Test getting risk alleles."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "riskAlleles": [
                    {
                        "riskAlleleName": "rs7903146-T",
                        "riskFrequency": "0.14",
                        "genomeWide": True,
                        "limitedList": False
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_risk_alleles("15608")
        
        mock_get.assert_called_once_with("/associations/15608/riskAlleles")
        assert len(result["_embedded"]["riskAlleles"]) == 1
        assert result["_embedded"]["riskAlleles"][0]["riskAlleleName"] == "rs7903146-T"

    @patch.object(GWASCatalogClient, 'get')
    def test_get_loci(self, mock_get):
        """Test getting loci."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "loci": [
                    {
                        "haplotypeSnpCount": 1,
                        "description": "Single SNP",
                        "strongestRiskAlleles": []
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_loci("15608")
        
        mock_get.assert_called_once_with("/associations/15608/loci")
        assert len(result["_embedded"]["loci"]) == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_summary_statistics(self, mock_get):
        """Test getting summary statistics."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "associations": [
                    {
                        "id": "15608",
                        "summaryStatistics": {
                            "file": "GCST000001_summary_statistics.tsv",
                            "downloadLink": "https://example.com/download"
                        }
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_summary_statistics("GCST000001")
        
        mock_get.assert_called_once_with(
            "/studies/GCST000001/associations",
            params={"projection": "summaryStatistics"}
        )
        assert len(result["_embedded"]["associations"]) == 1

    @patch.object(GWASCatalogClient, 'get')
    def test_get_top_associations_by_trait(self, mock_get):
        """Test getting top associations by trait."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "_embedded": {
                "associations": [
                    {
                        "id": "15608",
                        "pvalueMantissa": 2,
                        "pvalueExponent": -11,
                        "orPerCopyNum": 1.18
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = GWASCatalogClient()
        result = client.get_top_associations_by_trait("diabetes", p_value_threshold=1e-8)
        
        mock_get.assert_called_once_with(
            "/associations/search/findByEfoTrait",
            params={
                "efoTrait": "diabetes",
                "pvalueMantissaLessThan": 1e-8,
                "size": 100,
                "sort": "pvalueMantissa,asc"
            }
        )
        assert len(result["_embedded"]["associations"]) == 1

    @responses.activate
    def test_http_error_handling(self):
        """Test HTTP error handling."""
        responses.add(
            responses.GET,
            "https://www.ebi.ac.uk/gwas/rest/api/studies/INVALID",
            status=404
        )
        
        client = GWASCatalogClient()
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_study("INVALID")

    @responses.activate
    def test_successful_http_request(self):
        """Test successful HTTP request."""
        study_data = {
            "accessionId": "GCST000001",
            "snpCount": 2173762
        }
        
        responses.add(
            responses.GET,
            "https://www.ebi.ac.uk/gwas/rest/api/studies/GCST000001",
            json=study_data,
            status=200
        )
        
        client = GWASCatalogClient()
        result = client.get_study("GCST000001")
        
        assert result["accessionId"] == "GCST000001"
        assert result["snpCount"] == 2173762