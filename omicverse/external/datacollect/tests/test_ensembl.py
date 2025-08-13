"""Tests for Ensembl API client and collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from omicverse.external.datacollect.api.ensembl import EnsemblClient
from omicverse.external.datacollect.collectors.ensembl_collector import EnsemblCollector
from omicverse.external.datacollect.models.genomic import Gene, Variant
from omicverse.external.datacollect.models.protein import Protein


@pytest.fixture
def mock_ensembl_gene_response():
    """Sample Ensembl gene response."""
    return {
        "id": "ENSG00000141510",
        "display_name": "TP53",
        "description": "tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]",
        "species": "homo_sapiens",
        "assembly_name": "GRCh38",
        "source": "ensembl_havana",
        "logic_name": "ensembl_havana_gene",
        "version": 14,
        "canonical_transcript": "ENST00000269305.9",
        "start": 7661779,
        "end": 7687538,
        "strand": -1,
        "seq_region_name": "17",
        "biotype": "protein_coding",
        "Transcript": [
            {
                "id": "ENST00000269305",
                "version": 9,
                "biotype": "protein_coding",
                "Translation": {
                    "id": "ENSP00000269305",
                    "version": 4,
                    "length": 393
                }
            }
        ]
    }


@pytest.fixture
def mock_ensembl_xrefs_response():
    """Sample Ensembl cross-references response."""
    return [
        {
            "primary_id": "P04637",
            "display_id": "P04637",
            "version": "",
            "description": "",
            "dbname": "UniProtKB/Swiss-Prot",
            "synonyms": [],
            "info_type": "DIRECT",
            "info_text": ""
        },
        {
            "primary_id": "11998",
            "display_id": "TP53",
            "version": "",
            "description": "tumor protein p53",
            "dbname": "HGNC",
            "synonyms": ["p53"],
            "info_type": "DIRECT",
            "info_text": ""
        }
    ]


@pytest.fixture
def mock_ensembl_variant_response():
    """Sample Ensembl variant response."""
    return {
        "name": "rs28934578",
        "source": "Variants (including SNPs and indels) imported from dbSNP",
        "MAF": 0.0002,
        "minor_allele": "A",
        "ambiguity": "Y",
        "var_class": "SNP",
        "synonyms": [],
        "evidence": ["Multiple_observations", "1000Genomes", "ESP", "gnomAD"],
        "ancestral_allele": "C",
        "most_severe_consequence": "missense_variant",
        "mappings": [
            {
                "location": "17:7676154-7676154",
                "assembly_name": "GRCh38",
                "end": 7676154,
                "start": 7676154,
                "strand": 1,
                "coord_system": "chromosome",
                "allele_string": "C/A",
                "ancestral_allele": "C",
                "seq_region_name": "17"
            }
        ]
    }


class TestEnsemblClient:
    """Test Ensembl API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = EnsemblClient()
        assert "rest.ensembl.org" in client.base_url
        assert client.rate_limit == 15
    
    @patch.object(EnsemblClient, 'get')
    def test_lookup_id(self, mock_get, mock_ensembl_gene_response):
        """Test ID lookup."""
        mock_response = Mock()
        mock_response.json.return_value = mock_ensembl_gene_response
        mock_get.return_value = mock_response
        
        client = EnsemblClient()
        result = client.lookup_id("ENSG00000141510", expand=True)
        
        mock_get.assert_called_once_with(
            "/lookup/id/ENSG00000141510",
            params={"expand": 1}
        )
        assert result["id"] == "ENSG00000141510"
        assert result["display_name"] == "TP53"
        assert len(result["Transcript"]) == 1
    
    @patch.object(EnsemblClient, 'get')
    def test_lookup_symbol(self, mock_get, mock_ensembl_gene_response):
        """Test symbol lookup."""
        mock_response = Mock()
        mock_response.json.return_value = mock_ensembl_gene_response
        mock_get.return_value = mock_response
        
        client = EnsemblClient()
        result = client.lookup_symbol("TP53", "human")
        
        mock_get.assert_called_once_with(
            "/lookup/symbol/human/TP53",
            params={}
        )
        assert result["display_name"] == "TP53"
    
    @patch.object(EnsemblClient, 'get')
    def test_get_sequence(self, mock_get):
        """Test sequence retrieval."""
        mock_response = Mock()
        mock_response.text = ">ENSP00000269305\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
        mock_response.json.return_value = {"seq": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"}
        mock_get.return_value = mock_response
        
        client = EnsemblClient()
        
        # Test FASTA format
        result = client.get_sequence("ENSP00000269305", "protein", "fasta")
        assert ">ENSP00000269305" in result
        assert "MEEPQSDPSV" in result
        
        # Test JSON format
        result = client.get_sequence("ENSP00000269305", "protein", "json")
        assert "seq" in result
    
    @patch.object(EnsemblClient, 'get')
    def test_get_variant(self, mock_get, mock_ensembl_variant_response):
        """Test variant retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = mock_ensembl_variant_response
        mock_get.return_value = mock_response
        
        client = EnsemblClient()
        result = client.get_variant("human", "rs28934578")
        
        mock_get.assert_called_once_with("/variation/human/rs28934578")
        assert result["name"] == "rs28934578"
        assert result["var_class"] == "SNP"
        assert result["MAF"] == 0.0002
    
    @patch.object(EnsemblClient, 'get')
    def test_get_xrefs(self, mock_get, mock_ensembl_xrefs_response):
        """Test cross-references."""
        mock_response = Mock()
        mock_response.json.return_value = mock_ensembl_xrefs_response
        mock_get.return_value = mock_response
        
        client = EnsemblClient()
        result = client.get_xrefs("ENSG00000141510")
        
        mock_get.assert_called_once_with(
            "/xrefs/id/ENSG00000141510",
            params={}
        )
        assert len(result) == 2
        assert any(xref["dbname"] == "UniProtKB/Swiss-Prot" for xref in result)
        assert any(xref["dbname"] == "HGNC" for xref in result)
    
    @patch.object(EnsemblClient, 'get')
    def test_get_homology(self, mock_get):
        """Test homology retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "ENSG00000141510",
                    "homologies": [
                        {
                            "source": {"id": "ENSG00000141510"},
                            "target": {"id": "ENSMUSG00000059552", "species": "mus_musculus"},
                            "type": "ortholog_one2one"
                        }
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = EnsemblClient()
        result = client.get_homology("ENSG00000141510")
        
        assert "data" in result
        assert len(result["data"][0]["homologies"]) == 1


class TestEnsemblCollector:
    """Test Ensembl collector."""
    
    @patch('src.collectors.ensembl_collector.EnsemblClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = EnsemblCollector()
        mock_client_class.assert_called_once()
        assert collector.default_species == "human"
    
    def test_collect_gene(self, mock_ensembl_gene_response, mock_ensembl_xrefs_response):
        """Test collecting gene data."""
        collector = EnsemblCollector()
        
        with patch.object(collector.api_client, 'lookup_id') as mock_lookup:
            with patch.object(collector.api_client, 'get_xrefs') as mock_xrefs:
                with patch.object(collector.api_client, 'get_homology') as mock_homology:
                    with patch.object(collector.api_client, 'get_sequence') as mock_seq:
                        mock_lookup.return_value = mock_ensembl_gene_response
                        mock_xrefs.return_value = mock_ensembl_xrefs_response
                        mock_homology.return_value = {"data": []}
                        mock_seq.return_value = {"id": "ENSG00000141510", "seq": "ATCG"}
                        
                        result = collector.collect_gene("ENSG00000141510")
                        
                        assert result["gene_id"] == "ENSG00000141510"
                        assert result["gene_info"]["display_name"] == "TP53"
                        assert len(result["xrefs"]) == 2
                        assert len(result["transcripts"]) == 1
                        assert len(result["proteins"]) == 1
    
    def test_collect_gene_by_symbol(self):
        """Test collecting gene by symbol."""
        collector = EnsemblCollector()
        
        with patch.object(collector.api_client, 'lookup_symbol') as mock_lookup:
            with patch.object(collector, 'collect_gene') as mock_collect:
                mock_lookup.return_value = {"id": "ENSG00000141510"}
                mock_collect.return_value = {"gene_id": "ENSG00000141510"}
                
                result = collector.collect_gene_by_symbol("TP53", "human")
                
                mock_lookup.assert_called_once_with("TP53", "human", expand=True)
                mock_collect.assert_called_once_with("ENSG00000141510")
    
    def test_save_gene_to_database(self, test_db, mock_ensembl_xrefs_response):
        """Test saving gene to database."""
        collector = EnsemblCollector(db_session=test_db)
        
        data = {
            "gene_id": "ENSG00000141510",
            "gene_info": {
                "display_name": "TP53",
                "description": "tumor protein p53",
                "seq_region_name": "17",
                "start": 7661779,
                "end": 7687538,
                "strand": -1,
                "biotype": "protein_coding",
                "species": "homo_sapiens"
            },
            "xrefs": mock_ensembl_xrefs_response,
            "transcripts": [{"id": "ENST00000269305"}],
            "proteins": [{"id": "ENSP00000269305"}]
        }
        
        gene = collector.save_gene_to_database(data)
        
        assert isinstance(gene, Gene)
        assert gene.ensembl_id == "ENSG00000141510"
        assert gene.symbol == "TP53"
        assert gene.chromosome == "17"
        assert gene.start_position == 7661779
        assert gene.end_position == 7687538
        assert gene.transcript_count == 1
        assert gene.protein_count == 1
        assert gene.hgnc_id == "11998"
        assert "P04637" in gene.uniprot_ids
    
    def test_collect_variant(self, mock_ensembl_variant_response):
        """Test collecting variant data."""
        collector = EnsemblCollector()
        
        with patch.object(collector.api_client, 'get_variant') as mock_variant:
            mock_variant.return_value = mock_ensembl_variant_response
            
            result = collector.collect_variant("human", "rs28934578")
            
            assert result["variant_id"] == "rs28934578"
            assert result["species"] == "human"
            assert result["most_severe_consequence"] == "missense_variant"
            assert result["minor_allele"] == "A"
            assert result["minor_allele_freq"] == 0.0002
    
    def test_save_variant_to_database(self, test_db, mock_ensembl_variant_response):
        """Test saving variant to database."""
        collector = EnsemblCollector(db_session=test_db)
        
        data = {
            "variant_id": "rs28934578",
            "species": "human",
            "info": mock_ensembl_variant_response,
            "mappings": mock_ensembl_variant_response["mappings"],
            "most_severe_consequence": "missense_variant",
            "minor_allele": "A",
            "minor_allele_freq": 0.0002
        }
        
        variant = collector.save_variant_to_database(data)
        
        assert isinstance(variant, Variant)
        assert variant.rsid == "rs28934578"
        assert variant.variant_type == "SNP"
        assert variant.chromosome == "17"
        assert variant.position == 7676154
        assert variant.reference_allele == "C"
        assert variant.alternative_allele == "A"
        assert variant.minor_allele_frequency == 0.0002
        assert variant.clinical_significance == "missense_variant"
    
    @patch.object(EnsemblCollector, 'save_gene_to_database')
    def test_link_gene_to_proteins(self, mock_save, test_db):
        """Test linking genes to proteins."""
        collector = EnsemblCollector(db_session=test_db)
        
        # Create test protein
        protein = Protein(
            id="test_protein",
            accession="P04637",
            protein_name="TP53",
            sequence="MVLSEGEWQLVLHVWAK",
            sequence_length=17,
            source="test"
        )
        test_db.add(protein)
        test_db.commit()
        
        # Create test gene
        gene = Gene(
            id="test_gene",
            gene_id="ENSG00000141510",
            ensembl_id="ENSG00000141510",
            symbol="TP53",
            source="test"
        )
        
        data = {
            "xrefs": [
                {"dbname": "UniProtKB/Swiss-Prot", "primary_id": "P04637"}
            ]
        }
        
        collector._link_gene_to_proteins(gene, data)
        
        # Check protein was updated
        test_db.refresh(protein)
        assert protein.gene_name == "TP53"
        assert protein.ensembl_gene_id == "ENSG00000141510"